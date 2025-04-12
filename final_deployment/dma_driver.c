/*******************************************************************************
 * Example: AXI DMA MM2S -> S2MM Transfer (Polled) Using IDLE Bit Only
 *
 * Steps:
 *   1. Map AXI DMA registers at AXI_DMA_BASE
 *   2. Map DDR region at 0x00000000..(0x00200000 + 64KB)
 *   3. Copy X_test.bin (64KB) into DDR_SRC_ADDR
 *   4. Configure AXI DMA (MM2S and S2MM) for 64KB
 *   5. Wait for IDLE bit (no interrupts used)
 *   6. Copy result from DDR_DST_ADDR to Y_test.bin
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

// ---------------------------------------------------------------------
// AXI DMA Register Offsets (per Xilinx documentation)
// ---------------------------------------------------------------------
#define MM2S_DMACR      0x00
#define MM2S_DMASR      0x04
#define MM2S_SA         0x18
#define MM2S_LENGTH     0x28

#define S2MM_DMACR      0x30
#define S2MM_DMASR      0x34
#define S2MM_DA         0x48
#define S2MM_LENGTH     0x58

// Status bits
#define STATUS_HALTED   0x00000001
#define STATUS_IDLE     0x00000002
#define STATUS_ERR_IRQ  0x00004000  // just for debug

// Control bits
#define RUN_DMA         0x00000001
#define RESET_DMA       0x00000004
#define HALT_DMA        0x00000000

// ---------------------------------------------------------------------
// Addresses for your design
// ---------------------------------------------------------------------
#define AXI_DMA_BASE    0xA0000000UL  // DMA register base
#define DDR_SRC_ADDR    0x0e000000UL  // input buffer
#define DDR_DST_ADDR    0x0f000000UL  // output buffer
#define TRANSFER_SIZE   65536         // 64KB

// Helper to read/write registers
static inline void write_reg(volatile uint32_t *base, uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)((uintptr_t)base + offset) = value;
}
static inline uint32_t read_reg(volatile uint32_t *base, uint32_t offset)
{
    return *(volatile uint32_t *)((uintptr_t)base + offset);
}

// ---------------------------------------------------------------------
// Print DMA status for debugging
// ---------------------------------------------------------------------
static void print_mm2s_status(volatile uint32_t *dma_base)
{
    uint32_t s = read_reg(dma_base, MM2S_DMASR);
    printf("[MM2S] DMASR = 0x%08X => ", s);
    if (s & STATUS_HALTED)      printf("Halted. ");
    else                        printf("Running. ");
    if (s & STATUS_IDLE)        printf("Idle. ");
    if (s & STATUS_ERR_IRQ)     printf("ErrorIRQ. ");
    printf("\n");
}

static void print_s2mm_status(volatile uint32_t *dma_base)
{
    uint32_t s = read_reg(dma_base, S2MM_DMASR);
    printf("[S2MM] DMASR = 0x%08X => ", s);
    if (s & STATUS_HALTED)      printf("Halted. ");
    else                        printf("Running. ");
    if (s & STATUS_IDLE)        printf("Idle. ");
    if (s & STATUS_ERR_IRQ)     printf("ErrorIRQ. ");
    printf("\n");
}

// ---------------------------------------------------------------------
// Wait for channel to go Idle
// ---------------------------------------------------------------------
static int wait_mm2s_idle(volatile uint32_t *dma_base, int max_tries)
{
    for(int i=0; i<max_tries; i++) {
        uint32_t s = read_reg(dma_base, MM2S_DMASR);
        if (s & STATUS_IDLE) {
            printf("MM2S went Idle after %d polls.\n", i);
            return 0;
        }
        usleep(10000); // 10ms
    }
    printf("ERROR: MM2S not Idle (timeout)!\n");
    return -1;
}

static int wait_s2mm_idle(volatile uint32_t *dma_base, int max_tries)
{
    for(int i=0; i<max_tries; i++) {
        uint32_t s = read_reg(dma_base, S2MM_DMASR);
        if (s & STATUS_IDLE) {
            printf("S2MM went Idle after %d polls.\n", i);
            return 0;
        }
        usleep(10000); // 10ms
    }
    printf("ERROR: S2MM not Idle (timeout)!\n");
    return -1;
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(void)
{
    int fd;
    FILE *f_in=NULL, *f_out=NULL;
    off_t page_size = sysconf(_SC_PAGESIZE);

    printf("=== Polled DMA Example (No IOC Interrupt Check) ===\n");

    // 1) Open /dev/mem
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if(fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    // 2) Map DDR region from 0x00000000..(0x00200000+64KB)
    off_t ddr_base_aligned = (off_t)(DDR_SRC_ADDR & ~(page_size - 1));
    off_t ddr_map_size     = (DDR_DST_ADDR + TRANSFER_SIZE) - ddr_base_aligned;
    if(ddr_map_size <= 0) {
        fprintf(stderr, "Bad DDR map size!\n");
        close(fd);
        return 1;
    }

    void *ddr_map = mmap(NULL, ddr_map_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         fd, ddr_base_aligned);
    if(ddr_map == MAP_FAILED) {
        perror("mmap(DDR)");
        close(fd);
        return 1;
    }

    uintptr_t ddr_virt_base = (uintptr_t)ddr_map;
    uintptr_t ddr_src_ptr   = ddr_virt_base + (DDR_SRC_ADDR - ddr_base_aligned);
    uintptr_t ddr_dst_ptr   = ddr_virt_base + (DDR_DST_ADDR - ddr_base_aligned);

    // 3) Map AXI DMA registers at 0xA0000000
    off_t dma_base_aligned = (off_t)(AXI_DMA_BASE & ~(page_size - 1));
    void *dma_map = mmap(NULL, page_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         fd, dma_base_aligned);
    if(dma_map == MAP_FAILED) {
        perror("mmap(DMA)");
        munmap(ddr_map, ddr_map_size);
        close(fd);
        return 1;
    }
    uintptr_t dma_virt_base = (uintptr_t)dma_map + (AXI_DMA_BASE - dma_base_aligned);

    // 4) Copy X_test.bin into DDR_SRC_ADDR
    f_in = fopen("./data/data_stem_input.bin", "rb");
    if(!f_in) {
        perror("fopen(./data/data_stem_input.bin)");
        goto cleanup;
    }
    size_t bytes_read = fread((void*)ddr_src_ptr, 1, TRANSFER_SIZE, f_in);
    fclose(f_in);
    f_in = NULL;
    printf("Read %zu bytes into DDR@0x%08lX\n", bytes_read, (unsigned long)DDR_SRC_ADDR);

    // Clear destination
    memset((void*)ddr_dst_ptr, 0, TRANSFER_SIZE);

    // 5) Reset DMA
    volatile uint32_t *dma_regs = (volatile uint32_t *)dma_virt_base;
    printf("Resetting DMA...\n");
    write_reg(dma_regs, S2MM_DMACR, RESET_DMA);
    write_reg(dma_regs, MM2S_DMACR, RESET_DMA);
    usleep(1000);

    print_s2mm_status(dma_regs);
    print_mm2s_status(dma_regs);

    // 6) Halt DMA
    printf("Halting DMA...\n");
    write_reg(dma_regs, S2MM_DMACR, HALT_DMA);
    write_reg(dma_regs, MM2S_DMACR, HALT_DMA);
    usleep(1000);

    print_s2mm_status(dma_regs);
    print_mm2s_status(dma_regs);

    // 7) Set addresses
    printf("Setting Source=0x%08lX, Dest=0x%08lX\n",
           (unsigned long)DDR_SRC_ADDR,
           (unsigned long)DDR_DST_ADDR);

    write_reg(dma_regs, S2MM_DA, (uint32_t)DDR_DST_ADDR);
    write_reg(dma_regs, MM2S_SA, (uint32_t)DDR_SRC_ADDR);

    // 8) Run
    printf("Running DMA channels...\n");
    write_reg(dma_regs, S2MM_DMACR, RUN_DMA);
    write_reg(dma_regs, MM2S_DMACR, RUN_DMA);

    print_mm2s_status(dma_regs);
    print_s2mm_status(dma_regs);

    // 9) Transfer length
    printf("Setting transfer size = %d bytes\n", TRANSFER_SIZE);
    write_reg(dma_regs, S2MM_LENGTH, TRANSFER_SIZE);
    write_reg(dma_regs, MM2S_LENGTH, TRANSFER_SIZE);

    // 10) Wait for IDLE
    printf("Waiting for MM2S Idle...\n");
    if (wait_mm2s_idle(dma_regs, 200) != 0) {
        goto cleanup;
    }
    print_mm2s_status(dma_regs);

    printf("Waiting for S2MM Idle...\n");
    if (wait_s2mm_idle(dma_regs, 200) != 0) {
        goto cleanup;
    }
    print_s2mm_status(dma_regs);

    // 11) Write out Y_test.bin
    f_out = fopen("./data/result.bin", "wb");
    if(!f_out) {
        perror("fopen(./data/result.bin)");
        goto cleanup;
    }
    size_t bytes_written = fwrite((void*)ddr_dst_ptr, 1, TRANSFER_SIZE, f_out);
    fclose(f_out);
    f_out = NULL;

    printf("Wrote %zu bytes to ./data/result.bin\n", bytes_written);

cleanup:
    if(f_in) fclose(f_in);
    if(f_out) fclose(f_out);

    if(dma_map && dma_map != MAP_FAILED) {
        munmap(dma_map, page_size);
    }
    if(ddr_map && ddr_map != MAP_FAILED) {
        munmap(ddr_map, ddr_map_size);
    }
    close(fd);
    printf("Done.\n");
    return 0;
}
