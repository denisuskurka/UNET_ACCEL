#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

// AXI DMA register base address
#define AXI_DMA_BASE     0xA0000000UL

// Offsets for AXI DMA registers
#define MM2S_DMACR       0x00  // MM2S control register
#define MM2S_DMASR       0x04  // MM2S status register
#define MM2S_SA          0x18  // MM2S source address (lower 32 bits)
#define MM2S_LENGTH      0x28  // MM2S transfer length
#define S2MM_DMACR       0x30  // S2MM control register
#define S2MM_DMASR       0x34  // S2MM status register
#define S2MM_DA          0x48  // S2MM destination address (lower 32 bits)
#define S2MM_LENGTH      0x58  // S2MM transfer length

// DDR addresses
#define DDR_SRC_ADDR     0x00000000UL  // Where we store X_test.bin
#define DDR_DST_ADDR     0x00200000UL  // Where we want the output (Y_test.bin)
#define TRANSFER_SIZE    65536         // 64 KB

static inline void write_reg(volatile uint32_t *base, uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)((uintptr_t)base + offset) = value;
}

static inline uint32_t read_reg(volatile uint32_t *base, uint32_t offset)
{
    return *(volatile uint32_t *)((uintptr_t)base + offset);
}

int main(int argc, char *argv[])
{
    int fd;
    FILE *f_in, *f_out;
    void *dma_ctrl_base = NULL;
    void *ddr_map_base  = NULL;
    off_t page_size = sysconf(_SC_PAGESIZE);

    printf("Opening /dev/mem...\n");
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // 1) Map DDR region containing both source (0x0) and destination (0x200000).
    //    We'll map at least 0x00200000 + TRANSFER_SIZE to be safe.
    //    Round down to page boundary:
    off_t ddr_base_aligned = (off_t)(DDR_SRC_ADDR & ~(page_size - 1));
    off_t ddr_map_size     = DDR_DST_ADDR + TRANSFER_SIZE - ddr_base_aligned;
    ddr_map_base = mmap(NULL, ddr_map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, ddr_base_aligned);
    if (ddr_map_base == MAP_FAILED) {
        perror("mmap DDR");
        close(fd);
        return 1;
    }

    // 2) Map AXI DMA registers at 0xA0000000
    //    We'll map just one page (4 KB), which is typically enough for the DMA register set.
    off_t dma_base_aligned = (off_t)(AXI_DMA_BASE & ~(page_size - 1));
    dma_ctrl_base = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, dma_base_aligned);
    if (dma_ctrl_base == MAP_FAILED) {
        perror("mmap DMA");
        munmap(ddr_map_base, ddr_map_size);
        close(fd);
        return 1;
    }

    // Calculate the pointer offsets:
    // The pointer to DDR address 0x00000000 inside our mapped region:
    uintptr_t ddr_virt_base = (uintptr_t)ddr_map_base;
    uintptr_t ddr_src_ptr = ddr_virt_base + (DDR_SRC_ADDR - ddr_base_aligned);
    uintptr_t ddr_dst_ptr = ddr_virt_base + (DDR_DST_ADDR - ddr_base_aligned);

    // The pointer to AXI DMA base:
    uintptr_t dma_virt_base = (uintptr_t)dma_ctrl_base + (AXI_DMA_BASE - dma_base_aligned);

    printf("Copying X_test.bin to 0x%08lX in DDR...\n", (unsigned long)DDR_SRC_ADDR);
    f_in = fopen("X_test.bin", "rb");
    if (!f_in) {
        perror("fopen X_test.bin");
        goto cleanup;
    }
    // We expect 64KB in X_test.bin
    size_t bytes_read = fread((void *)ddr_src_ptr, 1, TRANSFER_SIZE, f_in);
    fclose(f_in);
    printf("Read %zu bytes into DDR.\n", bytes_read);

    // Optionally, flush changes to physical memory
    // This is often enough in MAP_SHARED, but let's be explicit:
    msync((void *)ddr_virt_base, ddr_map_size, MS_SYNC);

    // 3) Reset DMA
    printf("Resetting DMA...\n");
    write_reg((volatile uint32_t *)dma_virt_base, MM2S_DMACR, 0x4); // reset MM2S
    write_reg((volatile uint32_t *)dma_virt_base, S2MM_DMACR, 0x4); // reset S2MM
    usleep(1000); // short delay

    // 4) Configure source/dest addresses
    //    For 32-bit addresses, we just write lower 32 bits.
    write_reg((volatile uint32_t *)dma_virt_base, MM2S_SA, (uint32_t)DDR_SRC_ADDR);
    write_reg((volatile uint32_t *)dma_virt_base, S2MM_DA, (uint32_t)DDR_DST_ADDR);

    // 5) Start DMA (run bit = 1, no interrupts)
    write_reg((volatile uint32_t *)dma_virt_base, MM2S_DMACR, 0x1);
    write_reg((volatile uint32_t *)dma_virt_base, S2MM_DMACR, 0x1);

    // 6) Set transfer sizes
    write_reg((volatile uint32_t *)dma_virt_base, MM2S_LENGTH, TRANSFER_SIZE);
    write_reg((volatile uint32_t *)dma_virt_base, S2MM_LENGTH, TRANSFER_SIZE);

    // 7) Poll or wait
    printf("Waiting 1 second for DMA...\n");
    sleep(1);

    // Optionally read status registers
    uint32_t mm2s_status = read_reg((volatile uint32_t *)dma_virt_base, MM2S_DMASR);
    uint32_t s2mm_status = read_reg((volatile uint32_t *)dma_virt_base, S2MM_DMASR);
    printf("MM2S DMASR = 0x%08X\n", mm2s_status);
    printf("S2MM DMASR = 0x%08X\n", s2mm_status);

    // 8) Save the result from DDR 0x00200000 => Y_test.bin
    msync((void *)ddr_virt_base, ddr_map_size, MS_SYNC);
    f_out = fopen("Y_test.bin", "wb");
    if (!f_out) {
        perror("fopen Y_test.bin");
        goto cleanup;
    }
    size_t bytes_written = fwrite((void *)ddr_dst_ptr, 1, TRANSFER_SIZE, f_out);
    fclose(f_out);
    printf("Wrote %zu bytes to Y_test.bin.\n", bytes_written);

cleanup:
    munmap(dma_ctrl_base, page_size);
    munmap(ddr_map_base, ddr_map_size);
    close(fd);
    return 0;
}
