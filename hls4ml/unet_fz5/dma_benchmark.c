/*******************************************************************************
 * DMA Benchmark Example
 *
 * 1) Maps AXI DMA registers at AXI_DMA_BASE
 * 2) Maps DDR source region [DDR_SRC_ADDR .. DDR_SRC_ADDR + TRANSFER_SIZE]
 *    and DDR destination region [DDR_DST_ADDR .. DDR_DST_ADDR + TRANSFER_SIZE]
 * 3) Copies X_test.bin into the source buffer
 * 4) Repeatedly performs a DMA transfer (MM2S->S2MM) of TRANSFER_SIZE bytes:
 *      - Reset/Configure DMA
 *      - Start Timer
 *      - Start DMA
 *      - Wait for IDLE
 *      - Stop Timer
 *      - Compute latency, throughput, FPS
 * 5) Writes destination data to Y_test.bin (only once, after final iteration)
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <time.h>   // For clock_gettime

// ---------------------------------------------------------------------
// AXI DMA Register Offsets
// ---------------------------------------------------------------------
#define MM2S_DMACR      0x00
#define MM2S_DMASR      0x04
#define MM2S_SA         0x18
#define MM2S_LENGTH     0x28

#define S2MM_DMACR      0x30
#define S2MM_DMASR      0x34
#define S2MM_DA         0x48
#define S2MM_LENGTH     0x58

// ---------------------------------------------------------------------
// Status bits
// ---------------------------------------------------------------------
#define STATUS_HALTED   0x00000001
#define STATUS_IDLE     0x00000002

// ---------------------------------------------------------------------
// Control bits
// ---------------------------------------------------------------------
#define RUN_DMA         0x00000001
#define RESET_DMA       0x00000004
#define HALT_DMA        0x00000000

// ---------------------------------------------------------------------
// Addresses for your design
//   Adjust these if your hardware design is different
// ---------------------------------------------------------------------
#define AXI_DMA_BASE    0xA0000000UL  // DMA register base (AXI Lite)
#define DDR_SRC_ADDR    0x0E000000UL  // Source region in DDR
#define DDR_DST_ADDR    0x0F000000UL  // Destination region in DDR
#define TRANSFER_SIZE   65536         // 64 KB

// Number of benchmark iterations
#define NUM_ITERATIONS  10

// Helper read/write for DMA registers
static inline void write_reg(volatile uint32_t *base, uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)((uintptr_t)base + offset) = value;
}

static inline uint32_t read_reg(volatile uint32_t *base, uint32_t offset)
{
    return *(volatile uint32_t *)((uintptr_t)base + offset);
}

// ---------------------------------------------------------------------
// Wait for MM2S channel to go Idle
// ---------------------------------------------------------------------
static int wait_mm2s_idle(volatile uint32_t *dma_base, int max_tries)
{
    for(int i=0; i<max_tries; i++){
        uint32_t s = read_reg(dma_base, MM2S_DMASR);
        if(s & STATUS_IDLE) {
            return 0; // Idle found
        }
        usleep(1000); // 1ms
    }
    return -1; // Timeout
}

// ---------------------------------------------------------------------
// Wait for S2MM channel to go Idle
// ---------------------------------------------------------------------
static int wait_s2mm_idle(volatile uint32_t *dma_base, int max_tries)
{
    for(int i=0; i<max_tries; i++){
        uint32_t s = read_reg(dma_base, S2MM_DMASR);
        if(s & STATUS_IDLE){
            return 0; // Idle found
        }
        usleep(1000); // 1ms
    }
    return -1; // Timeout
}

// ---------------------------------------------------------------------
// Time helper: returns current time in seconds (double).
// ---------------------------------------------------------------------
static double now_in_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// ---------------------------------------------------------------------
// Print final results: throughput, FPS, latency
// ---------------------------------------------------------------------
static void print_results(double total_time, int iterations, size_t bytes)
{
    double avg_time = total_time / iterations;     // average sec per transfer
    double throughput = (double)bytes / (1024.0 * 1024.0) / avg_time;  // MB/s
    double fps = 1.0 / avg_time; // frames per second
    double lat_ms = avg_time * 1e3;

    printf("--------------------------------------------------------------\n");
    printf("DMA Benchmark Results (Avg over %d iterations)\n", iterations);
    printf(" Transfer Size  : %zu bytes (%.2f KB)\n", bytes, bytes/1024.0);
    printf(" Avg Latency    : %.3f ms\n", lat_ms);
    printf(" Throughput     : %.3f MB/s\n", throughput);
    printf(" FPS            : %.3f frames/s\n", fps);
    printf("--------------------------------------------------------------\n");
}

// ---------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------
int main()
{
    printf("=== DMA Benchmark Example ===\n");

    // 1) Open /dev/mem
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if(fd < 0){
        perror("open /dev/mem");
        return 1;
    }

    // 2) Map AXI DMA registers (assume 1 page is enough)
    size_t page_size = sysconf(_SC_PAGESIZE);
    off_t dma_base_aligned = AXI_DMA_BASE & ~(page_size - 1);
    void *dma_map = mmap(NULL, page_size,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, dma_base_aligned);
    if(dma_map == MAP_FAILED){
        perror("mmap(dma)");
        close(fd);
        return 1;
    }
    uintptr_t dma_virt_base = (uintptr_t)dma_map + (AXI_DMA_BASE - dma_base_aligned);
    volatile uint32_t *dma_regs = (volatile uint32_t *)dma_virt_base;

    // 3) Map Source DDR region
    off_t src_base_aligned = DDR_SRC_ADDR & ~(page_size - 1);
    size_t src_offset = DDR_SRC_ADDR - src_base_aligned;
    size_t src_map_size = src_offset + TRANSFER_SIZE;
    void *src_map = mmap(NULL, src_map_size,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, src_base_aligned);
    if(src_map == MAP_FAILED){
        perror("mmap(src)");
        munmap(dma_map, page_size);
        close(fd);
        return 1;
    }
    uintptr_t ddr_src_ptr = (uintptr_t)src_map + src_offset;

    // 4) Map Destination DDR region
    off_t dst_base_aligned = DDR_DST_ADDR & ~(page_size - 1);
    size_t dst_offset = DDR_DST_ADDR - dst_base_aligned;
    size_t dst_map_size = dst_offset + TRANSFER_SIZE;
    void *dst_map = mmap(NULL, dst_map_size,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, dst_base_aligned);
    if(dst_map == MAP_FAILED){
        perror("mmap(dst)");
        munmap(src_map, src_map_size);
        munmap(dma_map, page_size);
        close(fd);
        return 1;
    }
    uintptr_t ddr_dst_ptr = (uintptr_t)dst_map + dst_offset;

    // 5) Copy X_test.bin into the source buffer
    FILE *f_in = fopen("X_test.bin", "rb");
    if(!f_in){
        perror("fopen(X_test.bin)");
        goto cleanup;
    }
    size_t bytes_read = fread((void*)ddr_src_ptr, 1, TRANSFER_SIZE, f_in);
    fclose(f_in);
    f_in = NULL;
    printf("Loaded %zu bytes into DDR source 0x%08X\n", bytes_read, (unsigned)DDR_SRC_ADDR);

    // Clear the destination region
    memset((void*)ddr_dst_ptr, 0, TRANSFER_SIZE);

    // We'll do multiple iterations
    double total_time_sec = 0.0;

    for(int iteration=1; iteration <= NUM_ITERATIONS; iteration++){
        // --- Reset / Halt DMA each iteration to do a fresh start
        write_reg(dma_regs, S2MM_DMACR, RESET_DMA);
        write_reg(dma_regs, MM2S_DMACR, RESET_DMA);
        usleep(1000);

        // Halt
        write_reg(dma_regs, S2MM_DMACR, HALT_DMA);
        write_reg(dma_regs, MM2S_DMACR, HALT_DMA);
        usleep(1000);

        // Set addresses
        write_reg(dma_regs, S2MM_DA, (uint32_t)DDR_DST_ADDR);
        write_reg(dma_regs, MM2S_SA, (uint32_t)DDR_SRC_ADDR);

        // Start measuring time
        double t0 = now_in_seconds();

        // Run DMA channels
        write_reg(dma_regs, S2MM_DMACR, RUN_DMA);
        write_reg(dma_regs, MM2S_DMACR, RUN_DMA);

        // Set transfer size
        write_reg(dma_regs, S2MM_LENGTH, TRANSFER_SIZE);
        write_reg(dma_regs, MM2S_LENGTH, TRANSFER_SIZE);

        // Wait for idle
        if(wait_mm2s_idle(dma_regs, 1000) < 0){
            printf("ERROR: MM2S not idle. Iteration=%d\n", iteration);
            goto cleanup;
        }
        if(wait_s2mm_idle(dma_regs, 1000) < 0){
            printf("ERROR: S2MM not idle. Iteration=%d\n", iteration);
            goto cleanup;
        }

        double t1 = now_in_seconds();
        double elapsed = t1 - t0; // seconds

        total_time_sec += elapsed;

        // Print iteration stats
        double throughput = ((double)TRANSFER_SIZE / (1024.0*1024.0)) / elapsed; // MB/s
        double lat_ms     = elapsed * 1e3;
        double fps        = 1.0 / elapsed;
        printf("Iteration %d: Latency=%.3f ms, Throughput=%.3f MB/s, FPS=%.1f\n",
               iteration, lat_ms, throughput, fps);
    }

    // Print average results
    print_results(total_time_sec, NUM_ITERATIONS, TRANSFER_SIZE);

    // Finally, store the result once in Y_test.bin
    {
        FILE *f_out = fopen("Y_test.bin", "wb");
        if(!f_out){
            perror("fopen(Y_test.bin)");
            goto cleanup;
        }
        size_t bytes_written = fwrite((void*)ddr_dst_ptr, 1, TRANSFER_SIZE, f_out);
        fclose(f_out);
        printf("Wrote %zu bytes to Y_test.bin\n", bytes_written);
    }

cleanup:
    if(f_in) fclose(f_in);

    // Unmap everything
    if(dst_map && dst_map != MAP_FAILED){
        munmap(dst_map, dst_map_size);
    }
    if(src_map && src_map != MAP_FAILED){
        munmap(src_map, src_map_size);
    }
    if(dma_map && dma_map != MAP_FAILED){
        munmap(dma_map, page_size);
    }
    close(fd);

    printf("Done.\n");
    return 0;
}
