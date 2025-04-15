#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

// DMA constants
#define DMA_BASE_ADDR     0xA0000000
#define DDR_SRC_ADDR      0x0E000000
#define DDR_DST_ADDR      0x0F000000
#define TRANSFER_SIZE     65536 // 64KB
#define PAGE_SIZE         4096

// DMA register offsets
#define MM2S_CTRL         0x00
#define MM2S_STATUS       0x04
#define MM2S_SRC_ADDR     0x18
#define MM2S_LENGTH       0x28
#define S2MM_CTRL         0x30
#define S2MM_STATUS       0x34
#define S2MM_DST_ADDR     0x48
#define S2MM_LENGTH       0x58

#define RUN_DMA           0x01
#define RESET_DMA         0x04
#define STATUS_IDLE       0x02

// Helper functions
static inline void reg_write(volatile uint32_t *base, int offset, uint32_t val) {
    base[offset >> 2] = val;
}
static inline uint32_t reg_read(volatile uint32_t *base, int offset) {
    return base[offset >> 2];
}
void wait_idle(volatile uint32_t *base, int offset, const char *tag) {
    int timeout = 2000;
    while (!(reg_read(base, offset) & STATUS_IDLE) && timeout--) {
        usleep(1000);
    }
    if (timeout <= 0)
        fprintf(stderr, "[%s] Timeout waiting for IDLE.\n", tag);
}

// Check if file exists
int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

// Load binary into memory
int load_bin(const char *path, void *dst, size_t size) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t read_bytes = fread(dst, 1, size, f);
    fclose(f);
    return (read_bytes == size) ? 0 : -1;
}

// Save binary
int save_bin(const char *path, void *src, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(src, 1, size, f);
    fclose(f);
    return (written == size) ? 0 : -1;
}

int main() {
    printf("=== DMA Transfer Loop with mmap + Lock File ===\n");

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    // Map DMA registers
    volatile uint32_t *dma_regs = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DMA_BASE_ADDR);
    if (dma_regs == MAP_FAILED) {
        perror("mmap dma_regs");
        return 1;
    }

    // Map source and destination memory
    void *ddr_src = mmap(NULL, TRANSFER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DDR_SRC_ADDR);
    void *ddr_dst = mmap(NULL, TRANSFER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DDR_DST_ADDR);
    if (ddr_src == MAP_FAILED || ddr_dst == MAP_FAILED) {
        perror("mmap ddr");
        return 1;
    }

    while (1) {
        printf("Waiting for ./lock...\n");
        while (!file_exists("./lock")) usleep(50000);

        printf("Detected lock. Starting DMA transfer.\n");
        unlink("./lock"); // Remove lock file to allow further operations

        // Optional: remove previous output
        unlink("./data/result.bin");

        // Load input binary into source buffer
        if (load_bin("./data/data_stem_input.bin", ddr_src, TRANSFER_SIZE) != 0) {
            fprintf(stderr, "Failed to load input file.\n");
            continue;
        }

        memset(ddr_dst, 0, TRANSFER_SIZE); // Clear destination

        // Reset DMA
        reg_write(dma_regs, S2MM_CTRL, RESET_DMA);
        reg_write(dma_regs, MM2S_CTRL, RESET_DMA);
        usleep(1000);

        // Set source and destination addresses
        reg_write(dma_regs, MM2S_SRC_ADDR, DDR_SRC_ADDR);
        reg_write(dma_regs, S2MM_DST_ADDR, DDR_DST_ADDR);

        // Start DMA
        reg_write(dma_regs, MM2S_CTRL, RUN_DMA);
        reg_write(dma_regs, S2MM_CTRL, RUN_DMA);

        // Set transfer length
        reg_write(dma_regs, MM2S_LENGTH, TRANSFER_SIZE);
        reg_write(dma_regs, S2MM_LENGTH, TRANSFER_SIZE);

        // Wait for completion
        wait_idle(dma_regs, MM2S_STATUS, "MM2S");
        wait_idle(dma_regs, S2MM_STATUS, "S2MM");

        // Save result
        if (save_bin("./data/result.bin", ddr_dst, TRANSFER_SIZE) != 0) {
            fprintf(stderr, "Failed to save result file.\n");
        } else {
            printf("Saved output to ./data/result.bin\n");
        }
    }

    // Cleanup (unreachable in infinite loop, but useful if you later break)
    munmap((void *)dma_regs, PAGE_SIZE);
    munmap(ddr_src, TRANSFER_SIZE);
    munmap(ddr_dst, TRANSFER_SIZE);
    close(fd);

    return 0;
}
