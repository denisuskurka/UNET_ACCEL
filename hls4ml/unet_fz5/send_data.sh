#!/bin/bash

# Transfer input files to the remote machine
scp -P 8112 ./X_test1.bin ./dma_driver.c petalinux@85.70.252.121:/home/petalinux/

#gcc dma_driver.c -o dma_transfer_debug

# Execute commands on the remote machine
#ssh -p 8112 -t petalinux@85.70.252.121 << 'EOF'
## Run commands inside SSH
#
## Copy X_test.bin data to the memory
sudo dd if=/home/petalinux/X_test1.bin of=/dev/mem bs=1 seek=$((0x00000000)) count=65536

# Reset DMA controllers
sudo devmem 0xA0000000 w 0x4  # Reset MM2S 
sudo devmem 0xA0000030 w 0x4  # Reset S2MM

# Configure MM2S (Memory to Stream) and S2MM (Stream to Memory)
sudo devmem 0xA0000018 w 0x00000000  # MM2S source address
sudo devmem 0xA0000048 w 0x00200000  # S2MM destination address

# Start the DMA transfers
sudo devmem 0xA0000000 w 0x1  # Run MM2S 
sudo devmem 0xA0000030 w 0x1  # Run S2MM

# Set transfer sizes (64KB)
sudo devmem 0xA0000058 w 0x00010000  # S2MM transfer size
sudo devmem 0xA0000028 w 0x00010000  # MM2S transfer size

# Check DMA status
sudo devmem 0xA0000030
sudo devmem 0xA0000000

# Wait for the transfer to complete
sleep 1

# Retrieve output data from memory
sudo dd if=/dev/mem bs=1 skip=$((0x00200000)) count=65536 of=/home/petalinux/Y_test.bin
#
## End SSH session
#exit
#EOF

# Transfer the output file back to the local machine
# scp -P 8112 petalinux@85.70.252.121:/home/petalinux/Y_test.bin ./Y_test.bi
