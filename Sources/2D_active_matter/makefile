CC = /usr/bin/nvcc
CFLAGS = -std=c++11 -arch=sm_35

gpu:
	$(CC) $(CFLAGS) ActFlow_V3-2.cu -I ./ -L /usr/lib/cuda/lib64/ -l cufft -o ActFlow_V3-2
