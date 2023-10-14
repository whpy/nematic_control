CC = /usr/bin/nvcc
CFLAGS = -std=c++11 -arch=sm_35

gpu:
	$(CC) $(CFLAGS) QActFlow_ref.cu -I ./ -L /usr/lib/cuda/lib64/ -l cufft -o QActFlow_ref
