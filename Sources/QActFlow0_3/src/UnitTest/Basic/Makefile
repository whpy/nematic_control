NVCC = nvcc
ROOT = ../../..
INC = $(ROOT)/src
SRC = $(ROOT)/src

all: build

.PHONY: build
build: 
	


BasicOp:
	$(NVCC) -I $(INC) -c BasicOp.cu
	$(NVCC) -o main BasicOp.o -lcufft
#	rm *.csv *.o

BFwd:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c BFwd.cu
	$(NVCC) -o main BFwd.o Field.o Mesh.o -lcufft


.PHONY: clean
clean:
	rm *.o main *.csv
