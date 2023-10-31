NVCC = nvcc
ROOT = ../../..
INC = $(ROOT)/src
SRC = $(ROOT)/src

all: build

.PHONY: build
build: 
	

nonlterms:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c nonlterms.cu
	$(NVCC) -o main nonlterms.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft


.PHONY: clean
clean:
	rm *.o main *.csv
