NVCC = nvcc
INC = */nemacti_control/Sources/QActFlow0_2/src

all: build

.PHONY: build
build:
	$(NVCC) -I $(INC) -c Field/Field.cpp
	$(NVCC) -I $(INC) -c Basic/FldOp.cu
	$(NVCC) -I $(INC) -c Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c main.cu
	$(NVCC) -o main main.o FldOp.o Mesh.o Field.o -lcufft

.PHONY: clean
clean:
	rm *.o main *.csv 
