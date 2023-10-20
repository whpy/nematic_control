#ifndef QACTFLOW_H_
#define QACTFLOW_H_

#include <iostream>
#include <fstream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

#include <Basic/cudaErr.h>
#include <Basic/cuComplexBinOp.h>

typedef float* phys;
typedef cuComplex* spec;

#endif // end of header file