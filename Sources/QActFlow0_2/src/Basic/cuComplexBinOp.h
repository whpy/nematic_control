/****************************************************************************** 
 * This is a list of binary operators for the cuComplex class from the cuda   * 
 * toolkit. It requires cuComplex.h to be included first.                     * 
 *                                                                            * 
 *                                         Dominik Suchla, 2018               * 
 ******************************************************************************/

// This is a work in progress and feature integers only for multiplication.
// last edited 20th of September 2018

#ifndef CU_COMPLEX_BINOP_H_
#define CU_COMPLEX_BINOP_H_

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

// The complex unit:
inline __host__ __device__
cuComplex im()
{
	return make_cuComplex( 0., 1. );
}

inline __host__ __device__
cuDoubleComplex im2()
{
	return make_cuDoubleComplex( 0., 1. );
}


// The "+" operator:
inline __host__ __device__
cuComplex operator+( cuComplex a, cuComplex b )
{
	return make_cuComplex( a.x+b.x, a.y+b.y );
}

inline __host__ __device__
cuComplex operator+( float a, cuComplex b )
{
	return make_cuComplex( a+b.x, b.y );
}

inline __host__ __device__
cuComplex operator+( cuComplex a, float b )
{
	return make_cuComplex( a.x+b, a.y );
}

inline __host__ __device__
cuDoubleComplex operator+( cuDoubleComplex a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a.x+b.x, a.y+b.y );
}

inline __host__ __device__
cuDoubleComplex operator+( double a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a+b.x, b.y );
}

inline __host__ __device__
cuDoubleComplex operator+( cuDoubleComplex a, double b )
{
	return make_cuDoubleComplex( a.x+b, a.y );
}


// The "-" operator:
inline __host__ __device__
cuComplex operator-( cuComplex a, cuComplex b )
{
	return make_cuComplex( a.x-b.x, a.y-b.y );
}

inline __host__ __device__
cuComplex operator-( float a, cuComplex b )
{
	return make_cuComplex( a-b.x, -b.y );
}

inline __host__ __device__
cuComplex operator-( cuComplex a, float b )
{
	return make_cuComplex( a.x-b, a.y );
}

inline __host__ __device__
cuDoubleComplex operator-( cuDoubleComplex a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a.x-b.x, a.y-b.y );
}

inline __host__ __device__
cuDoubleComplex operator-( double a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a-b.x, -b.y );
}

inline __host__ __device__
cuDoubleComplex operator-( cuDoubleComplex a, double b )
{
	return make_cuDoubleComplex( a.x-b, a.y );
}


// The "*" operator:
inline __host__ __device__
cuComplex operator*( cuComplex a, cuComplex b )
{
	return make_cuComplex( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x );
}

inline __host__ __device__
cuComplex operator*( float a, cuComplex b )
{
	return make_cuComplex( a*b.x, a*b.y );
}

inline __host__ __device__
cuComplex operator*( cuComplex a, float b )
{
	return make_cuComplex( a.x*b, a.y*b );
}

inline __host__ __device__
cuDoubleComplex operator*( cuDoubleComplex a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x );
}

inline __host__ __device__
cuDoubleComplex operator*( double a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a*b.x, a*b.y );
}

inline __host__ __device__
cuDoubleComplex operator*( cuDoubleComplex a, double b )
{
	return make_cuDoubleComplex( a.x*b, a.y*b );
}
// with an integer:
inline __host__ __device__
cuDoubleComplex operator*( int a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a*b.x, a*b.y );
}

inline __host__ __device__
cuDoubleComplex operator*( cuDoubleComplex a, int b )
{
	return make_cuDoubleComplex( a.x*b, a.y*b );
}


// The "/" operator:
inline __host__ __device__
cuComplex operator/( cuComplex a, cuComplex b )
{
	return make_cuComplex( (a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y) );
}

inline __host__ __device__
cuComplex operator/( float a, cuComplex b )
{
	return make_cuComplex( a*b.x/(b.x*b.x + b.y*b.y), -a*b.y/(b.x*b.x + b.y*b.y) );
}

inline __host__ __device__
cuComplex operator/( cuComplex a, float b )
{
	return make_cuComplex( a.x/b, a.y/b );
}

inline __host__ __device__
cuDoubleComplex operator/( cuDoubleComplex a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( (a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y) );
}

inline __host__ __device__
cuDoubleComplex operator/( double a, cuDoubleComplex b )
{
	return make_cuDoubleComplex( a*b.x/(b.x*b.x + b.y*b.y), -a*b.y/(b.x*b.x + b.y*b.y) );
}

inline __host__ __device__
cuDoubleComplex operator/( cuDoubleComplex a, double b )
{
	return make_cuDoubleComplex( a.x/b, a.y/b );
}

#endif
