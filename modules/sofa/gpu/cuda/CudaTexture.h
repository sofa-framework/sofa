#ifndef CUDATEXTURE_H
#define CUDATEXTURE_H

#include "CudaMath.h"

/// Accesss to a vector in global memory using either direct access or linear texture
/// TIn is the data type as stored in memory
/// TOut is the data type as requested
template<class TIn, class TOut, bool useTexture>
class InputVector;

/// Direct access to a vector in global memory
/// TIn is the data type as stored in memory
/// TOut is the data type as requested
template<class TIn, class TOut>
class InputVectorDirect
{
public:
    __host__ void set(const TIn*) {}
    __inline__ __device__ TOut get(int i, const TIn* x) { return TOut::make(x[i]); }
};

/// Linear texture access to a vector in global memory
/// TIn is the data type as stored in memory
/// TOut is the data type as requested
template<class TIn, class TOut>
class InputVectorTexture;

template<class TIn, class TOut>
class InputVector<TIn, TOut, false> : public InputVectorDirect<TIn, TOut>
{
};

template<class TIn, class TOut>
class InputVector<TIn, TOut, true> : public InputVectorTexture<TIn, TOut>
{
};

template<class TOut>
class InputVectorTexture<float, TOut>
{
public:
    typedef float TIn;
    static texture<float,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
        static const void* cur = NULL;
        if (x != cur)
        {
            cudaBindTexture((size_t*)NULL, tex, x);
            cur = x;
        }
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
        return tex1Dfetch(tex, i);
    }
};

template<class TOut>
class InputVectorTexture<CudaVec2<float>, TOut>
{
public:
    typedef CudaVec2<float> TIn;
    static texture<float2,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
        static const void* cur = NULL;
        if (x != cur)
        {
            cudaBindTexture((size_t*)NULL, tex, x);
            cur = x;
        }
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
        return TOut::make(tex1Dfetch(tex, i));
    }
};

template<class TOut>
class InputVectorTexture<CudaVec3<float>, TOut>
{
public:
    typedef CudaVec3<float> TIn;
    static texture<float,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
        static const void* cur = NULL;
        if (x != cur)
        {
            cudaBindTexture((size_t*)NULL, tex, x);
            cur = x;
        }
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex, i3);
        float x2 = tex1Dfetch(tex, i3+1);
        float x3 = tex1Dfetch(tex, i3+2);
        return TOut::make(x1,x2,x3);
    }
};

template<class TOut>
class InputVectorTexture<CudaVec4<float>, TOut>
{
public:
    typedef CudaVec4<float> TIn;
    static texture<float4,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
        static const void* cur = NULL;
        if (x != cur)
        {
            cudaBindTexture((size_t*)NULL, tex, x);
            cur = x;
        }
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
        return TOut::make(tex1Dfetch(tex, i));
    }
};


#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

// no support for texturing with double yet...

template<class TOut>
class InputVectorTexture<double, TOut> : public InputVectorDirect<double, TOut>
{
public:
};

template<class TOut>
class InputVectorTexture<CudaVec2<double>, TOut> : public InputVectorDirect<CudaVec2<double>, TOut>
{
public:
};

template<class TOut>
class InputVectorTexture<CudaVec3<double>, TOut> : public InputVectorDirect<CudaVec3<double>, TOut>
{
public:
};

template<class TOut>
class InputVectorTexture<CudaVec4<double>, TOut> : public InputVectorDirect<CudaVec4<double>, TOut>
{
public:
};

/*
template<class TOut>
class InputVectorTexture<double, TOut>
{
public:
    typedef double TIn;
    static texture<double,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
	static const void* cur = NULL;
	if (x != cur)
	{
	    cudaBindTexture((size_t*)NULL, tex, x);
	    cur = x;
	}
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
	return tex1Dfetch(tex, i);
    }
};

template<class TOut>
class InputVectorTexture<CudaVec2<double>, TOut>
{
public:
    typedef CudaVec2<double> TIn;
    static texture<double2,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
	static const void* cur = NULL;
	if (x != cur)
	{
	    cudaBindTexture((size_t*)NULL, tex, x);
	    cur = x;
	}
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
	return TOut::make(tex1Dfetch(tex, i));
    }
};

template<class TOut>
class InputVectorTexture<CudaVec3<double>, TOut>
{
public:
    typedef CudaVec3<double> TIn;
    static texture<double,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
	static const void* cur = NULL;
	if (x != cur)
	{
	    cudaBindTexture((size_t*)NULL, tex, x);
	    cur = x;
	}
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
	int i3 = umul24(i,3);
	double x1 = tex1Dfetch(tex, i3);
	double x2 = tex1Dfetch(tex, i3+1);
	double x3 = tex1Dfetch(tex, i3+2);
	return TOut::make(x1,x2,x3);
    }
};

template<class TOut>
class InputVectorTexture<CudaVec4<double>, TOut>
{
public:
    typedef CudaVec4<double> TIn;
    static texture<double2,1,cudaReadModeElementType> tex;

    __host__ void set(const TIn* x)
    {
	static const void* cur = NULL;
	if (x != cur)
	{
	    cudaBindTexture((size_t*)NULL, tex, x);
	    cur = x;
	}
    }
    __inline__ __device__ TOut get(int i, const TIn* x)
    {
	int i2 = (i<<1);
	double2 x1 = tex1Dfetch(tex, i2);
	double2 x2 = tex1Dfetch(tex, i2+1);
	return TOut::make(make_double4(x1.x,x1.y,x2.x,x2.y));
    }
};
*/

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

#endif
