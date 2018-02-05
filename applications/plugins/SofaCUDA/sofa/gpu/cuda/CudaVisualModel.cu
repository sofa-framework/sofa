/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "CudaCommon.h"
#include "CudaMath.h"
#include "CudaTexture.h"
#include "cuda.h"

//#define umul24(x,y) ((x)*(y))

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void CudaVisualModelCuda3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    void CudaVisualModelCuda3f1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void CudaVisualModelCuda3d_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    void CudaVisualModelCuda3d1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

// #define USE_TEXTURE false
#ifdef USE_TEXTURE
#undef USE_TEXTURE
#endif

// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaCudaVisualModelTextures
{
public:

    static __host__ void setX(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setN(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getN(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }
};


#ifdef USE_TEXTURE

static texture<float,1,cudaReadModeElementType> tex_3f_x;
static texture<float,1,cudaReadModeElementType> tex_3f_n;

template<>
class CudaCudaVisualModelTextures<float, CudaVec3<float> >
{
public:
    typedef float real;
    typedef CudaVec3<real> TIn;

    static __host__ void setX(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_x, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_x, i3);
        float x2 = tex1Dfetch(tex_3f_x, i3+1);
        float x3 = tex1Dfetch(tex_3f_x, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }

    static __host__ void setN(const void* n)
    {
        static const void* cur = NULL;
        if (n!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_n, n);
            cur = n;
        }
    }

    static __inline__ __device__ CudaVec3<real> getN(int i, const TIn* n)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_n, i3);
        float x2 = tex1Dfetch(tex_3f_n, i3+1);
        float x3 = tex1Dfetch(tex_3f_n, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }
};

static texture<float4,1,cudaReadModeElementType> tex_3f1_x;
static texture<float4,1,cudaReadModeElementType> tex_3f1_n;

template<>
class CudaCudaVisualModelTextures<float, CudaVec4<float> >
{
public:
    typedef float real;
    typedef CudaVec4<real> TIn;

    static __host__ void setX(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_x, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_x, i));
    }

    static __host__ void setN(const void* n)
    {
        static const void* cur = NULL;
        if (n!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_n, n);
            cur = n;
        }
    }

    static __inline__ __device__ CudaVec3<real> getN(int i, const TIn* n)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_n, i));
    }
};

#endif

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t_calcTNormals_kernel(int nbElem, const int* elems, real* fnormals, const TIn* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;
    int index3 = umul24(index1,3);
    int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    __shared__  union
    {
        int itemp[3*BSIZE];
        real rtemp[3*BSIZE];
    } s;

    s.itemp[index1] = elems[iext];
    s.itemp[index1+BSIZE] = elems[iext+BSIZE];
    s.itemp[index1+2*BSIZE] = elems[iext+2*BSIZE];

    __syncthreads();

    CudaVec3<real> N = CudaVec3<real>::make(0,0,0);
    if (index < nbElem)
    {
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(s.itemp[index3+0], x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(s.itemp[index3+1], x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(s.itemp[index3+2], x);
        B -= A;
        C -= A;
        N = cross(B,C);
        N *= invnorm(N);
    }

    if (sizeof(real) != sizeof(int)) __syncthreads();

    s.rtemp[index3+0] = N.x;
    s.rtemp[index3+1] = N.y;
    s.rtemp[index3+2] = N.z;

    __syncthreads();

    fnormals[iext] = s.rtemp[index1];
    fnormals[iext+BSIZE] = s.rtemp[index1+BSIZE];
    fnormals[iext+2*BSIZE] = s.rtemp[index1+2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t1_calcTNormals_kernel(int nbElem, const int* elems, CudaVec4<real>* fnormals, const TIn* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;
    int index3 = umul24(index1,3);
    int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    __shared__ int itemp[3*BSIZE];

    itemp[index1] = elems[iext];
    itemp[index1+BSIZE] = elems[iext+BSIZE];
    itemp[index1+2*BSIZE] = elems[iext+2*BSIZE];

    __syncthreads();

    CudaVec3<real> N = CudaVec3<real>::make(0,0,0);
    if (index < nbElem)
    {
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(itemp[index3+0], x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(itemp[index3+1], x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(itemp[index3+2], x);
        B -= A;
        C -= A;
        N = cross(B,C);
        N *= invnorm(N);
    }

    fnormals[index] = CudaVec4<real>::make(N,0.0f);
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t_calcQNormals_kernel(int nbElem, const int4* elems, real* fnormals, const TIn* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;
    int index3 = umul24(index1,3);
    int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    __shared__ real rtemp[3*BSIZE];

    CudaVec3<real> N = CudaVec3<real>::make(0,0,0);
    if (index < nbElem)
    {
        int4 itemp = elems[index];
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.x, x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.y, x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.z, x);
        CudaVec3<real> D = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.w, x);
        C -= A;
        D -= B;
        N = cross(C,D);
        N *= invnorm(N);
    }

    rtemp[index3+0] = N.x;
    rtemp[index3+1] = N.y;
    rtemp[index3+2] = N.z;

    __syncthreads();

    fnormals[iext] = rtemp[index1];
    fnormals[iext+BSIZE] = rtemp[index1+BSIZE];
    fnormals[iext+2*BSIZE] = rtemp[index1+2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t1_calcQNormals_kernel(int nbElem, const int4* elems, CudaVec4<real>* fnormals, const TIn* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    CudaVec3<real> N = CudaVec3<real>::make(0,0,0);
    if (index < nbElem)
    {
        int4 itemp = elems[index];
        CudaVec3<real> A = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.x, x);
        CudaVec3<real> B = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.y, x);
        CudaVec3<real> C = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.z, x);
        CudaVec3<real> D = CudaCudaVisualModelTextures<real,TIn>::getX(itemp.w, x);
        C -= A;
        D -= B;
        N = cross(C,D);
        N *= invnorm(N);
    }

    fnormals[index] = CudaVec4<real>::make(N,0.0f);
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t_calcVNormals_kernel(int nbVertex, unsigned int nbElemPerVertex, const int* velems, real* vnormals, const TIn* fnormals)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index3 = umul24(index1,3); //3*index1;

    __shared__  real temp[3*BSIZE];

    int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    CudaVec3<real> n = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=umul24(index0,nbElemPerVertex)+index1;

    if (index0+index1 < nbVertex)
    {
        for (int s = 0; s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            velems+=BSIZE;
            if (i != -1)
            {
                n += CudaCudaVisualModelTextures<real,TIn>::getN(i,fnormals);
            }
        }
        real invn = invnorm(n);
        if (invn < 100000.0)
            n *= invn;
    }

    temp[index3  ] = n.x;
    temp[index3+1] = n.y;
    temp[index3+2] = n.z;

    __syncthreads();

    vnormals[iext        ] = temp[index1        ];
    vnormals[iext+  BSIZE] = temp[index1+  BSIZE];
    vnormals[iext+2*BSIZE] = temp[index1+2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModelCuda3t1_calcVNormals_kernel(int nbVertex, unsigned int nbElemPerVertex, const int* velems, CudaVec4<real>* vnormals, const TIn* fnormals)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;

    CudaVec3<real> n = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=umul24(index0,nbElemPerVertex)+index1;

    if (index < nbVertex)
    {
        for (int s = 0; s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            velems+=BSIZE;
            if (i != -1)
            {
                n += CudaCudaVisualModelTextures<real,TIn>::getN(i,fnormals);
            }
        }
        real invn = invnorm(n);
        if (invn < 100000.0)
            n *= invn;
    }
    vnormals[index] = CudaVec4<real>::make(n,0.0f);
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaVisualModelCuda3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec3<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcTNormals_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (float*)fnormals, (const CudaVec3<float>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcTNormals_kernel<float, CudaVec3<float> >");}
}

void CudaVisualModelCuda3f_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec3<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcQNormals_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (float*)fnormals, (const CudaVec3<float>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcQNormals_kernel<float, CudaVec3<float> >");}
}

void CudaVisualModelCuda3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcVNormals_kernel<float, CudaVec3<float> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (float*)vnormals, (const CudaVec3<float>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t_calcVNormals_kernel<float, CudaVec3<float> >");}
}

void CudaVisualModelCuda3f1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec4<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcTNormals_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (CudaVec4<float>*)fnormals, (const CudaVec4<float>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcTNormals_kernel<float, CudaVec4<float> >");}
}

void CudaVisualModelCuda3f1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<float,CudaVec4<float> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcQNormals_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (CudaVec4<float>*)fnormals, (const CudaVec4<float>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcQNormals_kernel<float, CudaVec4<float> >");}
}

void CudaVisualModelCuda3f1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcVNormals_kernel<float, CudaVec4<float> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (CudaVec4<float>*)vnormals, (const CudaVec4<float>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t1_calcVNormals_kernel<float, CudaVec4<float> >");}
}


#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaVisualModelCuda3d_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec3<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcTNormals_kernel<double, CudaVec3<double> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (double*)fnormals, (const CudaVec3<double>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcTNormals_kernel<double, CudaVec3<double> >");}
}

void CudaVisualModelCuda3d_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec3<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcQNormals_kernel<double, CudaVec3<double> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (double*)fnormals, (const CudaVec3<double>*)x); mycudaDebugError("CudaVisualModelCuda3t_calcQNormals_kernel<double, CudaVec3<double> >");}
}

void CudaVisualModelCuda3d_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t_calcVNormals_kernel<double, CudaVec3<double> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (double*)vnormals, (const CudaVec3<double>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t_calcVNormals_kernel<double, CudaVec3<double> >");}
}

void CudaVisualModelCuda3d1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec4<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcTNormals_kernel<double, CudaVec4<double> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (CudaVec4<double>*)fnormals, (const CudaVec4<double>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcTNormals_kernel<double, CudaVec4<double> >");}
}

void CudaVisualModelCuda3d1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    CudaCudaVisualModelTextures<double,CudaVec4<double> >::setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcQNormals_kernel<double, CudaVec4<double> ><<< grid1, threads1 >>>(nbElem, (const int4*)elems, (CudaVec4<double>*)fnormals, (const CudaVec4<double>*)x); mycudaDebugError("CudaVisualModelCuda3t1_calcQNormals_kernel<double, CudaVec4<double> >");}
}

void CudaVisualModelCuda3d1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaVisualModelCuda3t1_calcVNormals_kernel<double, CudaVec4<double> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (CudaVec4<double>*)vnormals, (const CudaVec4<double>*)fnormals); mycudaDebugError("CudaVisualModelCuda3t1_calcVNormals_kernel<double, CudaVec4<double> >");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
