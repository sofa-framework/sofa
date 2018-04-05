/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "cuda.h"
#include "CudaTexture.h"

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
    void SpringForceFieldCuda3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3f_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3f_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);

    void SpringForceFieldCuda3f1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3f1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3f1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3f1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3f1_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3f1_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);


#ifdef SOFA_GPU_CUDA_DOUBLE

    void SpringForceFieldCuda3d_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3d_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3d_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3d_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3d_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3d_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);

    void SpringForceFieldCuda3d1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3d1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3d1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3d1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3d1_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3d1_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);

#endif // SOFA_GPU_CUDA_DOUBLE

}

struct GPUSpring
{
    int index; ///< 0 if no spring
    float ks;
};

struct GPUSpring2
{
    float initpos;
    float kd;
};

//////////////////////
// GPU-side methods //
//////////////////////

//#define USE_TEXTURE
/*
#ifdef USE_TEXTURE

#define USE_TEXTURE_X true
#define USE_TEXTURE_V true
#define USE_TEXTURE_DX true
#define USE_TEXTURE_X2 true
#define USE_TEXTURE_V2 true
#define USE_TEXTURE_DX2 true

#else

#define USE_TEXTURE_X false
#define USE_TEXTURE_V false
#define USE_TEXTURE_DX false
#define USE_TEXTURE_X2 false
#define USE_TEXTURE_V2 false
#define USE_TEXTURE_DX2 false

#endif

template<typename real, class TIn>
class CudaSpringForceFieldInputTextures
{
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X> X;
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_V> V;
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_DX> DX;
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X2> X2;
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_V2> V2;
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_DX2> DX2;

public:

    static __host__ void setX(const void* x)
    {
	X.set((const TIn*)x);
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
	return X.get(i, (const TIn*)x);
    }

    static __host__ void setV(const void* v)
    {
	V.set((const TIn*)v);
    }

    static __inline__ __device__ CudaVec3<real> getV(int i, const TIn* v)
    {
	return V.get(i, (const TIn*)v);
    }

    static __host__ void setDX(const void* x)
    {
	DX.set((const TIn*)x);
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* x)
    {
	return DX.get(i, (const TIn*)x);
    }

    static __host__ void setX2(const void* x)
    {
	X2.set((const TIn*)x);
    }

    static __inline__ __device__ CudaVec3<real> getX2(int i, const TIn* x)
    {
	return X2.get(i, (const TIn*)x);
    }

    static __host__ void setV2(const void* v)
    {
	V2.set((const TIn*)v);
    }

    static __inline__ __device__ CudaVec3<real> getV2(int i, const TIn* v)
    {
	return V2.get(i, (const TIn*)v);
    }

    static __host__ void setDX2(const void* x)
    {
	DX2.set((const TIn*)x);
    }

    static __inline__ __device__ CudaVec3<real> getDX2(int i, const TIn* x)
    {
	return DX2.get(i, (const TIn*)x);
    }
};

template<typename real, class TIn> InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X> CudaSpringForceFieldInputTextures<real,TIn>::X;
template<typename real, class TIn> InputVector<TIn, CudaVec3<real>, USE_TEXTURE_V> CudaSpringForceFieldInputTextures<real,TIn>::V;
template<typename real, class TIn> InputVector<TIn, CudaVec3<real>, USE_TEXTURE_DX> CudaSpringForceFieldInputTextures<real,TIn>::DX;
template<typename real, class TIn> InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X2> CudaSpringForceFieldInputTextures<real,TIn>::X2;
template<typename real, class TIn> InputVector<TIn, CudaVec3<real>, USE_TEXTURE_V2> CudaSpringForceFieldInputTextures<real,TIn>::V2;
template<typename real, class TIn> InputVector<TIn, CudaVec3<real>, USE_TEXTURE_DX2> CudaSpringForceFieldInputTextures<real,TIn>::DX2;
*/


// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaSpringForceFieldInputTextures
{
public:

    static __host__ void setX(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setV(const void* /*v*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getV(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setDX(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setX2(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getX2(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setV2(const void* /*v*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getV2(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setDX2(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getDX2(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }
};

#ifdef USE_TEXTURE


static texture<float,1,cudaReadModeElementType> tex_3f_x;
static texture<float,1,cudaReadModeElementType> tex_3f_v;
static texture<float,1,cudaReadModeElementType> tex_3f_dx;
static texture<float,1,cudaReadModeElementType> tex_3f_x2;
static texture<float,1,cudaReadModeElementType> tex_3f_v2;
static texture<float,1,cudaReadModeElementType> tex_3f_dx2;

template<>
class CudaSpringForceFieldInputTextures<float, CudaVec3<float> >
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

    static __host__ void setV(const void* v)
    {
        static const void* cur = NULL;
        if (v!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_v, v);
            cur = v;
        }
    }

    static __inline__ __device__ CudaVec3<real> getV(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_v, i3);
        float x2 = tex1Dfetch(tex_3f_v, i3+1);
        float x3 = tex1Dfetch(tex_3f_v, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }

    static __host__ void setDX(const void* dx)
    {
        static const void* cur = NULL;
        if (dx!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_dx, dx);
            cur = dx;
        }
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* dx)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_dx, i3);
        float x2 = tex1Dfetch(tex_3f_dx, i3+1);
        float x3 = tex1Dfetch(tex_3f_dx, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }


    static __host__ void setX2(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_x2, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX2(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_x2, i3);
        float x2 = tex1Dfetch(tex_3f_x2, i3+1);
        float x3 = tex1Dfetch(tex_3f_x2, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }

    static __host__ void setV2(const void* v)
    {
        static const void* cur = NULL;
        if (v!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_v2, v);
            cur = v;
        }
    }

    static __inline__ __device__ CudaVec3<real> getV2(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_v2, i3);
        float x2 = tex1Dfetch(tex_3f_v2, i3+1);
        float x3 = tex1Dfetch(tex_3f_v2, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }

    static __host__ void setDX2(const void* dx)
    {
        static const void* cur = NULL;
        if (dx!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_dx2, dx);
            cur = dx;
        }
    }

    static __inline__ __device__ CudaVec3<real> getDX2(int i, const TIn* dx)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_dx2, i3);
        float x2 = tex1Dfetch(tex_3f_dx2, i3+1);
        float x3 = tex1Dfetch(tex_3f_dx2, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }

};


static texture<float4,1,cudaReadModeElementType> tex_3f1_x;
static texture<float4,1,cudaReadModeElementType> tex_3f1_v;
static texture<float4,1,cudaReadModeElementType> tex_3f1_dx;
static texture<float4,1,cudaReadModeElementType> tex_3f1_x2;
static texture<float4,1,cudaReadModeElementType> tex_3f1_v2;
static texture<float4,1,cudaReadModeElementType> tex_3f1_dx2;

template<>
class CudaSpringForceFieldInputTextures<float, CudaVec4<float> >
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

    static __host__ void setV(const void* v)
    {
        static const void* cur = NULL;
        if (v!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_v, v);
            cur = v;
        }
    }

    static __inline__ __device__ CudaVec3<real> getV(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_v, i));
    }

    static __host__ void setDX(const void* dx)
    {
        static const void* cur = NULL;
        if (dx!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_dx, dx);
            cur = dx;
        }
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* dx)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_dx, i));
    }


    static __host__ void setX2(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_x2, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX2(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_x2, i));
    }

    static __host__ void setV2(const void* v)
    {
        static const void* cur = NULL;
        if (v!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_v2, v);
            cur = v;
        }
    }

    static __inline__ __device__ CudaVec3<real> getV2(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_v2, i));
    }

    static __host__ void setDX2(const void* dx)
    {
        static const void* cur = NULL;
        if (dx!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_dx2, dx);
            cur = dx;
        }
    }

    static __inline__ __device__ CudaVec3<real> getDX2(int i, const TIn* dx)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_dx2, i));
    }

};

#endif



template<typename real>
__global__ void SpringForceFieldCuda3t_addExternalForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, real* f1, const real* x1, const real* v1, const real* x2, const real* v2)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    temp[index1        ] = x1[iext        ];
    temp[index1+  BSIZE] = x1[iext+  BSIZE];
    temp[index1+2*BSIZE] = x1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v1[iext        ];
    temp[index1+4*BSIZE] = v1[iext+  BSIZE];
    temp[index1+5*BSIZE] = v1[iext+2*BSIZE];

    __syncthreads();

    const int index3 = umul24(index1,3); //3*index1;
    CudaVec3<real> pos1 = CudaVec3<real>::make(temp[index3  ],temp[index3+1],temp[index3+2]);
    CudaVec3<real> vel1 = CudaVec3<real>::make(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring2.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring2.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            CudaVec3<real> u, relativeVelocity;

            {
                // general case
                u = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getX2(spring.index, (const CudaVec3<real>*)x2); //((const CudaVec3<real>*)x2)[spring.index];
                relativeVelocity = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getV2(spring.index, (const CudaVec3<real>*)v2); //((const CudaVec3<real>*)v2)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            real inverseLength = 1/sqrt(dot(u,u));
            real d = 1/inverseLength;
            u *= inverseLength;
            real elongation = d - spring2.initpos;
            real elongationVelocity = dot(u,relativeVelocity);
            real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;
        }
    }

    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename real>
__global__ void SpringForceFieldCuda3t1_addForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, CudaVec4<real>* f1, const CudaVec4<real>* x1, const CudaVec4<real>* v1, const CudaVec4<real>* x2, const CudaVec4<real>* v2)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;
    CudaVec3<real> pos1 = CudaVec3<real>::make(x1[index]);
    CudaVec3<real> vel1 = CudaVec3<real>::make(v1[index]);
    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring2.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring2.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            CudaVec3<real> u, relativeVelocity;

            {
                // general case
                u = CudaSpringForceFieldInputTextures<real,CudaVec4<real> >::getX2(spring.index, x2); //((const CudaVec3<real>*)x2)[spring.index];
                relativeVelocity = CudaSpringForceFieldInputTextures<real,CudaVec4<real> >::getV2(spring.index, v2); //((const CudaVec3<real>*)v2)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            real inverseLength = 1/sqrt(dot(u,u));
            real d = 1/inverseLength;
            u *= inverseLength;
            real elongation = d - spring2.initpos;
            real elongationVelocity = dot(u,relativeVelocity);
            real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;
        }
    }

    CudaVec4<real> fi = f1[index];
    fi.x = force.x;
    fi.y = force.y;
    fi.z = force.z;
    f1[index] = fi;
}

template<typename real>
__global__ void SpringForceFieldCuda3t_addForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, real* f, const real* x, const real* v)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    temp[index1        ] = x[iext        ];
    temp[index1+  BSIZE] = x[iext+  BSIZE];
    temp[index1+2*BSIZE] = x[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v[iext        ];
    temp[index1+4*BSIZE] = v[iext+  BSIZE];
    temp[index1+5*BSIZE] = v[iext+2*BSIZE];

    __syncthreads();

    const int index3 = umul24(index1,3); //3*index1;
    CudaVec3<real> pos1 = CudaVec3<real>::make(temp[index3  ],temp[index3+1],temp[index3+2]);
    CudaVec3<real> vel1 = CudaVec3<real>::make(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring2.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring2.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            CudaVec3<real> u, relativeVelocity;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int i = spring.index - index0;
                u = CudaVec3<real>::make(temp[3*i  ], temp[3*i+1], temp[3*i+2]);
                relativeVelocity = CudaVec3<real>::make(temp[3*i  +3*BSIZE], temp[3*i+1+3*BSIZE], temp[3*i+2+3*BSIZE]);
            }
            else
            {
                // general case
                u = ((const CudaVec3<real>*)x)[spring.index];
                relativeVelocity = ((const CudaVec3<real>*)v)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            real inverseLength = 1/sqrt(dot(u,u));
            real d = 1/inverseLength;
            u *= inverseLength;
            real elongation = d - spring2.initpos;
            real elongationVelocity = dot(u,relativeVelocity);
            real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;
        }
    }

    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename real>
__global__ void StiffSpringForceFieldCuda3t_addExternalForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, real* f1, const real* x1, const real* v1, const real* x2, const real* v2, real* dfdx)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    temp[index1        ] = x1[iext        ];
    temp[index1+  BSIZE] = x1[iext+  BSIZE];
    temp[index1+2*BSIZE] = x1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v1[iext        ];
    temp[index1+4*BSIZE] = v1[iext+  BSIZE];
    temp[index1+5*BSIZE] = v1[iext+2*BSIZE];

    __syncthreads();

    const int index3 = umul24(index1,3); //3*index1;
    CudaVec3<real> pos1 = CudaVec3<real>::make(temp[index3  ],temp[index3+1],temp[index3+2]);
    CudaVec3<real> vel1 = CudaVec3<real>::make(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;
    dfdx+=umul24(index0,nbSpringPerVertex)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring2.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring2.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            CudaVec3<real> u, relativeVelocity;

            {
                // general case
                u = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getX2(spring.index, (const CudaVec3<real>*)x2); //((const CudaVec3<real>*)x2)[spring.index];
                relativeVelocity = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getV2(spring.index, (const CudaVec3<real>*)v2); //((const CudaVec3<real>*)v2)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            real inverseLength = 1/sqrt(dot(u,u));
            real d = 1/inverseLength;
            u *= inverseLength;
            real elongation = d - spring2.initpos;
            real elongationVelocity = dot(u,relativeVelocity);
            real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;

            *dfdx = forceIntensity*inverseLength;
        }
        dfdx+=BSIZE;
    }

    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename real>
__global__ void StiffSpringForceFieldCuda3t1_addForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, CudaVec4<real>* f1, const CudaVec4<real>* x1, const CudaVec4<real>* v1, const CudaVec4<real>* x2, const CudaVec4<real>* v2, real* dfdx)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;
    CudaVec3<real> pos1 = CudaVec3<real>::make(x1[index]);
    CudaVec3<real> vel1 = CudaVec3<real>::make(v1[index]);
    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;
    dfdx+=umul24(index0,nbSpringPerVertex)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring2.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring2.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            CudaVec3<real> u, relativeVelocity;

            {
                // general case
                u = CudaSpringForceFieldInputTextures<real,CudaVec4<real> >::getX2(spring.index, x2); //((const CudaVec3<real>*)x2)[spring.index];
                relativeVelocity = CudaSpringForceFieldInputTextures<real,CudaVec4<real> >::getV2(spring.index, v2); //((const CudaVec3<real>*)v2)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            real inverseLength = 1/sqrt(dot(u,u));
            real d = 1/inverseLength;
            u *= inverseLength;
            real elongation = d - spring2.initpos;
            real elongationVelocity = dot(u,relativeVelocity);
            real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;

            *dfdx = forceIntensity*inverseLength;
        }
        dfdx+=BSIZE;
    }
    CudaVec4<real> fi = f1[index];
    fi.x += force.x;
    fi.y += force.y;
    fi.z += force.z;
    f1[index] = fi;
}

template<typename real>
__global__ void StiffSpringForceFieldCuda3t_addForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, real* f, const real* x, const real* v, real* dfdx)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    temp[index1        ] = x[iext        ];
    temp[index1+  BSIZE] = x[iext+  BSIZE];
    temp[index1+2*BSIZE] = x[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v[iext        ];
    temp[index1+4*BSIZE] = v[iext+  BSIZE];
    temp[index1+5*BSIZE] = v[iext+2*BSIZE];

    __syncthreads();

    const int index3 = umul24(index1,3); //3*index1;
    CudaVec3<real> pos1 = CudaVec3<real>::make(temp[index3  ],temp[index3+1],temp[index3+2]);
    CudaVec3<real> vel1 = CudaVec3<real>::make(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;
    dfdx+=umul24(index0,nbSpringPerVertex)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring2.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring2.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            CudaVec3<real> u, relativeVelocity;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int i = spring.index - index0;
                u = CudaVec3<real>::make(temp[3*i  ], temp[3*i+1], temp[3*i+2]);
                relativeVelocity = CudaVec3<real>::make(temp[3*i  +3*BSIZE], temp[3*i+1+3*BSIZE], temp[3*i+2+3*BSIZE]);
            }
            else
            {
                // general case
                u = ((const CudaVec3<real>*)x)[spring.index];
                relativeVelocity = ((const CudaVec3<real>*)v)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            //real inverseLength = 1/sqrt(dot(u,u));
            //real d = __fdividef(1,inverseLength);
            real d = sqrt(dot(u,u));
            real inverseLength = 1.0f/d;
            u *= inverseLength;
            real elongation = d - spring2.initpos;
            real elongationVelocity = dot(u,relativeVelocity);
            real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;

            *dfdx = forceIntensity*inverseLength;
        }
        dfdx+=BSIZE;
    }

    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename real>
__global__ void StiffSpringForceFieldCuda3t_addExternalDForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, real* f1, const real* dx1, const real* x1, const real* dx2, const real* x2, const real* dfdx, real factor)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*6];

    // First copy dx and x inside temp
    const int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    temp[index1        ] = dx1[iext        ];
    temp[index1+  BSIZE] = dx1[iext+  BSIZE];
    temp[index1+2*BSIZE] = dx1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = x1[iext        ];
    temp[index1+4*BSIZE] = x1[iext+  BSIZE];
    temp[index1+5*BSIZE] = x1[iext+2*BSIZE];

    __syncthreads();

    const int index3 = umul24(index1,3); //3*index1;
    CudaVec3<real> dpos1 = CudaVec3<real>::make(temp[index3  ],temp[index3+1],temp[index3+2]);
    CudaVec3<real> pos1 = CudaVec3<real>::make(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    CudaVec3<real> dforce = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;
    dfdx+=umul24(index0,nbSpringPerVertex)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        //GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            real tgt = *dfdx;
            CudaVec3<real> du;
            CudaVec3<real> u;

            {
                // general case
                du = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getDX2(spring.index, (const CudaVec3<real>*)dx2); //((const CudaVec3<real>*)dx2)[spring.index];
                u = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getX2(spring.index, (const CudaVec3<real>*)x2); //((const CudaVec3<real>*)x2)[spring.index];
            }

            du -= dpos1;
            u -= pos1;

            real uxux = u.x*u.x;
            real uyuy = u.y*u.y;
            real uzuz = u.z*u.z;
            real uxuy = u.x*u.y;
            real uxuz = u.x*u.z;
            real uyuz = u.y*u.z;
            real fact = (spring.ks-tgt)/(uxux+uyuy+uzuz);
            dforce.x += fact*(uxux*du.x+uxuy*du.y+uxuz*du.z)+tgt*du.x;
            dforce.y += fact*(uxuy*du.x+uyuy*du.y+uyuz*du.z)+tgt*du.y;
            dforce.z += fact*(uxuz*du.x+uyuz*du.y+uzuz*du.z)+tgt*du.z;
        }
        dfdx+=BSIZE;
    }

    __syncthreads();

    temp[index3  ] = dforce.x*factor;
    temp[index3+1] = dforce.y*factor;
    temp[index3+2] = dforce.z*factor;

    __syncthreads();

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename real>
__global__ void StiffSpringForceFieldCuda3t1_addDForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, CudaVec4<real>* f1, const CudaVec4<real>* dx1, const CudaVec4<real>* x1, const CudaVec4<real>* dx2, const CudaVec4<real>* x2, const real* dfdx, real factor)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;
    CudaVec3<real> dpos1 = CudaVec3<real>::make(dx1[index]);
    CudaVec3<real> pos1 = CudaVec3<real>::make(x1[index]);
    CudaVec3<real> dforce = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;
    dfdx+=umul24(index0,nbSpringPerVertex)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        //GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            real tgt = *dfdx;
            CudaVec3<real> du;
            CudaVec3<real> u;

            {
                // general case
                du = CudaSpringForceFieldInputTextures<real,CudaVec4<real> >::getDX2(spring.index, dx2); //((const CudaVec3<real>*)dx2)[spring.index];
                u = CudaSpringForceFieldInputTextures<real,CudaVec4<real> >::getX2(spring.index, x2); //((const CudaVec3<real>*)x2)[spring.index];
            }

            du -= dpos1;
            u -= pos1;

            real uxux = u.x*u.x;
            real uyuy = u.y*u.y;
            real uzuz = u.z*u.z;
            real uxuy = u.x*u.y;
            real uxuz = u.x*u.z;
            real uyuz = u.y*u.z;
            real fact = (spring.ks-tgt)/(uxux+uyuy+uzuz);
            dforce.x += fact*(uxux*du.x+uxuy*du.y+uxuz*du.z)+tgt*du.x;
            dforce.y += fact*(uxuy*du.x+uyuy*du.y+uyuz*du.z)+tgt*du.y;
            dforce.z += fact*(uxuz*du.x+uyuz*du.y+uzuz*du.z)+tgt*du.z;
        }
        dfdx+=BSIZE;
    }
    CudaVec4<real> fi = f1[index];
    fi.x += dforce.x*factor;
    fi.y += dforce.y*factor;
    fi.z += dforce.z*factor;
    f1[index] = fi;
}

template<typename real>
__global__ void StiffSpringForceFieldCuda3t_addDForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, real* f, const real* dx, const real* x, const real* dfdx, real factor)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*6];
    int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    int index3 = umul24(index1,3); //3*index1;

#ifdef USE_TEXTURE
    CudaVec3<real> dpos1 = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getDX(index0+index1, (const CudaVec3<real>*)dx); //((const CudaVec3<real>*)dx)[index0+index1];
    CudaVec3<real> pos1 = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getX(index0+index1, (const CudaVec3<real>*)x); //((const CudaVec3<real>*)x)[index0+index1];
#else
    // First copy dx and x inside temp
    temp[index1        ] = dx[iext        ];
    temp[index1+  BSIZE] = dx[iext+  BSIZE];
    temp[index1+2*BSIZE] = dx[iext+2*BSIZE];
    temp[index1+3*BSIZE] = x[iext        ];
    temp[index1+4*BSIZE] = x[iext+  BSIZE];
    temp[index1+5*BSIZE] = x[iext+2*BSIZE];

    __syncthreads();

    CudaVec3<real> dpos1 = CudaVec3<real>::make(temp[index3  ],temp[index3+1],temp[index3+2]);
    CudaVec3<real> pos1 = CudaVec3<real>::make(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
#endif
    CudaVec3<real> dforce = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    springs+=(umul24(index0,nbSpringPerVertex)<<1)+index1;
    dfdx+=umul24(index0,nbSpringPerVertex)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        //GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        real tgt = *dfdx;
        dfdx+=BSIZE;
        if (spring.index != -1)
        {
#ifdef USE_TEXTURE
            CudaVec3<real> du = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getDX(spring.index, (const CudaVec3<real>*)dx); //((const CudaVec3<real>*)dx)[spring.index];
            CudaVec3<real> u = CudaSpringForceFieldInputTextures<real,CudaVec3<real> >::getX(spring.index, (const CudaVec3<real>*)x); //((const CudaVec3<real>*)x)[spring.index];
#else
            CudaVec3<real> du;
            CudaVec3<real> u;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int i3 = umul24(spring.index - index0, 3);
                du = CudaVec3<real>::make(temp[i3  ], temp[i3+1], temp[i3+2]);
                u = CudaVec3<real>::make(temp[i3  +3*BSIZE], temp[i3+1+3*BSIZE], temp[i3+2+3*BSIZE]);
            }
            else
            {
                // general case
                du = ((const CudaVec3<real>*)dx)[spring.index];
                u = ((const CudaVec3<real>*)x)[spring.index];
            }
#endif
            du -= dpos1;
            u -= pos1;

            real uxux = u.x*u.x;
            real uyuy = u.y*u.y;
            real uzuz = u.z*u.z;
            real uxuy = u.x*u.y;
            real uxuz = u.x*u.z;
            real uyuz = u.y*u.z;
            real fact = (spring.ks-tgt)/(uxux+uyuy+uzuz);
            dforce.x += fact*(uxux*du.x+uxuy*du.y+uxuz*du.z)+tgt*du.x;
            dforce.y += fact*(uxuy*du.x+uyuy*du.y+uyuz*du.z)+tgt*du.y;
            dforce.z += fact*(uxuz*du.x+uyuz*du.y+uzuz*du.z)+tgt*du.z;
        }
    }

    __syncthreads();

    temp[index3  ] = dforce.x*factor;
    temp[index3+1] = dforce.y*factor;
    temp[index3+2] = dforce.z*factor;

    __syncthreads();

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void SpringForceFieldCuda3f_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
{
    //setX(x);
    //setV(v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t_addForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f, (const float*)x, (const float*)v); mycudaDebugError("SpringForceFieldCuda3t_addForce_kernel<float>");}
}

void SpringForceFieldCuda3f1_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
{
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setX2((const CudaVec4<float>*)x);
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setV2((const CudaVec4<float>*)v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t1_addForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<float>*)f, (const CudaVec4<float>*)x, (const CudaVec4<float>*)v, (const CudaVec4<float>*)x, (const CudaVec4<float>*)v); mycudaDebugError("SpringForceFieldCuda3t1_addForce_kernel<float>");}
}

void SpringForceFieldCuda3f_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
{
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setX2((const CudaVec3<float>*)x2);
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setV2((const CudaVec3<float>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t_addExternalForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f1, (const float*)x1, (const float*)v1, (const float*)x2, (const float*)v2); mycudaDebugError("SpringForceFieldCuda3t_addExternalForce_kernel<float>");}
}

void SpringForceFieldCuda3f1_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
{
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setX2((const CudaVec4<float>*)x2);
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setV2((const CudaVec4<float>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t1_addForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<float>*)f1, (const CudaVec4<float>*)x1, (const CudaVec4<float>*)v1, (const CudaVec4<float>*)x2, (const CudaVec4<float>*)v2); mycudaDebugError("SpringForceFieldCuda3t1_addForce_kernel<float>");}
}

void StiffSpringForceFieldCuda3f_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
{
    //setX(x);
    //setV(v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t_addForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f, (const float*)x, (const float*)v, (float*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t_addForce_kernel<float>");}
}

void StiffSpringForceFieldCuda3f1_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
{
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setX2((const CudaVec4<float>*)x);
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setV2((const CudaVec4<float>*)v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<float>*)f, (const CudaVec4<float>*)x, (const CudaVec4<float>*)v, (const CudaVec4<float>*)x, (const CudaVec4<float>*)v, (float*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t1_addForce_kernel<float>");}
}

void StiffSpringForceFieldCuda3f_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
{
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setX2((const CudaVec3<float>*)x2);
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setV2((const CudaVec3<float>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t_addExternalForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f1, (const float*)x1, (const float*)v1, (const float*)x2, (const float*)v2, (float*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t_addExternalForce_kernel<float>");}
}

void StiffSpringForceFieldCuda3f1_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
{
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setX2((const CudaVec4<float>*)x2);
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setV2((const CudaVec4<float>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<float>*)f1, (const CudaVec4<float>*)x1, (const CudaVec4<float>*)v1, (const CudaVec4<float>*)x2, (const CudaVec4<float>*)v2, (float*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t1_addForce_kernel<float>");}
}

void StiffSpringForceFieldCuda3f_addDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setX((const CudaVec3<float>*)x);
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setDX((const CudaVec3<float>*)dx);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    StiffSpringForceFieldCuda3t_addDForce_kernel<float><<< grid, threads,
#ifdef USE_TEXTURE
                                                 BSIZE*3*sizeof(float)
#else
                                                 BSIZE*6*sizeof(float)
#endif
                                                 >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f, (const float*)dx, (const float*)x, (const float*)dfdx, (float)factor);
}

void StiffSpringForceFieldCuda3f1_addDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setX2((const CudaVec4<float>*)x);
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setDX2((const CudaVec4<float>*)dx);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addDForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<float>*)f, (const CudaVec4<float>*)dx, (const CudaVec4<float>*)x, (const CudaVec4<float>*)dx, (const CudaVec4<float>*)x, (const float*)dfdx, (float)factor); mycudaDebugError("StiffSpringForceFieldCuda3t1_addDForce_kernel<float>");}
}

void StiffSpringForceFieldCuda3f_addExternalDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setX2((const CudaVec3<float>*)x2);
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setDX2((const CudaVec3<float>*)dx2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t_addExternalDForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f1, (const float*)dx1, (const float*)x1, (const float*)dx2, (const float*)x2, (const float*)dfdx, (float)factor); mycudaDebugError("StiffSpringForceFieldCuda3t_addExternalDForce_kernel<float>");}
}

void StiffSpringForceFieldCuda3f1_addExternalDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setX2((const CudaVec4<float>*)x2);
    CudaSpringForceFieldInputTextures<float,CudaVec4<float> >::setDX2((const CudaVec4<float>*)dx2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addDForce_kernel<float><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<float>*)f1, (const CudaVec4<float>*)dx1, (const CudaVec4<float>*)x1, (const CudaVec4<float>*)dx2, (const CudaVec4<float>*)x2, (const float*)dfdx, (float)factor); mycudaDebugError("StiffSpringForceFieldCuda3t1_addDForce_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void SpringForceFieldCuda3d_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
{
    //setX(x);
    //setV(v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t_addForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (double*)f, (const double*)x, (const double*)v); mycudaDebugError("SpringForceFieldCuda3t_addForce_kernel<double>");}
}

void SpringForceFieldCuda3d1_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
{
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setX2((const CudaVec4<double>*)x);
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setV2((const CudaVec4<double>*)v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t1_addForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<double>*)f, (const CudaVec4<double>*)x, (const CudaVec4<double>*)v, (const CudaVec4<double>*)x, (const CudaVec4<double>*)v); mycudaDebugError("SpringForceFieldCuda3t1_addForce_kernel<double>");}
}

void SpringForceFieldCuda3d_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
{
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setX2((const CudaVec3<double>*)x2);
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setV2((const CudaVec3<double>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t_addExternalForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (double*)f1, (const double*)x1, (const double*)v1, (const double*)x2, (const double*)v2); mycudaDebugError("SpringForceFieldCuda3t_addExternalForce_kernel<double>");}
}

void SpringForceFieldCuda3d1_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
{
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setX2((const CudaVec4<double>*)x2);
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setV2((const CudaVec4<double>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SpringForceFieldCuda3t1_addForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<double>*)f1, (const CudaVec4<double>*)x1, (const CudaVec4<double>*)v1, (const CudaVec4<double>*)x2, (const CudaVec4<double>*)v2); mycudaDebugError("SpringForceFieldCuda3t1_addForce_kernel<double>");}
}

void StiffSpringForceFieldCuda3d_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
{
    //setX(x);
    //setV(v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t_addForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (double*)f, (const double*)x, (const double*)v, (double*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t_addForce_kernel<double>");}
}

void StiffSpringForceFieldCuda3d1_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
{
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setX2((const CudaVec4<double>*)x);
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setV2((const CudaVec4<double>*)v);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<double>*)f, (const CudaVec4<double>*)x, (const CudaVec4<double>*)v, (const CudaVec4<double>*)x, (const CudaVec4<double>*)v, (double*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t1_addForce_kernel<double>");}
}

void StiffSpringForceFieldCuda3d_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
{
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setX2((const CudaVec3<double>*)x2);
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setV2((const CudaVec3<double>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t_addExternalForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (double*)f1, (const double*)x1, (const double*)v1, (const double*)x2, (const double*)v2, (double*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t_addExternalForce_kernel<double>");}
}

void StiffSpringForceFieldCuda3d1_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
{
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setX2((const CudaVec4<double>*)x2);
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setV2((const CudaVec4<double>*)v2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<double>*)f1, (const CudaVec4<double>*)x1, (const CudaVec4<double>*)v1, (const CudaVec4<double>*)x2, (const CudaVec4<double>*)v2, (double*)dfdx); mycudaDebugError("StiffSpringForceFieldCuda3t1_addForce_kernel<double>");}
}

void StiffSpringForceFieldCuda3d_addDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setX((const CudaVec3<double>*)x);
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setDX((const CudaVec3<double>*)dx);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    StiffSpringForceFieldCuda3t_addDForce_kernel<double><<< grid, threads,
#ifdef USE_TEXTURE
                                                 BSIZE*3*sizeof(double)
#else
                                                 BSIZE*6*sizeof(double)
#endif
                                                 >>>(nbSpringPerVertex, (const GPUSpring*)springs, (double*)f, (const double*)dx, (const double*)x, (const double*)dfdx, (double)factor);
}

void StiffSpringForceFieldCuda3d1_addDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setX2((const CudaVec4<double>*)x);
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setDX2((const CudaVec4<double>*)dx);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addDForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<double>*)f, (const CudaVec4<double>*)dx, (const CudaVec4<double>*)x, (const CudaVec4<double>*)dx, (const CudaVec4<double>*)x, (const double*)dfdx, (double)factor); mycudaDebugError("StiffSpringForceFieldCuda3t1_addDForce_kernel<double>");}
}

void StiffSpringForceFieldCuda3d_addExternalDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setX2((const CudaVec3<double>*)x2);
    CudaSpringForceFieldInputTextures<double,CudaVec3<double> >::setDX2((const CudaVec3<double>*)dx2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t_addExternalDForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (double*)f1, (const double*)dx1, (const double*)x1, (const double*)dx2, (const double*)x2, (const double*)dfdx, (double)factor); mycudaDebugError("StiffSpringForceFieldCuda3t_addExternalDForce_kernel<double>");}
}

void StiffSpringForceFieldCuda3d1_addExternalDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
{
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setX2((const CudaVec4<double>*)x2);
    CudaSpringForceFieldInputTextures<double,CudaVec4<double> >::setDX2((const CudaVec4<double>*)dx2);
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {StiffSpringForceFieldCuda3t1_addDForce_kernel<double><<< grid, threads >>>(nbSpringPerVertex, (const GPUSpring*)springs, (CudaVec4<double>*)f1, (const CudaVec4<double>*)dx1, (const CudaVec4<double>*)x1, (const CudaVec4<double>*)dx2, (const CudaVec4<double>*)x2, (const double*)dfdx, (double)factor); mycudaDebugError("StiffSpringForceFieldCuda3t1_addDForce_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
