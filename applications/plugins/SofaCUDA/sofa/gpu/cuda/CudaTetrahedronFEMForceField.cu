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
#include "mycuda.h"
#include <stdio.h>

//#define umul24(x,y) ((x)*(y))

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void TetrahedronFEMForceFieldCuda3f_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);


    void TetrahedronFEMForceFieldCuda3f1_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f1_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
    void TetrahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void TetrahedronFEMForceFieldCuda3d_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3d_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3d1_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3d1_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3d_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
    void TetrahedronFEMForceFieldCuda3d_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

#endif // SOFA_GPU_CUDA_DOUBLE
}

template<class real>
class __align__(16) GPUElement
{
public:
    /// index of the 4 connected vertices
    //Vec<4,int> tetra;
    int ia[BSIZE];
    int ib[BSIZE];
    int ic[BSIZE];
    int id[BSIZE];
    /// material stiffness matrix
    //Mat<6,6,Real> K;
    real gamma_bx2[BSIZE], mu2_bx2[BSIZE];
    /// initial position of the vertices in the local (rotated) coordinate system
    //Vec3f initpos[4];
    real bx[BSIZE],cx[BSIZE];
    real cy[BSIZE],dx[BSIZE],dy[BSIZE],dz[BSIZE];
    /// strain-displacement matrix
    //Mat<12,6,Real> J;
    real Jbx_bx[BSIZE],Jby_bx[BSIZE],Jbz_bx[BSIZE];
    /// unused value to align to 64 bytes
    //real dummy[BSIZE];
};

template<class real>
class GPUElementForce
{
public:
    CudaVec4<real> fA,fB,fC,fD;
};

//////////////////////
// GPU-side methods //
//////////////////////

//#define USE_TEXTURE_X false
//#define USE_TEXTURE_ELEMENT_FORCE false

/*
template<typename real, class TIn>
class CudaTetrahedronFEMForceFieldInputTextures
{
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X>& X()
    {
	static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X> v; return v;
    }
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X>& DX()
    {
	static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_X> v; return v;
    }

public:

    static __host__ void setX(const void* x)
    {
	X().set((const TIn*)x);
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
	return X().get(i, (const TIn*)x);
    }

    static __host__ void setDX(const void* x)
    {
	DX().set((const TIn*)x);
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* x)
    {
	return DX().get(i, (const TIn*)x);
    }
};

template<typename real, class TIn>
class CudaTetrahedronFEMForceFieldTempTextures
{
    static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_ELEMENT_FORCE>& ElementForce();
    {
	static InputVector<TIn, CudaVec3<real>, USE_TEXTURE_ELEMENT_FORCE> v; return v;
    }

public:

    static __host__ void setElementForce(const void* x)
    {
	ElementForce().set((const TIn*)x);
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
	return ElementForce().get(i, (const TIn*)x);
    }

};
*/

// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaTetrahedronFEMForceFieldInputTextures
{
public:

    static __host__ void setX(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
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
};


// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaTetrahedronFEMForceFieldTempTextures
{
public:

    static __host__ void setElementForce(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
        return CudaVec3<real>::make(x[i]);
    }
};

#if defined(USE_TEXTURE_X)

static texture<float,1,cudaReadModeElementType> tex_3f_x;
static texture<float,1,cudaReadModeElementType> tex_3f_dx;

template<>
class CudaTetrahedronFEMForceFieldInputTextures<float, CudaVec3<float> >
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
};

static texture<float4,1,cudaReadModeElementType> tex_3f1_x;
static texture<float4,1,cudaReadModeElementType> tex_3f1_dx;

template<>
class CudaTetrahedronFEMForceFieldInputTextures<float, CudaVec4<float> >
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
};

//////////////////////////////////////////DOUBLE

static texture<double,1,cudaReadModeElementType> tex_3d_x;
static texture<double,1,cudaReadModeElementType> tex_3d_dx;

template<>
class CudaTetrahedronFEMForceFieldInputTextures<double, CudaVec3<double> >
{
public:
    typedef double real;
    typedef CudaVec3<real> TIn;

    static __host__ void setX(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3d_x, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        double x1 = tex1Dfetch(tex_3d_x, i3);
        double x2 = tex1Dfetch(tex_3d_x, i3+1);
        double x3 = tex1Dfetch(tex_3d_x, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }

    static __host__ void setDX(const void* dx)
    {
        static const void* cur = NULL;
        if (dx!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3d_dx, dx);
            cur = dx;
        }
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* dx)
    {
        int i3 = umul24(i,3);
        double x1 = tex1Dfetch(tex_3d_dx, i3);
        double x2 = tex1Dfetch(tex_3d_dx, i3+1);
        double x3 = tex1Dfetch(tex_3d_dx, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }
};

static texture<double4,1,cudaReadModeElementType> tex_3d1_x;
static texture<double4,1,cudaReadModeElementType> tex_3d1_dx;

template<>
class CudaTetrahedronFEMForceFieldInputTextures<double, CudaVec4<double> >
{
public:
    typedef double real;
    typedef CudaVec4<real> TIn;

    static __host__ void setX(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3d1_x, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3d1_x, i));
    }

    static __host__ void setDX(const void* dx)
    {
        static const void* cur = NULL;
        if (dx!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3d1_dx, dx);
            cur = dx;
        }
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* dx)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3d1_dx, i));
    }
};

#endif


#if defined(USE_TEXTURE_ELEMENT_FORCE)

static texture<float,1,cudaReadModeElementType> tex_3f_eforce;

template<>
class CudaTetrahedronFEMForceFieldTempTextures<float, CudaVec3<float> >
{
public:
    typedef float real;
    typedef CudaVec3<real> TIn;

    static __host__ void setElementForce(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f_eforce, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        float x1 = tex1Dfetch(tex_3f_eforce, i3);
        float x2 = tex1Dfetch(tex_3f_eforce, i3+1);
        float x3 = tex1Dfetch(tex_3f_eforce, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }
};

static texture<float4,1,cudaReadModeElementType> tex_3f1_eforce;

template<>
class CudaTetrahedronFEMForceFieldTempTextures<float, CudaVec4<float> >
{
public:
    typedef float real;
    typedef CudaVec4<real> TIn;
    typedef CudaVec3<real> TOut;

    static __host__ void setElementForce(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3f1_eforce, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3f1_eforce, i));
    }
};


//////////////////////////////////////////////DOUBLE
static texture<double,1,cudaReadModeElementType> tex_3d_eforce;

template<>
class CudaTetrahedronFEMForceFieldTempTextures<double, CudaVec3<double> >
{
public:
    typedef double real;
    typedef CudaVec3<real> TIn;

    static __host__ void setElementForce(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3d_eforce, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
        int i3 = umul24(i,3);
        double x1 = tex1Dfetch(tex_3d_eforce, i3);
        double x2 = tex1Dfetch(tex_3d_eforce, i3+1);
        double x3 = tex1Dfetch(tex_3d_eforce, i3+2);
        return CudaVec3<real>::make(x1,x2,x3);
    }
};

static texture<double4,1,cudaReadModeElementType> tex_3d1_eforce;

template<>
class CudaTetrahedronFEMForceFieldTempTextures<double, CudaVec4<double> >
{
public:
    typedef double real;
    typedef CudaVec4<real> TIn;
    typedef CudaVec3<real> TOut;

    static __host__ void setElementForce(const void* x)
    {
        static const void* cur = NULL;
        if (x!=cur)
        {
            cudaBindTexture((size_t*)NULL, tex_3d1_eforce, x);
            cur = x;
        }
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
        return CudaVec3<real>::make(tex1Dfetch(tex_3d1_eforce, i));
    }
};

#endif


template<typename real, class TIn>
__global__ void TetrahedronFEMForceFieldCuda3t_calcForce_kernel(int nbElem, const GPUElement<real>* elems, real* rotations, real* eforce, const TIn* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    //GPUElement<real> e = elems[index];
    //GPUElementState<real> s;
    const GPUElement<real>* e = elems + blockIdx.x;
    matrix3<real> Rt;
    rotations += umul24(index0,9)+index1;
    //GPUElementForce<real> f;
    CudaVec3<real> fB,fC,fD;

    if (index < nbElem)
    {
        CudaVec3<real> A = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ia[index1], x); //((const CudaVec3<real>*)x)[e.ia];
        CudaVec3<real> B = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ib[index1], x); //((const CudaVec3<real>*)x)[e.ib];
        B -= A;

        // Compute R
        real bx = norm(B);
        Rt.x = B/bx;
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        CudaVec3<real> JtRtX0,JtRtX1;

        bx -= e->bx[index1];
        //                    ( bx)
        // RtB =              ( 0 )
        //                    ( 0 )
        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        real e_Jbx_bx = e->Jbx_bx[index1];
        real e_Jby_bx = e->Jby_bx[index1];
        real e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x = e_Jbx_bx * bx;
        JtRtX0.y = 0;
        JtRtX0.z = 0;
        JtRtX1.x = e_Jby_bx * bx;
        JtRtX1.y = 0;
        JtRtX1.z = e_Jbz_bx * bx;

        CudaVec3<real> C = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ic[index1], x); //((const CudaVec3<real>*)x)[e.ic];
        C -= A;
        Rt.z = cross(B,C);
        Rt.y = cross(Rt.z,B);
        Rt.y *= invnorm(Rt.y);
        Rt.z *= invnorm(Rt.z);

        real e_cx = e->cx[index1];
        real e_cy = e->cy[index1];
        real cx = Rt.mulX(C) - e_cx;
        real cy = Rt.mulY(C) - e_cy;
        //                    ( cx)
        // RtC =              ( cy)
        //                    ( 0 )
        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        real e_dy = e->dy[index1];
        real e_dz = e->dz[index1];
        //JtRtX0.x += 0;
        JtRtX0.y += e_dz * cy;
        //JtRtX0.z += 0;
        JtRtX1.x += e_dz * cx;
        JtRtX1.y -= e_dy * cy;
        JtRtX1.z -= e_dy * cx;

        CudaVec3<real> D = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->id[index1], x); //((const CudaVec3<real>*)x)[e.id];
        D -= A;

        real e_dx = e->dx[index1];
        real dx = Rt.mulX(D) - e_dx;
        real dy = Rt.mulY(D) - e_dy;
        real dz = Rt.mulZ(D) - e_dz;
        //                    ( dx)
        // RtD =              ( dy)
        //                    ( dz)
        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        //JtRtX0.x += 0;
        //JtRtX0.y += 0;
        JtRtX0.z += e_cy * dz;
        //JtRtX1.x += 0;
        JtRtX1.y += e_cy * dy;
        JtRtX1.z += e_cy * dx;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        real e_mu2_bx2 = e->mu2_bx2[index1];
        CudaVec3<real> S0  = JtRtX0*e_mu2_bx2;
        S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e->gamma_bx2[index1];
        CudaVec3<real> S1  = JtRtX1*(e_mu2_bx2*0.5f);

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.mulT(CudaVec3<real>::make(
                e_cy * S1.z,
                e_cy * S1.y,
                e_cy * S0.z)));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.mulT(CudaVec3<real>::make(
                e_dz * S1.x - e_dy * S1.z,
                e_dz * S0.y - e_dy * S1.y,
                e_dz * S1.y - e_dy * S0.z)));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.mulT(CudaVec3<real>::make(
                e_Jbx_bx * S0.x                                     + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
                e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
                e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z)));
        //fA.x = -(fB.x+fC.x+fD.x);
        //fA.y = -(fB.y+fC.y+fD.y);
        //fA.z = -(fB.z+fC.z+fD.z);
    }

    //state[index] = s;
    Rt.writeAoS(rotations);
    //((rmatrix3*)rotations)[index] = Rt;
    //((GPUElementForce<real>*)eforce)[index] = f;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*13];
    int index13 = umul24(index1,13);
    temp[index13+0 ] = -(fB.x+fC.x+fD.x);
    temp[index13+1 ] = -(fB.y+fC.y+fD.y);
    temp[index13+2 ] = -(fB.z+fC.z+fD.z);
    temp[index13+3 ] = fB.x;
    temp[index13+4 ] = fB.y;
    temp[index13+5 ] = fB.z;
    temp[index13+6 ] = fC.x;
    temp[index13+7 ] = fC.y;
    temp[index13+8 ] = fC.z;
    temp[index13+9 ] = fD.x;
    temp[index13+10] = fD.y;
    temp[index13+11] = fD.z;
    __syncthreads();
    real* out = ((real*)eforce)+(umul24(blockIdx.x,BSIZE*16))+index1;
    real v = 0;
    bool read = true; //(index1&4)<3;
    index1 += (index1>>4) - (index1>>2); // remove one for each 4-values before this thread, but add an extra one each 16 threads (so each 12 input cells, to align to 13)

    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;

}

template<typename real,int BSIZE>
__global__ void TetrahedronFEMForceFieldCuda3t_addForce1_kernel(int nbVertex, unsigned int nbElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index3 = fastmul(index1,3); //3*index1;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE*3];

    int iext = fastmul(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=fastmul(index0,nbElemPerVertex)+index1;

    if (index0+index1 < nbVertex)
        for (int s = 0; s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            if (i == -1) break;
            velems+=BSIZE;
            //if (i != -1)
            {
                force -= CudaTetrahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
            }
        }

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename real,int BSIZE>
__global__ void TetrahedronFEMForceFieldCuda3t_addForce4_kernel(int nbVertex, unsigned int nb4ElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE*3];

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=(index0*nb4ElemPerVertex)+index1;

    //if (index0+index1 < (nbVertex<<2))
    for (int s = 0; s < nb4ElemPerVertex; s++)
    {
        int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;
        //if (i != -1)
        {
            force -= CudaTetrahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
        }
    }

    //int iout = (index1>>2)*3 + (index1&3)*((BSIZE/4)*3);
    int iout = fastmul((index1>>2) + ((index1&3)*(BSIZE/4)),3);
    temp[iout  ] = force.x;
    temp[iout+1] = force.y;
    temp[iout+2] = force.z;

    __syncthreads();

    // we need to merge 4 values together
    if (index1 < (BSIZE/4)*3)
    {

        real res = temp[index1] + temp[index1+ (BSIZE/4)*3] + temp[index1+ 2*(BSIZE/4)*3] + temp[index1+ 3*(BSIZE/4)*3];

        int iext = fastmul(blockIdx.x,(BSIZE/4)*3)+index1; //index0*3+index1;

        f[iext] += res;
    }
}

template<typename real,int BSIZE>
__global__ void TetrahedronFEMForceFieldCuda3t_addForce8_kernel(int nbVertex, unsigned int nb8ElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE/2*3];

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=(index0*nb8ElemPerVertex)+index1;

    //if (index0+index1 < (nbVertex<<2))
    for (int s = 0; s < nb8ElemPerVertex; s++)
    {
        int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;
        //if (i != -1)
        {
            force -= CudaTetrahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
        }
    }

    //int iout = (index1>>2)*3 + (index1&7)*((BSIZE/8)*3);
    int iout = fastmul((index1>>3) + ((index1&3)*(BSIZE/8)),3);
    if (index1&4)
    {
        temp[iout  ] = force.x;
        temp[iout+1] = force.y;
        temp[iout+2] = force.z;
    }
    __syncthreads();
    if (!(index1&4))
    {
        temp[iout  ] += force.x;
        temp[iout+1] += force.y;
        temp[iout+2] += force.z;
    }
    __syncthreads();

    if (index1 < (BSIZE/8)*3)
    {
        // we need to merge 4 values together
        real res = temp[index1] + temp[index1+ (BSIZE/8)*3] + temp[index1+ 2*(BSIZE/8)*3] + temp[index1+ 3*(BSIZE/8)*3];

        int iext = fastmul(blockIdx.x,(BSIZE/8)*3)+index1; //index0*3+index1;

        f[iext] += res;
    }
}

template<typename real, int BSIZE>
__global__ void TetrahedronFEMForceFieldCuda3t1_addForce1_kernel(int nbVertex, unsigned int nbElemPerVertex, const CudaVec4<real>* eforce, const int* velems, CudaVec4<real>* f)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=umul24(index0,nbElemPerVertex)+index1;

    if (index < nbVertex)
        for (int s = 0; s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            velems+=BSIZE;
            if (i != -1)
            {
                force -= CudaTetrahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
            }
        }
    CudaVec4<real> fi = f[index];
    fi.x += force.x;
    fi.y += force.y;
    fi.z += force.z;
    f[index] = fi;
}

template<typename real, class TIn>
__global__ void TetrahedronFEMForceFieldCuda3t_calcDForce_kernel(int nbElem, const GPUElement<real>* elems, const real* rotations, real* eforce, const TIn* x, real factor)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    //GPUElement<real> e = elems[index];
    const GPUElement<real>* e = elems + blockIdx.x;
    //GPUElementState<real> s = state[index];
    //GPUElementForce<real> f;
    CudaVec3<real> fB,fC,fD;
    matrix3<real> Rt;
    rotations += umul24(index0,9)+index1;
    Rt.readAoS(rotations);
    //Rt = ((const rmatrix3*)rotations)[index];

    if (index < nbElem)
    {
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        CudaVec3<real> A = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->ia[index1], x); //((const CudaVec3<real>*)x)[e.ia];
        CudaVec3<real> JtRtX0,JtRtX1;


        CudaVec3<real> B = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->ib[index1], x); //((const CudaVec3<real>*)x)[e.ib];
        B = Rt * (B-A);

        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        real e_Jbx_bx = e->Jbx_bx[index1];
        real e_Jby_bx = e->Jby_bx[index1];
        real e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x = e_Jbx_bx * B.x;
        JtRtX0.y =                  e_Jby_bx * B.y;
        JtRtX0.z =                                   e_Jbz_bx * B.z;
        JtRtX1.x = e_Jby_bx * B.x + e_Jbx_bx * B.y;
        JtRtX1.y =                  e_Jbz_bx * B.y + e_Jby_bx * B.z;
        JtRtX1.z = e_Jbz_bx * B.x                  + e_Jbx_bx * B.z;

        CudaVec3<real> C = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->ic[index1], x); //((const CudaVec3<real>*)x)[e.ic];
        C = Rt * (C-A);

        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        real e_dy = e->dy[index1];
        real e_dz = e->dz[index1];
        //JtRtX0.x += 0;
        JtRtX0.y +=              e_dz * C.y;
        JtRtX0.z +=                         - e_dy * C.z;
        JtRtX1.x += e_dz * C.x;
        JtRtX1.y +=            - e_dy * C.y + e_dz * C.z;
        JtRtX1.z -= e_dy * C.x;

        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        CudaVec3<real> D = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->id[index1], x); //((const CudaVec3<real>*)x)[e.id];
        D = Rt * (D-A);

        real e_cy = e->cy[index1];
        //JtRtX0.x += 0;
        //JtRtX0.y += 0;
        JtRtX0.z +=                           e_cy * D.z;
        //JtRtX1.x += 0;
        JtRtX1.y +=              e_cy * D.y;
        JtRtX1.z += e_cy * D.x;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        real e_mu2_bx2 = e->mu2_bx2[index1];
        CudaVec3<real> S0  = JtRtX0*e_mu2_bx2;
        S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e->gamma_bx2[index1];
        CudaVec3<real> S1  = JtRtX1*(e_mu2_bx2*0.5f);

        S0 *= factor;
        S1 *= factor;

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.mulT(CudaVec3<real>::make(
                e_cy * S1.z,
                e_cy * S1.y,
                e_cy * S0.z)));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.mulT(CudaVec3<real>::make(
                e_dz * S1.x - e_dy * S1.z,
                e_dz * S0.y - e_dy * S1.y,
                e_dz * S1.y - e_dy * S0.z)));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.mulT(CudaVec3<real>::make(
                e_Jbx_bx * S0.x                                     + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
                e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
                e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z)));
        //fA.x = -(fB.x+fC.x+fD.x);
        //fA.y = -(fB.y+fC.y+fD.y);
        //fA.z = -(fB.z+fC.z+fD.z);
    }

    //state[index] = s;
    //((GPUElementForce<real>*)eforce)[index] = f;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*13];
    int index13 = umul24(index1,13);
    temp[index13+0 ] = -(fB.x+fC.x+fD.x);
    temp[index13+1 ] = -(fB.y+fC.y+fD.y);
    temp[index13+2 ] = -(fB.z+fC.z+fD.z);
    temp[index13+3 ] = fB.x;
    temp[index13+4 ] = fB.y;
    temp[index13+5 ] = fB.z;
    temp[index13+6 ] = fC.x;
    temp[index13+7 ] = fC.y;
    temp[index13+8 ] = fC.z;
    temp[index13+9 ] = fD.x;
    temp[index13+10] = fD.y;
    temp[index13+11] = fD.z;
    __syncthreads();
    real* out = ((real*)eforce)+(umul24(blockIdx.x,BSIZE*16))+index1;
    real v = 0;
    bool read = true; //(index1&4)<3;
    index1 += (index1>>4) - (index1>>2); // remove one for each 4-values before this thread, but add an extra one each 16 threads (so each 12 input cells, to align to 13)

    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
}

template<typename real>
__global__ void TetrahedronFEMForceFieldCuda3t_getRotations_kernel(int nbVertex, const real* initState, const real* state, const int* rotationIdx, real* rotations)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index>=nbVertex) return;

    const int elemIdx = rotationIdx[index];
    const int stateIdx = ((elemIdx / BSIZE) * (9*BSIZE)) + (elemIdx % BSIZE);

    matrix3<real> initRt, curRt, R;

    initRt.readAoS(initState + stateIdx);
    curRt.readAoS(state + stateIdx);

    // R = transpose(curRt) * initRt
    R = curRt.mulT(initRt);

    //R.x.x = dot (curR.x, initR.x); R.x.y = dot (curR.x, initR.y); R.x.z = dot (curR.x, initR.z);
    //R.y.x = dot (curR.y, initR.x); R.y.y = dot (curR.y, initR.y); R.y.z = dot (curR.y, initR.z);
    //R.z.x = dot (curR.z, initR.x); R.z.y = dot (curR.z, initR.y); R.z.z = dot (curR.z, initR.z);

    rotations += 9*index;
    rotations[0] = R.x.x;
    rotations[1] = R.x.y;
    rotations[2] = R.x.z;

    rotations[3] = R.y.x;
    rotations[4] = R.y.y;
    rotations[5] = R.y.z;

    rotations[6] = R.z.x;
    rotations[7] = R.z.y;
    rotations[8] = R.z.z;
}

template<typename real>
__global__ void TetrahedronFEMForceFieldCuda3t_getElementRotations_kernel(unsigned nbElem,const real* rotationsAos, real* rotations)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index>=nbElem) return;

    matrix3<real> R;
    rotationsAos += umul24(index0,9)+index1;
    R.readAoS(rotationsAos);

    rotations += 9*index;
    rotations[0] = R.x.x;
    rotations[1] = R.y.x;
    rotations[2] = R.z.x;

    rotations[3] = R.x.y;
    rotations[4] = R.y.y;
    rotations[5] = R.z.y;

    rotations[6] = R.x.z;
    rotations[7] = R.y.z;
    rotations[8] = R.z.z;
}

//////////////////////
// CPU-side methods //
//////////////////////

template<typename real, int BSIZE>
inline void TetrahedronFEMForceFieldCuda3t_addForce_launch2(int pt,unsigned int nbVertex, unsigned int nbElemPerVertex, void* eforce, const void* velems, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbVertex*pt+BSIZE-1)/BSIZE,1);
    nbElemPerVertex = (nbElemPerVertex + pt-1)/pt;

    switch (pt)
    {
    case 1 : {TetrahedronFEMForceFieldCuda3t_addForce1_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (real*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_addForce1_kernel<real,BSIZE>");} return;
    case 4 : {TetrahedronFEMForceFieldCuda3t_addForce4_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (real*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_addForce4_kernel<real,BSIZE>");} return;
    case 8 : {TetrahedronFEMForceFieldCuda3t_addForce8_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (real*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_addForce8_kernel<real,BSIZE>");} return;
    }
    mycudaPrintf("Error : gatherPt should be 1 or 4 or 8, current value is %d\n",pt);
}

template<typename real>
inline void TetrahedronFEMForceFieldCuda3t_addForce_launch1(int bsize,int pt,unsigned int nbVertex, unsigned int nbElemPerVertex, void* eforce, const void* velems, void* f)
{
    switch (bsize)
    {
    case  32 : TetrahedronFEMForceFieldCuda3t_addForce_launch2<real, 32>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    case  64 : TetrahedronFEMForceFieldCuda3t_addForce_launch2<real, 64>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    case 128 : TetrahedronFEMForceFieldCuda3t_addForce_launch2<real,128>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    case 256 : TetrahedronFEMForceFieldCuda3t_addForce_launch2<real,256>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    }
    mycudaPrintf("Error : gatherBsize should be 32 or 64 or 128 or 256, current value is %d\n",bsize);
}

template<typename real, int BSIZE>
inline void TetrahedronFEMForceFieldCuda3t1_addForce_launch2(int pt,unsigned int nbVertex, unsigned int nbElemPerVertex, void* eforce, const void* velems, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbVertex*pt+BSIZE-1)/BSIZE,1);

    switch (pt)
    {
    case 1 : {TetrahedronFEMForceFieldCuda3t1_addForce1_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (CudaVec4<real>*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t1_addForce1_kernel<real,BSIZE>");} return;
//       case 4 : {TetrahedronFEMForceFieldCuda3t1_addForce4_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (CudaVec4<real>*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t1_addForce4_kernel<real,BSIZE>");} return;
//       case 8 : {TetrahedronFEMForceFieldCuda3t1_addForce8_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (CudaVec4<real>*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t1_addForce8_kernel<real,BSIZE>");} return;
    }
    mycudaPrintf("Error : gatherPt should be 1 or 4 or 8, current value is %d\n",pt);
}

template<typename real>
inline void TetrahedronFEMForceFieldCuda3t1_addForce_launch1(int bsize,int pt,unsigned int nbVertex, unsigned int nbElemPerVertex, void* eforce, const void* velems, void* f)
{
    //nbElemPerVertex = (nbElemPerVertex + bsize-1)/bsize;
    switch (bsize)
    {
    case  32 : TetrahedronFEMForceFieldCuda3t1_addForce_launch2<real, 32>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    case  64 : TetrahedronFEMForceFieldCuda3t1_addForce_launch2<real, 64>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    case 128 : TetrahedronFEMForceFieldCuda3t1_addForce_launch2<real,128>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    case 256 : TetrahedronFEMForceFieldCuda3t1_addForce_launch2<real,256>(pt,nbVertex, nbElemPerVertex, eforce, velems, f); return;
    }
    mycudaPrintf("Error : gatherBsize should be 32 or 64 or 128 or 256, current value is %d\n",bsize);
}

void TetrahedronFEMForceFieldCuda3f_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
{
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec3<float> >::setX(x);
    CudaTetrahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcForce_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (float*)state, (float*)eforce, (const CudaVec3<float>*)x); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcForce_kernel<float, CudaVec3<float> >");}
    TetrahedronFEMForceFieldCuda3t_addForce_launch1<float>(bsize,pt, nbVertex, nbElemPerVertex, eforce, velems, f);
}

void TetrahedronFEMForceFieldCuda3f_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
{
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec3<float> >::setDX(dx);
    CudaTetrahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (const float*)state, (float*)eforce, (const CudaVec3<float>*)dx, (float) factor); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<float, CudaVec3<float> >");}
    TetrahedronFEMForceFieldCuda3t_addForce_launch1<float>(bsize,pt,nbVertex, nbElemPerVertex, eforce, velems, df);
}

void TetrahedronFEMForceFieldCuda3f1_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
{
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec4<float> >::setX(x);
    CudaTetrahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcForce_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (float*)state, (float*)eforce, (const CudaVec4<float>*)x); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcForce_kernel<float, CudaVec4<float> >");}
    TetrahedronFEMForceFieldCuda3t1_addForce_launch1<float>(bsize,pt, nbVertex, nbElemPerVertex, eforce, velems, f);
}

void TetrahedronFEMForceFieldCuda3f1_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
{
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec4<float> >::setDX(dx);
    CudaTetrahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (const float*)state, (float*)eforce, (const CudaVec4<float>*)dx, (float) factor); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<float, CudaVec4<float> >");}
    TetrahedronFEMForceFieldCuda3t1_addForce_launch1<float>(bsize,pt, nbVertex, nbElemPerVertex, eforce, velems, df);
}

void TetrahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbVertex+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_getRotations_kernel<float><<< grid, threads >>>(nbVertex, (const float*)initState, (const float*)state, (const int*)rotationIdx, (float*)rotations); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_getRotations_kernel<float>");}
}

void TetrahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_getElementRotations_kernel<float><<< grid, threads >>>(nbElem,(const float*)rotationsAos, (float*)rotations); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_getElementRotations_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void TetrahedronFEMForceFieldCuda3d_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
{
    CudaTetrahedronFEMForceFieldInputTextures<double,CudaVec3<double> >::setX(x);
    CudaTetrahedronFEMForceFieldTempTextures<double,CudaVec4<double> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcForce_kernel<double, CudaVec3<double> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<double>*)elems, (double*)state, (double*)eforce, (const CudaVec3<double>*)x); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcForce_kernel<double, CudaVec3<double> >");}
    TetrahedronFEMForceFieldCuda3t_addForce_launch1<double>(bsize,pt,nbVertex, nbElemPerVertex, eforce, velems, f);
}

void TetrahedronFEMForceFieldCuda3d_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
{
    CudaTetrahedronFEMForceFieldInputTextures<double,CudaVec3<double> >::setDX(dx);
    CudaTetrahedronFEMForceFieldTempTextures<double,CudaVec4<double> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<double, CudaVec3<double> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<double>*)elems, (const double*)state, (double*)eforce, (const CudaVec3<double>*)dx, (double) factor); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<double, CudaVec3<double> >");}
    TetrahedronFEMForceFieldCuda3t_addForce_launch1<double>(bsize,pt,nbVertex, nbElemPerVertex, eforce, velems, df);
}

void TetrahedronFEMForceFieldCuda3d1_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
{
    CudaTetrahedronFEMForceFieldInputTextures<double,CudaVec4<double> >::setX(x);
    CudaTetrahedronFEMForceFieldTempTextures<double,CudaVec4<double> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcForce_kernel<double, CudaVec4<double> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<double>*)elems, (double*)state, (double*)eforce, (const CudaVec4<double>*)x); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcForce_kernel<double, CudaVec4<double> >");}
    TetrahedronFEMForceFieldCuda3t1_addForce_launch1<double>(bsize,pt,nbVertex, nbElemPerVertex, eforce, velems, f);
}

void TetrahedronFEMForceFieldCuda3d1_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
{
    CudaTetrahedronFEMForceFieldInputTextures<double,CudaVec4<double> >::setDX(dx);
    CudaTetrahedronFEMForceFieldTempTextures<double,CudaVec4<double> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<double, CudaVec4<double> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<double>*)elems, (const double*)state, (double*)eforce, (const CudaVec4<double>*)dx, (double) factor); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_calcDForce_kernel<double, CudaVec4<double> >");}
    TetrahedronFEMForceFieldCuda3t1_addForce_launch1<double>(bsize,pt,nbVertex, nbElemPerVertex, eforce, velems, df);
}

void TetrahedronFEMForceFieldCuda3d_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbVertex+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_getRotations_kernel<double><<< grid, threads >>>(nbVertex, (const double*)initState, (const double*)state, (const int*)rotationIdx, (double*)rotations); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_getRotations_kernel<double>");}
}

void TetrahedronFEMForceFieldCuda3d_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbElem+BSIZE-1)/BSIZE,1);
    {TetrahedronFEMForceFieldCuda3t_getElementRotations_kernel<double><<< grid, threads >>>(nbElem,(const double*)rotationsAos, (double*)rotations); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_getElementRotations_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
