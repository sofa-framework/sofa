/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>
#include <cuda.h>
#include <stdio.h>

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

template<class real>
class __align__(16) GPUElement
{
public:
    /// index of the 8 connected vertices
    int ia[BSIZE];
    int ib[BSIZE];
    int ic[BSIZE];
    int id[BSIZE];
    int ig[BSIZE];
    int ih[BSIZE];
    int ii[BSIZE];
    int ij[BSIZE];

    /// initial position of the vertices in the local (rotated) coordinate system
    real ax[BSIZE],ay[BSIZE],az[BSIZE];
    real bx[BSIZE],by[BSIZE],bz[BSIZE];
    real cx[BSIZE],cy[BSIZE],cz[BSIZE];
    real dx[BSIZE],dy[BSIZE],dz[BSIZE];
    real gx[BSIZE],gy[BSIZE],gz[BSIZE];
    real hx[BSIZE],hy[BSIZE],hz[BSIZE];
    real ix[BSIZE],iy[BSIZE],iz[BSIZE];
    real jx[BSIZE],jy[BSIZE],jz[BSIZE];

};

template<class real>
class  GPURotation
{
public:
    /// transposed rotation matrix
    matrix3<real> Rt;
};

template<class real>
class  GPUKMatrix
{
public:
    matrix3<real> K0_0;
    matrix3<real> K0_1;
    matrix3<real> K0_2;
    matrix3<real> K0_3;
    matrix3<real> K0_4;
    matrix3<real> K0_5;
    matrix3<real> K0_6;
    matrix3<real> K0_7;

    matrix3<real> K1_0;
    matrix3<real> K1_1;
    matrix3<real> K1_2;
    matrix3<real> K1_3;
    matrix3<real> K1_4;
    matrix3<real> K1_5;
    matrix3<real> K1_6;
    matrix3<real> K1_7;

    matrix3<real> K2_0;
    matrix3<real> K2_1;
    matrix3<real> K2_2;
    matrix3<real> K2_3;
    matrix3<real> K2_4;
    matrix3<real> K2_5;
    matrix3<real> K2_6;
    matrix3<real> K2_7;

    matrix3<real> K3_0;
    matrix3<real> K3_1;
    matrix3<real> K3_2;
    matrix3<real> K3_3;
    matrix3<real> K3_4;
    matrix3<real> K3_5;
    matrix3<real> K3_6;
    matrix3<real> K3_7;

    matrix3<real> K4_0;
    matrix3<real> K4_1;
    matrix3<real> K4_2;
    matrix3<real> K4_3;
    matrix3<real> K4_4;
    matrix3<real> K4_5;
    matrix3<real> K4_6;
    matrix3<real> K4_7;

    matrix3<real> K5_0;
    matrix3<real> K5_1;
    matrix3<real> K5_2;
    matrix3<real> K5_3;
    matrix3<real> K5_4;
    matrix3<real> K5_5;
    matrix3<real> K5_6;
    matrix3<real> K5_7;

    matrix3<real> K6_0;
    matrix3<real> K6_1;
    matrix3<real> K6_2;
    matrix3<real> K6_3;
    matrix3<real> K6_4;
    matrix3<real> K6_5;
    matrix3<real> K6_6;
    matrix3<real> K6_7;

    matrix3<real> K7_0;
    matrix3<real> K7_1;
    matrix3<real> K7_2;
    matrix3<real> K7_3;
    matrix3<real> K7_4;
    matrix3<real> K7_5;
    matrix3<real> K7_6;
    matrix3<real> K7_7;
};


template<class real>
class GPUElementForce
{
public:
    CudaVec4<real> fA,fB,fC,fD,fG,fH,fI,fJ;
};
 
extern "C"
{

void HexahedronFEMForceFieldCuda3f_addForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* rotation, const void * kmatrix, void* eforce, const void* velems, void* f, const void* x, const void* v);
void HexahedronFEMForceFieldCuda3f_addDForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* rotation, const void * kmatrix,  void* eforce, const void* velems, void* df, const void* dx,double kfactor);
void HexaahedronFEMForceFieldCuda3f_getRotations(int gatherpt,int gatherbs,int nbVertex, unsigned int nbElemPerVertex, const void * velems, const void * erotation, const void * irotation,void * nrotation);

void HexahedronFEMForceFieldCuda3f1_addForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* rotation, const void * kmatrix,  void* eforce, const void* velems, void* f, const void* x, const void* v);
void HexahedronFEMForceFieldCuda3f1_addDForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* rotation, const void * kmatrix, void* eforce, const void* velems, void* df, const void* dx,double kfactor);
void HexaahedronFEMForceFieldCuda3f1_getRotations(int gatherpt,int gatherbs,int nbVertex, unsigned int nbElemPerVertex, const void * velems, const void * erotation, const void * irotation,void * nrotation);

} // extern "C"

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
class SharedMemory;

template<>
class SharedMemory<float>
{
public:
    static __device__ float* get()
    {
	extern  __shared__  float ftemp[];
	return ftemp;
    }
};
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 130
template<>
class SharedMemory<double>
{
public:
    static __device__ double* get()
    {
	extern  __shared__  double dtemp[];
	return dtemp;
    }
};
#endif
// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaHexahedronFEMForceFieldInputTextures
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
class CudaHexahedronFEMForceFieldTempTextures
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
class CudaHexahedronFEMForceFieldInputTextures<float, CudaVec3<float> >
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
	int i3 = i * 3;
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
	int i3 = i * 3;
	float x1 = tex1Dfetch(tex_3f_dx, i3);
	float x2 = tex1Dfetch(tex_3f_dx, i3+1);
	float x3 = tex1Dfetch(tex_3f_dx, i3+2);
	return CudaVec3<real>::make(x1,x2,x3);
    }
};

static texture<float4,1,cudaReadModeElementType> tex_3f1_x;
static texture<float4,1,cudaReadModeElementType> tex_3f1_dx;

template<>
class CudaHexahedronFEMForceFieldInputTextures<float, CudaVec4<float> >
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
#endif

#if defined(USE_TEXTURE_ELEMENT_FORCE)

static texture<float,1,cudaReadModeElementType> tex_3f_eforce;

template<>
class CudaHexahedronFEMForceFieldTempTextures<float, CudaVec3<float> >
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
	int i3 = i * 3;
	float x1 = tex1Dfetch(tex_3f_eforce, i3);
	float x2 = tex1Dfetch(tex_3f_eforce, i3+1);
	float x3 = tex1Dfetch(tex_3f_eforce, i3+2);
	return CudaVec3<real>::make(x1,x2,x3);
    }
};

static texture<float4,1,cudaReadModeElementType> tex_3f1_eforce;

template<>
class CudaHexahedronFEMForceFieldTempTextures<float, CudaVec4<float> >
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
#endif


template<typename real, class TIn>
__global__ void HexahedronFEMForceFieldCuda3t_calcForce_kernel(int nbElem, const GPUElement<real>* elems, GPURotation<real>* rotation,const GPUKMatrix<real>* kmatrix, real* eforce, const TIn* x)
{
  int index0 = blockIdx.x * BSIZE; //blockDim.x;
  int index1 = threadIdx.x;
  int index = index0+index1;

  const GPUElement<real>* e = elems + blockIdx.x;
  matrix3<real> Rt;

  register CudaVec3<real> fA, fB, fC, fD;
  register CudaVec3<real> fG, fH, fI, fJ;
 
  if (index < nbElem) 
  {
    CudaVec3<real> IA = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ia[index1], x);
    CudaVec3<real> IB = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ib[index1], x);
    CudaVec3<real> IC = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ic[index1], x);
    CudaVec3<real> ID = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->id[index1], x);
    CudaVec3<real> IG = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ig[index1], x);
    CudaVec3<real> IH = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ih[index1], x);
    CudaVec3<real> II = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ii[index1], x);
    CudaVec3<real> IJ = CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ij[index1], x);

    CudaVec3<real> horizontal = IB-IA + IC-ID + IH-IG + II-IJ;
    CudaVec3<real> vertical = ID-IA + IC-IB + IJ-IG + II-IH;

    CudaVec3<real> dA, dB, dC, dD, dG, dH, dI, dJ;
    //real nor = norm(horizontal);
    Rt.x = horizontal; // /nor;
    Rt.x *= invnorm(Rt.x);

    Rt.z = cross(horizontal, vertical);
    Rt.y = cross(Rt.z, horizontal);
    Rt.y *= invnorm(Rt.y);
    Rt.z *= invnorm(Rt.z);

    real e_x = e->ax[index1];
    real e_y = e->ay[index1];
    real e_z = e->az[index1];
    dA.x = e_x - Rt.mulX(IA);
    dA.y = e_y - Rt.mulY(IA);
    dA.z = e_z - Rt.mulZ(IA);

    e_x = e->bx[index1];
    e_y = e->by[index1];
    e_z = e->bz[index1];
    dB.x = e_x - Rt.mulX(IB);
    dB.y = e_y - Rt.mulY(IB);
    dB.z = e_z - Rt.mulZ(IB);

    e_x = e->cx[index1];
    e_y = e->cy[index1];
    e_z = e->cz[index1];
    dC.x = e_x - Rt.mulX(IC);
    dC.y = e_y - Rt.mulY(IC);
    dC.z = e_z - Rt.mulZ(IC);

    e_x = e->dx[index1];
    e_y = e->dy[index1];
    e_z = e->dz[index1];
    dD.x = e_x - Rt.mulX(ID);
    dD.y = e_y - Rt.mulY(ID);
    dD.z = e_z - Rt.mulZ(ID);

    e_x = e->gx[index1];
    e_y = e->gy[index1];
    e_z = e->gz[index1];
    dG.x = e_x - Rt.mulX(IG);
    dG.y = e_y - Rt.mulY(IG);
    dG.z = e_z - Rt.mulZ(IG);

    e_x = e->hx[index1];
    e_y = e->hy[index1];
    e_z = e->hz[index1];
    dH.x = e_x - Rt.mulX(IH);
    dH.y = e_y - Rt.mulY(IH);
    dH.z = e_z - Rt.mulZ(IH);

    e_x = e->ix[index1];
    e_y = e->iy[index1];
    e_z = e->iz[index1];
    dI.x = e_x - Rt.mulX(II);
    dI.y = e_y - Rt.mulY(II);
    dI.z = e_z - Rt.mulZ(II);

    e_x = e->jx[index1];
    e_y = e->jy[index1];
    e_z = e->jz[index1];
    dJ.x = e_x - Rt.mulX(IJ);
    dJ.y = e_y - Rt.mulY(IJ);
    dJ.z = e_z - Rt.mulZ(IJ);


    GPURotation<real>* s = (GPURotation<real>*)(rotation + index);
    s->Rt.x = Rt.x;
    s->Rt.y = Rt.y;
    s->Rt.z = Rt.z;

    const GPUKMatrix<real>* k = (const GPUKMatrix<real>*)(kmatrix + index);
    fA = k->K0_0*dA;
    fA += k->K0_1*dB;
    fA += k->K0_2*dC;
    fA += k->K0_3*dD;
    fA += k->K0_4*dG;
    fA += k->K0_5*dH;
    fA += k->K0_6*dI;
    fA += k->K0_7*dJ;

    fA = Rt.mulT(fA);

    fB = k->K1_0*dA;
    fB += k->K1_1*dB;
    fB += k->K1_2*dC;
    fB += k->K1_3*dD;
    fB += k->K1_4*dG;
    fB += k->K1_5*dH;
    fB += k->K1_6*dI;
    fB += k->K1_7*dJ;
 		
    fB = Rt.mulT(fB);

    fC = k->K2_0*dA;
    fC += k->K2_1*dB;
    fC += k->K2_2*dC;
    fC += k->K2_3*dD;
    fC += k->K2_4*dG;
    fC += k->K2_5*dH;
    fC += k->K2_6*dI;
    fC += k->K2_7*dJ;
 
    fC = Rt.mulT(fC);


    fD = k->K3_0*dA;
    fD += k->K3_1*dB;
    fD += k->K3_2*dC;
    fD += k->K3_3*dD;
    fD += k->K3_4*dG;
    fD += k->K3_5*dH;
    fD += k->K3_6*dI;
    fD += k->K3_7*dJ;

    fD = Rt.mulT(fD);


    fG = k->K4_0*dA;
    fG += k->K4_1*dB;
    fG += k->K4_2*dC;
    fG += k->K4_3*dD;
    fG += k->K4_4*dG;
    fG += k->K4_5*dH;
    fG += k->K4_6*dI;
    fG += k->K4_7*dJ;
    
    fG = Rt.mulT(fG);

    fH = k->K5_0*dA;
    fH += k->K5_1*dB;
    fH += k->K5_2*dC;
    fH += k->K5_3*dD;
    fH += k->K5_4*dG;
    fH += k->K5_5*dH;
    fH += k->K5_6*dI;
    fH += k->K5_7*dJ;
    
    fH = Rt.mulT(fH);

    fI = k->K6_0*dA;
    fI += k->K6_1*dB;
    fI += k->K6_2*dC;
    fI += k->K6_3*dD;
    fI += k->K6_4*dG;
    fI += k->K6_5*dH;
    fI += k->K6_6*dI;
    fI += k->K6_7*dJ;
    
    fI = Rt.mulT(fI);

    fJ = k->K7_0*dA;
    fJ += k->K7_1*dB;
    fJ += k->K7_2*dC;
    fJ += k->K7_3*dD;
    fJ += k->K7_4*dG;
    fJ += k->K7_5*dH;
    fJ += k->K7_6*dI;
    fJ += k->K7_7*dJ;

    fJ = Rt.mulT(fJ);
  }
    
  int HALFBSIZE = BSIZE<<1;
  extern __shared__ real temp[];
  int index13 = index1 * 13;

  temp[index13+0 ] = fA.x;
  temp[index13+1 ] = fA.y;
  temp[index13+2 ] = fA.z;
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

  // we write only the 4th first vertices of hexahedron
  real* out = ((real*)eforce)+(blockIdx.x*BSIZE*32)+index1+(index1>>4)*16;
  real v = 0;
  index1 += (index1>>4) - (index1>>2);

   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;

  __syncthreads();

  temp[index13+0 ] = fG.x;
  temp[index13+1 ] = fG.y;
  temp[index13+2 ] = fG.z;
  temp[index13+3 ] = fH.x;
  temp[index13+4 ] = fH.y;
  temp[index13+5 ] = fH.z;
  temp[index13+6 ] = fI.x;
  temp[index13+7 ] = fI.y;
  temp[index13+8 ] = fI.z;
  temp[index13+9 ] = fJ.x;
  temp[index13+10] = fJ.y;
  temp[index13+11] = fJ.z;

  __syncthreads();

  index1 = threadIdx.x;
  out = ((real*)eforce)+(blockIdx.x*BSIZE*32)+index1+(index1>>4)*16+16;
  index1 += (index1>>4) - (index1>>2);

   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;

}


template<typename real, class TIn>
__global__ void HexahedronFEMForceFieldCuda3t_calcDForce_kernel(int nbElem, const GPUElement<real>* elems, const GPURotation<real>* rotation,const GPUKMatrix<real>* kmatrix, real* eforce, const TIn* x, real fact) {
  int index0 = blockIdx.x * BSIZE; //blockDim.x;
  int index1 = threadIdx.x;
  int index = index0+index1;

  const GPUElement<real>* e = elems + blockIdx.x;
  matrix3<real> Rt;
  CudaVec3<real> fA = CudaVec3<real>::make(0.0f,0.0f,0.0f);
  CudaVec3<real> fB = CudaVec3<real>::make(0.0f,0.0f,0.0f);
  CudaVec3<real> fC = CudaVec3<real>::make(0.0f,0.0f,0.0f);
  CudaVec3<real> fD = CudaVec3<real>::make(0.0f,0.0f,0.0f);
  CudaVec3<real> fG = CudaVec3<real>::make(0.0f,0.0f,0.0f);
  CudaVec3<real> fH = CudaVec3<real>::make(0.0f,0.0f,0.0f);
  CudaVec3<real> fI = CudaVec3<real>::make(0.0f,0.0f,0.0f);
  CudaVec3<real> fJ = CudaVec3<real>::make(0.0f,0.0f,0.0f);

  if (index < nbElem) 
  {
    GPURotation<real>* s = (GPURotation<real>*)(rotation + index);
    Rt.x = s->Rt.x;
    Rt.y = s->Rt.y;
    Rt.z = s->Rt.z;

    CudaVec3<real> dA = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ia[index1], x);
    CudaVec3<real> dB = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ib[index1], x);
    CudaVec3<real> dC = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ic[index1], x);
    CudaVec3<real> dD = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->id[index1], x);
    CudaVec3<real> dG = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ig[index1], x);
    CudaVec3<real> dH = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ih[index1], x);
    CudaVec3<real> dI = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ii[index1], x);
    CudaVec3<real> dJ = Rt*CudaHexahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ij[index1], x);

    const GPUKMatrix<real>* k = (const GPUKMatrix<real>*)(kmatrix + index);

    fA = k->K0_0*dA;
    fA += k->K0_1*dB;
    fA += k->K0_2*dC;
    fA += k->K0_3*dD;
    fA += k->K0_4*dG;
    fA += k->K0_5*dH;
    fA += k->K0_6*dI;
    fA += k->K0_7*dJ;

    fA = Rt.mulT(fA)*(-fact);

    fB = k->K1_0*dA;
    fB += k->K1_1*dB;
    fB += k->K1_2*dC;
    fB += k->K1_3*dD;
    fB += k->K1_4*dG;
    fB += k->K1_5*dH;
    fB += k->K1_6*dI;
    fB += k->K1_7*dJ;
 		
    fB = Rt.mulT(fB)*(-fact);

    fC = k->K2_0*dA;
    fC += k->K2_1*dB;
    fC += k->K2_2*dC;
    fC += k->K2_3*dD;
    fC += k->K2_4*dG;
    fC += k->K2_5*dH;
    fC += k->K2_6*dI;
    fC += k->K2_7*dJ;
 
    fC = Rt.mulT(fC)*(-fact);


    fD = k->K3_0*dA;
    fD += k->K3_1*dB;
    fD += k->K3_2*dC;
    fD += k->K3_3*dD;
    fD += k->K3_4*dG;
    fD += k->K3_5*dH;
    fD += k->K3_6*dI;
    fD += k->K3_7*dJ;

    fD = Rt.mulT(fD)*(-fact);


    fG = k->K4_0*dA;
    fG += k->K4_1*dB;
    fG += k->K4_2*dC;
    fG += k->K4_3*dD;
    fG += k->K4_4*dG;
    fG += k->K4_5*dH;
    fG += k->K4_6*dI;
    fG += k->K4_7*dJ;
    
    fG = Rt.mulT(fG)*(-fact);

    fH = k->K5_0*dA;
    fH += k->K5_1*dB;
    fH += k->K5_2*dC;
    fH += k->K5_3*dD;
    fH += k->K5_4*dG;
    fH += k->K5_5*dH;
    fH += k->K5_6*dI;
    fH += k->K5_7*dJ;
    
    fH = Rt.mulT(fH)*(-fact);

    fI = k->K6_0*dA;
    fI += k->K6_1*dB;
    fI += k->K6_2*dC;
    fI += k->K6_3*dD;
    fI += k->K6_4*dG;
    fI += k->K6_5*dH;
    fI += k->K6_6*dI;
    fI += k->K6_7*dJ;
    
    fI = Rt.mulT(fI)*(-fact);

    fJ = k->K7_0*dA;
    fJ += k->K7_1*dB;
    fJ += k->K7_2*dC;
    fJ += k->K7_3*dD;
    fJ += k->K7_4*dG;
    fJ += k->K7_5*dH;
    fJ += k->K7_6*dI;
    fJ += k->K7_7*dJ;

    fJ = Rt.mulT(fJ)*(-fact);

  }

  int HALFBSIZE = BSIZE<<1;
  real* temp = SharedMemory<real>::get();

  int index13 = index1 * 13;
  temp[index13+0 ] = fA.x;
  temp[index13+1 ] = fA.y;
  temp[index13+2 ] = fA.z;
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

  // we write only the 4th first vertices of hexahedron
  real* out = ((real*)eforce)+(blockIdx.x*BSIZE*32)+index1+(index1>>4)*16;
  real v = 0;
  index1 += (index1>>4) - (index1>>2);

   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;

  __syncthreads();

  temp[index13+0 ] = fG.x;
  temp[index13+1 ] = fG.y;
  temp[index13+2 ] = fG.z;
  temp[index13+3 ] = fH.x;
  temp[index13+4 ] = fH.y;
  temp[index13+5 ] = fH.z;
  temp[index13+6 ] = fI.x;
  temp[index13+7 ] = fI.y;
  temp[index13+8 ] = fI.z;
  temp[index13+9 ] = fJ.x;
  temp[index13+10] = fJ.y;
  temp[index13+11] = fJ.z;

  __syncthreads();


  index1 = threadIdx.x;
  out = ((real*)eforce)+(blockIdx.x*BSIZE*32)+index1+(index1>>4)*16+16;
  index1 += (index1>>4) - (index1>>2);

  v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;
   v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2)); 
  *out = v; out += HALFBSIZE;

}

template<typename real, int BSIZE>
__global__ void HexahedronFEMForceFieldCuda3t_addForce1_kernel(int nbVertex, unsigned int nbElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;
    int index3 = 3 * index1;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE*3];

    int iext = blockIdx.x*BSIZE*3+index1; //index0*3+index1;

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=index0*nbElemPerVertex+index1;

    if (index0+index1 < nbVertex)
        for (int s = 0; s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            if (i == -1) break;
            velems+=BSIZE;
            //if (i != -1)
            {
                force += CudaHexahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
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

template<typename real, int BSIZE>
__global__ void HexahedronFEMForceFieldCuda3t_addForce4_kernel(int /*nbVertex*/, unsigned int nb4ElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE*3];

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=(index0*nb4ElemPerVertex)+index1;

    for (int s = 0; s < nb4ElemPerVertex; s++)
    {
        int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;
        //if (i != -1)
        {
            force += CudaHexahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
        }
    }

    int iout = ((index1>>2) + ((index1&3)*(BSIZE/4)))*3;
    temp[iout  ] = force.x;
    temp[iout+1] = force.y;
    temp[iout+2] = force.z;

    __syncthreads();

    // we need to merge 4 values together
    if (index1 < (BSIZE/4)*3)
    {

        real res = temp[index1] + temp[index1+ (BSIZE/4)*3] + temp[index1+ 2*(BSIZE/4)*3] + temp[index1+ 3*(BSIZE/4)*3];

        int iext = blockIdx.x*(BSIZE/4)*3+index1; //index0*3+index1;

        f[iext] += res;
    }
}

template<typename real, int BSIZE>
__global__ void HexahedronFEMForceFieldCuda3t_addForce8_kernel(int /*nbVertex*/, unsigned int nb8ElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE/2*3];

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=(index0*nb8ElemPerVertex)+index1;

    for (int s = 0; s < nb8ElemPerVertex; s++)
    {
        int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;
        //if (i != -1)
        {
            force += CudaHexahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
        }
    }

    int iout = ((index1>>3) + ((index1&3)*(BSIZE/8)))*3;
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

        int iext = blockIdx.x*(BSIZE/8)*3+index1; //index0*3+index1;

        f[iext] += res;
    }
}

template<typename real, int BSIZE>
__global__ void HexahedronFEMForceFieldCuda3t_getRotations1_kernel(int nbVertex, unsigned int nbElemPerVertex, const int* velems, const GPURotation<real>* erotation, const GPURotation<real>* irotation,real * nrotation)
{
    int index0 = blockIdx.x * BSIZE; //blockDim.x;
    int index1 = threadIdx.x;
    int index9 = index1 * 9;

    __shared__  real temp[BSIZE*9];

    int iext = index0*9+index1;

    matrix3<real> R = matrix3<real>::make();
    int ne=0;

    velems += index0 * nbElemPerVertex + index1;

    if (index0+index1<nbVertex) {
        for (int s = 0;s < nbElemPerVertex; s++)
        {
            const int i = *velems -1;
            if (i == -1) break;
            velems+=BSIZE;

            int eid = i/8; // divide by the num of point per elements to get the index of the element
            R += erotation[eid].Rt.mulT(irotation[eid].Rt);
            ne++;
        }
    }

    if (ne>0) {
        real norm = 1.0/ne;
        R.x.x *= norm;R.x.y *= norm;R.x.z *= norm;
        R.y.x *= norm;R.y.y *= norm;R.y.z *= norm;
        R.z.x *= norm;R.z.y *= norm;R.z.z *= norm;
    }

    temp[index9  ] = R.x.x;
    temp[index9+1] = R.x.y;
    temp[index9+2] = R.x.z;
    temp[index9+3] = R.y.x;
    temp[index9+4] = R.y.y;
    temp[index9+5] = R.y.z;
    temp[index9+6] = R.z.x;
    temp[index9+7] = R.z.y;
    temp[index9+8] = R.z.z;

    __syncthreads();

    nrotation[iext        ] = temp[index1        ];
    nrotation[iext+  BSIZE] = temp[index1+  BSIZE];
    nrotation[iext+2*BSIZE] = temp[index1+2*BSIZE];
    nrotation[iext+3*BSIZE] = temp[index1+3*BSIZE];
    nrotation[iext+4*BSIZE] = temp[index1+4*BSIZE];
    nrotation[iext+5*BSIZE] = temp[index1+5*BSIZE];
    nrotation[iext+6*BSIZE] = temp[index1+6*BSIZE];
    nrotation[iext+7*BSIZE] = temp[index1+7*BSIZE];
    nrotation[iext+8*BSIZE] = temp[index1+8*BSIZE];
}

template<typename real, int BSIZE>
__global__ void HexahedronFEMForceFieldCuda3t_getRotations4_kernel(int nbVertex9, unsigned int nb4ElemPerVertex, const int* velems, const GPURotation<real>* erotation, const GPURotation<real>* irotation,real * nrotation)
{
    int index0 = blockIdx.x * BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    __shared__  real temp[BSIZE*9];

    matrix3<real> R = matrix3<real>::make();
    int ne=0;

    velems+=index0 * nb4ElemPerVertex+index1;

    for (int s = 0;s < nb4ElemPerVertex; s++)
    {
        const int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;

        int eid = i/8; // divide by the num of point per elements to get the index of the element
        R += erotation[eid].Rt.mulT(irotation[eid].Rt);
        ne++;
    }

    if (ne>0) {
        real norm = 1.0/ne;
        R.x.x *= norm;R.x.y *= norm;R.x.z *= norm;
        R.y.x *= norm;R.y.y *= norm;R.y.z *= norm;
        R.z.x *= norm;R.z.y *= norm;R.z.z *= norm;
    } else {
        R.x.x = 1.0;R.y.y = 1.0;R.z.z = 1.0;
    }

    //int iout = (index1>>2)*3 + (index1&3)*((BSIZE/4)*3);
    int iout = ((index1>>2) + ((index1&3)*(BSIZE/4)))*9;
    temp[iout  ] = R.x.x;
    temp[iout+1] = R.x.y;
    temp[iout+2] = R.x.z;
    temp[iout+3] = R.y.x;
    temp[iout+4] = R.y.y;
    temp[iout+5] = R.y.z;
    temp[iout+6] = R.z.x;
    temp[iout+7] = R.z.y;
    temp[iout+8] = R.z.z;

    __syncthreads();

    int iext = blockIdx.x*(BSIZE/4)*9+index1; //index0*3+index1;

    if (iext>=nbVertex9) return;
    nrotation[iext] = (temp[index1] + temp[index1+ (BSIZE/4)*9] + temp[index1+ 2*(BSIZE/4)*9] + temp[index1+ 3*(BSIZE/4)*9])*0.25;
    index1+=BSIZE;
    iext  +=BSIZE;

    if (iext>=nbVertex9) return;
    nrotation[iext] = (temp[index1] + temp[index1+ (BSIZE/4)*9] + temp[index1+ 2*(BSIZE/4)*9] + temp[index1+ 3*(BSIZE/4)*9])*0.25;
    index1+=BSIZE;
    iext  +=BSIZE;

    if (index1 >= (BSIZE/4)*9 || iext>=nbVertex9) return;
    nrotation[iext] = (temp[index1] + temp[index1+ (BSIZE/4)*9] + temp[index1+ 2*(BSIZE/4)*9] + temp[index1+ 3*(BSIZE/4)*9])*0.25;
}

template<typename real, int BSIZE>
__global__ void HexahedronFEMForceFieldCuda3t_getRotations8_kernel(int nbVertex9, unsigned int nb8ElemPerVertex, const int* velems, const GPURotation<real>* erotation, const GPURotation<real>* irotation,real * nrotation)
{
    int index0 = blockIdx.x * BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    __shared__  real temp[BSIZE/2*9];

    matrix3<real> R = matrix3<real>::make();
    int ne=0;

    velems+=index0 * nb8ElemPerVertex+index1;

    for (int s = 0;s < nb8ElemPerVertex; s++)
    {
        const int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;

        int eid = i/8; // divide by the num of point per elements to get the index of the element
        R += erotation[eid].Rt.mulT(irotation[eid].Rt);
        ne++;
    }

    if (ne>0) {
        real norm = 1.0/ne;
        R.x.x *= norm;R.x.y *= norm;R.x.z *= norm;
        R.y.x *= norm;R.y.y *= norm;R.y.z *= norm;
        R.z.x *= norm;R.z.y *= norm;R.z.z *= norm;
    } else {
        R.x.x = 1.0;R.y.y = 1.0;R.z.z = 1.0;
    }

    int iout = ((index1>>3) + ((index1&3)*(BSIZE/8)))*9;
    if (index1&4)
    {
        temp[iout  ] = R.x.x;
        temp[iout+1] = R.x.y;
        temp[iout+2] = R.x.z;
        temp[iout+3] = R.y.x;
        temp[iout+4] = R.y.y;
        temp[iout+5] = R.y.z;
        temp[iout+6] = R.z.x;
        temp[iout+7] = R.z.y;
        temp[iout+8] = R.z.z;
    }

    __syncthreads();

    if (!(index1&4))
    {
        temp[iout  ] = R.x.x;
        temp[iout+1] = R.x.y;
        temp[iout+2] = R.x.z;
        temp[iout+3] = R.y.x;
        temp[iout+4] = R.y.y;
        temp[iout+5] = R.y.z;
        temp[iout+6] = R.z.x;
        temp[iout+7] = R.z.y;
        temp[iout+8] = R.z.z;
    }

    __syncthreads();

    int iext = blockIdx.x*(BSIZE/8)*9+index1; //index0*3+index1;
    if (iext>=nbVertex9) return;
    nrotation[iext] = (temp[index1] + temp[index1+ (BSIZE/8)*9] + temp[index1+ 2*(BSIZE/8)*9] + temp[index1+ 3*(BSIZE/8)*9])*0.25;

    index1+=BSIZE;
    iext  +=BSIZE;

    if (index1 >= (BSIZE/8)*9 || iext>=nbVertex9) return;
    nrotation[iext] = (temp[index1] + temp[index1+ (BSIZE/8)*9] + temp[index1+ 2*(BSIZE/8)*9] + temp[index1+ 3*(BSIZE/8)*9])*0.25;
}

//////////////////////
// CPU-side methods //
//////////////////////

template<typename real,int BSIZE>
inline void HexahedronFEMForceFieldCuda3t_addForce_launch2(int gatherpt,unsigned int nbVertex, unsigned int nbElemPerVertex, void* eforce, const void* velems, void* f)
{
    if (gatherpt==1) {
        dim3 threads(BSIZE,1);
        dim3 grid((nbVertex+BSIZE-1)/BSIZE,1);
        nbElemPerVertex = nbElemPerVertex;
        {HexahedronFEMForceFieldCuda3t_addForce1_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (real*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_addForce1_kernel<real>");}
    } else if (gatherpt==4) {
        dim3 threads(BSIZE,1);
        dim3 grid((nbVertex*4+BSIZE-1)/BSIZE,1);
        nbElemPerVertex = (nbElemPerVertex + 4-1)/4;
        {HexahedronFEMForceFieldCuda3t_addForce4_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (real*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_addForce4_kernel<real>");}
    } else if (gatherpt==8) {
        dim3 threads(BSIZE,1);
        dim3 grid((nbVertex*8+BSIZE-1)/BSIZE,1);
        nbElemPerVertex = (nbElemPerVertex + 8-1)/8;
        {HexahedronFEMForceFieldCuda3t_addForce8_kernel<real,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, (const CudaVec4<real>*)eforce, (const int*)velems, (real*)f); mycudaDebugError("TetrahedronFEMForceFieldCuda3t_addForce8_kernel<real>");}
    } else {
        printf("ERROR GATHERPT must be equal to 1 4 or 8\n");
    }
}

template<typename real>
inline void HexahedronFEMForceFieldCuda3t_addForce_launch(int gatherpt,int gatherbs,unsigned int nbVertex, unsigned int nbElemPerVertex, void* eforce, const void* velems, void* f)
{
    if (gatherbs==32) {
        HexahedronFEMForceFieldCuda3t_addForce_launch2<real, 32>(gatherpt,nbVertex, nbElemPerVertex, eforce, velems, f);
    } else if (gatherbs==64) {
        HexahedronFEMForceFieldCuda3t_addForce_launch2<real, 64>(gatherpt,nbVertex, nbElemPerVertex, eforce, velems, f);
    } else if (gatherbs==128) {
        HexahedronFEMForceFieldCuda3t_addForce_launch2<real, 128>(gatherpt,nbVertex, nbElemPerVertex, eforce, velems, f);
    } else if (gatherbs==256) {
        HexahedronFEMForceFieldCuda3t_addForce_launch2<real, 256>(gatherpt,nbVertex, nbElemPerVertex, eforce, velems, f);
    } else {
        printf("ERROR GATHERBSIZE must be equal to 32 64 128 or 256\n");
    }
}

template<typename real,int BSIZE>
inline void HexahedronFEMForceFieldCuda3t_getRotations_launch2(int gatherpt,int nbVertex, unsigned int nbElemPerVertex, const void * velems, const void * erotation, const void * irotation,void * nrotation)
{
    if (gatherpt==1) {
        dim3 threads(BSIZE,1);
        dim3 grid((nbVertex+BSIZE-1)/BSIZE,1);
        nbElemPerVertex = nbElemPerVertex;
        { HexahedronFEMForceFieldCuda3t_getRotations1_kernel<real,BSIZE><<< grid, threads >>>(nbVertex,nbElemPerVertex,(const int *) velems,(const GPURotation<real>*)erotation,(const GPURotation<real>*)irotation,(real*) nrotation);mycudaDebugError("HexaahedronFEMForceFieldCuda3t_getRotations_kernel<real, CudaVec3<real> >");}
    } else if (gatherpt==4) {
        dim3 threads(BSIZE,1);
        dim3 grid((nbVertex*4+BSIZE-1)/BSIZE,1);
        nbElemPerVertex = (nbElemPerVertex + 4-1)/4;
        { HexahedronFEMForceFieldCuda3t_getRotations4_kernel<real,BSIZE><<< grid, threads >>>(nbVertex*9,nbElemPerVertex,(const int *) velems,(const GPURotation<real>*)erotation,(const GPURotation<real>*)irotation,(real*) nrotation);mycudaDebugError("HexaahedronFEMForceFieldCuda3t_getRotations_kernel<real, CudaVec3<real> >");}
    } else if (gatherpt==8) {
        dim3 threads(BSIZE,1);
        dim3 grid((nbVertex*8+BSIZE-1)/BSIZE,1);
        nbElemPerVertex = (nbElemPerVertex + 8-1)/8;
        { HexahedronFEMForceFieldCuda3t_getRotations8_kernel<real,BSIZE><<< grid, threads >>>(nbVertex*9,nbElemPerVertex,(const int *) velems,(const GPURotation<real>*)erotation,(const GPURotation<real>*)irotation,(real*) nrotation);mycudaDebugError("HexaahedronFEMForceFieldCuda3t_getRotations_kernel<real, CudaVec3<real> >");}
    } else {
        printf("ERROR GATHERPT must be equal to 1 4 or 8\n");
    }
}

template<typename real>
inline void HexahedronFEMForceFieldCuda3t_getRotations_launch(int gatherpt,int gatherbs,int nbVertex, unsigned int nbElemPerVertex, const void * velems, const void * erotation, const void * irotation,void * nrotation)
{
    if (gatherbs==32) {
        HexahedronFEMForceFieldCuda3t_getRotations_launch2<real, 32>(gatherpt,nbVertex,nbElemPerVertex,velems,erotation,irotation,nrotation);
    } else if (gatherbs==64) {
        HexahedronFEMForceFieldCuda3t_getRotations_launch2<real, 64>(gatherpt,nbVertex,nbElemPerVertex,velems,erotation,irotation,nrotation);
    } else if (gatherbs==128) {
        HexahedronFEMForceFieldCuda3t_getRotations_launch2<real, 128>(gatherpt,nbVertex,nbElemPerVertex,velems,erotation,irotation,nrotation);
    } else if (gatherbs==256) {
        HexahedronFEMForceFieldCuda3t_getRotations_launch2<real, 256>(gatherpt,nbVertex,nbElemPerVertex,velems,erotation,irotation,nrotation);
    } else {
        printf("ERROR GATHERBSIZE must be equal to 32 64 128 or 256\n");
    }
}


void HexahedronFEMForceFieldCuda3f_addForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* rotation, const void * kmatrix,void* eforce, const void* velems, void* f, const void* x, const void* /*v*/)
{
    CudaHexahedronFEMForceFieldInputTextures<float,CudaVec3<float> >::setX(x);
    CudaHexahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
        {HexahedronFEMForceFieldCuda3t_calcForce_kernel<float, CudaVec3<float> ><<< grid1, threads1, BSIZE*13*sizeof(float)>>>(nbElem, (const GPUElement<float>*)elems, (GPURotation<float>*)rotation, (const GPUKMatrix<float>*)kmatrix, (float*)eforce, (const CudaVec3<float>*)x); mycudaDebugError("HexahedronFEMForceFieldCuda3t_calcForce_kernel<float, CudaVec3<float> >");}

    HexahedronFEMForceFieldCuda3t_addForce_launch<float>(gatherpt,gatherbs,nbVertex, nbElemPerVertex, eforce, velems, f);
}

void HexahedronFEMForceFieldCuda3f_addDForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* rotation, const void * kmatrix, void* eforce, const void* velems, void* df, const void* dx,double kfactor)
{
    CudaHexahedronFEMForceFieldInputTextures<float,CudaVec3<float> >::setDX(dx);
    CudaHexahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
        {HexahedronFEMForceFieldCuda3t_calcDForce_kernel<float, CudaVec3<float> ><<< grid1, threads1, BSIZE*13*sizeof(float)>>>(nbElem, (const GPUElement<float>*)elems, (const GPURotation<float>*)rotation, (const GPUKMatrix<float>*)kmatrix, (float*)eforce, (const CudaVec3<float>*)dx,kfactor); mycudaDebugError("HexahedronFEMForceFieldCuda3t_calcDForce_kernel<float, CudaVec3<float> >");}

    HexahedronFEMForceFieldCuda3t_addForce_launch<float>(gatherpt,gatherbs,nbVertex, nbElemPerVertex, eforce, velems, df);
}


void HexaahedronFEMForceFieldCuda3f_getRotations(int gatherpt,int gatherbs,int nbVertex, unsigned int nbElemPerVertex, const void * velems, const void * erotation, const void * irotation,void * nrotation)
{
    HexahedronFEMForceFieldCuda3t_getRotations_launch<float>(gatherpt,gatherbs,nbVertex,nbElemPerVertex,velems,erotation,irotation,nrotation);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
