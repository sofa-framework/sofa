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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>
#include "cuda.h"

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

template<class real>
class GPUSPHFluid
{
public:
    real h;         ///< particles radius
    real h2;        ///< particles radius squared
    real inv_h2;    ///< particles radius squared inverse
    real stiffness; ///< pressure stiffness
    real mass;      ///< particles mass
    real mass2;     ///< particles mass squared
    real density0;  ///< 1000 kg/m3 for water
    real viscosity;
    real surfaceTension;

    // Precomputed constants for smoothing kernels
    real CWd;          ///< =     constWd(h)
    //real CgradWd;      ///< = constGradWd(h)
    real CgradWp;      ///< = constGradWp(h)
    real ClaplacianWv; ///< =  constLaplacianWv(h)
};

typedef GPUSPHFluid<float> GPUSPHFluid3f;
typedef GPUSPHFluid<double> GPUSPHFluid3d;

extern "C"
{

    void SPHFluidForceFieldCuda3f_computeDensity(int kernelType, int pressureType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3f_addForce (int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v);
//void SPHFluidForceFieldCuda3f_addDForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v, const void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void SPHFluidForceFieldCuda3d_computeDensity(int kernelType, int pressureType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3d_addForce (int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v);
//void SPHFluidForceFieldCuda3d_addDForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v, const void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE

}

//////////////////////
// GPU-side methods //
//////////////////////

#ifndef R_PI
#ifdef M_PI
#define R_PI M_PI
#else
#define R_PI 3.141592653589793238462
#endif
#endif

template<class real, int kernelType>
class SPH;

template<class real>
class SPH<real,1>
{
public:
    typedef real Real;
    typedef CudaVec3<Real> Deriv;
    static Real constW(Real h)
    {
        return (Real) (48/R_PI)/(h*h*h);
    }

    static __device__ Real  W0(Real C)
    {
        return C*((Real)(1.0/6.0));
    }
    static __device__ Real  W0()
    {
        return ((Real)(1.0/6.0));
    }
    static __device__ Real W(Real r_h, Real C)
    {
        Real s = 1-r_h;
        if (r_h < (Real)0.5)   return C*((Real)(1.0/6.0) - r_h*r_h*s);
        else                   return C*((Real)(1.0/3.0) * (s*s*s));
    }
    static __device__ Real W2(Real r2_h2, Real C)
    {
        Real r_h = sqrtf(r2_h2);
        Real s = 1-r_h;
        if (r2_h2 < (Real)0.25) return C*((Real)(1.0/6.0) - r2_h2*s);
        else                    return C*((Real)(1.0/3.0) * (s*s*s));
    }
    static __device__ Real W2(Real r2_h2)
    {
        Real r_h = sqrtf(r2_h2);
        Real s = 1-r_h;
        if (r2_h2 < (Real)0.25) return   ((Real)(1.0/6.0) - r2_h2*s);
        else                    return   ((Real)(1.0/3.0) * (s*s*s));
    }
    static Real constGradW(Real h)
    {
        return constW(h)/(h*h);
    }
    static __device__ Deriv gradW(const Deriv& d, Real r_h, Real C)
    {
        Real g;
        if (r_h < (Real)0.5)    g = 3*r_h - 2;
        else  { Real s = 1-r_h; g = -s*s/r_h; }
        return d*(C*g);
    }
    static Real  constLaplacianW(Real h)
    {
        return constW(h)/(h*h);
    }
    static __device__ Real  laplacianW(Real r_h, Real C)
    {
        if (r_h < (Real)0.5)    return C*(12*r_h-6);
        else if (r_h < (Real)1) return C*(6-4*r_h-2/r_h);
        else return 0;
    }
    /// Density Smoothing Kernel
    static Real  constWd(Real h)
    {
        return constW(h);
    }
    static __device__ Real  Wd(Real r_h, Real C)
    {
        return W(r_h,C);
    }
    static __device__ Real  Wd2(Real r2_h2, Real C)
    {
        return W2(r2_h2,C);
    }
    static __device__ Real  Wd2(Real r2_h2)
    {
        return W2(r2_h2);
    }
    static __device__ Real  Wd0()
    {
        return W0();
    }
    static Real  constGradWd(Real h)
    {
        return constGradW(h);
    }
    static __device__ Deriv gradWd(const Deriv& d, Real r_h, Real C)
    {
        return gradW(d,r_h,C);
    }
    static __device__ Deriv gradWd2(const Deriv& d, Real r2_h2, Real C)
    {
        return gradW(d,sqrtf(r2_h2),C);
    }
    static Real  constLaplacianWd(Real h)
    {
        return constLaplacianW(h);
    }
    static __device__ Real  laplacianWd(Real r_h, Real C)
    {
        return laplacianW(r_h,C);
    }
    static __device__ Real  laplacianWd2(Real r2_h2, Real C)
    {
        return laplacianW(sqrtf(r2_h2),C);
    }

    /// Pressure Smoothing Kernel
    static Real  constWp(Real h)
    {
        return constW(h);
    }
    static __device__ Real  Wp(Real r_h, Real C)
    {
        return W(r_h,C);
    }
    static Real  constGradWp(Real h)
    {
        return constGradW(h);
    }
    static __device__ Deriv gradWp(const Deriv& d, Real r_h, Real C)
    {
        return gradW(d,r_h,C);
    }

    /// Viscosity Smoothing Kernel
    static Real  constWv(Real h)
    {
        return constW(h);
    }
    static __device__ Real  Wv(Real r_h, Real C)
    {
        return W(r_h,C);
    }
    static __device__ Real  Wv2(Real r2_h2, Real r_h, Real C)
    {
        return W2(r2_h2,r_h,C);
    }
    static Real  constGradWv(Real h)
    {
        return constGradW(h);
    }
    static __device__ Deriv gradWv(const Deriv& d, Real r_h, Real C)
    {
        return gradW(d,r_h,C);
    }
    static __device__ Deriv gradWv2(const Deriv& d, Real r2_h2, Real r_h, Real C)
    {
        return gradW(d,r_h,C);
    }
    static Real  constLaplacianWv(Real h)
    {
        return constLaplacianW(h);
    }

    static __device__ Real  laplacianWv(Real r_h, Real C)
    {
        return laplacianW(r_h,C);
    }
};

template<class real>
class SPH<real,0>
{
public:
    typedef real Real;
    typedef CudaVec3<Real> Deriv;
    /// Density Smoothing Kernel:  W = 315 / 64pih9 * (h2 - r2)3 = 315 / 64pih3 * (1 - (r/h)2)3
    static Real  constWd(Real h)
    {
        return (Real)(315 / (64*R_PI*h*h*h));
    }
    static __device__ Real  Wd(Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        return  C*a*a*a;
    }
    static __device__ Real  Wd2(Real r2_h2, Real C)
    {
        Real a = (1-r2_h2);
        return  C*a*a*a;
    }
    static __device__ Real  Wd2(Real r2_h2)
    {
        Real a = (1-r2_h2);
        return  a*a*a;
    }
    static __device__ Real  Wd0(Real C)
    {
        return C;
    }
    static __device__ Real  Wd0()
    {
        return 1;
    }
    static Real  constGradWd(Real h)
    {
        return -6*constWd(h)/(h*h);
    }
    static __device__ Deriv gradWd(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        return d*(C*a*a);
    }
    static __device__ Deriv gradWd2(const Deriv& d, Real r2_h2, Real C)
    {
        Real a = (1-r2_h2);
        return d*(C*a*a);
    }
    static Real  constLaplacianWd(Real h)
    {
        return -6*constWd(h)/(h*h);
    }
    static __device__ Real  laplacianWd(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        return C*((1-r2_h2)*(3-7*r2_h2));
    }
    static __device__ Real  laplacianWd2(Real r2_h2, Real C)
    {
        return C*((1-r2_h2)*(3-7*r2_h2));
    }

    /// Pressure Smoothing Kernel:  W = 15 / pih6 (h - r)3 = 15 / pih3 (1 - r/h)3
    static Real  constWp(Real h)
    {
        return (Real)(15 / (R_PI*h*h*h));
    }
    static __device__ Real  Wp(Real r_h, Real C)
    {
        Real a = (1-r_h);
        return  C*a*a*a;
    }
    static Real  constGradWp(Real h)
    {
        return (-3*constWp(h)) / (h*h);
    }
    static __device__ Deriv gradWp(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h);
        return d * (C*a*a/r_h);
    }

    /// Viscosity Smoothing Kernel:  W = 15/(2pih3) (-r3/2h3 + r2/h2 + h/2r - 1)
    static Real  constWv(Real h)
    {
        return (Real)(15/(2*R_PI*h*h*h));
    }
    static __device__ Real  Wv(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        Real r3_h3 = r2_h2*r_h;
        return C*(-0.5f*r3_h3 + r2_h2 + 0.5f/r_h - 1);
    }
    static __device__ Real  Wv2(Real r2_h2, Real r_h, Real C)
    {
        Real r3_h3 = r2_h2*r_h;
        return C*(-0.5f*r3_h3 + r2_h2 + 0.5f/r_h - 1);
    }
    static Real  constGradWv(Real h)
    {
        return constWv(h)/(h*h);
    }
    static __device__ Deriv gradWv(const Deriv& d, Real r_h, Real C)
    {
        Real r3_h3 = r_h*r_h*r_h;
        return d * (C*(2.0f -3.0f*r_h  - 0.5f/r3_h3));
    }
    static __device__ Deriv gradWv2(const Deriv& d, Real r2_h2, Real r_h, Real C)
    {
        Real r3_h3 = r2_h2*r_h;
        return d * (C*(-3*r_h  + 4 - 1/r3_h3));
    }
    static Real  constLaplacianWv(Real h)
    {
        return 6*constWv(h)/(h*h);
    }

    static __device__ Real  laplacianWv(Real r_h, Real C)
    {
        return C*(1-r_h);
    }
};

template<class real, int kernelType, int pressureType>
__device__ void SPHFluidInitDensity(CudaVec3<real> /*x1*/, real& density, GPUSPHFluid<real>& params)
{
    density = 0; //params.mass * params.CWd; //SPH<real,kernelType>::Wd(0,params.CWd);
}

template<class real, int kernelType, int pressureType>
__device__ void SPHFluidCalcDensity(CudaVec3<real> x1, CudaVec3<real> x2, real& density, GPUSPHFluid<real>& params)
{
    CudaVec3<real> n = x2-x1;
    real d2 = norm2(n);
    if (sqrtf(d2) < params.h)
        //if (d2 < params.h2)
    {
        //real r_h = rsqrtf(params.h2/d2);
        //real r_h = sqrtf(d2/params.h2);
        real r2_h2 = (d2/params.h2);
        real d = SPH<real,kernelType>::Wd2(r2_h2);
        density += d;
    }
}

template<class real, int kernelType, int pressureType>
__device__ void SPHFluidFinalDensity(CudaVec3<real> /*x1*/, real& density, GPUSPHFluid<real>& params)
{
    density = (SPH<real,kernelType>::Wd0()+density) * params.CWd * params.mass; //SPH<real,kernelType>::Wd(0,params.CWd);
}

template<class real, int kernelType, int pressureType, int viscosityType, int surfaceTensionType>
__device__ void SPHFluidCalcForce(CudaVec4<real> x1, CudaVec3<real> v1, CudaVec4<real> x2, CudaVec3<real> v2, CudaVec3<real>& force, GPUSPHFluid<real>& params)
{
    CudaVec3<real> n = CudaVec3<real>::make(x2.x-x1.x,x2.y-x1.y,x2.z-x1.z);
    real d2 = norm2(n);
    if (d2 < params.h2)
    {
        //real r_h = rsqrtf(params.h2/d2);
        //real r_h = sqrtf(d2/params.h2);
        real r2_h2 = (d2/params.h2);
        real r_h = sqrtf(r2_h2);
        real density1 = x1.w;
        real density2 = x2.w;

        // Pressure
        real pressure1 = params.stiffness * (density1 - params.density0);
        real pressure2 = params.stiffness * (density2 - params.density0);
        real pressureFV = params.mass2 * (pressure1 / (density1*density1) + pressure2 / (density2*density2) );

        // Viscosity
        if (viscosityType == 1)
        {
            force += ( v2 - v1 ) * ( params.mass2 * params.viscosity * SPH<real,kernelType>::laplacianWv(r_h,params.ClaplacianWv) / (density1 * density2) );
        }
        else if (viscosityType == 2)
        {
            real vx = dot(n,v2-v1);
            if (vx < 0)
                pressureFV += - (vx * params.viscosity * params.h * params.mass / ((r2_h2 + 0.01f*params.h2)*(density1+density2)*0.5f));
        }

        force += SPH<real,kernelType>::gradWp(n, r_h, params.CgradWp) * pressureFV;
    }
}
/*
template<class real, int kernelType, int pressureType, int viscosityType, int surfaceTensionType>
__device__ void SPHFluidCalcDForce(CudaVec4<real> x1, CudaVec3<real> v1, CudaVec3<real> dx1, CudaVec4<real> x2, CudaVec3<real> v2,  CudaVec3<real> dx2, CudaVec3<real>& dforce, GPUSPHFluid<real>& params)
{
    CudaVec3<real> n = CudaVec3<real>::make(x2.x-x1.x,x2.y-x1.y,x2.z-x1.z);
    real d2 = norm2(n);
    if (d2 < params.h2)
    {
        CudaVec3<real> dxi = dx2-dx1;

        real inv_d = rsqrtf(d2);

        real r2_h2 = (d2/params.h2);
        real r_h = sqrtf(r2_h2);

        real density1 = x1.w;
        real density2 = x2.w;

        // Pressure
        real pressure1 = params.stiffness * (density1 - params.density0);
        real pressure2 = params.stiffness * (density2 - params.density0);

        real fpressure = params.mass2 * (pressure1 / (density1*density1) + pressure2 / (density2*density2) );

        // Derivatives

        CudaVec3<real> dn = dxi * inv_d;

        real dr_h = dot(dn,n)/params.h;
        //real dr2_h2 = 2*r_h*dr_h;

        real ddensity = dot(dxi,SPH<real,kernelType>::gradWd2(n, r2_h2, params.CgradWd));
        real dpressure = params.stiffness*ddensity;
        // d(a/b^2) = (d(a)*b-2*a*d(b))/b^3
        real dfpressure = params.mass2 * ( (dpressure*density1-2*pressure1*ddensity)/(density1*density1*density1)
                                          +(dpressure*density2-2*pressure2*ddensity)/(density2*density2*density2) );
        real a = (1-r_h);
        real dWp = params.CgradWp*a*a;
        real ddWp = -2*params.CgradWp*a*dr_h;

        // f = n * dWp * fpressure
        // df = dn * (dWp * fpressure) + n * (ddWp * fpressure + dWp * dfpressure);

        dforce += dn * (dWp  * fpressure) + n * (ddWp * fpressure + dWp * dfpressure);

        // Viscosity
        // force += ( v2 - v1 ) * ( params.mass2 * params.viscosity * params.ClaplacianWv * (1-r_h) / (density1 * density2) );
        real d1d2 = density1*density2;
        dforce += dxi * ( params.mass2 * params.viscosity * params.ClaplacianWv * (1-r_h) / (density1 * density2) )
            + (v2-v1) * ( params.mass2 * params.viscosity * params.ClaplacianWv * (-dr_h*d1d2 - (1-r_h)*ddensity*(density1+density2))/(d1d2*d1d2));

    }
}
*/
template<class real, int kernelType, int pressureType>
__global__ void SPHFluidForceFieldCuda3t_computeDensity_kernel(int size, const int *cells, const int *cellGhost, GPUSPHFluid<real> params, real* pos4, const real* x)
{
    __shared__ int2 range;
    __shared__ int ghost;
    __shared__ real temp_x[BSIZE*3];
    int tx3 = __umul24(threadIdx.x,3);
    for (int cell = blockIdx.x; cell < size; cell += gridDim.x)
    {
        __syncthreads();
        if (!threadIdx.x)
        {
            //range = *(const int2*)(cells+cell);
            range.x = cells[cell];
            range.y = cells[cell+1];
            range.y &= ~(1U<<31);
            ghost = cellGhost[cell];
        }
        __syncthreads();
        if (range.x <= 0) continue; // no actual particle in this cell
        for (int px0 = range.x; px0 < ghost; px0 += BSIZE)
        {
            int px = px0 + threadIdx.x;
            CudaVec3<real> xi;
            real density;
            int index;
            if (px < range.y)
            {
                index = cells[px];
                xi = ((const CudaVec3<real>*)x)[index];
                temp_x[tx3  ] = xi.x;
                temp_x[tx3+1] = xi.y;
                temp_x[tx3+2] = xi.z;
            }
            __syncthreads();
            if (px < ghost)
            {
                // actual particle -> compute interactions
                SPHFluidInitDensity<real,kernelType,pressureType>(xi, density, params);

                int np = min(range.y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != threadIdx.x)
                        SPHFluidCalcDensity<real,kernelType,pressureType>(xi, ((const CudaVec3<real>*)temp_x)[i], density, params);
                }
            }
            __syncthreads();
            // loop through other groups of particles
            for (int py0 = range.x; py0 < range.y; py0 += BSIZE)
            {
                if (py0 == px0) continue;
                int py = py0 + threadIdx.x;
                if (py < range.y)
                {
                    int index2 = cells[py];
                    CudaVec3<real> xj = ((const CudaVec3<real>*)x)[index2];
                    temp_x[tx3  ] = xj.x;
                    temp_x[tx3+1] = xj.y;
                    temp_x[tx3+2] = xj.z;
                }
                __syncthreads();
                if (px < ghost)
                {
                    // actual particle -> compute interactions
                    int np = min(range.y-py0,BSIZE);
                    for (int i=0; i < np; ++i)
                    {
                        SPHFluidCalcDensity<real,kernelType,pressureType>(xi, ((const CudaVec3<real>*)temp_x)[i], density, params);
                    }
                }
                __syncthreads();
            }
            if (px < ghost)
            {
                // actual particle -> write computed density
                SPHFluidFinalDensity<real,kernelType,pressureType>(xi, density, params);
                CudaVec4<real> res = CudaVec4<real>::make(xi.x,xi.y,xi.z,density);
                ((CudaVec4<real>*)pos4)[index] = res;
            }
        }
    }
}

template<class real, int kernelType, int pressureType, int viscosityType, int surfaceTensionType>
__global__ void SPHFluidForceFieldCuda3t_addForce_kernel(int size, const int *cells, const int *cellGhost, GPUSPHFluid<real> params, real* f, const real* pos4, const real* v)
{
    __shared__ int2 range;
    __shared__ int ghost;
    __shared__ real temp_x[BSIZE*4];
    __shared__ real temp_v[BSIZE*3];
    int tx3 = __umul24(threadIdx.x,3);
    int tx4 = threadIdx.x << 2;
    for (int cell = blockIdx.x; cell < size; cell += gridDim.x)
    {
        if (!threadIdx.x)
        {
            //range = *(const int2*)(cells+cell);
            range.x = cells[cell];
            range.y = cells[cell+1];
            range.y &= ~(1U<<31);
            ghost = cellGhost[cell];
        }
        __syncthreads();
        if (range.x <= 0) continue; // no actual particle in this cell
        for (int px0 = range.x; px0 < ghost; px0 += BSIZE)
        {
            int px = px0 + threadIdx.x;
            CudaVec4<real> xi;
            CudaVec3<real> vi;
            CudaVec3<real> force;
            int index;
            if (px < range.y)
            {
                index = cells[px];
                xi = ((const CudaVec4<real>*)pos4)[index];
                temp_x[tx4  ] = xi.x;
                temp_x[tx4+1] = xi.y;
                temp_x[tx4+2] = xi.z;
                temp_x[tx4+3] = xi.w;
                vi = ((const CudaVec3<real>*)v)[index];
                temp_v[tx3  ] = vi.x;
                temp_v[tx3+1] = vi.y;
                temp_v[tx3+2] = vi.z;
            }
            __syncthreads();
            if (px < ghost)
            {
                // actual particle -> compute interactions
                force = CudaVec3<real>::make(0,0,0);
                int np = min(range.y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != threadIdx.x)
                        SPHFluidCalcForce<real,kernelType,pressureType,viscosityType,surfaceTensionType>(xi, vi, ((const CudaVec4<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], force, params);
                }
            }
            __syncthreads();
            // loop through other groups of particles
            for (int py0 = range.x; py0 < range.y; py0 += BSIZE)
            {
                if (py0 == px0) continue;
                int py = py0 + threadIdx.x;
                if (py < range.y)
                {
                    int index2 = cells[py];
                    CudaVec4<real> xj = ((const CudaVec4<real>*)pos4)[index2];
                    temp_x[tx4  ] = xj.x;
                    temp_x[tx4+1] = xj.y;
                    temp_x[tx4+2] = xj.z;
                    temp_x[tx4+3] = xj.w;
                    CudaVec3<real> vj = ((const CudaVec3<real>*)v)[index2];
                    temp_v[tx3  ] = vj.x;
                    temp_v[tx3+1] = vj.y;
                    temp_v[tx3+2] = vj.z;
                }
                __syncthreads();
                if (px < ghost)
                {
                    // actual particle -> compute interactions
                    int np = min(range.y-py0,BSIZE);
                    for (int i=0; i < np; ++i)
                    {
                        SPHFluidCalcForce<real,kernelType,pressureType,viscosityType,surfaceTensionType>(xi, vi, ((const CudaVec4<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], force, params);
                    }
                }
                __syncthreads();
            }
            if (px < ghost)
            {
                // actual particle -> write computed force
                ((CudaVec3<real>*)f)[index] += force;
            }
        }
    }
}
/*
template<class real, int kernelType, int pressureType, int viscosityType, int surfaceTensionType>
__global__ void SPHFluidForceFieldCuda3t_addDForce_kernel(int size, const int *cells, const int *cellGhost, GPUSPHFluid<real> params, real* df, const real* pos4, const real* v, const real* dx)
{
    __shared__ int2 range;
    __shared__ int ghost;
    __shared__ real temp_x[BSIZE*4];
    __shared__ real temp_v[BSIZE*3];
    __shared__ real temp_dx[BSIZE*3];
    int tx3 = __umul24(threadIdx.x,3);
    int tx4 = threadIdx.x << 2;
    for (int cell = blockIdx.x; cell < size; cell += gridDim.x)
    {
        if (!threadIdx.x)
        {
            //range = *(const int2*)(cells+cell);
            range.x = cells[cell];
            range.y = cells[cell+1];
            range.y &= ~(1U<<31);
            ghost = cellGhost[cell];
        }
        __syncthreads();
        if (range.x <= 0) continue; // no actual particle in this cell
        for (int px0 = range.x; px0 < ghost; px0 += BSIZE)
        {
            int px = px0 + threadIdx.x;
            CudaVec4<real> xi;
            CudaVec3<real> vi;
            CudaVec3<real> dxi;
            CudaVec3<real> dforce;
            int index;
            if (px < range.y)
            {
                index = cells[px];
                xi = ((const CudaVec4<real>*)pos4)[index];
                temp_x[tx4  ] = xi.x;
                temp_x[tx4+1] = xi.y;
                temp_x[tx4+2] = xi.z;
                temp_x[tx4+3] = xi.w;
                vi = ((const CudaVec3<real>*)v)[index];
                temp_v[tx3  ] = vi.x;
                temp_v[tx3+1] = vi.y;
                temp_v[tx3+2] = vi.z;
                dxi = ((const CudaVec3<real>*)dx)[index];
                temp_dx[tx3  ] = dxi.x;
                temp_dx[tx3+1] = dxi.y;
                temp_dx[tx3+2] = dxi.z;
            }
            __syncthreads();
            if (px < ghost)
            { // actual particle -> compute interactions
                dforce = CudaVec3<real>::make(0,0,0);
                int np = min(range.y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != threadIdx.x)
                        SPHFluidCalcDForce(xi, vi, dxi, ((const CudaVec4<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], ((const CudaVec3<real>*)temp_dx)[i], dforce, params);
                }
            }
            __syncthreads();
            // loop through other groups of particles
            for (int py0 = range.x; py0 < range.y; py0 += BSIZE)
            {
                if (py0 == px0) continue;
                int py = py0 + threadIdx.x;
                if (py < range.y)
                {
                    int index2 = cells[py];
                    CudaVec4<real> xj = ((const CudaVec4<real>*)pos4)[index2];
                    temp_x[tx4  ] = xj.x;
                    temp_x[tx4+1] = xj.y;
                    temp_x[tx4+2] = xj.z;
                    temp_x[tx4+3] = xj.w;
                    CudaVec3<real> vj = ((const CudaVec3<real>*)v)[index2];
                    temp_v[tx3  ] = vj.x;
                    temp_v[tx3+1] = vj.y;
                    temp_v[tx3+2] = vj.z;
                    CudaVec3<real> dxj = ((const CudaVec3<real>*)dx)[index2];
                    temp_dx[tx3  ] = dxj.x;
                    temp_dx[tx3+1] = dxj.y;
                    temp_dx[tx3+2] = dxj.z;
                }
                __syncthreads();
                if (px < ghost)
                { // actual particle -> compute interactions
                    int np = min(range.y-py0,BSIZE);
                    for (int i=0; i < np; ++i)
                    {
                        SPHFluidCalcDForce(xi, vi, dxi, ((const CudaVec4<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], ((const CudaVec3<real>*)temp_dx)[i], dforce, params);
                    }
                }
                __syncthreads();
            }
            if (px < ghost)
            { // actual particle -> write computed force
                ((CudaVec3<real>*)df)[index] += dforce;
            }
        }
    }
}
*/
//////////////////////
// CPU-side methods //
//////////////////////

void SPHFluidForceFieldCuda3f_computeDensity(int kernelType, int pressureType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* pos4, const void* x)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    switch(kernelType)
    {
    case 0:
        switch(pressureType)
        {
        case 0: //SPHFluidForceFieldCuda3t_computeDensity_kernel<float,0,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)pos4, (const float*)x);
            break;
        case 1: SPHFluidForceFieldCuda3t_computeDensity_kernel<float,0,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)pos4, (const float*)x);
            break;
        default: break;
        }
        break;
    case 1:
        switch(pressureType)
        {
        case 0: //SPHFluidForceFieldCuda3t_computeDensity_kernel<float,1,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)pos4, (const float*)x);
            break;
        case 1: SPHFluidForceFieldCuda3t_computeDensity_kernel<float,1,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)pos4, (const float*)x);
            break;
        default: break;
        }
        break;
    default: break;
    }
    mycudaDebugError("SPHFluidForceFieldCuda3t_computeDensity_kernel<float>");
}

void SPHFluidForceFieldCuda3f_addForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    switch(kernelType)
    {
    case 0:
        switch(pressureType)
        {
        case 0:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,0,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,0,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,0,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,1,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,1,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,1,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,2,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,2,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,0,2,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        case 1:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,0,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,0,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,0,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,1,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,1,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,1,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,2,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,2,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,0,1,2,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        default: break;
        }
        break;
    case 1:
        switch(pressureType)
        {
        case 0:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,0,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,0,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,0,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,1,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,1,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,1,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,2,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,2,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,0,2,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        case 1:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,0,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,0,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,0,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,1,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,1,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,1,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,2,0><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,2,1><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<float,1,1,2,2><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        default: break;
        }
        break;
    default: break;
    }
    mycudaDebugError("SPHFluidForceFieldCuda3t_addForce_kernel<float>");
}
/*
void SPHFluidForceFieldCuda3f_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* df, const void* x, const void* v, const void* dx)
{
	dim3 threads(BSIZE,1);
	dim3 grid(60/BSIZE,1);
	{SPHFluidForceFieldCuda3t_addDForce_kernel<float><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)df, (const float*)x, (const float*)v, (const float*)dx); mycudaDebugError("SPHFluidForceFieldCuda3t_addDForce_kernel<float>");}
}
*/
#ifdef SOFA_GPU_CUDA_DOUBLE

void SPHFluidForceFieldCuda3d_computeDensity(int kernelType, int pressureType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* pos4, const void* x)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    switch(kernelType)
    {
    case 0:
        switch(pressureType)
        {
        case 0: //SPHFluidForceFieldCuda3t_computeDensity_kernel<double,0,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)pos4, (const double*)x);
            break;
        case 1: SPHFluidForceFieldCuda3t_computeDensity_kernel<double,0,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)pos4, (const double*)x);
            break;
        default: break;
        }
        break;
    case 1:
        switch(pressureType)
        {
        case 0: //SPHFluidForceFieldCuda3t_computeDensity_kernel<double,1,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)pos4, (const double*)x);
            break;
        case 1: SPHFluidForceFieldCuda3t_computeDensity_kernel<double,1,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)pos4, (const double*)x);
            break;
        default: break;
        }
        break;
    default: break;
    }
    mycudaDebugError("SPHFluidForceFieldCuda3t_computeDensity_kernel<double>");

}

void SPHFluidForceFieldCuda3d_addForce (int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    switch(kernelType)
    {
    case 0:
        switch(pressureType)
        {
        case 0:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,0,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,0,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,0,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,1,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,1,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,1,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,2,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,2,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,0,2,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        case 1:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,0,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,0,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,0,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,1,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,1,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,1,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,2,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,2,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,0,1,2,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        default: break;
        }
        break;
    case 1:
        switch(pressureType)
        {
        case 0:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,0,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,0,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,0,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,1,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,1,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,1,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,2,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,2,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,0,2,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        case 1:
            switch(viscosityType)
            {
            case 0:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,0,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,0,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,0,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 1:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,1,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,1,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,1,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            case 2:
                switch(surfaceTensionType)
                {
                case 0: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,2,0><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 1: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,2,1><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                case 2: SPHFluidForceFieldCuda3t_addForce_kernel<double,1,1,2,2><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
                    break;
                default: break;
                }
                break;
            default: break;
            }
            break;
        default: break;
        }
        break;
    default: break;
    }
    mycudaDebugError("SPHFluidForceFieldCuda3t_addForce_kernel<double>");
    /*
    	dim3 threads(BSIZE,1);
    	dim3 grid(60/BSIZE,1);
        if (params->surfaceTension > 0)
    	{SPHFluidForceFieldCuda3t_addForce_kernel<double,true><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v); mycudaDebugError("SPHFluidForceFieldCuda3t_addForce_kernel<double,true>");}
        else
    	{SPHFluidForceFieldCuda3t_addForce_kernel<double,false><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v); mycudaDebugError("SPHFluidForceFieldCuda3t_addForce_kernel<double,false>");}
    */
}
/*
void SPHFluidForceFieldCuda3d_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* df, const void* x, const void* v, const void* dx)
{
	dim3 threads(BSIZE,1);
	dim3 grid(60/BSIZE,1);
	{SPHFluidForceFieldCuda3t_addDForce_kernel<double><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)df, (const double*)x, (const double*)v, (const double*)dx); mycudaDebugError("SPHFluidForceFieldCuda3t_addDForce_kernel<double>");}
}
*/
#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
