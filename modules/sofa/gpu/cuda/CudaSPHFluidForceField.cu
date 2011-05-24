/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "CudaCommon.h"
#include "CudaMath.h"
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
    real stiffness; ///< pressure stiffness
    real mass;      ///< particles mass
    real mass2;     ///< particles mass squared
    real density0;  ///< 1000 kg/m3 for water
    real viscosity;
    real surfaceTension;

    // Precomputed constants for smoothing kernels
    real CWd;          ///< =     constWd(h)
    real CgradWd;      ///< = constGradWd(h)
    real CgradWp;      ///< = constGradWp(h)
    real ClaplacianWv; ///< =  constLaplacianWv(h)
    real CgradWc;      ///< = constGradWc(h)
    real ClaplacianWc; ///< =  constLaplacianWc(h)
};

typedef GPUSPHFluid<float> GPUSPHFluid3f;
typedef GPUSPHFluid<double> GPUSPHFluid3d;

extern "C"
{

    void SPHFluidForceFieldCuda3f_computeDensity(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3f_addForce (unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v);
    void SPHFluidForceFieldCuda3f_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v, const void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void SPHFluidForceFieldCuda3d_computeDensity(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3d_addForce (unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v);
    void SPHFluidForceFieldCuda3d_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v, const void* dx);

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

template<class real>
class SPH
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
        if (a<=0) return 0;
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

    // grad W = d(W)/dr Ur            in spherical coordinates, with Ur = D/|D| = D/r
    //        = d( C(1-r2/h2)3 )/dr D/r
    //        = d( C/h6 (h2-r2)3 )/dr D/r
    //        = d( C/h6 (h2-r2)(h4+r4-2h2r2) )/dr D/r
    //        = ( C/h6 (h2-r2)(4r3-4h2r) + (-2r)(h4+r4-2h2r2) ) D/r
    //        = C/h6 ( 4h2r3-4h4r-4r5+4h2r3 -2h4r -2r5 +4h2r3 ) D/r
    //        = C/h6 ( -6r5 +12h2r3 -6h4r ) D/r
    //        = -6C/h6 ( r4 -2h2r2 +h4 ) D
    //        = -6C/h6 ( h2 - r2 )2 D
    //        = -6C/h2 ( 1 - r2/h2 )2 D
    static Real  constGradWd(Real h)
    {
        return -6*constWd(h)/(h*h);
    }
    static __device__ Deriv gradWd(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        if (a<0) a=0;
        return d*(C*a*a);
    }
    static __device__ Deriv gradWd2(const Deriv& d, Real r2_h2, Real C)
    {
        Real a = (1-r2_h2);
        return d*(C*a*a);
    }

    // laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
    //              = 1/r d2(rW)/dr2                 in spherical coordinate, as f only depends on r
    //              = C/r d2(r(1-r2/h2)3)/dr2
    //              = C/rh6 d2(r(h2-r2)3)/dr2
    //              = C/rh6 d2(r(h2-r2)(h4-2h2r2+r4))/dr2
    //              = C/rh6 d2(r(h6-3h4r2+3h2r4-r6))/dr2
    //              = C/rh6 d2(h6r-3h4r3+3h2r5-r7)/dr2
    //              = C/rh6 d(h6-9h4r2+15h2r4-7r6)/dr
    //              = C/rh6 (-18h4r+60h2r3-42r5)
    //              = C/h6 (-18h4+60h2r2-42r4)
    //              = 6C/h2 (-3+10r2/h2-7r4/h4)
    //              = CL (-3+10r2/h2-7r4/h4)
    static Real  constLaplacianWd(Real h)
    {
        return 6*constWd(h)/(h*h);
    }
    static __device__ Real  laplacianWd(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
    }
    static __device__ Real  laplacianWd2(Real r2_h2, Real C)
    {
        return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
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

    // grad W = d(W)/dr Ur            in spherical coordinates, with Ur = D/|D| = D/r
    //        = d( C(1-r/h)3 )/dr D/r
    //        = d( C/h3 (h-r)3 )/dr D/r
    //        = d( C/h6 (h-r)(h2+r2-2hr) )/dr D/r
    //        = C/h6 ( (h-r)(2r-2h) -(h2+r2-2hr) ) D/r
    //        = C/h6 ( -2r2+4hr-2h2 -r2+2hr-h2 ) D/r
    //        = C/h6 ( -2r2+4hr-2h2 -r2+2hr-h2 ) D/r
    //        = C/h6 ( -3r2+6hr-3h2 ) D/r
    //        = 3C/h4 ( -r2/h2+2r/h-1 ) D/r
    //        = -3C/h4 ( 1-r/h )2 D/r
    static Real  constGradWp(Real h)
    {
        return (-3*constWp(h)) / (h*h*h*h);
    }
    static __device__ Deriv gradWp(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h);
        return d * (C*a*a);
    }

    //Real  laplacianWp(Real r_h, Real C);

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

    // grad W = d(W)/dr Ur            in spherical coordinates, with Ur = D/|D| = D/r
    //        = d( C(-r3/2h3 + r2/h2 + h/2r - 1) )/dr D/r
    //        = C(-3r2/2h3 + 2r/h2 - h/2r2) D/r
    //        = C(-3r/2h3 + 2/h2 - h/2r3) D
    //        = C/2h2 (-3r/h + 4 - h3/r3) D

    static Real  constGradWv(Real h)
    {
        return constWv(h)/(2*h*h);
    }
    static __device__ Deriv gradWv(const Deriv& d, Real r_h, Real C)
    {
        Real r3_h3 = r_h*r_h*r_h;
        return d * (C*(-3*r_h  + 4 - 1/r3_h3));
    }
    static __device__ Deriv gradWv2(const Deriv& d, Real r2_h2, Real r_h, Real C)
    {
        Real r3_h3 = r2_h2*r_h;
        return d * (C*(-3*r_h  + 4 - 1/r3_h3));
    }

    // laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
    //              = 1/r d2(rW)/dr2                 in spherical coordinate, as f only depends on r
    //              = C/r d2(r(-r3/2h3 + r2/h2 + h/2r - 1))/dr2
    //              = C/r d2(-r4/2h3 + r3/h2 + h/2 - r)/dr2
    //              = C/r d(-4r3/2h3 + 3r2/h2 - 1)/dr
    //              = C/r (-6r2/h3 + 6r/h2)
    //              = C (-6r/h3 + 6/h2)
    //              = 6C/h2 (1 - r/h)

    // laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
    //              = 1/r2 d(r2 d(W)/dr)/dr                 in spherical coordinate, as f only depends on r
    //              = C/r2 d(r2 d(-r3/2h3 + r2/h2 + h/2r - 1)/dr)/dr
    //              = C/r2 d(r2 (-3r2/2h3 + 2r/h2 - h/2r2))/dr
    //              = C/r2 d(-3r4/2h3 + 2r3/h2 - h/2))/dr
    //              = C/r2 (-6r3/h3 + 6r2/h2)
    //              = 6C/h2 (1 -r/h)

    static Real  constLaplacianWv(Real h)
    {
        return 6*constWv(h)/(h*h);
        //return 75/(R_PI*h*h*h*h*h);
    }

    static __device__ Real  laplacianWv(Real r_h, Real C)
    {
        return C*(1-r_h);
    }

    /// Color Smoothing Kernel: same as Density
    static Real  constWc(Real h)
    {
        return (Real)(315 / (64*R_PI*h*h*h));
    }
    static __device__ Real  Wc(Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        return  C*a*a*a;
    }
    static Real  constGradWc(Real h)
    {
        return -6*constWc(h)/(h*h);
    }
    static __device__ Deriv gradWc(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        return d*(C*a*a);
    }
    static Real  constLaplacianWc(Real h)
    {
        return 6*constWc(h)/(h*h);
    }
    static __device__ Real  laplacianWc(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
    }
};

template<class real>
__device__ void SPHFluidInitDensity(CudaVec3<real> /*x1*/, real& density, GPUSPHFluid<real>& params)
{
    density = 0; //params.mass * params.CWd; //SPH<real>::Wd(0,params.CWd);
}

template<class real>
__device__ void SPHFluidCalcDensity(CudaVec3<real> x1, CudaVec3<real> x2, real& density, GPUSPHFluid<real>& params)
{
    CudaVec3<real> n = x2-x1;
    real d2 = norm2(n);
    if (d2 < params.h2)
    {
        //real r_h = rsqrtf(params.h2/d2);
        //real r_h = sqrtf(d2/params.h2);
        real r2_h2 = (d2/params.h2);
        //real inv_d = rsqrtf(d2);
        //n *= inv_d;
        //real d = d2*inv_d;
        //real d = params.mass * SPH<real>::Wd2(r2_h2,params.CWd);
        real d = SPH<real>::Wd2(r2_h2);
        density += d;
    }
}

template<class real>
__device__ void SPHFluidFinalDensity(CudaVec3<real> /*x1*/, real& density, GPUSPHFluid<real>& params)
{
    density = (1+density) * params.CWd * params.mass; //SPH<real>::Wd(0,params.CWd);
}

template<class real, bool surface>
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
        force += SPH<real>::gradWp(n, r_h, params.CgradWp) * ( params.mass2 * (pressure1 / (density1*density1) + pressure2 / (density2*density2) ));

        // Viscosity
        force += ( v2 - v1 ) * ( params.mass2 * params.viscosity * SPH<real>::laplacianWv(r_h,params.ClaplacianWv) / (density1 * density2) );
    }
}

template<class real>
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

        real ddensity = dot(dxi,SPH<real>::gradWd2(n, r2_h2, params.CgradWd));
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

template<class real>
__global__ void SPHFluidForceFieldCuda3t_computeDensity_kernel(int size, const int *cells, const int *cellGhost, GPUSPHFluid<real> params, real* pos4, const real* x)
{
    __shared__ int2 range;
    __shared__ int ghost;
    __shared__ real temp_x[BSIZE*3];
    int tx3 = __umul24(threadIdx.x,3);
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
                SPHFluidInitDensity(xi, density, params);

                int np = min(range.y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != threadIdx.x)
                        SPHFluidCalcDensity(xi, ((const CudaVec3<real>*)temp_x)[i], density, params);
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
                        SPHFluidCalcDensity(xi, ((const CudaVec3<real>*)temp_x)[i], density, params);
                    }
                }
                __syncthreads();
            }
            if (px < ghost)
            {
                // actual particle -> write computed density
                SPHFluidFinalDensity(xi, density, params);
                CudaVec4<real> res = CudaVec4<real>::make(xi.x,xi.y,xi.z,density);
                ((CudaVec4<real>*)pos4)[index] = res;
            }
        }
    }
}

template<class real, bool surface>
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
                        SPHFluidCalcForce<real,surface>(xi, vi, ((const CudaVec4<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], force, params);
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
                        SPHFluidCalcForce<real,surface>(xi, vi, ((const CudaVec4<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], force, params);
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

template<class real>
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
            {
                // actual particle -> compute interactions
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
                {
                    // actual particle -> compute interactions
                    int np = min(range.y-py0,BSIZE);
                    for (int i=0; i < np; ++i)
                    {
                        SPHFluidCalcDForce(xi, vi, dxi, ((const CudaVec4<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], ((const CudaVec3<real>*)temp_dx)[i], dforce, params);
                    }
                }
                __syncthreads();
            }
            if (px < ghost)
            {
                // actual particle -> write computed force
                ((CudaVec3<real>*)df)[index] += dforce;
            }
        }
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void SPHFluidForceFieldCuda3f_computeDensity(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* pos4, const void* x)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    SPHFluidForceFieldCuda3t_computeDensity_kernel<float><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)pos4, (const float*)x);
}

void SPHFluidForceFieldCuda3f_addForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    if (params->surfaceTension > 0)
        SPHFluidForceFieldCuda3t_addForce_kernel<float,true><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);
    else
        SPHFluidForceFieldCuda3t_addForce_kernel<float,false><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)f, (const float*)x, (const float*)v);

}

void SPHFluidForceFieldCuda3f_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* df, const void* x, const void* v, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60/BSIZE,1);
    SPHFluidForceFieldCuda3t_addDForce_kernel<float><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (float*)df, (const float*)x, (const float*)v, (const float*)dx);
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void SPHFluidForceFieldCuda3d_computeDensity(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* pos4, const void* x)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    SPHFluidForceFieldCuda3t_computeDensity_kernel<double><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)pos4, (const double*)x);
}

void SPHFluidForceFieldCuda3d_addForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60/BSIZE,1);
    if (params->surfaceTension > 0)
        SPHFluidForceFieldCuda3t_addForce_kernel<double,true><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
    else
        SPHFluidForceFieldCuda3t_addForce_kernel<double,false><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)f, (const double*)x, (const double*)v);
}

void SPHFluidForceFieldCuda3d_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* df, const void* x, const void* v, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60/BSIZE,1);
    SPHFluidForceFieldCuda3t_addDForce_kernel<double><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *params, (double*)df, (const double*)x, (const double*)v, (const double*)dx);
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
