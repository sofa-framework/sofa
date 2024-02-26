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
#ifndef SOFA_COMPONENT_FORCEFIELD_SPHKERNEL_H
#define SOFA_COMPONENT_FORCEFIELD_SPHKERNEL_H
#include <SofaSphFluid/config.h>
#include <vector>
#include <cmath>


namespace sofa::component::forcefield
{

enum SPHKernels
{
    SPH_KERNEL_DEFAULT_DENSITY,
    SPH_KERNEL_DEFAULT_PRESSURE,
    SPH_KERNEL_DEFAULT_VISCOSITY,
    SPH_KERNEL_CUBIC
};


template<SPHKernels KT, class Deriv>
class BaseSPHKernel;


template <class Deriv>
class BaseSPHKernel<SPH_KERNEL_DEFAULT_DENSITY, Deriv>
{
public:
    typedef typename Deriv::value_type Real;
    enum { N = Deriv::spatial_dimensions };

    static const char* Name() { return "d"; }

    /// Density Smoothing Kernel:  W = 315 / 64pih9 * (h2 - r2)3 = 315 / 64pih3 * (1 - (r/h)2)3
    static Real  constW(Real h)
    {
        return (Real)(315 / (64*R_PI*h*h*h));
    }
    static Real  W2(Real r2_h2, Real C)
    {
        Real a = (1-r2_h2);
        return  C*a*a*a;
    }
    static Real  W(Real r_h, Real C)
    {
        return W2(r_h*r_h, C);
    }

    // grad W = d(W)/dr Ur            in spherical coordinates, with Ur = D/|D| = D/r
    // grad W = 1/h d(W)/dq Ur        with q = r/h
    // d(W)/dq = d( C(1-q^2)^3 )/dq
    //         = d( C(1-q^2)^3 )/dq
    //         = -6C q(1-q^2)^2
    // grad W = -6C/h q(1-q^2)^2 D/qh
    // grad W = -6C/h^2 (1-q^2)^2 D
    static Real constGradW(Real h)
    {
        return -6*constW(h)/(h*h);
    }

    static Deriv gradW2(const Deriv& d, Real r2_h2, Real C)
    {
        Real a = (1-r2_h2);
        return d*(C*a*a);
    }

    static Deriv gradW(const Deriv& d, Real r_h, Real C)
    {
        return gradW2(d, r_h*r_h, C);
    }

    // laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
    //              = d2(W)/dr2 + 2/r d(W)/dr      in spherical coordinate, as W only depends on r
    //              = 1/h2 d2(W)/dq2 + 2/r 1/h d(W)/dq      with q = r/h
    //              = 1/h2 (d2(W)/dq2 + 2/q d(W)/dq)
    //              = -6C/h2 ((1-q2)(1-5q2) + 2/q q(1-q2)^2)
    //              = -6C/h2 ((1-q2)(1-5q2  + 2-2q2))
    //              = -6C/h2 ((1-q2)(3-7q2))
    static Real  constLaplacianW(Real h)
    {
        return -6*constW(h)/(h*h);
    }
    static Real  laplacianW2(Real r2_h2, Real C)
    {
        return C*((1-r2_h2)*(3-7*r2_h2));
    }
    static Real  laplacianW(Real r_h, Real C)
    {
        return laplacianW2(r_h*r_h, C);
    }
};


template <class Deriv>
class BaseSPHKernel<SPH_KERNEL_DEFAULT_PRESSURE, Deriv>
{
public:
    typedef typename Deriv::value_type Real;
    enum { N = Deriv::spatial_dimensions };

    static const char* Name() { return "p"; }

    /// Pressure Smoothing Kernel:  W = 15 / pih6 (h - r)3 = 15 / pih3 (1 - r/h)3
    static Real  constW(Real h)
    {
        return (Real)(15 / (R_PI*h*h*h));
    }
    static Real  W(Real r_h, Real C)
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
    static Real  constGradW(Real h)
    {
        return (-3*constW(h)) / (h*h);
    }
    static Deriv gradW(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h);
        return d * (C*a*a/r_h);
    }

    static Real  constLaplacianW(Real /*h*/)
    {
        return 0;
    }

    static Real  laplacianW(Real /*r_h*/, Real /*C*/)
    {
        return 0;
    }


    static Real  W2(Real r2_h2, Real C)
    {
        return W(helper::rsqrt(r2_h2), C);
    }

    static Deriv gradW2(const Deriv& d, Real r2_h2, Real C)
    {
        return gradW(d, helper::rsqrt(r2_h2), C);
    }

    static Real  laplacianW2(Real r2_h2, Real C)
    {
        return laplacianW(helper::rsqrt(r2_h2), C);
    }
};


template <class Deriv>
class BaseSPHKernel<SPH_KERNEL_DEFAULT_VISCOSITY, Deriv>
{
public:
    typedef typename Deriv::value_type Real;
    enum { N = Deriv::spatial_dimensions };

    static const char* Name() { return "v"; }

    /// Viscosity Smoothing Kernel:  W = 15/(2pih3) (-r3/2h3 + r2/h2 + h/2r - 1)
    static Real  constW(Real h)
    {
        return (Real)(15/(2*R_PI*h*h*h));
    }

    static Real  W(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        Real r3_h3 = r2_h2*r_h;
        return C*(-0.5f*r3_h3 + r2_h2 + 0.5f/r_h - 1);
    }

    // grad W = d(W)/dr Ur            in spherical coordinates, with Ur = D/|D| = D/r
    //        = d( C(-r3/2h3 + r2/h2 + h/2r - 1) )/dr D/r
    //        = C/h (-1.5r2/h2 + 2r/h - 0.5h2/r2) D

    static Real  constGradW(Real h)
    {
        return constW(h)/(h*h);
    }
    static Deriv gradW(const Deriv& d, Real r_h, Real C)
    {
        Real r3_h3 = r_h*r_h*r_h;
        return d * (C*(2.0f - 1.5f*r_h - 0.5f/r3_h3));
    }

    // laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
    //              = d2(W)/dr2 + 2/r d(W)/dr         in spherical coordinate, as f only depends on r
    //              = 1/h2 (d2(W)/dq2 + 2/q d(W)/dq)  with q = r/h

    static Real  constLaplacianW(Real h)
    {
        return 6*constW(h)/(h*h);
    }

    static Real  laplacianW(Real r_h, Real C)
    {
        return C*(1-r_h);
    }

    static Real  W2(Real r2_h2, Real C)
    {
        return W(helper::rsqrt(r2_h2), C);
    }

    static Deriv gradW2(const Deriv& d, Real r2_h2, Real C)
    {
        return gradW2(d, helper::rsqrt(r2_h2), C);
    }

    static Real  laplacianW2(Real r2_h2, Real C)
    {
        return laplacianW(helper::rsqrt(r2_h2), C);
    }

};


template <class Deriv>
class BaseSPHKernel<SPH_KERNEL_CUBIC, Deriv>
{
public:
    typedef typename Deriv::value_type Real;
    enum { N = Deriv::spatial_dimensions };

    static const char* Name() { return "cubic"; }

    // Cubic spline kernel
    // Originally defined between 0 and 2h as:
    // W(q) = \omega { 2/3 - q^2 + 1/2 q^3  if 0 <= q <= 1
    //               { 1/6 (2-q)^3          if 1 <= q <= 2
    //               { 0                    if q >= 2
    // with \omega = 3/2 Pi h^3 in 3D
    // If we want the same kernel but between 0 and h', we have
    // W(q') = \omega' { 2/3 - (2q')^2 + 1/2 (2q')^3  if 0 <= q' <= 1/2
    //               { 1/6 (2-(2q'))^3                if 1/2 <= q' <= 1
    //               { 0                              if q' >= 1
    // with \omega' = 8 \omega = 12 Pi h'^3 in 3D
    // W(q') = 4\omega' { 1/6 - q'^2 + q'^3  if 0 <= q' <= 1/2
    //                  { 1/3 (1-q')^3       if 1/2 <= q' <= 1
    //                  { 0                  if q' >= 1
    static Real constW(Real h)
    {
        return (Real) (48/R_PI)/(h*h*h);
    }

    static Real W(Real r_h, Real C)
    {
        if (r_h < (Real)0.5) return C*((Real)(1.0/6.0) - r_h*r_h + r_h*r_h*r_h);
        else if (r_h < (Real)1) { Real s = 1-r_h; return C*((Real)(1.0/3.0) * (s*s*s)); }
        else return (Real)0;
    }

    // grad W = d(W)/dr Ur            in spherical coordinates, with Ur = D/|D| = D/r
    //        = 1/h d(W)/dq Ur        with q = r/h
    // if q < 0.5 :  d(W)/dq = Cq(3q - 2)
    // if q < 1   :  d(W)/dq = C(- (1-q)^2)
    static Real constGradW(Real h)
    {
        return constW(h)/(h*h);
    }

    static Deriv gradW(const Deriv& d, Real r_h, Real C)
    {
        Real g;
        if (r_h < (Real)0.5)    g = 3*r_h - 2;
        else if (r_h < (Real)1) { Real s = 1-r_h; g = -s*s/r_h; }
        else return Deriv();
        return d*(C*g);
    }

    // laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
    //              = d2(W)/dr2 + 2/r d(W)/dr      in spherical coordinate, as W only depends on r
    //              = 1/h2 d2(W)/dq2 + 2/r 1/h d(W)/dq      with q = r/h
    //              = 1/h2 (d2(W)/dq2 + 2/q d(W)/dq)
    // if q < 0.5 : d2(W)/dq2 = C(6q - 2)
    // laplacian(W) = 1/h2 (C(6q-2) + 2C(3q-2)
    //              = C/h2 (12q-6)
    // if q < 0.5 : d2(W)/dq2 = C(-2q + 2)
    // laplacian(W) = 1/h2 (C(-2q+2) - 2C/q (1-q)^2)
    //              = 2C/h2 (1-q)(1 - (1-q)/q)
    //              = 2C/h2 (1-q)(1 - 1/q + 1)
    //              = C/h2 (1-q)(4 - 2/q)
    //              = C/h2 (4 - 2/q - 4q + 2)
    //              = C/h2 (6 - 4q - 2/q)

    static Real  constLaplacianW(Real h)
    {
        return constW(h)/(h*h);
    }

    static Real  laplacianW(Real r_h, Real C)
    {
        if (r_h < (Real)0.5)    return C*(12*r_h-6);
        else if (r_h < (Real)1) return C*(6-4*r_h-2/r_h);
        else return 0;
    }

    static Real  W2(Real r2_h2, Real C)
    {
        return W(helper::rsqrt(r2_h2), C);
    }

    static Deriv gradW2(const Deriv& d, Real r2_h2, Real C)
    {
        return gradW(d, helper::rsqrt(r2_h2), C);
    }

    static Real  laplacianW2(Real r2_h2, Real C)
    {
        return laplacianW(helper::rsqrt(r2_h2), C);
    }

};


template<SPHKernels KT, class Deriv>
class SPHKernel: public BaseSPHKernel<KT, Deriv>
{
public:
    typedef BaseSPHKernel<KT, Deriv> K;
    typedef typename Deriv::value_type Real;
    enum { N = Deriv::spatial_dimensions };

    // Instanced methods, storing constants as member variables

    const Real H;
    const Real cW;
    const Real cGW;
    const Real cLW;

    SPHKernel(Real h)
        : H(h),
          cW( K::constW(h) ),
          cGW( K::constGradW(h) ),
          cLW( K::constLaplacianW(h) )
    {
    }

    Real W(Real r_h) const
    {
        return K::W(r_h, cW);
    }

    Deriv gradW(const Deriv& d, Real r_h) const
    {
        return K::gradW(d, r_h, cGW);
    }

    Real laplacianW(Real r_h) const
    {
        return K::laplacianW(r_h, cLW);
    }


    Real W2(Real r2_h2) const
    {
        return K::W2(r2_h2, cW);
    }

    Deriv gradW2(const Deriv& d, Real r2_h2) const
    {
        return K::gradW2(d, r2_h2, cGW);
    }

    Real laplacianW2(Real r2_h2) const
    {
        return K::laplacianW2(r2_h2, cLW);
    }

    // Check kernel constants and derivatives

    bool CheckKernel(std::ostream& sout, std::ostream& serr)
    {
        const int log2S = 5;
        const int iS = 1 << (log2S);
        const Real S = (Real)iS;
        const int iT = 1 << (log2S*N);

        double sum = 0.0;
        for (int i = 0; i<iT; ++i)
        {
            double norm2 = 0;
            double area = 1;
            for (int c = 0; c<N; ++c)
            {
                int ix = (i >> log2S * c) & ((1 << log2S) - 1);
                Real x = (ix) / S;
                norm2 += x * x;
                area *= (ix == 0) ? H / S : 2 * H / S;
            }
            Real q = (Real)sqrt(norm2);
            if (q > 1) continue;
            Real w = W(q);
            if (w > 1000000000.f)
            {
                if (q == 0) sout << "W" << K::Name() << "(" << q << ") = " << w << std::endl;
                else serr << "W" << K::Name() << "(" << q << ") = " << w << std::endl;
            }
            else if (w < 0) serr << "W" << K::Name() << "(" << q << ") = " << w << std::endl;
            else sum += area * w;
        }
        if (fabs(sum - 1) > 0.01)
        {
            serr << "sum(" << "W" << K::Name() << ") = " << sum << std::endl;
            return false;
        }
        else
        {
            sout << "Kernel " << "W" << K::Name() << "  OK" << std::endl;
            return true;
        }
    }


    bool CheckGrad(std::ostream& sout, std::ostream& serr)
    {
        const int iG = 4 * 1024;
        const Real G = (Real)iG;

        Deriv D;
        int nerr = 0;
        Real err0 = 0, err1 = -1;
        Real maxerr = 0, maxerr_q = 0, maxerr_grad = 0, maxerr_dw = 0;
        for (int r = 2; r < iG; ++r)
        {
            Real q = r / G;
            D[0] = q * H;
            Deriv grad = gradW(D, q);
            Real dw = (W(q + 0.5f / G) - W(q - 0.5f / G)) * G / H;
            if (fabs(grad[0] - dw) > 0.000001f && fabs(grad[0] - dw) > 0.1f*fabs(dw))
            {
                if (!nerr)
                {
                    serr << "grad" << "W" << K::Name() << "(" << q << ") = " << grad[0] << " != " << dw << std::endl;
                    err0 = err1 = q;
                }
                else err1 = q;
                if (fabs(grad[0] - dw) > maxerr)
                {
                    maxerr = fabs(grad[0] - dw); maxerr_q = q; maxerr_grad = grad[0]; maxerr_dw = dw;
                }
                ++nerr;
            }
            else if (err1 == (r - 1) / G)
                serr << "grad" << "W" << K::Name() << "(" << q << ") = " << grad[0] << " ~ " << dw << std::endl;
        }
        if (nerr > 0)
        {
            serr << "grad" << "W" << K::Name() << " failed within q = [" << err0 << " " << err1 << "] (" << 0.01*(nerr * 10000 / (iG - 2)) << "%) :  " << "grad" << "W" << K::Name() << "(" << maxerr_q << ") = " << maxerr_grad << " != " << maxerr_dw << std::endl;
            return false;
        }
        else
        {
            sout << "grad" << "W" << K::Name() << " OK" << std::endl;
            return true;
        }
    }

    bool CheckLaplacian(std::ostream& sout, std::ostream& serr)
    {
        const int iG = 4 * 1024;
        const Real G = (Real)iG;

        int nerr = 0;
        Real err0 = 0, err1 = -1;
        Real maxerr = 0, maxerr_q = 0, maxerr_lap = 0, maxerr_l = 0;
        for (int r = 2; r < iG; ++r)
        {
            Real q = r * 1.0f / G;
            Real w0 = W(q);
            Real wa = W(q - 0.5f / G);
            Real wb = W(q + 0.5f / G);
            Real lap = laplacianW(q);
            Real dw = (wb - wa) * G / H;
            Real dw2 = (wb - 2 * w0 + wa) * (2 * 2 * G*G / (H*H));
            Real l = dw2 + 2 / (q*H)*dw;
            if (fabs(lap - l) > 0.00001f && fabs(lap - l) > 0.1f * fabs(l))
            {
                if (!nerr)
                {
                    serr << "laplacian" << "W" << K::Name() << "(" << q << ") = " << lap << " != " << l << std::endl;
                    err0 = err1 = q;
                }
                else err1 = q;
                ++nerr;
                if (fabs(lap - dw2) > maxerr)
                {
                    maxerr = fabs(lap - dw2); maxerr_q = q; maxerr_lap = lap; maxerr_l = l;
                }
            }
            else if (err1 == (r - 1) / G)
                serr << "laplacian" << "W" << K::Name() << "(" << q << ") = " << lap << " ~ " << l << std::endl;
        }
        if (nerr > 0)
        {
            serr << "laplacian" << "W" << K::Name() << " failed within q = [" << err0 << " " << err1 << "] (" << 0.01*(nerr * 10000 / (iG - 2)) << "%):  " << "laplacian" << "W" << K::Name() << "(" << maxerr_q << ") = " << maxerr_lap << " != " << maxerr_l << std::endl;
            return false;
        }
        else
        {
            sout << "laplacian" << "W" << K::Name() << " OK" << std::endl;
            return true;
        }
    }

    bool CheckAll(int order, std::ostream& sout, std::ostream& serr)
    {
        bool ok = true;
        if (order >= 0)
            ok &= CheckKernel(sout, serr);
        if (order >= 1)
            ok &= CheckGrad(sout, serr);
        if (order >= 2)
            ok &= CheckLaplacian(sout, serr);
        return ok;
    }
};

} // namespace sofa::component::forcefield


#endif // SOFA_COMPONENT_FORCEFIELD_SPHKERNEL_H
