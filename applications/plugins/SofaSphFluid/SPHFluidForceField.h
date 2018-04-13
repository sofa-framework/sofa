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
#ifndef SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
#include "config.h"

#include <sofa/helper/system/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaSphFluid/SpatialGridContainer.h>
#include <sofa/helper/rmath.h>
#include <vector>
#include <math.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class SPHFluidForceFieldInternalData
{
public:
};

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

    bool CheckKernel(std::ostream& sout, std::ostream& serr);
    bool CheckGrad(std::ostream& sout, std::ostream& serr);
    bool CheckLaplacian(std::ostream& sout, std::ostream& serr);
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

template<class DataTypes>
class SPHFluidForceField : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SPHFluidForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef sofa::core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

public:
    Data< Real > particleRadius; ///< Radius of a Particle
    Data< Real > particleMass; ///< Mass of a Particle
    Data< Real > pressureStiffness; ///< 100 - 1000 m2/s2
    Data< Real > density0; ///< 1000 kg/m3 for water
    Data< Real > viscosity; ///< Viscosity
    Data< Real > surfaceTension; ///< Surface Tension
    //Data< int  > pressureExponent;
    Data< int > kernelType; ///< 0 = default kernels, 1 = cubic spline
    Data< int > pressureType; ///< 0 = none, 1 = default pressure
    Data< int > viscosityType; ///< 0 = none, 1 = default viscosity using kernel Laplacian, 2 = artificial viscosity
    Data< int > surfaceTensionType; ///< 0 = none, 1 = default surface tension using kernel Laplacian, 2 = cohesion forces surface tension from Becker et al. 2007

protected:
    struct Particle
    {
        Real density;
        Real pressure;
        Deriv normal;
        Real curvature;
        sofa::helper::vector< std::pair<int,Real> > neighbors; ///< indice + r/h
#ifdef SOFA_DEBUG_SPATIALGRIDCONTAINER
        sofa::helper::vector< std::pair<int,Real> > neighbors2; ///< indice + r/h
#endif
    };

    Real lastTime;
    sofa::helper::vector<Particle> particles;

    typedef sofa::component::container::SpatialGridContainer<DataTypes> Grid;

    Grid* grid;

    SPHFluidForceFieldInternalData<DataTypes> data;
    friend class SPHFluidForceFieldInternalData<DataTypes>;

public:
    /// this method is called by the SpatialGrid when w connection between two particles is detected
    void addNeighbor(int i1, int i2, Real r2, Real h2)
    {
        Real r_h = (Real)sqrt(r2/h2);
        if (i1<i2)
            particles[i1].neighbors.push_back(std::make_pair(i2,r_h));
        else
            particles[i2].neighbors.push_back(std::make_pair(i1,r_h));
    }

protected:

    /// Color Smoothing Kernel: same as Density
    Real  constWc(Real h) const
    {
        return (Real)(315 / (64*R_PI*h*h*h));
    }
    Real  Wc(Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        return  C*a*a*a;
    }
    Real  constGradWc(Real h) const
    {
        return -6*constWc(h)/h;
    }
    Deriv gradWc(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        return d*(C*a*a)*r_h;
    }
    Real  constLaplacianWc(Real h) const
    {
        return -6*constWc(h)/(h*h);
    }
    Real  laplacianWc(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        return C*((1-r2_h2)*(1-5*r2_h2));
    }


    struct DForce
    {
        unsigned int a,b;
        Real df;
    };

    sofa::helper::vector<DForce> dforces;


    SPHFluidForceField();
public:
    Real getParticleRadius() const { return particleRadius.getValue(); }
    void setParticleRadius(Real v) { particleRadius.setValue(v);    }
    Real getParticleMass() const { return particleMass.getValue(); }
    void setParticleMass(Real v) { particleMass.setValue(v);    }
    Real getPressureStiffness() const { return pressureStiffness.getValue(); }
    void setPressureStiffness(Real v) { pressureStiffness.setValue(v);    }
    Real getDensity0() const { return density0.getValue(); }
    void setDensity0(Real v) { density0.setValue(v);    }
    Real getViscosity() const { return viscosity.getValue(); }
    void setViscosity(Real v) { viscosity.setValue(v);    }
    Real getSurfaceTension() const { return surfaceTension.getValue(); }
    void setSurfaceTension(Real v) { surfaceTension.setValue(v);    }

    Real getParticleField(int i, Real r2_h2)
    {
        Real a = 1-r2_h2;
        return (a*a*a)/particles[i].density;
    }

    Real getParticleFieldConstant(Real h)
    {
        return constWc(h)*particleMass.getValue();
    }

    virtual void init() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const override;


    void draw(const core::visual::VisualParams* vparams) override;

protected:
    void computeNeighbors(const core::MechanicalParams* mparams, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    template<class Kd, class Kp, class Kv, class Kc>
    void computeForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
};

#ifndef SOFA_FLOAT
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec2dTypes;
#endif

#ifndef SOFA_DOUBLE
using sofa::defaulttype::Vec2fTypes;
using sofa::defaulttype::Vec3fTypes;
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec3dTypes>;
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec3fTypes>;
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec2fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
