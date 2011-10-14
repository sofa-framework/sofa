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
#ifndef SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H

#include <sofa/helper/system/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/container/SpatialGridContainer.h>
#include <sofa/helper/rmath.h>
#include <vector>
#include <math.h>

#include <sofa/component/component.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::component::container;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class SPHFluidForceFieldInternalData
{
public:
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
    Data< Real > particleRadius;
    Data< Real > particleMass;
    Data< Real > pressureStiffness; ///< 100 - 1000 m2/s2
    Data< Real > density0; ///< 1000 kg/m3 for water
    Data< Real > viscosity;
    Data< Real > surfaceTension;
    //Data< int  > pressureExponent;

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

    typedef SpatialGridContainer<DataTypes> Grid;

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
    /// Density Smoothing Kernel:  W = 315 / 64pih9 * (h2 - r2)3 = 315 / 64pih3 * (1 - (r/h)2)3
    Real  constWd(Real h) const
    {
        return (Real)(315 / (64*R_PI*h*h*h));

    }
    Real  Wd(Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        if(a<=0)return 0;
        return  C*a*a*a;
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
    Real constGradWd(Real h) const
    {
        return -6*constWd(h)/(h*h);
        //		return -6*constWd(h)/h;
    }

    Deriv gradWd(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        if(a<=0)return Deriv();
        return d*(C*a*a);
        //		return d*(C*a*a)*r_h;
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
    Real  constLaplacianWd(Real h) const
    {
        return 6*constWd(h)/(h*h);
    }
    Real  laplacianWd(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
    }

    /// Pressure Smoothing Kernel:  W = 15 / pih6 (h - r)3 = 15 / pih3 (1 - r/h)3
    Real  constWp(Real h) const
    {
        return (Real)(15 / (R_PI*h*h*h));
    }
    Real  Wp(Real r_h, Real C)
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
    Real  constGradWp(Real h) const
    {
        return (-3*constWp(h)) / (h*h*h*h);
    }
    Deriv gradWp(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h);
        return d * (C*a*a);
    }

    //Real  laplacianWp(Real r_h, Real C);

    /// Viscosity Smoothing Kernel:  W = 15/(2pih3) (-r3/2h3 + r2/h2 + h/2r - 1)
    Real  constWv(Real h)
    {
        return (Real)(15/(2*R_PI*h*h*h));
    }
    Real  Wv(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        Real r3_h3 = r2_h2*r_h;
        return C*(-0.5f*r3_h3 + r2_h2 + 0.5f/r_h - 1);
    }

    // grad W = d(W)/dr Ur            in spherical coordinates, with Ur = D/|D| = D/r
    //        = d( C(-r3/2h3 + r2/h2 + h/2r - 1) )/dr D/r
    //        = C(-3r2/2h3 + 2r/h2 - h/2r2) D/r
    //        = C(-3r/2h3 + 2/h2 - h/2r3) D
    //        = C/2h2 (-3r/h + 4 - h3/r3) D

    Real  constGradWv(Real h)
    {
        return constWv(h)/(2*h*h);
    }
    Deriv gradWv(const Deriv& d, Real r_h, Real C)
    {
        Real r3_h3 = r_h*r_h*r_h;
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

    Real  constLaplacianWv(Real h)
    {
        return 6*constWv(h)/(h*h);
        //return 75/(R_PI*h*h*h*h*h);
    }

    Real  laplacianWv(Real r_h, Real C)
    {
        return C*(1-r_h);
    }

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
        return -6*constWc(h)/(h*h);
    }
    Deriv gradWc(const Deriv& d, Real r_h, Real C)
    {
        Real a = (1-r_h*r_h);
        return d*(C*a*a);
    }
    Real  constLaplacianWc(Real h) const
    {
        return 6*constWc(h)/(h*h);
    }
    Real  laplacianWc(Real r_h, Real C)
    {
        Real r2_h2 = r_h*r_h;
        return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
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

    virtual void init();

    virtual void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

    void draw(const core::visual::VisualParams* vparams);
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::Vec2dTypes;
using sofa::defaulttype::Vec2fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_CPP)
#pragma warning(disable : 4231)

#ifndef SOFA_FLOAT
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec3dTypes>;
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec3fTypes>;
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec2fTypes>;
#endif

#endif // defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
