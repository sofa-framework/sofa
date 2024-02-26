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
#ifndef SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
#include <SofaSphFluid/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaSphFluid/SpatialGridContainer.h>
#include <SofaSphFluid/SPHKernel.h>
#include <sofa/helper/rmath.h>
#include <vector>
#include <cmath>


namespace sofa::component::forcefield
{

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
    Data< Real > d_particleRadius; ///< Radius of a Particle
    Data< Real > d_particleMass; ///< Mass of a Particle
    Data< Real > d_pressureStiffness; ///< 100 - 1000 m2/s2
    Data< Real > d_density0; ///< 1000 kg/m3 for water
    Data< Real > d_viscosity; ///< Viscosity
    Data< Real > d_surfaceTension; ///< Surface Tension
    //Data< int  > pressureExponent;
    Data< int > d_kernelType; ///< 0 = default kernels, 1 = cubic spline
    Data< int > d_pressureType; ///< 0 = none, 1 = default pressure
    Data< int > d_viscosityType; ///< 0 = none, 1 = default viscosity using kernel Laplacian, 2 = artificial viscosity
    Data< int > d_surfaceTensionType; ///< 0 = none, 1 = default surface tension using kernel Laplacian, 2 = cohesion forces surface tension from Becker et al. 2007
    Data< bool > d_debugGrid; ///< If true will store additionnal information on the grid to check neighbors and draw them
protected:
    struct Particle
    {
        Real density;
        Real pressure;
        Deriv normal;
        Real curvature;
        sofa::type::vector< std::pair<int,Real> > neighbors; ///< indice + r/h
        sofa::type::vector< std::pair<int,Real> > neighbors2; ///< indice + r/h
    };

    Real m_lastTime;
    sofa::type::vector<Particle> m_particles;

    typedef sofa::component::container::SpatialGridContainer<DataTypes> Grid;

    Grid* m_grid;

    SPHFluidForceFieldInternalData<DataTypes> data;
    friend class SPHFluidForceFieldInternalData<DataTypes>;

public:
    /// this method is called by the SpatialGrid when w connection between two particles is detected
    void addNeighbor(int i1, int i2, Real r2, Real h2)
    {
        Real r_h = (Real)sqrt(r2/h2);
        if (i1<i2)
            m_particles[i1].neighbors.push_back(std::make_pair(i2,r_h));
        else
            m_particles[i2].neighbors.push_back(std::make_pair(i1,r_h));
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

    sofa::type::vector<DForce> dforces;


    SPHFluidForceField();
public:
    Real getParticleRadius() const { return d_particleRadius.getValue(); }
    void setParticleRadius(Real v) { d_particleRadius.setValue(v);    }
    Real getParticleMass() const { return d_particleMass.getValue(); }
    void setParticleMass(Real v) { d_particleMass.setValue(v);    }
    Real getPressureStiffness() const { return d_pressureStiffness.getValue(); }
    void setPressureStiffness(Real v) { d_pressureStiffness.setValue(v);    }
    Real getDensity0() const { return d_density0.getValue(); }
    void setDensity0(Real v) { d_density0.setValue(v);    }
    Real getViscosity() const { return d_viscosity.getValue(); }
    void setViscosity(Real v) { d_viscosity.setValue(v);    }
    Real getSurfaceTension() const { return d_surfaceTension.getValue(); }
    void setSurfaceTension(Real v) { d_surfaceTension.setValue(v);    }

    Real getParticleField(int i, Real r2_h2)
    {
        Real a = 1-r2_h2;
        return (a*a*a)/m_particles[i].density;
    }

    Real getParticleFieldConstant(Real h)
    {
        return constWc(h)*d_particleMass.getValue();
    }

    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const override;


    void draw(const core::visual::VisualParams* vparams) override;

protected:
    void computeNeighbors(const core::MechanicalParams* mparams, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    template<class Kd, class Kp, class Kv, class Kc>
    void computeForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
};

#if  !defined(SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_CPP)
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_SPH_FLUID_API SPHFluidForceField<sofa::defaulttype::Vec2Types>;
#endif // !defined(SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_CPP)

} // namespace sofa::component::forcefield


#endif // SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_H
