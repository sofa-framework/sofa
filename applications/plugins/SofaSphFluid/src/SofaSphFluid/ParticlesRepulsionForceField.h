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
#ifndef SOFA_COMPONENT_FORCEFIELD_PARTICLESREPULSIONFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_PARTICLESREPULSIONFORCEFIELD_H
#include <SofaSphFluid/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaSphFluid/SpatialGridContainer.h>
#include <sofa/helper/rmath.h>
#include <vector>
#include <cmath>


namespace sofa::component::forcefield
{

template<class DataTypes>
class ParticlesRepulsionForceField : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParticlesRepulsionForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef sofa::core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

public:
    Data< Real > distance; ///< Distance to maintain between particles
    Data< Real > stiffness; ///< Stiffness
    Data< Real > damping; ///< Damping

    typedef sofa::component::container::SpatialGridContainer<DataTypes> Grid;

    Grid* grid;

protected:

    struct Particle
    {
        sofa::type::vector< int > neighbors; ///< indice + r/h
    };

    sofa::type::vector<Particle> particles;

public:
    /// this method is called by the SpatialGrid when w connection between two particles is detected
    void addNeighbor(int i1, int i2, Real /*r2*/, Real /*h2*/)
    {
        //Real r_h = (Real)sqrt(r2/h2);
        if (i1<i2)
            particles[i1].neighbors.push_back(i2);
        else
            particles[i2].neighbors.push_back(i1);
    }
protected:
    ParticlesRepulsionForceField();
public:
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    SReal getPotentialEnergy(const sofa::core::MechanicalParams* /*mparams*/, const DataVecCoord& /* x */) const override
    {
        msg_error() << "getPotentialEnergy not implemented";

        return 0.0;
    }

    void draw(const core::visual::VisualParams* vparams) override;
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec2dTypes;


#if  !defined(SOFA_COMPONENT_FORCEFIELD_PARTICLESREPULSIONFORCEFIELD_CPP)

extern template class SOFA_SPH_FLUID_API ParticlesRepulsionForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_SPH_FLUID_API ParticlesRepulsionForceField<sofa::defaulttype::Vec2Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_PARTICLESREPULSIONFORCEFIELD_CPP)

} // namespace sofa::component::forcefield


#endif // SOFA_COMPONENT_FORCEFIELD_PARTICLESREPULSIONFORCEFIELD_H
