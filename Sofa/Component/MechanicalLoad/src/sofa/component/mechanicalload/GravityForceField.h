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
#pragma once

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/type/Vec.h>

namespace sofa::component::mechanicalload
{

/// ForceField applying the external force due to the gravitational acceleration.
/// This class requires a link towards a Mass to compute the space integration with the mass density.
template<class DataTypes>
class GravityForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(GravityForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::DPos DPos;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data< DPos > d_gravitationalAcceleration; ///< Value corresponding to the gravitational acceleration
    SingleLink<GravityForceField<DataTypes>, sofa::core::behavior::Mass<DataTypes>, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_mass; ///< Link to be set to the mass in the component graph

    /// Init function
    void init() override;

    /// Add the external forces due to the gravitational acceleration.
    void addForce (const core::MechanicalParams* params, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    /// Gravity force has null variation
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df , const DataVecDeriv& d_dx) override;

    /// Gravity force has null variation
    void addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal k, unsigned int &offset) override;

    SReal getPotentialEnergy(const core::MechanicalParams* params, const DataVecCoord& x) const override;

    /// Set the gravitational acceleration
    void setGravitationalAcceleration(const DPos grav);

protected:
    GravityForceField();
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_GRAVITYFORCEFIELD_CPP)
extern template class SOFA_CORE_API GravityForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_CORE_API GravityForceField<sofa::defaulttype::Vec2Types>;
extern template class SOFA_CORE_API GravityForceField<sofa::defaulttype::Vec1Types>;
extern template class SOFA_CORE_API GravityForceField<sofa::defaulttype::Vec6Types>;
extern template class SOFA_CORE_API GravityForceField<sofa::defaulttype::Rigid3Types>;
extern template class SOFA_CORE_API GravityForceField<sofa::defaulttype::Rigid2Types>;
#endif

} // namespace sofa::component::mechanicalload
