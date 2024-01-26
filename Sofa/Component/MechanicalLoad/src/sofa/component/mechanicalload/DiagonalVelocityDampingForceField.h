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
#include <sofa/component/mechanicalload/config.h>

#include <sofa/core/behavior/ForceField.h>

namespace sofa::component::mechanicalload
{

/// Apply damping forces to given degrees of freedom.
template<class DataTypes>
class DiagonalVelocityDampingForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DiagonalVelocityDampingForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::vector<unsigned int> VecIndex;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    /// velocity damping coefficients (by cinematic dof)
    Data< VecDeriv > d_dampingCoefficients;

    SOFA_ATTRIBUTE_DISABLED("v23.12", "v24.06", "This Data is now replaced by d_dampingCoefficients")
    DeprecatedAndRemoved dampingCoefficients;

protected:

    DiagonalVelocityDampingForceField();

public:

    void addForce (const core::MechanicalParams*, DataVecDeriv&, const DataVecCoord&, const DataVecDeriv&) override;

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df , const DataVecDeriv& d_dx) override;

    void addKToMatrix(sofa::linearalgebra::BaseMatrix * /*m*/, SReal /*kFactor*/, unsigned int &/*offset*/) override {}
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* /* matrix */) override;

    void addBToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal bFact, unsigned int& offset) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* matrix) override;

    SReal getPotentialEnergy(const core::MechanicalParams* params, const DataVecCoord& x) const override;

};


#if !defined(SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<defaulttype::Rigid2Types>;

#endif

} // namespace sofa::component::mechanicalload
