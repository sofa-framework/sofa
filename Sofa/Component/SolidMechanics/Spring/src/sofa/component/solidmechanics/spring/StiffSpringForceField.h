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
#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologySubsetIndices.h>


namespace sofa::component::solidmechanics::spring
{

/** SpringForceField able to evaluate and apply its stiffness.
This allows to perform implicit integration.
Stiffness is evaluated and stored by the addForce method.
When explicit integration is used, SpringForceField is slightly more efficient.
*/

template<class DataTypes>
class StiffSpringForceField : public SpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(StiffSpringForceField,DataTypes), SOFA_TEMPLATE(SpringForceField,DataTypes));

    typedef SpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;
    typedef type::vector<sofa::Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;


    typedef typename Inherit::Spring Spring;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    static constexpr auto N = DataTypes::spatial_dimensions;
    typedef type::Mat<N,N,Real> Mat;

    SetIndex d_indices1; ///< Indices of the source points on the first model
    SetIndex d_indices2; ///< Indices of the fixed points on the second model

    core::objectmodel::Data<sofa::type::vector<SReal> > d_lengths; ///< List of lengths to create the springs. Must have the same than indices1 & indices2, or if only one element, it will be applied to all springs. If empty, 0 will be applied everywhere
protected:
    sofa::type::vector<Mat>  dfdx;


    StiffSpringForceField(SReal ks=100.0, SReal kd=5.0);
    StiffSpringForceField(MechanicalState* object1, MechanicalState* object2, SReal ks=100.0, SReal kd=5.0);

    /// Will create the set of springs using \sa d_indices1 and \sa d_indices2 with \sa d_length
    void createSpringsFromInputs();

public:
    void init() override;



protected:


};

#if !defined(SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::solidmechanics::spring
