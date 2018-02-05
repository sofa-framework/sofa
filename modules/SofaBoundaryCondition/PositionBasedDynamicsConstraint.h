/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <set>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class PositionBasedDynamicsConstraintInternalData
{
};

/** Position-based dynamics as described in [Muller06]:
input: target positions X
output : x(t) <- x(t) + stiffness.( X - x(t) )
		 v(t) <- [ x(t) - x(t-1) ] / dt = v(t) + stiffness.( X - x(t) ) /dt

*/

template <class DataTypes>
class PositionBasedDynamicsConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PositionBasedDynamicsConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;

protected:
    PositionBasedDynamicsConstraintInternalData<DataTypes> data;
    friend class PositionBasedDynamicsConstraintInternalData<DataTypes>;

public:
    Data< Real > stiffness;
    Data< VecCoord > position;

    Data < VecDeriv > velocity;
    Data < VecCoord > old_position;

    PositionBasedDynamicsConstraint();

    virtual ~PositionBasedDynamicsConstraint();

    // -- Constraint interface
    virtual void init() override;
    virtual void reset() override;

    void projectResponse(const core::MechanicalParams* , DataVecDeriv& ) override {}
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

//    void applyConstraint(defaulttype::BaseMatrix *, unsigned int ) {}
//    void applyConstraint(defaulttype::BaseVector *, unsigned int ) {}

    // Handle topological changes
    virtual void handleTopologyChange() override;

protected :



};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec6dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Vec6fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
