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
#include <sofa/component/constraint/projective/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/vector.h>
#include <set>

namespace sofa::component::constraint::projective
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class PositionBasedDynamicsProjectiveConstraintInternalData
{
};

/** Position-based dynamics as described in [Muller06]:
input: target positions X
output : x(t) <- x(t) + stiffness.( X - x(t) )
		 v(t) <- [ x(t) - x(t-1) ] / dt = v(t) + stiffness.( X - x(t) ) /dt

*/

template <class DataTypes>
class PositionBasedDynamicsProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PositionBasedDynamicsProjectiveConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

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
    PositionBasedDynamicsProjectiveConstraintInternalData<DataTypes> data;
    friend class PositionBasedDynamicsProjectiveConstraintInternalData<DataTypes>;

public:
    Data< Real > stiffness; ///< Blending between current pos and target pos.
    Data< VecCoord > position; ///< Target positions.

    Data < VecDeriv > velocity; ///< Velocities.
    Data < VecCoord > old_position; ///< Old positions.

    PositionBasedDynamicsProjectiveConstraint();

    virtual ~PositionBasedDynamicsProjectiveConstraint();

    // -- Constraint interface
    void init() override;
    void reset() override;

    void projectResponse(const core::MechanicalParams* , DataVecDeriv& ) override {}
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    // Handle topological changes
    void handleTopologyChange() override;

protected :



};


#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<defaulttype::Rigid3Types>;
//extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<defaulttype::Rigid2Types>;
#endif


} // namespace sofa::component::constraint::projective
