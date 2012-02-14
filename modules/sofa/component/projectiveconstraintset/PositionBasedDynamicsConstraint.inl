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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_INL

#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/component/projectiveconstraintset/PositionBasedDynamicsConstraint.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace core::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::behavior;



template <class DataTypes>
PositionBasedDynamicsConstraint<DataTypes>::PositionBasedDynamicsConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , stiffness(initData(&stiffness,(Real)1.0,"stiffness","Blending between current pos and target pos."))
    , position(initData(&position,"position","Target positions."))
{
    // stiffness.setWidget("0to1RatioWidget");
}


// Handle topological changes
template <class DataTypes> void PositionBasedDynamicsConstraint<DataTypes>::handleTopologyChange()
{
    this->reinit();
}

template <class DataTypes>
PositionBasedDynamicsConstraint<DataTypes>::~PositionBasedDynamicsConstraint()
{
}


// -- Constraint interface


template <class DataTypes>
void PositionBasedDynamicsConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
    if ((int)position.getValue().size() != (int)this->mstate->getSize())    serr << "Invalid target position vector size." << sendl;
}


template <class DataTypes>
void PositionBasedDynamicsConstraint<DataTypes>::reset()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::reset();
    const VecCoord& x = *this->mstate->getX();	old_position.assign(x.begin(),x.end());
    for( unsigned i=0; i<velocity.size(); i++ ) velocity[i]=Deriv();
}



template <class DataTypes>
void PositionBasedDynamicsConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataMatrixDeriv& cData)
{
    helper::WriteAccessor<DataMatrixDeriv> c ( mparams, cData );

    /*
    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();
    { // fix everything
        while (rowIt != rowItEnd)
        {
            rowIt.row().clear();
            ++rowIt;
        }
    }*/
}



template <class DataTypes>
void PositionBasedDynamicsConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> res ( mparams, vData );

    if (velocity.size() != res.size()) 	{ serr << "Invalid target position vector size." << sendl;		return; }
    for( unsigned i=0; i<res.size(); i++ )	res[i] = velocity[i];
}

template <class DataTypes>
void PositionBasedDynamicsConstraint<DataTypes>::projectPosition(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> res ( mparams, xData );
    helper::ReadAccessor<DataVecCoord> tpos = position ;
    if (tpos.size() != res.size()) 	{ serr << "Invalid target position vector size." << sendl;		return; }

    Real dt =  (Real)this->getContext()->getDt();
    if(!dt) return;

    velocity.resize(res.size());

    if(old_position.size() != res.size()) old_position.assign(res.begin(),res.end());

    for( unsigned i=0; i<res.size(); i++ )
    {
        res[i] += ( tpos[i] - res[i]) * stiffness.getValue();
        velocity[i] = (res[i] - old_position[i])/dt;
        old_position[i] = res[i];
    }
}

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void PositionBasedDynamicsConstraint<Rigid3dTypes >::projectPosition(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecCoord& xData);
#endif
#ifndef SOFA_DOUBLE
template <>
void PositionBasedDynamicsConstraint<Rigid3fTypes >::projectPosition(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecCoord& xData);
#endif





} // namespace constraint

} // namespace component

} // namespace sofa

#endif


