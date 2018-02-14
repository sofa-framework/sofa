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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_INL

#include <SofaBoundaryCondition/PositionBasedDynamicsConstraint.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{


template <class DataTypes>
PositionBasedDynamicsConstraint<DataTypes>::PositionBasedDynamicsConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , stiffness(initData(&stiffness,(Real)1.0,"stiffness","Blending between current pos and target pos."))
    , position(initData(&position,"position","Target positions."))
    , velocity(initData(&velocity,"velocity","Velocities."))
    , old_position(initData(&old_position,"old_position","Old positions."))
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

	helper::WriteAccessor<DataVecDeriv> vel ( velocity );
	std::fill(vel.begin(),vel.end(),Deriv());

	helper::WriteAccessor<DataVecCoord> old_pos ( old_position );
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
	old_pos.resize(x.size());
	std::copy(x.begin(),x.end(),old_pos.begin());
}



template <class DataTypes>
void PositionBasedDynamicsConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
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
void PositionBasedDynamicsConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> res ( mparams, vData );
	helper::ReadAccessor<DataVecDeriv> vel ( mparams, velocity );

    if (vel.size() != res.size()) 	{ serr << "Invalid target position vector size." << sendl;		return; }
    std::copy(vel.begin(),vel.end(),res.begin());
}

template <class DataTypes>
void PositionBasedDynamicsConstraint<DataTypes>::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> res ( mparams, xData );
	helper::WriteAccessor<DataVecDeriv> vel ( mparams, velocity );
	helper::WriteAccessor<DataVecCoord> old_pos ( mparams, old_position );
    helper::ReadAccessor<DataVecCoord> tpos = position ;
    if (tpos.size() != res.size()) 	{ serr << "Invalid target position vector size." << sendl;		return; }

    Real dt =  (Real)this->getContext()->getDt();
    if(!dt) return;
    Real invdt=(Real)(1./dt);

    vel.resize(res.size());

	if(old_pos.size() != res.size()) {
		old_pos.resize(res.size());
		std::copy(res.begin(),res.end(),old_pos.begin());
	}

    for( size_t i=0; i<res.size(); i++ )
    {
        res[i] += ( tpos[i] - res[i]) * stiffness.getValue();
        vel[i] = (res[i] - old_pos[i]) * invdt;
        old_pos[i] = res[i];
    }
}

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void PositionBasedDynamicsConstraint<defaulttype::Rigid3dTypes >::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData);
#endif
#ifndef SOFA_DOUBLE
template <>
void PositionBasedDynamicsConstraint<defaulttype::Rigid3fTypes >::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData);
#endif





} // namespace constraint

} // namespace component

} // namespace sofa

#endif


