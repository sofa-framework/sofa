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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSCONSTRAINT_CPP
#include <SofaBoundaryCondition/PositionBasedDynamicsConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


SOFA_DECL_CLASS(PositionBasedDynamicsConstraint)

int PositionBasedDynamicsConstraintClass = core::RegisterObject("Position-based dynamics")

#ifndef SOFA_FLOAT
        .add< PositionBasedDynamicsConstraint<Vec3dTypes> >(true)
        .add< PositionBasedDynamicsConstraint<Vec2dTypes> >()
        .add< PositionBasedDynamicsConstraint<Vec1dTypes> >()
        .add< PositionBasedDynamicsConstraint<Vec6dTypes> >()
        .add< PositionBasedDynamicsConstraint<Rigid3dTypes> >()
//.add< PositionBasedDynamicsConstraint<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PositionBasedDynamicsConstraint<Vec3fTypes> >(true)
        .add< PositionBasedDynamicsConstraint<Vec2fTypes> >()
        .add< PositionBasedDynamicsConstraint<Vec1fTypes> >()
        .add< PositionBasedDynamicsConstraint<Vec6fTypes> >()
        .add< PositionBasedDynamicsConstraint<Rigid3fTypes> >()
//.add< PositionBasedDynamicsConstraint<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Rigid3dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Rigid3fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Rigid2fTypes>;
#endif


// specialization for rigids

#ifndef SOFA_FLOAT
template <>
void PositionBasedDynamicsConstraint<Rigid3dTypes>::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> res ( mparams, xData );
    helper::ReadAccessor<DataVecCoord> tpos = position ;
	helper::WriteAccessor<DataVecDeriv> vel ( mparams, velocity );
	helper::WriteAccessor<DataVecCoord> old_pos ( mparams, old_position );
    if (tpos.size() != res.size()) 	{ serr << "Invalid target position vector size." << sendl;		return; }

    Real dt =  (Real)this->getContext()->getDt();
    if(!dt) return;

    vel.resize(res.size());

	if(old_pos.size() != res.size()) {
		old_pos.resize(res.size());
		std::copy(res.begin(),res.end(),old_pos.begin());
	}

    Vec<3,Real> a; Real phi;

    Real s = stiffness.getValue();
    for( size_t i=0; i<res.size(); i++ )
    {
        res[i].getCenter() += ( tpos[i].getCenter() - res[i].getCenter()) * s;

        if(s==(Real)1.) res[i].getOrientation() = tpos[i].getOrientation();
        else 	res[i].getOrientation().slerp(res[i].getOrientation(),tpos[i].getOrientation(),(float)s,false);

        getLinear(vel[i]) = (res[i].getCenter() - old_pos[i].getCenter())/dt;
        ( res[i].getOrientation() * old_pos[i].getOrientation().inverse() ).quatToAxis(a , phi) ;
        getAngular(vel[i]) = a * phi / dt;

        old_pos[i] = res[i];
    }
}

#endif

#ifndef SOFA_DOUBLE
template <>
void PositionBasedDynamicsConstraint<Rigid3fTypes>::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> res ( mparams, xData );
    helper::ReadAccessor<DataVecCoord> tpos = position ;
	helper::WriteAccessor<DataVecDeriv> vel ( mparams, velocity );
	helper::WriteAccessor<DataVecCoord> old_pos ( mparams, old_position );
    if (tpos.size() != res.size()) 	{ serr << "Invalid target position vector size." << sendl;		return; }

    Real dt =  (Real)this->getContext()->getDt();
    if(!dt) return;

    vel.resize(res.size());

	if(old_pos.size() != res.size()) {
		old_pos.resize(res.size());
		std::copy(res.begin(),res.end(),old_pos.begin());
	}

    Vec<3,Real> a; Real phi;

    Real s = stiffness.getValue();
    for( size_t i=0; i<res.size(); i++ )
    {
        res[i].getCenter() += ( tpos[i].getCenter() - res[i].getCenter()) * s;

        if(s==(Real)1.) res[i].getOrientation() = tpos[i].getOrientation();
        else 	res[i].getOrientation().slerp(res[i].getOrientation(),tpos[i].getOrientation(),(float)s,false);

        getLinear(vel[i]) = (res[i].getCenter() - old_pos[i].getCenter())/dt;
        ( res[i].getOrientation() * old_pos[i].getOrientation().inverse() ).quatToAxis(a , phi) ;
        getAngular(vel[i]) = a * phi / dt;

        old_pos[i] = res[i];
    }
}

#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

