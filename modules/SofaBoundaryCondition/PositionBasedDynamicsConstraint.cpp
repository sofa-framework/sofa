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

#include <sofa/defaulttype/VecTypes.h>
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


int PositionBasedDynamicsConstraintClass = core::RegisterObject("Position-based dynamics")

        .add< PositionBasedDynamicsConstraint<Vec3Types> >(true)
        .add< PositionBasedDynamicsConstraint<Vec2Types> >()
        .add< PositionBasedDynamicsConstraint<Vec1Types> >()
        .add< PositionBasedDynamicsConstraint<Vec6Types> >()
        .add< PositionBasedDynamicsConstraint<Rigid3Types> >()
//.add< PositionBasedDynamicsConstraint<Rigid2Types> >()

        ;

template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec3Types>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec2Types>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec1Types>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Vec6Types>;
template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Rigid3Types>;
//template class SOFA_BOUNDARY_CONDITION_API PositionBasedDynamicsConstraint<Rigid2Types>;



// specialization for rigids

template <>
void PositionBasedDynamicsConstraint<Rigid3Types>::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData)
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



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

