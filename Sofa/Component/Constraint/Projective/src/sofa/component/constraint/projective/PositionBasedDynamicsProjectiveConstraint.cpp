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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POSITIONBASEDDYNAMICSPROJECTIVECONSTRAINT_CPP
#include <sofa/component/constraint/projective/PositionBasedDynamicsProjectiveConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>

namespace sofa::component::constraint::projective
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::helper;


int PositionBasedDynamicsProjectiveConstraintClass = core::RegisterObject("Position-based dynamics")

        .add< PositionBasedDynamicsProjectiveConstraint<Vec3Types> >(true)
        .add< PositionBasedDynamicsProjectiveConstraint<Vec2Types> >()
        .add< PositionBasedDynamicsProjectiveConstraint<Vec1Types> >()
        .add< PositionBasedDynamicsProjectiveConstraint<Vec6Types> >()
        .add< PositionBasedDynamicsProjectiveConstraint<Rigid3Types> >()
        ;

template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<Vec2Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<Vec1Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<Vec6Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<Rigid3Types>;
//template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PositionBasedDynamicsProjectiveConstraint<Rigid2Types>;



// specialization for rigids

template <>
void PositionBasedDynamicsProjectiveConstraint<Rigid3Types>::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData)
{
    SOFA_UNUSED(mparams);

    helper::WriteAccessor<DataVecCoord> res ( xData );
    const helper::ReadAccessor<DataVecCoord> tpos = position ;
    helper::WriteAccessor<DataVecDeriv> vel ( velocity );
    helper::WriteAccessor<DataVecCoord> old_pos ( old_position );
    if (tpos.size() != res.size()) { msg_error() << "Invalid target position vector size."; return; }

    const Real dt =  (Real)this->getContext()->getDt();
    if(!dt) return;

    vel.resize(res.size());

	if(old_pos.size() != res.size()) {
		old_pos.resize(res.size());
		std::copy(res.begin(),res.end(),old_pos.begin());
	}

    Vec<3,Real> a; Real phi;

    const Real s = stiffness.getValue();
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

} // namespace sofa::component::constraint::projective
