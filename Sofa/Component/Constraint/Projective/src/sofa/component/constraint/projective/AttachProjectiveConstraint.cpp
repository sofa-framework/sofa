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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHPROJECTIVECONSTRAINT_CPP
#include <sofa/component/constraint/projective/AttachProjectiveConstraint.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::constraint::projective
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::helper;

int AttachProjectiveConstraintClass = core::RegisterObject("Attach given pair of particles, projecting the positions of the second particles to the first ones")
        .add< AttachProjectiveConstraint<Vec3Types> >()
        .add< AttachProjectiveConstraint<Vec2Types> >()
        .add< AttachProjectiveConstraint<Vec1Types> >()
        .add< AttachProjectiveConstraint<Rigid3Types> >()
        .add< AttachProjectiveConstraint<Rigid2Types> >()
        ;

template <> SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API
void AttachProjectiveConstraint<Rigid3Types>::calcRestRotations()
{
    const SetIndexArray & indices2 = f_indices2.getValue();
    const VecCoord& x0 = this->mstate2->read(core::ConstVecCoordId::restPosition())->getValue();
    restRotations.resize(indices2.size());
    for (unsigned int i=0; i<indices2.size(); ++i)
    {
        Quat<SReal> q(0,0,0,1);
        if (indices2[i] < x0.size()-1)
        {
            Vec3 dp0 = x0[indices2[i]].unprojectVector(x0[indices2[i]+1].getCenter()-x0[indices2[i]].getCenter());
            dp0.normalize();
            Vec3 y = cross(dp0, Vec3(1_sreal,0_sreal,0_sreal));
            y.normalize();
            const double alpha = acos(dp0[0]);
            q = Quat<SReal>(y,alpha);
            msg_info() << "restRotations x2["<<indices2[i]<<"]="<<q<<" dp0="<<dp0<<" qx="<<q.rotate(Vec3(1_sreal,0_sreal,0_sreal));
        }
        restRotations[i] = q;
    }
}

template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<Vec2Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<Vec1Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<Rigid3Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<Rigid2Types>;



} // namespace sofa::component::constraint::projective
