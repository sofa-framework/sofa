/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHCONSTRAINT_CPP
#include <SofaGeneralObjectInteraction/AttachConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(AttachConstraint)

int AttachConstraintClass = core::RegisterObject("Attach given pair of particles, projecting the positions of the second particles to the first ones")
#ifndef SOFA_FLOAT
        .add< AttachConstraint<Vec3dTypes> >()
        .add< AttachConstraint<Vec2dTypes> >()
        .add< AttachConstraint<Vec1dTypes> >()
        .add< AttachConstraint<Rigid3dTypes> >()
        .add< AttachConstraint<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< AttachConstraint<Vec3fTypes> >()
        .add< AttachConstraint<Vec2fTypes> >()
        .add< AttachConstraint<Vec1fTypes> >()
        .add< AttachConstraint<Rigid3fTypes> >()
        .add< AttachConstraint<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec3dTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec2dTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec1dTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Rigid3dTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec3fTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec2fTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec1fTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Rigid3fTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Rigid2fTypes>;
#endif

#ifndef SOFA_FLOAT
template <> SOFA_GENERAL_OBJECT_INTERACTION_API
void AttachConstraint<Rigid3dTypes>::calcRestRotations()
{
    const SetIndexArray & indices2 = f_indices2.getValue();
    const VecCoord& x0 = this->mstate2->read(core::ConstVecCoordId::restPosition())->getValue();
    restRotations.resize(indices2.size());
    for (unsigned int i=0; i<indices2.size(); ++i)
    {
        Quat q(0,0,0,1);
        if (indices2[i] < x0.size()-1)
        {
            Vector3 dp0 = x0[indices2[i]].unprojectVector(x0[indices2[i]+1].getCenter()-x0[indices2[i]].getCenter());
            dp0.normalize();
            Vector3 y = cross(dp0, Vector3(1,0,0));
            y.normalize();
            double alpha = acos(dp0[0]);
            q = Quat(y,alpha);
            sout << "restRotations x2["<<indices2[i]<<"]="<<q<<" dp0="<<dp0<<" qx="<<q.rotate(Vector3(1,0,0))<<sendl;
        }
        restRotations[i] = q;
    }
}
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

