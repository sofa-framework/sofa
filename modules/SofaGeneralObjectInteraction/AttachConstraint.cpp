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

int AttachConstraintClass = core::RegisterObject("Attach given pair of particles, projecting the positions of the second particles to the first ones")
        .add< AttachConstraint<Vec3Types> >()
        .add< AttachConstraint<Vec2Types> >()
        .add< AttachConstraint<Vec1Types> >()
        .add< AttachConstraint<Rigid3Types> >()
        .add< AttachConstraint<Rigid2Types> >()

        ;

template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec3Types>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec2Types>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Vec1Types>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Rigid3Types>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<Rigid2Types>;


template <> SOFA_GENERAL_OBJECT_INTERACTION_API
void AttachConstraint<Rigid3Types>::calcRestRotations()
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


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

