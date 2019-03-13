/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_CONSTRAINTSET_DISTANCELMCONSTRAINT_CPP
#include <SofaConstraint/DistanceLMConstraint.inl>

#include <sofa/core/behavior/LMConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

///TODO: handle combinaison of Rigid and Deformable bodies.

int DistanceLMConstraintClass = core::RegisterObject("Maintain constant the length of some edges of a pair of objects")
        .add< DistanceLMConstraint<Vec3Types> >()
        .add< DistanceLMConstraint<Rigid3Types> >()

        ;

template class SOFA_CONSTRAINT_API DistanceLMConstraint<Vec3Types>;
template class SOFA_CONSTRAINT_API DistanceLMConstraint<Rigid3Types>;



//TODO(dmarchal) Yet again this ugly code duplication between float and double.
// To fix this you can use the same design of UniformMass.
template<>
Rigid3Types::Deriv DistanceLMConstraint<Rigid3Types>::getDirection(const Edge &e, const VecCoord &x1, const VecCoord &x2) const
{
    Vector3 V12=(x2[e[1]].getCenter() - x1[e[0]].getCenter()); V12.normalize();
    return Deriv(V12, Vector3());
}
template<>
void DistanceLMConstraint<Rigid3Types>::draw(const core::visual::VisualParams* vparams)
{
    if (this->l0.size() != vecConstraint.getValue().size()) updateRestLength();

    if (vparams->displayFlags().getShowBehaviorModels())
    {
        const VecCoord &x1= this->constrainedObject1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord &x2= this->constrainedObject2->read(core::ConstVecCoordId::position())->getValue();

        std::vector< Vector3 > points;
        const SeqEdges &edges =  vecConstraint.getValue();
        for (unsigned int i=0; i<edges.size(); ++i)
        {
            points.push_back(x1[edges[i][0]].getCenter());
            points.push_back(x2[edges[i][1]].getCenter());
        }
        vparams->drawTool()->drawLines(points, 1, Vec<4,float>(0.0,1.0,0.0f,1.0f));
    }
}


} // namespace constraintset

} // namespace component

} // namespace sofa

