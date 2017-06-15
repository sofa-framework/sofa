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
#define SOFA_COMPONENT_CONSTRAINTSET_DISTANCELMCONSTRAINT_CPP
#include <SofaConstraint/DistanceLMConstraint.inl>

#include <sofa/core/behavior/LMConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
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

SOFA_DECL_CLASS(DistanceLMConstraint)

int DistanceLMConstraintClass = core::RegisterObject("Maintain constant the length of some edges of a pair of objects")
#ifndef SOFA_FLOAT
        .add< DistanceLMConstraint<Vec3dTypes> >()
        .add< DistanceLMConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceLMConstraint<Vec3fTypes> >()
        .add< DistanceLMConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_CONSTRAINT_API DistanceLMConstraint<Vec3dTypes>;
template class SOFA_CONSTRAINT_API DistanceLMConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API DistanceLMConstraint<Vec3fTypes>;
template class SOFA_CONSTRAINT_API DistanceLMConstraint<Rigid3fTypes>;
#endif


//TODO(dmarchal) Yet again this ugly code duplication between float and double.
// To fix this you can use the same design of UniformMass.
#ifndef SOFA_FLOAT
template<>
Rigid3dTypes::Deriv DistanceLMConstraint<Rigid3dTypes>::getDirection(const Edge &e, const VecCoord &x1, const VecCoord &x2) const
{
    Vector3 V12=(x2[e[1]].getCenter() - x1[e[0]].getCenter()); V12.normalize();
    return Deriv(V12, Vector3());
}
template<>
void DistanceLMConstraint<Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
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
//                 double length     = lengthEdge(edges[i],x1,x2);
//                 double restLength = this->l0[i];
//                 double factor = fabs(length - restLength)/length;
            points.push_back(x1[edges[i][0]].getCenter());
            points.push_back(x2[edges[i][1]].getCenter());
        }
        vparams->drawTool()->drawLines(points, 1, Vec<4,float>(0.0,1.0,0.0f,1.0f));
    }
}
#endif

#ifndef SOFA_DOUBLE
template<>
Rigid3fTypes::Deriv DistanceLMConstraint<Rigid3fTypes>::getDirection(const Edge &e, const VecCoord &x1, const VecCoord &x2) const
{
    Vector3 V12=(x2[e[1]].getCenter() - x1[e[0]].getCenter()); V12.normalize();
    return Deriv(V12, Vector3());
}
template<>
void DistanceLMConstraint<Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
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
//                 double length     = lengthEdge(edges[i],x1,x2);
//                 double restLength = this->l0[i];
//                 double factor = fabs(length - restLength)/length;
            points.push_back(x1[edges[i][0]].getCenter());
            points.push_back(x2[edges[i][1]].getCenter());
        }
        vparams->drawTool()->drawLines(points, 1, Vec<4,float>(0.0,1.0,0.0f,1.0f));
    }
}
#endif
} // namespace constraintset

} // namespace component

} // namespace sofa

