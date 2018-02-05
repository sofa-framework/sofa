/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_CPP
#include <SofaBoundaryCondition/PartialFixedConstraint.inl>
#include <sofa/core/visual/VisualParams.h>
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


SOFA_DECL_CLASS(PartialFixedConstraint)

int PartialFixedConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
#ifndef SOFA_FLOAT
        .add< PartialFixedConstraint<Vec3dTypes> >()
        .add< PartialFixedConstraint<Vec2dTypes> >()
        .add< PartialFixedConstraint<Vec1dTypes> >()
        .add< PartialFixedConstraint<Vec6dTypes> >()
        .add< PartialFixedConstraint<Rigid3dTypes> >()
        .add< PartialFixedConstraint<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PartialFixedConstraint<Vec3fTypes> >()
        .add< PartialFixedConstraint<Vec2fTypes> >()
        .add< PartialFixedConstraint<Vec1fTypes> >()
        .add< PartialFixedConstraint<Vec6fTypes> >()
        .add< PartialFixedConstraint<Rigid3fTypes> >()
        .add< PartialFixedConstraint<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Rigid3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Rigid3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<Rigid2fTypes>;
#endif

#ifndef SOFA_FLOAT
template <>
void PartialFixedConstraint<Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = d_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    std::vector< Vector3 > points;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    if( d_fixAll.getValue()==true )
        for (unsigned i=0; i<x.size(); i++ )
            points.push_back(x[i].getCenter());
    else
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            points.push_back(x[*it].getCenter());

    if( d_drawSize.getValue() == 0) // old classical drawing by points
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    else
        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
}

template <>
void PartialFixedConstraint<Rigid2dTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    const SetIndexArray & indices = d_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    if( d_fixAll.getValue()==true )
        for (unsigned i=0; i<x.size(); i++ )
            gl::glVertexT(x[i].getCenter());
    else
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            gl::glVertexT(x[*it].getCenter());
    glEnd();
    glPointSize(1);
#endif /* SOFA_NO_OPENGL */
}
#endif

#ifndef SOFA_DOUBLE
template <>
void PartialFixedConstraint<Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = d_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    std::vector< Vector3 > points;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    if( d_fixAll.getValue()==true )
        for (unsigned i=0; i<x.size(); i++ )
            points.push_back(x[i].getCenter());
    else
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            points.push_back(x[*it].getCenter());

    if( d_drawSize.getValue() == 0) // old classical drawing by points
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    else
        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
}

template <>
void PartialFixedConstraint<Rigid2fTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    const SetIndexArray & indices = d_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    if( d_fixAll.getValue()==true )
        for (unsigned i=0; i<x.size(); i++ )
            gl::glVertexT(x[i].getCenter());
    else
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            gl::glVertexT(x[*it].getCenter());
    glEnd();
    glPointSize(1);
#endif /* SOFA_NO_OPENGL */
}
#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

