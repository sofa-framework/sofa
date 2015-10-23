/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_CPP
#include <SofaBoundaryCondition/FixedConstraint.inl>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


SOFA_DECL_CLASS(FixedConstraint)

int FixedConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
#ifndef SOFA_FLOAT
        .add< FixedConstraint<Vec3dTypes> >()
        .add< FixedConstraint<Vec2dTypes> >()
        .add< FixedConstraint<Vec1dTypes> >()
        .add< FixedConstraint<Vec6dTypes> >()
        .add< FixedConstraint<Rigid3dTypes> >()
        .add< FixedConstraint<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FixedConstraint<Vec3fTypes> >()
        .add< FixedConstraint<Vec2fTypes> >()
        .add< FixedConstraint<Vec1fTypes> >()
        .add< FixedConstraint<Vec6fTypes> >()
        .add< FixedConstraint<Rigid3fTypes> >()
        .add< FixedConstraint<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Rigid3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Rigid3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Rigid2fTypes>;
#endif

#ifndef SOFA_FLOAT
template <>
void FixedConstraint<Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    std::vector< Vector3 > points;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    if( f_fixAll.getValue()==true )
        for (unsigned i=0; i<x.size(); i++ )
            points.push_back(x[i].getCenter());
    else
    {
        if( x.size() < indices.size() )
        {
            for (unsigned i=0; i<x.size(); i++ )
                points.push_back(x[indices[i]].getCenter());
        }
        else
        {
            for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
                points.push_back(x[*it].getCenter());
        }
    }

    if( f_drawSize.getValue() == 0) // old classical drawing by points
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    else
        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
}

template <>
void FixedConstraint<Rigid2dTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    if( f_fixAll.getValue()==true )
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
void FixedConstraint<Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    std::vector< Vector3 > points;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    if( f_fixAll.getValue()==true )
        for (unsigned i=0; i<x.size(); i++ )
            points.push_back(x[i].getCenter());
    else
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            points.push_back(x[*it].getCenter());

    if( f_drawSize.getValue() == 0) // old classical drawing by points
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    else
        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
}

template <>
void FixedConstraint<Rigid2fTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    if( f_fixAll.getValue()==true )
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

