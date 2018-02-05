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
#define SOFA_COMPONENT_CONSTRAINTSET_FIXEDLMCONSTRAINT_CPP
#include <SofaConstraint/FixedLMConstraint.inl>
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

SOFA_DECL_CLASS(FixedLMConstraint)

int FixedLMConstraintClass = core::RegisterObject("Maintain a set of particle to a fixed position using LMConstraint")
#ifndef SOFA_FLOAT
        .add< FixedLMConstraint<Vec3dTypes> >()
        .add< FixedLMConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FixedLMConstraint<Vec3fTypes> >()
        .add< FixedLMConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_CONSTRAINT_API FixedLMConstraint<Vec3dTypes>;
template class SOFA_CONSTRAINT_API FixedLMConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API FixedLMConstraint<Vec3fTypes>;
template class SOFA_CONSTRAINT_API FixedLMConstraint<Rigid3fTypes>;
#endif


#ifndef SOFA_FLOAT
template <>
void FixedLMConstraint<Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =constrainedObject1->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
        gl::glVertexT(x[*it].getCenter());
    glEnd();
#endif /* SOFA_NO_OPENGL */
}

#endif

#ifndef SOFA_DOUBLE
template <>
void FixedLMConstraint<Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =constrainedObject1->read(core::ConstVecCoordId::position())->getValue();

    std::vector< Vector3 > points;
    for (SetIndexArray::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        points.push_back(x[*it].getCenter());
    }

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else
    {
        vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
    }
}

#endif



} // namespace constraintset

} // namespace component

} // namespace sofa

