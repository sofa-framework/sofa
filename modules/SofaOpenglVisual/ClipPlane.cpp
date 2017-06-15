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
#include <SofaOpenglVisual/ClipPlane.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <cmath>
using sofa::core::objectmodel::ComponentState ;

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(ClipPlane)

int ClipPlaneClass = core::RegisterObject("OpenGL Clipping Plane")
        .add< ClipPlane >()
        ;


ClipPlane::ClipPlane()
    : position(initData(&position, sofa::defaulttype::Vector3(0,0,0), "position", "Point crossed by the clipping plane"))
    , normal(initData(&normal, sofa::defaulttype::Vector3(1,0,0), "normal", "Normal of the clipping plane, pointing toward the clipped region"))
    , id(initData(&id, 0, "id", "Clipping plane OpenGL ID"))
    , active(initData(&active,true,"active","Control whether the clipping plane should be applied or not"))
{
}

ClipPlane::~ClipPlane()
{
}

sofa::core::objectmodel::ComponentState ClipPlane::checkDataValues()
{
    if(id.getValue() < 0)
    {
        msg_error() << "plane ID cannot be negative. The component is disabled." ;
        return ComponentState::Invalid;
    }
    return ComponentState::Valid;
}

void ClipPlane::init()
{
    m_componentstate = checkDataValues() ;
}

void ClipPlane::reinit()
{
    if(m_componentstate==ComponentState::Invalid)
        msg_error() << "Reiniting an invalid component is not allowed. It must be inited first" ;
}

void ClipPlane::fwdDraw(core::visual::VisualParams*)
{
    if(m_componentstate==ComponentState::Invalid)
        return ;

#ifndef PS3
    wasActive = glIsEnabled(GL_CLIP_PLANE0+id.getValue());
    if (active.getValue())
    {
        glGetClipPlane(GL_CLIP_PLANE0+id.getValue(), saveEq);
        sofa::defaulttype::Vector3 p = position.getValue();
        sofa::defaulttype::Vector3 n = normal.getValue();
        GLdouble c[4] = { (GLdouble) -n[0], (GLdouble)-n[1], (GLdouble)-n[2], (GLdouble)(p*n) };
        glClipPlane(GL_CLIP_PLANE0+id.getValue(), c);
        if (!wasActive)
            glEnable(GL_CLIP_PLANE0+id.getValue());
    }
    else
    {
        if (wasActive)
            glDisable(GL_CLIP_PLANE0+id.getValue());
    }
#else
    ///\todo save clip plane in this class and restore it because PS3 SDK doesn't have glGetClipPlane
    if (active.getValue())
    {
        sofa::defaulttype::Vector3 p = position.getValue();
        sofa::defaulttype::Vector3 n = normal.getValue();
        GLdouble c[4] = { (GLdouble) -n[0], (GLdouble)-n[1], (GLdouble)-n[2], (GLdouble)(p*n) };
        glClipPlanef(GL_CLIP_PLANE0+id.getValue(), c);
        glEnable(GL_CLIP_PLANE0+id.getValue());
    }
    else
    {
        glDisable(GL_CLIP_PLANE0+id.getValue());
    }
#endif
}

void ClipPlane::bwdDraw(core::visual::VisualParams*)
{
#ifndef PS3
    if (active.getValue())
    {
        glClipPlane(GL_CLIP_PLANE0+id.getValue(), saveEq);
        if (!wasActive)
            glDisable(GL_CLIP_PLANE0+id.getValue());
    }
    else
    {
        if (wasActive)
            glEnable(GL_CLIP_PLANE0+id.getValue());
    }
#else
    if(active.getValue())
    {
        glDisable(GL_CLIP_PLANE0+id.getValue());
    }
    else
    {
        glEnable(GL_CLIP_PLANE0+id.getValue());
    }
#endif
}

} // namespace visualmodel

} //namespace component

} //namespace sofa
