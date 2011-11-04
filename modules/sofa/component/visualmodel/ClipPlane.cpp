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
#include <sofa/component/visualmodel/ClipPlane.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

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
    : position(initData(&position, Vector3(0,0,0), "position", "Point crossed by the clipping plane"))
    , normal(initData(&normal, Vector3(1,0,0), "normal", "Normal of the clipping plane, pointing toward the clipped region"))
    , id(initData(&id, 0, "id", "Clipping plane OpenGL ID"))
    , active(initData(&active,true,"active","Control whether the clipping plane should be applied or not"))
{

}

ClipPlane::~ClipPlane()
{
}

void ClipPlane::init()
{
}

void ClipPlane::reinit()
{
}

void ClipPlane::fwdDraw(core::visual::VisualParams*)
{
    wasActive = glIsEnabled(GL_CLIP_PLANE0+id.getValue());
    if (active.getValue())
    {
        glGetClipPlane(GL_CLIP_PLANE0+id.getValue(), saveEq);
        Vector3 p = position.getValue();
        Vector3 n = normal.getValue();
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
}

void ClipPlane::bwdDraw(core::visual::VisualParams*)
{
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
}

} // namespace visualmodel

} //namespace component

} //namespace sofa
