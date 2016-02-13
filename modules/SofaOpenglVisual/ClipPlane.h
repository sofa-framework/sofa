/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CLIPPLANE_H
#define SOFA_COMPONENT_CLIPPLANE_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API ClipPlane : public core::visual::VisualModel
{
public:
    SOFA_CLASS(ClipPlane, core::visual::VisualModel);

    Data<sofa::defaulttype::Vector3> position;
    Data<sofa::defaulttype::Vector3> normal;
    Data<int> id;
    Data<bool> active;
protected:
    ClipPlane();
    virtual ~ClipPlane();
public:
    virtual void init();
    virtual void reinit();
    virtual void fwdDraw(core::visual::VisualParams*);
    virtual void bwdDraw(core::visual::VisualParams*);

protected:
    GLboolean wasActive;
    double saveEq[4];
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
