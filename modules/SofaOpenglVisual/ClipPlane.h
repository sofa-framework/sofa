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

    virtual sofa::core::objectmodel::ComponentState checkDataValues();
    virtual void init() override;
    virtual void reinit() override;
    virtual void fwdDraw(core::visual::VisualParams*) override;
    virtual void bwdDraw(core::visual::VisualParams*) override;

protected:
    ClipPlane();
    virtual ~ClipPlane();

    GLboolean wasActive;
    double saveEq[4];
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
