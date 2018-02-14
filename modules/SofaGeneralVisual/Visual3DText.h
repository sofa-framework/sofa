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
#ifndef SOFA_COMPONENT_VISUALMODEL_3DTEXT_H
#define SOFA_COMPONENT_VISUALMODEL_3DTEXT_H

#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/RGBAColor.h>
namespace sofa
{
namespace core
{
namespace topology
{
class BaseMeshTopology;
}
namespace behavior
{
class BaseMechanicalState;
}
}

namespace component
{

namespace visualmodel
{


/// Draw camera-oriented (billboard) 3D text
class SOFA_GENERAL_VISUAL_API Visual3DText : public core::visual::VisualModel
{

public:
    SOFA_CLASS(Visual3DText,core::visual::VisualModel);

protected:
    Visual3DText();

public:
    virtual void init() override;

    virtual void reinit() override;

    virtual void drawTransparent(const core::visual::VisualParams* vparams) override;

public:
    Data<std::string> d_text; ///< Test to display
    Data<defaulttype::Vec3f> d_position; ///< 3d position
    Data<float> d_scale; ///< text scale
    Data<defaulttype::RGBAColor> d_color; ///< text color. (default=[1.0,1.0,1.0,1.0])
    Data<bool> d_depthTest; ///< perform depth test


};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
