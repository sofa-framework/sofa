/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/component/visual/VisualBoundingBox.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::visual
{

void registerVisualBoundingBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Display an Axis Aligned Bounding Box (AABB).")
        .add< VisualBoundingBox >());
}

VisualBoundingBox::VisualBoundingBox()
    : d_color(initData(&d_color, sofa::type::RGBAColor::yellow(),  "color", "Color of the lines of the box."))
    , d_thickness(initData(&d_thickness, 1.0f,  "thickness", "Thickness of the lines of the box."))
{

}

void VisualBoundingBox::doDrawVisual(const core::visual::VisualParams* vparams)
{
    
    const auto& bbox = f_bbox.getValue();
    vparams->drawTool()->disableLighting();
    vparams->drawTool()->setMaterial(d_color.getValue());
    vparams->drawTool()->drawBoundingBox(bbox.minBBox(), bbox.maxBBox(), d_thickness.getValue());
}

} // namespace sofa::component::visual
