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
#include <sofa/component/visual/VisualTransform.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::visual
{

int VisualTransformClass = sofa::core::RegisterObject("TODO")
        .add<VisualTransform>();

VisualTransform::VisualTransform()
    : transform(initData(&transform,"transform","Transformation to apply"))
    , recursive(initData(&recursive,false,"recursive","True to apply transform to all nodes below"))
    , nbpush(0)
{
}

VisualTransform::~VisualTransform()
{
}

void VisualTransform::push(const sofa::core::visual::VisualParams* vparams)
{
    const Coord xform = transform.getValue();
    vparams->drawTool()->pushMatrix();
    ++nbpush;
    float glTransform[16];
    xform.writeOpenGlMatrix ( glTransform );
    vparams->drawTool()->multMatrix( glTransform );

}

void VisualTransform::pop(const sofa::core::visual::VisualParams* vparams)
{
    if (nbpush > 0)
    {
        vparams->drawTool()->popMatrix();
        --nbpush;
    }
}

void VisualTransform::fwdDraw(sofa::core::visual::VisualParams* vparams)
{
    push(vparams);
}

void VisualTransform::draw(const sofa::core::visual::VisualParams* /*vparams*/)
{
    //pop(vparams);
}

void VisualTransform::doDrawVisual(const sofa::core::visual::VisualParams* vparams)
{
    if (!recursive.getValue())
        pop(vparams);
}

void VisualTransform::drawTransparent(const sofa::core::visual::VisualParams* vparams)
{
    if (!recursive.getValue())
        pop(vparams);
}

void VisualTransform::bwdDraw(sofa::core::visual::VisualParams* vparams)
{
    pop(vparams);
}

} // namespace sofa::component::visual
