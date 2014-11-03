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
#include <SofaBaseVisual/VisualTransform.h>
#include <sofa/core/visual/VisualParams.h>
//#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/ObjectFactory.h>
//#include <sofa/simulation/common/UpdateContextVisitor.h>

#include <sofa/core/visual/DrawTool.h>

namespace sofa
{
namespace component
{
namespace visualmodel
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
    Coord xform = transform.getValue();
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

void VisualTransform::drawVisual(const sofa::core::visual::VisualParams* vparams)
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

}
}
}

