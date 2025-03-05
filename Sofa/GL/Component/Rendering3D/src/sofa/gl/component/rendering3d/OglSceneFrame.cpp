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
#include <sofa/gl/component/rendering3d/OglSceneFrame.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gl/gl.h>

namespace sofa::gl::component::rendering3d
{

void registerOglSceneFrame(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Display a frame at the corner of the scene view.")
        .add< OglSceneFrame >());
}

using namespace sofa::defaulttype;

static constexpr OglSceneFrame::Alignment defaultAlignment("BottomRight");
static constexpr OglSceneFrame::Style defaultStyle("Cylinders");

OglSceneFrame::OglSceneFrame()
    : d_drawFrame(initData(&d_drawFrame, true,  "draw", "Display the frame or not"))
    , d_style(initData(&d_style, defaultStyle, "style", ("Style of the frame\n" + Style::dataDescription()).c_str()))
    , d_alignment(initData(&d_alignment, defaultAlignment, "alignment", ("Alignment of the frame in the view\n" + Alignment::dataDescription()).c_str()))
    , d_viewportSize(initData(&d_viewportSize, 150, "viewportSize", "Size of the viewport where the frame is rendered"))
{}

void OglSceneFrame::drawArrows(const core::visual::VisualParams* vparams)
{
    for (unsigned int i = 0; i < 3; ++i)
    {
        vparams->drawTool()->drawArrow(
             {}, {i == 0, i == 1, i == 2},
            0.05f,
            sofa::core::visual::DrawTool::RGBAColor(i == 0, i == 1, i == 2, 1.)
        );
    }
}

void OglSceneFrame::drawCylinders(const core::visual::VisualParams* vparams)
{
    for (unsigned int i = 0; i < 3; ++i)
    {
        vparams->drawTool()->drawCylinder(
             {}, {i == 0, i == 1, i == 2},
            0.05f,
            sofa::core::visual::DrawTool::RGBAColor(i == 0, i == 1, i == 2, 1.)
        );
    }
}

void OglSceneFrame::drawCubeCones(const core::visual::VisualParams* vparams)
{
    using sofa::type::Vec3;
    static constexpr SReal s = 0.25;
    static constexpr Vec3 p0 {-s, -s, -s};
    static constexpr Vec3 p1 {s, -s, -s};
    static constexpr Vec3 p2 {s, s, -s};
    static constexpr Vec3 p3 {-s, s, -s};
    static constexpr Vec3 p4 {-s, -s, s};
    static constexpr Vec3 p5 {s, -s, s};
    static constexpr Vec3 p6 {s, s, s};
    static constexpr Vec3 p7 {-s, s, s};

    vparams->drawTool()->drawHexahedron(p0, p1, p2, p3, p4, p5, p6, p7,
        sofa::core::visual::DrawTool::RGBAColor::darkgray());

    for (unsigned int i = 0; i < 3; ++i)
    {
        vparams->drawTool()->drawCone(
             s * Vec3{i == 0, i == 1, i == 2}, 3_sreal * s * Vec3{i == 0, i == 1, i == 2},
            0, s,
            sofa::core::visual::DrawTool::RGBAColor(i == 0, i == 1, i == 2, 1.)
        );
        vparams->drawTool()->drawCone(
             - s * Vec3{i == 0, i == 1, i == 2}, - 3_sreal * s * Vec3{i == 0, i == 1, i == 2},
            0, s,
            sofa::core::visual::DrawTool::RGBAColor::gray()
        );
    }
}

void OglSceneFrame::doDrawVisual(const core::visual::VisualParams* vparams)
{
    if (!d_drawFrame.getValue()) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const Viewport& viewport = vparams->viewport();

    const auto viewportSize = d_viewportSize.getValue();

    switch(d_alignment.getValue())
    {
        case Alignment("BottomLeft"):
        default:
            glViewport(0,0,viewportSize,viewportSize);
            glScissor(0,0,viewportSize,viewportSize);
            break;
        case Alignment("BottomRight"):
            glViewport(viewport[2]-viewportSize,0,viewportSize,viewportSize);
            glScissor(viewport[2]-viewportSize,0,viewportSize,viewportSize);
            break;
        case Alignment("TopRight"):
            glViewport(viewport[2]-viewportSize,viewport[3]-viewportSize,viewportSize,viewportSize);
            glScissor(viewport[2]-viewportSize,viewport[3]-viewportSize,viewportSize,viewportSize);
            break;
        case Alignment("TopLeft"):
            glViewport(0,viewport[3]-viewportSize,viewportSize,viewportSize);
            glScissor(0,viewport[3]-viewportSize,viewportSize,viewportSize);
            break;
    }
        
    glEnable(GL_SCISSOR_TEST);
    // only reset depth to appear on front
    glClear(GL_DEPTH_BUFFER_BIT );
    
    glMatrixMode(GL_PROJECTION);
    vparams->drawTool()->pushMatrix();
    glLoadIdentity();
    gluPerspective(60.0, 1.0, 0.5, 10.0);

    GLdouble matrix[16];
    vparams->getModelViewMatrix(matrix);

    matrix[12] = 0;
    matrix[13] = 0;
    matrix[14] = -3;
    matrix[15] = 1;

    glMatrixMode(GL_MODELVIEW);
    vparams->drawTool()->pushMatrix();
    glLoadMatrixd(matrix);

    vparams->drawTool()->disableLighting();

    switch (d_style.getValue())
    {
    case Style("Arrows"):
    default:
        drawArrows(vparams);
        break;

    case Style("Cylinders"):
        drawCylinders(vparams);
        break;

    case Style("CubeCones"):
        drawCubeCones(vparams);
        break;
    }

    glMatrixMode(GL_PROJECTION);
    vparams->drawTool()->popMatrix();
    glMatrixMode(GL_MODELVIEW);
    vparams->drawTool()->popMatrix();


    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);

}

} // namespace sofa::gl::component::rendering3d
