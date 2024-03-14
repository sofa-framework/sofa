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
#pragma once

#include <sofa/gl/component/shader/config.h>

#include <sofa/core/visual/VisualManager.h>
#include <sofa/gl/GLSLShader.h>
#include <sofa/gl/component/shader/OglOITShader.h>

namespace sofa::gl::component::shader
{

/**
 *  \brief Utility to manage transparency (translucency) into an Opengl scene
 *  \note Reference: http://jcgt.org/published/0002/02/09/paper.pdf
 */

class SOFA_GL_COMPONENT_SHADER_API OrderIndependentTransparencyManager : public core::visual::VisualManager
{
    class FrameBufferObject
    {
    public:
        FrameBufferObject();

        void init(int w, int h);
        void destroy();

        void copyDepth(GLuint fromFBO);

        void bind();

        void bindTextures();
        void releaseTextures();

    private:
        GLuint id;

        int width;
        int height;

        GLuint depthRenderbuffer;
        GLuint accumulationTexture;
        GLuint revealageTexture;

    };

public:
    SOFA_CLASS(OrderIndependentTransparencyManager, core::visual::VisualManager);

public:
    Data<float> depthScale; ///< Depth scale

protected:
    OrderIndependentTransparencyManager();
    ~OrderIndependentTransparencyManager() override;

public:
    void init() override;
    void bwdInit() override;
    void reinit() override;
    void initVisual() override;

    void preDrawScene(core::visual::VisualParams* vp) override;
    bool drawScene(core::visual::VisualParams* vp) override;
    void postDrawScene(core::visual::VisualParams* vp) override;

    void draw(const core::visual::VisualParams* vparams) override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;

protected:
    void drawOpaques(core::visual::VisualParams* vp);
    void drawTransparents(core::visual::VisualParams* vp, sofa::gl::GLSLShader* oitShader);

private:
    FrameBufferObject            fbo;
    sofa::gl::GLSLShader accumulationShader;
    sofa::gl::GLSLShader compositionShader;

};

}// namespace sofa::gl::component::shader
