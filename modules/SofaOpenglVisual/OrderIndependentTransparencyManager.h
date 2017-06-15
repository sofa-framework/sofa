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
//
// C++ Interface: OrderIndependentTransparencyManager
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_ORDERINDEPENDENTTRANSPARENCYMANAGER_H
#define SOFA_COMPONENT_ORDERINDEPENDENTTRANSPARENCYMANAGER_H

#include "config.h"

#include <sofa/core/visual/VisualManager.h>
#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/helper/gl/FrameBufferObject.h>
#include <SofaOpenglVisual/OglOITShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Utility to manage transparency (translucency) into an Opengl scene
 *  \note Reference: http://jcgt.org/published/0002/02/09/paper.pdf
 */

class SOFA_OPENGL_VISUAL_API OrderIndependentTransparencyManager : public core::visual::VisualManager
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
    Data<float> depthScale;

protected:
    OrderIndependentTransparencyManager();
    virtual ~OrderIndependentTransparencyManager();

public:
    void init();
    void bwdInit();
    void reinit();
    void initVisual();

    void preDrawScene(core::visual::VisualParams* vp);
    bool drawScene(core::visual::VisualParams* vp);
    void postDrawScene(core::visual::VisualParams* vp);

    void draw(const core::visual::VisualParams* vparams);
    void fwdDraw(core::visual::VisualParams*);
    void bwdDraw(core::visual::VisualParams*);

protected:
    void drawOpaques(core::visual::VisualParams* vp);
    void drawTransparents(core::visual::VisualParams* vp, helper::gl::GLSLShader* oitShader);

private:
    FrameBufferObject            fbo;
    sofa::helper::gl::GLSLShader accumulationShader;
    sofa::helper::gl::GLSLShader compositionShader;

};

}// namespace visualmodel

}// namespace component

}// namespace sofa

#endif //SOFA_COMPONENT_ORDERINDEPENDENTTRANSPARENCYMANAGER_H
