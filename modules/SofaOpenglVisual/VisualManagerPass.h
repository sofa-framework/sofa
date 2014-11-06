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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * VisualManagerPass.h
 *
 *  Created on: 16 janv. 2012
 *      Author: Jeremy Ringard
 */

#ifndef SOFA_COMPONENT_VISUALMANAGERPASS_H
#define SOFA_COMPONENT_VISUALMANAGERPASS_H

#include <sofa/SofaGeneral.h>
#include <SofaOpenglVisual/CompositingVisualLoop.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/FrameBufferObject.h>
#include <SofaOpenglVisual/OglShader.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Render pass element: render the relevant tagged objects in a FBO
 */

class SOFA_OPENGL_VISUAL_API VisualManagerPass : public core::visual::VisualManager
{
public:
    SOFA_CLASS(VisualManagerPass, core::visual::VisualManager);

    Data<float> factor;
    Data<bool> renderToScreen;
    Data<std::string> outputName;
protected:
    bool checkMultipass(sofa::core::objectmodel::BaseContext* con);
    bool multiPassEnabled;

    helper::gl::FrameBufferObject* fbo;
    bool prerendered;

    GLint passWidth;
    GLint passHeight;
public:
    VisualManagerPass();
    virtual ~VisualManagerPass();


    virtual void init();
    virtual void initVisual();

    virtual void preDrawScene(core::visual::VisualParams* vp);
    virtual bool drawScene(core::visual::VisualParams* vp);
    virtual void postDrawScene(core::visual::VisualParams* vp);


    virtual void draw(const core::visual::VisualParams* vparams);
    virtual void fwdDraw(core::visual::VisualParams*);
    virtual void bwdDraw(core::visual::VisualParams*);

    virtual void handleEvent(sofa::core::objectmodel::Event* /*event*/);

    virtual bool isPrerendered() {return prerendered;};

    virtual helper::gl::FrameBufferObject* getFBO() {return fbo;};
    bool hasFilledFbo();
    std::string getOutputName();
};

}//namespace visualmodel

}//namespace component

}//namespace sofa

#endif //SOFA_COMPONENT_LIGHT_MANAGER_H
