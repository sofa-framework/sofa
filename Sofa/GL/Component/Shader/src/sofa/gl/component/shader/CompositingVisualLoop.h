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

#include <sofa/simulation/DefaultVisualManagerLoop.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/gl/component/shader/OglShader.h>
#include <sofa/gl/component/shader/VisualManagerPass.h>

#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Event.h>

namespace sofa::gl::component::shader
{

/**
 *  \Compositing visual loop: render multiple passes and composite them into one single rendered frame
 */

class SOFA_GL_COMPONENT_SHADER_API CompositingVisualLoop : public simulation::DefaultVisualManagerLoop
{
public:
    SOFA_CLASS(CompositingVisualLoop,simulation::DefaultVisualManagerLoop);

    ///Files where vertex shader is defined
    sofa::core::objectmodel::DataFileName vertFilename;
    ///Files where fragment shader is defined
    sofa::core::objectmodel::DataFileName fragFilename;

private:

    void traceFullScreenQuad();
    void defaultRendering(sofa::core::visual::VisualParams* vparams);

protected:
    CompositingVisualLoop();
    ~CompositingVisualLoop() override;

public:

    void init() override;
    void initVisual() override;
    void drawStep(sofa::core::visual::VisualParams* vparams) override;
};

} // namespace sofa::gl::component::shader
