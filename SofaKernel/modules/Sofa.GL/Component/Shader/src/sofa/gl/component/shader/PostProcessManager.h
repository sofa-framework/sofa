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
#include <sofa/core/visual/VisualParams.h>
#include <sofa/gl/FrameBufferObject.h>
#include <sofa/gl/component/shader/OglShader.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa::gl::component::shader
{

class SOFA_GL_COMPONENT_SHADER_API PostProcessManager : public core::visual::VisualManager
{
public:
    SOFA_CLASS(PostProcessManager,core::visual::VisualModel);

private:
    static const std::string DEPTH_OF_FIELD_VERTEX_SHADER;
    static const std::string DEPTH_OF_FIELD_FRAGMENT_SHADER;
    Data<double> zNear; ///< Set zNear distance (for Depth Buffer)
    Data<double> zFar; ///< Set zFar distance (for Depth Buffer)
    std::unique_ptr<sofa::gl::FrameBufferObject> fbo;
    OglShader* dofShader;
    bool postProcessEnabled;

public:
    ///Files where vertex shader is defined
    sofa::core::objectmodel::DataFileName vertFilename;
    ///Files where fragment shader is defined
    sofa::core::objectmodel::DataFileName fragFilename;
protected:
    PostProcessManager();
    ~PostProcessManager() override;
public:
    void init() override;
    void reinit() override { };
    void initVisual() override;

    void preDrawScene(core::visual::VisualParams* vp) override;
    bool drawScene(core::visual::VisualParams* vp) override;
    void postDrawScene(core::visual::VisualParams* vp) override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;
};

} // namespace sofa::gl::component::shader
