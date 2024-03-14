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

#include <sofa/gl/component/rendering3d/OglModel.h>
#include <sofa/gl/component/shader/OglShader.h>
#include <sofa/gl/component/shader/OglAttribute.h>
#include <sofa/gl/component/shader/OglVariable.h>

namespace sofa::gl::component::shader
{


class SOFA_GL_COMPONENT_SHADER_API OglShaderVisualModel : public component::rendering3d::OglModel
{
public:
    using Inherit = component::rendering3d::OglModel;
    SOFA_CLASS(OglShaderVisualModel, Inherit);

protected:
    GLuint abo;
    OglShader* shader;
    int restPosition_lastUpdate;
public:
    // These attributes are public due to dynamic topologies updates.
    OglFloat3Attribute* vrestpositions;
    OglFloat3Attribute* vrestnormals;

    OglMatrix4Variable* modelMatrixUniform;
protected:
    OglShaderVisualModel();
    ~OglShaderVisualModel() override;
public:
    void init() override;
    void initVisual() override;

    void updateVisual() override;

    //void putRestPositions(const Vec3fTypes::VecCoord& positions);

    void bwdDraw(core::visual::VisualParams*) override;
    void fwdDraw(core::visual::VisualParams*) override;

    // handle topological changes
    void handleTopologyChange() override;
    void computeRestPositions();
    void computeRestNormals();

private:
    void pushTransformMatrix(float* matrix) override;
    void popTransformMatrix() override;


};

} // namespace sofa::gl::component::shader
