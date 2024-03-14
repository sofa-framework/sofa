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
#include <sofa/gl/component/rendering3d/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/gl/template.h>

namespace sofa::gl::component::rendering3d
{

class SOFA_GL_COMPONENT_RENDERING3D_API ClipPlane : public core::visual::VisualModel
{
public:
    SOFA_CLASS(ClipPlane, core::visual::VisualModel);

    Data<sofa::type::Vec3> position; ///< Point crossed by the clipping plane
    Data<sofa::type::Vec3> normal; ///< Normal of the clipping plane, pointing toward the clipped region
    Data<int> id; ///< Clipping plane OpenGL ID
    Data<bool> active; ///< Control whether the clipping plane should be applied or not

    virtual sofa::core::objectmodel::ComponentState checkDataValues();
    void init() override;
    void reinit() override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;

protected:
    ClipPlane();
    ~ClipPlane() override;

    GLboolean wasActive;
    double saveEq[4];
};

} // namespace sofa::gl::component::rendering3d
