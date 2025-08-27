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
#include <sofa/gl/component/rendering2d/config.h>

#include <sofa/core/visual/VisualManager.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/BoundingBox.h>

#include <sofa/gl/FrameBufferObject.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::gl::component::rendering2d
{

class OglViewport : public core::visual::VisualManager
{
public:
    typedef defaulttype::RigidCoord<3,double> RigidCoord;
    typedef core::visual::VisualParams::Viewport Viewport;

    SOFA_CLASS(OglViewport, core::visual::VisualManager);

    Data<type::Vec<2, int> > p_screenPosition; ///< Viewport position
    Data<type::Vec<2, unsigned int> > p_screenSize; ///< Viewport size
    Data<type::Vec3> p_cameraPosition; ///< Camera's position in eye's space
    Data<type::Quat<SReal>> p_cameraOrientation; ///< Camera's orientation
    Data<RigidCoord > p_cameraRigid; ///< Camera's rigid coord
    Data<double> p_zNear; ///< Camera's ZNear
    Data<double> p_zFar; ///< Camera's ZFar
    Data<double> p_fovy; ///< Field of View (Y axis)
    Data<bool> p_enabled; ///< Enable visibility of the viewport
    Data<bool> p_advancedRendering; ///< If true, viewport will be hidden if advancedRendering visual flag is not enabled
    Data<bool> p_useFBO; ///< Use a FBO to render the viewport
    Data<bool> p_swapMainView; ///< Swap this viewport with the main view
    Data<bool> p_drawCamera; ///< Draw a frame representing the camera (see it in main viewport)

    std::unique_ptr<sofa::gl::FrameBufferObject> fbo;

protected:
    OglViewport();
    ~OglViewport() override;
public:
    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void doInitVisual(const core::visual::VisualParams*) override;
    void preDrawScene(core::visual::VisualParams* vp) override;
    bool drawScene(core::visual::VisualParams* vp) override;
    void postDrawScene(core::visual::VisualParams* vp) override;

    bool isVisible(const core::visual::VisualParams* vparams);

protected:
    void renderToViewport(core::visual::VisualParams* vp);
    void renderFBOToScreen(core::visual::VisualParams* vp);

};

} // namespace sofa::gl::component::rendering2d
