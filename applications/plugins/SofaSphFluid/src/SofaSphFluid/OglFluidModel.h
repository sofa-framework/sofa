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

#include <SofaSphFluid/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/gl/FrameBufferObject.h>
#include <sofa/gl/GLSLShader.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/RGBAColor.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{
/**
 *  \brief Render volume using particles
 *
 */

// http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf

template<class DataTypes>
class SOFA_SPH_FLUID_API OglFluidModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglFluidModel, core::visual::VisualModel);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Real Real;

private:
    Data< VecCoord > m_positions; ///< Vertices coordinates
	VecCoord m_previousPositions;

    GLuint m_posVBO;
    sofa::gl::FrameBufferObject* m_spriteDepthFBO;
    sofa::gl::FrameBufferObject* m_spriteThicknessFBO;
    sofa::gl::FrameBufferObject* m_spriteNormalFBO;
    sofa::gl::FrameBufferObject* m_spriteBlurDepthHFBO;
    sofa::gl::FrameBufferObject* m_spriteBlurDepthVFBO;
    sofa::gl::FrameBufferObject* m_spriteBlurThicknessHFBO;
    sofa::gl::FrameBufferObject* m_spriteBlurThicknessVFBO;
    sofa::gl::FrameBufferObject* m_spriteShadeFBO;

    sofa::gl::GLSLShader m_spriteShader;
    sofa::gl::GLSLShader m_spriteNormalShader;
    sofa::gl::GLSLShader m_spriteBlurDepthShader;
    sofa::gl::GLSLShader m_spriteBlurThicknessShader;
    sofa::gl::GLSLShader m_spriteShadeShader;
    sofa::gl::GLSLShader m_spriteFinalPassShader;

    void drawSprites(const core::visual::VisualParams* vparams);
    void updateVertexBuffer();
protected:
    OglFluidModel();
    virtual ~OglFluidModel();
public:
    Data<unsigned int> d_debugFBO; ///< DEBUG FBO
    Data<float> d_spriteRadius; ///< Radius of sprites
    Data<float> d_spriteThickness; ///< Thickness of sprites
    Data<float> d_spriteBlurRadius; ///< Blur radius (in pixels)
    Data<float> d_spriteBlurScale; ///< Blur scale
    Data<float> d_spriteBlurDepthFalloff; ///< Blur Depth Falloff
    Data<sofa::type::RGBAColor> d_spriteDiffuseColor; ///< Diffuse Color


    void init() override;
    void initVisual() override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;
    void doDrawVisual(const core::visual::VisualParams* vparams) override;
    void drawTransparent(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams* params, bool onlyVisible = false) override;

    virtual void updateVisual() override;

};

}

}

}

