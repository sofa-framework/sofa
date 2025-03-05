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

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/gl/template.h>
#include <sofa/gl/Texture.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/gl/component/shader/OglShader.h>

namespace sofa::gl::component::shader
{

/**
 *  \brief Defines an uniform sampler (texture) for a OglShader.
 *
 *  This is an abstract class which passes a texture id to an uniform
 *  sampler variable defined into the shader and load the image into OpenGL.
 *  At the moment, only texture2D is supported.
 */

class SOFA_GL_COMPONENT_SHADER_API OglTexture :  public core::visual::VisualModel, public OglShaderElement
{
public:
    SOFA_CLASS2(OglTexture, core::visual::VisualModel, OglShaderElement);

protected:
    sofa::core::objectmodel::DataFileName textureFilename;
    Data<unsigned short> textureUnit; ///< Set the texture unit
    Data<bool> enabled; ///< enabled ?
    Data<bool> repeat; ///< Repeat Texture ?
    Data<bool> linearInterpolation; ///< Interpolate Texture ?
    Data<bool> generateMipmaps; ///< Generate mipmaps ?
    Data<bool> srgbColorspace; ///< SRGB colorspace ?
    Data<float> minLod; ///< Minimum mipmap lod ?
    Data<float> maxLod; ///< Maximum mipmap lod ?
    Data<unsigned int> proceduralTextureWidth; ///< Width of procedural Texture
    Data<unsigned int> proceduralTextureHeight; ///< Height of procedural Texture
    Data<unsigned int> proceduralTextureNbBits; ///< Nb bits per color
    Data<type::vector<unsigned int> > proceduralTextureData; ///< Data of procedural Texture 
    sofa::core::objectmodel::DataFileName cubemapFilenamePosX;
    sofa::core::objectmodel::DataFileName cubemapFilenamePosY;
    sofa::core::objectmodel::DataFileName cubemapFilenamePosZ;
    sofa::core::objectmodel::DataFileName cubemapFilenameNegX;
    sofa::core::objectmodel::DataFileName cubemapFilenameNegY;
    sofa::core::objectmodel::DataFileName cubemapFilenameNegZ;

    sofa::gl::Texture* texture;
    helper::io::Image* img;

public:
    static GLint MAX_NUMBER_OF_TEXTURE_UNIT;
protected:
    OglTexture();
    ~OglTexture() override;
public:
    void init() override;
    void doInitVisual(const core::visual::VisualParams* vparams) override;
    void reinit() override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;

    std::string getTextureName();
    unsigned short getTextureUnit() { return textureUnit.getValue(); }

    void bind();
    void unbind();

    ///Utility function to set current active texture
    static void setActiveTexture(unsigned short unit);

    /// Returns the type of shader element (texture, macro, variable, or attribute)
    ShaderElementType getSEType() const override { return core::visual::ShaderElement::SE_TEXTURE; }
    // Returns the value of the shader element
    const core::objectmodel::BaseData* getSEValue() const override { return &textureFilename; }
    // Returns the value of the shader element
    core::objectmodel::BaseData* getSEValue() override { return &textureFilename; }
};

class SOFA_GL_COMPONENT_SHADER_API OglTexture2D : public OglTexture
{
public:
    SOFA_CLASS(OglTexture2D, OglTexture);

private:
    sofa::core::objectmodel::DataFileName texture2DFilename;

public:
    OglTexture2D();
    ~OglTexture2D() override;

    void init() override;
};

} // namespace sofa::gl::component::shader
