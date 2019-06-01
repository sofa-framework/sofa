/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef OGLTEXTURE_H_
#define OGLTEXTURE_H_
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <SofaOpenglVisual/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Defines an uniform sampler (texture) for a OglShader.
 *
 *  This is an abstract class which passes a texture id to an uniform
 *  sampler variable defined into the shader and load the image into OpenGL.
 *  At the moment, only texture2D is supported.
 */

class SOFA_OPENGL_VISUAL_API OglTexture :  public core::visual::VisualModel, public OglShaderElement
{
public:
    SOFA_CLASS2(OglTexture, core::visual::VisualModel, OglShaderElement);

protected:
    sofa::core::objectmodel::DataFileName d_textureFilename;
    Data<unsigned short> d_textureUnit; ///< Set the texture unit
    Data<bool> d_enabled; ///< enabled ?
    Data<bool> d_repeat; ///< Repeat Texture ?
    Data<bool> d_linearInterpolation; ///< Interpolate Texture ?
    Data<bool> d_generateMipmaps; ///< Generate mipmaps ?
    Data<bool> d_srgbColorspace; ///< SRGB colorspace ?
    Data<float> d_minLod; ///< Minimum mipmap lod ?
    Data<float> d_maxLod; ///< Maximum mipmap lod ?
    Data<unsigned int> d_proceduralTextureWidth; ///< Width of procedural Texture
    Data<unsigned int> d_proceduralTextureHeight; ///< Height of procedural Texture
    Data<unsigned int> d_proceduralTextureNbBits; ///< Nb bits per color
    Data<helper::vector<unsigned int> > d_proceduralTextureData; ///< Data of procedural Texture
    sofa::core::objectmodel::DataFileName d_cubemapFilenamePosX;
    sofa::core::objectmodel::DataFileName d_cubemapFilenamePosY;
    sofa::core::objectmodel::DataFileName d_cubemapFilenamePosZ;
    sofa::core::objectmodel::DataFileName d_cubemapFilenameNegX;
    sofa::core::objectmodel::DataFileName d_cubemapFilenameNegY;
    sofa::core::objectmodel::DataFileName d_cubemapFilenameNegZ;

    helper::gl::Texture* texture;
    helper::io::Image* img;

public:
    static GLint MAX_NUMBER_OF_TEXTURE_UNIT;
protected:
    OglTexture();
    ~OglTexture() override;
public:
    void init() override;
    void initVisual() override;
    void reinit() override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;

    std::string getTextureName();
    unsigned short getTextureUnit() { return d_textureUnit.getValue(); }

    void bind();
    void unbind();

    ///Utility function to set current active texture
    static void setActiveTexture(unsigned short unit);

    /// Returns the type of shader element (texture, macro, variable, or attribute)
    ShaderElementType getSEType() const override { return core::visual::ShaderElement::SE_TEXTURE; }
    // Returns the value of the shader element
    const core::objectmodel::BaseData* getSEValue() const override { return &d_textureFilename; }
    // Returns the value of the shader element
    core::objectmodel::BaseData* getSEValue() override { return &d_textureFilename; }
};

class SOFA_OPENGL_VISUAL_API OglTexture2D : public OglTexture
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

}

}

}

#endif /*OGLTEXTURE_H_*/
