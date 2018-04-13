/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef OGLTEXTUREPOINTER_H_
#define OGLTEXTUREPOINTER_H_
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <SofaOpenglVisual/OglShader.h>
#include <SofaOpenglVisual/OglTexture.h>

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

class SOFA_OPENGL_VISUAL_API OglTexturePointer :  public core::visual::VisualModel, public OglShaderElement
{
public:
    SOFA_CLASS2(OglTexturePointer, core::visual::VisualModel, OglShaderElement);

protected:
    typedef SingleLink< OglTexturePointer, component::visualmodel::OglTexture, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkTexture;
    LinkTexture l_oglTexture;

    OglTexturePointer();
    virtual ~OglTexturePointer();

    Data<unsigned short> textureUnit; ///< Set the texture unit
    Data<bool> enabled; ///< enabled ?

public:
    void init() override;
    void initVisual() override;
    void reinit() override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;

    unsigned short getTextureUnit() { return textureUnit.getValue(); }

    void bind();
    void unbind();

    ///Utility function to set current active texture
    static void setActiveTexture(unsigned short unit);

    /// Returns the type of shader element (texture, macro, variable, or attribute)
    virtual ShaderElementType getSEType() const override { return core::visual::ShaderElement::SE_TEXTURE; }
    // Returns the value of the shader element
    virtual const core::objectmodel::BaseData* getSEValue() const override { return &textureUnit; }
    // Returns the value of the shader element
    virtual core::objectmodel::BaseData* getSEValue() override { return &textureUnit; }
};

}

}

}

#endif /*OGLTEXTUREPOINTER_H_*/
