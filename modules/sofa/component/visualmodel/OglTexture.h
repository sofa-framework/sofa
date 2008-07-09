/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef OGLTEXTURE_H_
#define OGLTEXTURE_H_

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/GLshader.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/component/visualmodel/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class OglTexture :  public core::VisualModel, public OglShaderElement
{

protected:
    Data<int> textureUnit;
    Data<bool> enabled;

public:
    static unsigned short MAX_NUMBER_OF_TEXTURE_UNIT;

    OglTexture();
    virtual ~OglTexture();

    virtual void init();
    virtual void initVisual();
    virtual void reinit();
    void fwdDraw(Pass);
    void bwdDraw(Pass);

    std::string getTextureName();
    void getTextureUnit();

    virtual void bind() = 0;
    virtual void unbind() = 0;

    static void setActiveTexture(unsigned short unit);

    //virtual void setInShader(OglShader& s) = 0;

protected:
    virtual void forwardDraw() = 0;
    virtual void backwardDraw() = 0;
};

class OglTexture2D : public OglTexture
{
private:
    Data<std::string> texture2DFilename;
    Data<bool> repeat;
    helper::gl::Texture* texture2D;

    helper::io::Image* img;

public:
    OglTexture2D();
    virtual ~OglTexture2D();

    void parse(core::objectmodel::BaseObjectDescription* arg);
    void initVisual();
    void reinit() { };

    void bind();
    void unbind();


protected:
    void forwardDraw();
    void backwardDraw();
};

}

}

}

#endif /*OGLTEXTURE_H_*/
