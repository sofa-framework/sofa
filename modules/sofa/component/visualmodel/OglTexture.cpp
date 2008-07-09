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
#include <sofa/component/visualmodel/OglTexture.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{
namespace component
{

namespace visualmodel
{


SOFA_DECL_CLASS(OglTexture2D)

//Register OglTexture2D in the Object Factory
int OglTexture2DClass = core::RegisterObject("OglTexture2D")
        .add< OglTexture2D >()
        ;

unsigned short OglTexture::MAX_NUMBER_OF_TEXTURE_UNIT = 1;

OglTexture::OglTexture()
    :textureUnit(initData(&textureUnit, 1, "textureUnit", "Set the texture unit"))
    ,enabled(initData(&enabled, (bool) true, "enabled", "enabled ?"))
{

}

OglTexture::~OglTexture()
{

}

void OglTexture::setActiveTexture(unsigned short unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
}

void OglTexture::init()
{
    OglShaderElement::init();
}

void OglTexture::initVisual()
{
    GLint maxTextureUnits;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);
    MAX_NUMBER_OF_TEXTURE_UNIT = maxTextureUnits;

    if (textureUnit.getValue() > MAX_NUMBER_OF_TEXTURE_UNIT)
    {
        std::cerr << "Unit Texture too high ; set it at the unit texture n°1" << std::endl;
        textureUnit.setValue(1);
    }

    /*if (textureUnit.getValue() < 1)
    	{
    		std::cerr << "Unit Texture 0 not permitted ; set it at the unit texture n°1" << std::endl;
    		textureUnit.setValue(1);
    	}*/
}

void OglTexture::reinit()
{
    if (textureUnit.getValue() > MAX_NUMBER_OF_TEXTURE_UNIT)
    {
        std::cerr << "Unit Texture too high ; set it at the unit texture n°1" << std::endl;
        textureUnit.setValue(1);
    }

}
void OglTexture::fwdDraw(Pass)
{
    if (enabled.getValue())
    {
        setActiveTexture(textureUnit.getValue());
        bind();
        forwardDraw();
        setActiveTexture(0);
    }
}
void OglTexture::bwdDraw(Pass)
{
    if (enabled.getValue())
    {
        setActiveTexture(textureUnit.getValue());
        unbind();
        backwardDraw();
        ///TODO ?
        setActiveTexture(0);
    }
}

OglTexture2D::OglTexture2D()
    :texture2DFilename(initData(&texture2DFilename, (std::string) "", "texture2DFilename", "Texture2D Filename"))
    ,repeat(initData(&repeat, (bool) false, "repeat", "Repeat Texture ?"))
{

}

OglTexture2D::~OglTexture2D()
{
    if (!texture2D)delete texture2D;
}

void OglTexture2D::parse(core::objectmodel::BaseObjectDescription* arg)
{
    helper::system::FileRepository fp;
    if (arg->getAttribute("texture2DFilename"))
    {
        texture2DFilename.setValue( arg->getAttribute("texture2DFilename") );
    }
    img = helper::io::Image::Create(texture2DFilename.getValue());

    if (arg->getAttribute("id"))
    {
        id.setValue( arg->getAttribute("id") );
    }

    if (arg->getAttribute("textureUnit"))
    {
        textureUnit.setValue( atoi(arg->getAttribute("textureUnit") ) );
    }
}

void OglTexture2D::initVisual()
{
    OglTexture::initVisual();

    if (!img)
    {
        std::cerr << "OglTexture2D: Error : OglTexture2D file " << texture2DFilename.getValue() << " not found." << std::endl;
        return;
    }

    texture2D = new helper::gl::Texture(img);

    texture2D->init();
    setActiveTexture(textureUnit.getValue());

    bind();

    if (!repeat.getValue())
    {
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP );
    }
    else
    {
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT );
    }

    unbind();

    shader->setTexture(id.getValue().c_str(), textureUnit.getValue());

    setActiveTexture(0);
}

void OglTexture2D::bind()
{
    texture2D->bind();
    glEnable(GL_TEXTURE_2D);
}

void OglTexture2D::unbind()
{
    texture2D->unbind();
    glDisable(GL_TEXTURE_2D);
}

void OglTexture2D::forwardDraw()
{

}

void OglTexture2D::backwardDraw()
{

}


}

}

}
