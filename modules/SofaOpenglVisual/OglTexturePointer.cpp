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
#include <SofaOpenglVisual/OglTexturePointer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{
namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OglTexturePointer)

// Register the OglTexturePointer class in the Factory
int OglTexturePointerClass = core::RegisterObject("OglTexturePointer").add< OglTexturePointer >();

OglTexturePointer::OglTexturePointer()
    :l_oglTexture( initLink( "oglTexture", "OglTexture" ) )
    ,textureUnit(initData(&textureUnit, (unsigned short) 1, "textureUnit", "Set the texture unit"))
    ,enabled(initData(&enabled, (bool) true, "enabled", "enabled ?"))
{
    
}

OglTexturePointer::~OglTexturePointer()
{

}

void OglTexturePointer::setActiveTexture(unsigned short unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
}

void OglTexturePointer::init()
{
    OglShaderElement::init();
}

void OglTexturePointer::initVisual()
{

}

void OglTexturePointer::reinit()
{

}

void OglTexturePointer::fwdDraw(core::visual::VisualParams*)
{
    if (enabled.getValue() && !l_oglTexture.empty())
    {
        setActiveTexture(textureUnit.getValue());
        bind();
        setActiveTexture(0);
    }
}

void OglTexturePointer::bwdDraw(core::visual::VisualParams*)
{
    if (enabled.getValue() && !l_oglTexture.empty())
    {
        setActiveTexture(textureUnit.getValue());
        unbind();
        setActiveTexture(0);
    }
}

void OglTexturePointer::bind()
{
    if(!l_oglTexture.empty())
        l_oglTexture->bind();
}

void OglTexturePointer::unbind()
{
    if(!l_oglTexture.empty())
        l_oglTexture->unbind();
}

}//end of namespaces
}
}
