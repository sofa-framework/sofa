/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "OglShaderMacro.h"
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OglShaderDefineMacro)

//Register OglIntVariable in the Object Factory
int OglShaderDefineMacroClass = core::RegisterObject("OglShaderDefineMacro")
        .add< OglShaderDefineMacro >();

OglShaderMacro::OglShaderMacro()
{

}

OglShaderMacro::~OglShaderMacro()
{
}

void OglShaderMacro::init()
{
    OglShaderElement::init();
}

OglShaderDefineMacro::OglShaderDefineMacro()
    : value(initData(&value, (std::string) "", "value", "Set a value for define macro"))
{

}

OglShaderDefineMacro::~OglShaderDefineMacro()
{
}

void OglShaderDefineMacro::init()
{
    OglShaderMacro::init();

    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->addDefineMacro(indexShader.getValue(), id.getValue(), value.getValue());
}

}

}

}
