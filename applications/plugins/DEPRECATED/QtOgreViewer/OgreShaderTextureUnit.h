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
#ifndef OGRESHADERTEXTUREUNIT_H
#define OGRESHADERTEXTUREUNIT_H

#include <sofa/core/objectmodel/DataFileName.h>
#include "OgreShaderEntryPoint.h"

namespace sofa
{

namespace component
{

namespace visualmodel
{

class OgreShaderTextureUnit : public OgreShaderEntryPoint
{
public:
    SOFA_CLASS(OgreShaderTextureUnit, OgreShaderEntryPoint);

    OgreShaderTextureUnit():
        textureIndex(initData(&textureIndex, 0, "textureIndex", "Index of the texture in the pass"))
        , textureName(initData(&textureName,"textureName", "File to use for the Texture"))
    {
    };
    virtual ~OgreShaderTextureUnit() {};

    void setTextureIndex(int entry) {textureIndex.setValue(entry);}
    int getTextureIndex() const {return textureIndex.getValue();}

    void setTextureName(const std::string &filename) {textureName.setValue(filename);}
    const std::string &getTextureName() const {return textureName.getValue();}

protected:
    Data<int> textureIndex; ///< Index of the texture in the pass
    core::objectmodel::DataFileName textureName;

};
}
}
}

#endif

