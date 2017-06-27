/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef OGLTEXTURECACHE_H_
#define OGLTEXTURECACHE_H_

#include "config.h"

#include <sofa/helper/gl/Texture.h>

#include <unordered_map>

namespace sofa
{

namespace component
{

namespace visualmodel
{


/**
 *  \brief Manager class able to cache textures to avoid multiple reloading
 */

class SOFA_OPENGL_VISUAL_API OglTextureCache
{
protected:
    OglTextureCache();

public:
    static OglTextureCache* Instance();

    helper::gl::Texture* findTexture(const std::string& path) const;
    helper::gl::Texture* addTexture(const std::string& path, helper::gl::Texture* texture); /// \return the added texture
    helper::gl::Texture* removeTexture(const std::string& path); /// \return the texture if it is not used anymore or nullptr, it is the user responsability to free it
    helper::gl::Texture* removeTexture(helper::gl::Texture* texture); /// \return the texture if it is not used anymore or nullptr, it is the user responsability to free it

private:
    struct CachedTexture
    {
        helper::gl::Texture* texture;
        int count;
    };

    helper::gl::Texture* removeTexture(std::unordered_map<std::string, CachedTexture>::iterator iterator);

private:
    std::unordered_map<std::string, CachedTexture> myTextures;

};

}

}

}

#endif /*OGLTEXTURECACHE_H_*/
