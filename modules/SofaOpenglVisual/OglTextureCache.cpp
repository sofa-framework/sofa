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
#include <SofaOpenglVisual/OglTextureCache.h>

namespace sofa {

namespace component {

namespace visualmodel {

using namespace helper::gl;

OglTextureCache::OglTextureCache() :
    myTextures()
{

}

OglTextureCache* OglTextureCache::Instance()
{
    static OglTextureCache* instance = new OglTextureCache();

    return instance;
}

Texture* OglTextureCache::findTexture(const std::string& path) const
{
    auto iterator = myTextures.find(path);
    if(myTextures.end() != iterator)
        return iterator->second.texture;

    return nullptr;
}

Texture* OglTextureCache::addTexture(const std::string& path, Texture* texture)
{
    auto iterator = myTextures.find(path);
    if(myTextures.end() != iterator)
        iterator->second.count++;
    else
        myTextures[path] = CachedTexture({texture, 1});

    return texture;
}

Texture* OglTextureCache::removeTexture(const std::string& path)
{
    return removeTexture(myTextures.find(path));
}

Texture* OglTextureCache::removeTexture(Texture* texture)
{
    return removeTexture(std::find_if(myTextures.begin(), myTextures.end(), [=](const std::pair<std::string, CachedTexture>& element) { return element.second.texture == texture; }));
}

Texture* OglTextureCache::removeTexture(std::unordered_map<std::string, CachedTexture>::iterator iterator)
{
    Texture* texture = nullptr;

    if(myTextures.end() != iterator && --iterator->second.count == 0)
    {
        texture = iterator->second.texture;
        myTextures.erase(iterator);
    }

    return texture;
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
