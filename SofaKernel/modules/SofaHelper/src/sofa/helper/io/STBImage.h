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

#include <sofa/helper/config.h>

#include <sofa/helper/io/Image.h>
#include <string>
#include <vector>

namespace sofa::helper::io
{

class SOFA_HELPER_API STBImageCreators
{
    std::vector<sofa::helper::io::Image::FactoryImage::Creator*> creators;
public:
    static const std::vector<std::string> stbSupportedExtensions;
    static const std::vector<std::string> stbWriteSupportedExtensions;

    STBImageCreators();

};

class SOFA_HELPER_API STBImage : public Image
{
public:
    STBImage() = default;

    STBImage(const std::string &filename)
    {
        if(!filename.empty())
            load(filename);
    }

    static void setSTBCreators();

    bool load(std::string filename);
    bool save(std::string filename, int compression_level = -1);
};

} // namespace sofa::helper::io
