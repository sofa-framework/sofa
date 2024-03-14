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
#include <array>

namespace sofa::helper::io
{

class SOFA_HELPER_API STBImageCreators
{
    std::vector<std::shared_ptr<sofa::helper::io::Image::FactoryImage::Creator> > creators;
public:
    inline static constexpr std::array<const char*, 8> stbSupportedExtensions
    {
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tga",
        "gif",
        "psd",
        "pnm"
    };

    inline static constexpr std::array<const char*, 5> stbWriteSupportedExtensions
    {
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tga"
    };

    STBImageCreators();

};

class SOFA_HELPER_API STBImage : public Image
{
public:
    STBImage() = default;

    explicit STBImage(const std::string &filename)
    {
        if (!filename.empty())
        {
            load(filename);
        }
    }

    static void setSTBCreators();

    bool load(std::string filename) override;
    bool save(std::string filename, int compression_level = -1) override;
};

} // namespace sofa::helper::io
