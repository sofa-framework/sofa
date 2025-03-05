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
#include <sofa/helper/io/STBImage.h>

#include <sofa/helper/Factory.inl>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileRepository.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_writer.h>

namespace sofa::helper::io
{

STBImageCreators::STBImageCreators()
{
    using ImageFactoryCreator = Creator<helper::io::Image::FactoryImage, STBImage>;
    for(const auto& ext : stbSupportedExtensions)
    {
        if (!sofa::helper::io::Image::FactoryImage::HasKey(ext))
        {
            creators.push_back(std::make_shared<ImageFactoryCreator>(ext));
        }
    }

}

void STBImage::setSTBCreators()
{
    static STBImageCreators stbCreators;
}

bool STBImage::load(std::string filename)
{
    m_bLoaded = 0;

    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error() << "File " << filename << " not found ";
        return false;
    }

    int width, height;
    int comp;
    stbi_set_flip_vertically_on_load((int)true);
    unsigned char* image = stbi_load(filename.c_str(), &width, &height, &comp, STBI_default);

    if (image == nullptr)
    {
        msg_error() << "Could not load: " << filename.c_str();
        return false;
    }

    Image::ChannelFormat format;
    switch (comp)
    {
    case 1:
        format = Image::L;
        break;
    case 2:
        format = Image::LA;
        break;
    case 3:
        format = Image::RGB;
        break;
    case 4:
        format = Image::RGBA;
        break;
    default:
        msg_error() << "in " << filename << ", unsupported number of channels: " << comp;
        return false;
    }

    init(width, height, 1, 1, UNORM8, format);

    unsigned char* data = getPixels();
    const unsigned int totalSize = width * height;
    std::memcpy(data, image, totalSize * comp);
    stbi_image_free(image);

    m_bLoaded = 1;
    return true;
}

bool STBImage::save(std::string filename, int compression_level )
{
    bool res = false;
    std::string ext;
    //check extension
    for(const auto& currExt : STBImageCreators::stbWriteSupportedExtensions)
    {
        if(filename.substr(filename.find_last_of(".") + 1) == currExt)
            ext = currExt;
    }

    if(ext.empty())
    {
        msg_warning() << "Cannot recognize extension or file format not supported,"
                                << "image will be saved as a PNG file.";
        ext = "png";
    }

    stbi_flip_vertically_on_write((int)true);
    if (ext == "png" )
    {
        res = stbi_write_png(filename.c_str(), getWidth(), getHeight(), getChannelCount(), getPixels(), getWidth() * getChannelCount());
    }
    if (ext == "jpg" || ext == "jpeg")
    {
        res = stbi_write_jpg(filename.c_str(), getWidth(), getHeight(), getChannelCount(), getPixels(), compression_level);
    }
    if (ext == "bmp")
    {
        res = stbi_write_bmp(filename.c_str(), getWidth(), getHeight(), getChannelCount(), getPixels());
    }
    if (ext == "tga")
    {
        res = stbi_write_tga(filename.c_str(), getWidth(), getHeight(), getChannelCount(), getPixels());
    }

    return res;
}

// call the list of image creators using stb
const auto stbImageCreation = STBImageCreators();

} // namespace sofa::helper::io
