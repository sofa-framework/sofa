/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/helper/gl/Capture.h>
#include <sofa/helper/system/SetDirectory.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <cstdio>		// sprintf and friends
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

namespace gl
{

Capture::Capture()
    : prefix("capture"), counter(-1)
{
}

bool Capture::saveScreen(const std::string& filename, int compression_level)
{
    std::string extension = sofa::helper::system::SetDirectory::GetExtension(filename.c_str());
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    //test if we can export in lossless image
    bool imageSupport = helper::io::Image::FactoryImage::getInstance()->hasKey(extension);
    if (!imageSupport)
    {
        msg_error("Capture") << "Could not write " << extension << "image format (no support found)";
        return false;
    }

    helper::io::Image* img = helper::io::Image::FactoryImage::getInstance()->createObject(extension, "");
    bool success = false;
    if (img)
    {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        img->init(viewport[2], viewport[3], 1, 1, io::Image::UNORM8, io::Image::RGB);
        glReadBuffer(GL_FRONT);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, img->getPixels());

        success = img->save(filename, compression_level);

        if (success)
        {
            msg_info("Capture") << "Saved " << img->getWidth() << "x" << img->getHeight() << " screen image to " << filename;
        }

        glReadBuffer(GL_BACK);
        delete img;
    }

    if(!success)
    {
        msg_error("Capture") << "Unknown error while saving screen image to " << filename;
    }

    return success;
}

std::string Capture::findFilename()
{
    bool pngSupport = helper::io::Image::FactoryImage::getInstance()->hasKey("png")
            || helper::io::Image::FactoryImage::getInstance()->hasKey("PNG");

    bool bmpSupport = helper::io::Image::FactoryImage::getInstance()->hasKey("bmp")
            || helper::io::Image::FactoryImage::getInstance()->hasKey("BMP");

    std::string filename = "";

    if(!pngSupport && !bmpSupport)
        return filename;

#ifndef PS3
    char buf[32];
    int c;
    c = 0;
    struct stat st;
    do
    {
        ++c;
        sprintf(buf, "%08d",c);
        filename = prefix;
        filename += buf;
        if(pngSupport)
            filename += ".png";
        else
            filename += ".bmp";
    }
    while (stat(filename.c_str(),&st)==0);
    counter = c+1;

    sprintf(buf, "%08d",c);
    filename = prefix;
    filename += buf;
    if(pngSupport)
        filename += ".png";
    else
        filename += ".bmp";
#endif

    return filename;
}


bool Capture::saveScreen(int compression_level)
{
    return saveScreen(findFilename(), compression_level);
}

} // namespace gl

} // namespace helper

} // namespace sofa

