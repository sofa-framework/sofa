/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaOpenglVisual/OglTexture.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{
namespace component
{

namespace visualmodel
{

// Register the OglTexture* objects in the Object Factory
int OglTextureClass = core::RegisterObject("OglTexture").add< OglTexture >();
int OglTexture2DClass = core::RegisterObject("OglTexture2D").add< OglTexture2D >();

GLint OglTexture::MAX_NUMBER_OF_TEXTURE_UNIT = 1;

OglTexture::OglTexture()
    :d_textureFilename(initData(&d_textureFilename, (std::string) "", "filename", "Texture Filename"))
    ,d_textureUnit(initData(&d_textureUnit, (unsigned short) 1, "textureUnit", "Set the texture unit"))
    ,d_enabled(initData(&d_enabled, (bool) true, "enabled", "enabled ?"))
    ,d_repeat(initData(&d_repeat, (bool) false, "repeat", "Repeat Texture ?"))
    ,d_linearInterpolation(initData(&d_linearInterpolation, (bool) true, "linearInterpolation", "Interpolate Texture ?"))
    ,d_generateMipmaps(initData(&d_generateMipmaps, (bool) true, "generateMipmaps", "Generate mipmaps ?"))
    ,d_srgbColorspace(initData(&d_srgbColorspace, (bool) false, "srgbColorspace", "SRGB colorspace ?"))
    ,d_minLod(initData(&d_minLod, (float) -1000, "minLod", "Minimum mipmap lod ?"))
    ,d_maxLod(initData(&d_maxLod, (float)  1000, "maxLod", "Maximum mipmap lod ?"))
    ,d_proceduralTextureWidth(initData(&d_proceduralTextureWidth, (unsigned int) 0, "proceduralTextureWidth", "Width of procedural Texture"))
    ,d_proceduralTextureHeight(initData(&d_proceduralTextureHeight, (unsigned int) 0, "proceduralTextureHeight", "Height of procedural Texture"))
    ,d_proceduralTextureNbBits(initData(&d_proceduralTextureNbBits, (unsigned int) 1, "proceduralTextureNbBits", "Nb bits per color"))
    ,d_proceduralTextureData(initData(&d_proceduralTextureData,  "proceduralTextureData", "Data of procedural Texture "))
    ,d_cubemapFilenamePosX(initData(&d_cubemapFilenamePosX, (std::string) "", "cubemapFilenamePosX", "Texture filename of positive-X cubemap face"))
    ,d_cubemapFilenamePosY(initData(&d_cubemapFilenamePosY, (std::string) "", "cubemapFilenamePosY", "Texture filename of positive-Y cubemap face"))
    ,d_cubemapFilenamePosZ(initData(&d_cubemapFilenamePosZ, (std::string) "", "cubemapFilenamePosZ", "Texture filename of positive-Z cubemap face"))
    ,d_cubemapFilenameNegX(initData(&d_cubemapFilenameNegX, (std::string) "", "cubemapFilenameNegX", "Texture filename of negative-X cubemap face"))
    ,d_cubemapFilenameNegY(initData(&d_cubemapFilenameNegY, (std::string) "", "cubemapFilenameNegY", "Texture filename of negative-Y cubemap face"))
    ,d_cubemapFilenameNegZ(initData(&d_cubemapFilenameNegZ, (std::string) "", "cubemapFilenameNegZ", "Texture filename of negative-Z cubemap face"))
    ,texture(nullptr)
    ,img(nullptr)
{
    this->addAlias(&d_textureFilename, "textureFilename");
}

OglTexture::~OglTexture()
{
    if (texture) delete texture;
//    if (img) delete img; // should be deleted by the texture (but what happens if the texture is never created ?) Why not use smart pointers for that kind of stuff?
}

void OglTexture::setActiveTexture(unsigned short unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
}

void OglTexture::init()
{
    if (d_textureFilename.getFullPath().empty())
    {
        if (d_cubemapFilenamePosX.getFullPath().empty() &&
            d_cubemapFilenamePosY.getFullPath().empty() &&
            d_cubemapFilenamePosZ.getFullPath().empty() &&
            d_cubemapFilenameNegX.getFullPath().empty() &&
            d_cubemapFilenameNegY.getFullPath().empty() &&
            d_cubemapFilenameNegZ.getFullPath().empty())
        {
            // "Procedural" texture (actually inline texture data inside the scene file).
            unsigned int height = d_proceduralTextureHeight.getValue();
            unsigned int width = d_proceduralTextureWidth.getValue();
            helper::vector<unsigned int> textureData = d_proceduralTextureData.getValue();
            unsigned int nbb = d_proceduralTextureNbBits.getValue();

            if (height > 0 && width > 0 && !textureData.empty() )
            {
                //Init texture
                if (img) { delete img; img=nullptr; }
                img = new helper::io::Image();
                img->init(width, height, nbb);

                //Make texture
                unsigned char* data = img->getPixels();

                for(std::size_t i=0 ; i<textureData.size() && i < height*width*(nbb/8); i++)
                    data[i] = (unsigned char)textureData[i];

                for (std::size_t i=textureData.size() ; i<height*width*(nbb/8) ; i++)
                    data[i] = 0;
            }
        }
        else
        {
            // A cubemap with faces stored in separate files.
            std::string filename[6] =
            {
                d_cubemapFilenamePosX.getFullPath(),
                d_cubemapFilenameNegX.getFullPath(),
                d_cubemapFilenamePosY.getFullPath(),
                d_cubemapFilenameNegY.getFullPath(),
                d_cubemapFilenamePosZ.getFullPath(),
                d_cubemapFilenameNegZ.getFullPath()
            };

            if (img) delete img;
            img = nullptr;
            helper::io::Image *tmp = nullptr;

            for (unsigned i = 0; i < 6; i++)
                if (!filename[i].empty())
                {
                    if (tmp) delete tmp;
                    tmp = helper::io::Image::Create(filename[i].c_str());

                    if (tmp->getTextureType() != helper::io::Image::TEXTURE_2D)
                    {
                        msg_error() << "Invalid texture type in '" << filename[i] <<"'";
                        continue;
                    }

                    if (!img)
                    {
                        img = new helper::io::Image();
                        img->init(tmp->getWidth(), tmp->getHeight(), 0, 1, tmp->getDataType(), tmp->getChannelFormat());
                        memset(img->getPixels(), 0, img->getImageSize());
                    }
                    else
                    {
                        if (img->getWidth() != tmp->getWidth() ||
                            img->getHeight() != tmp->getHeight())
                        {
                            msg_error() << "Inconsistent cubemap dimensions in '" << filename[i] << "'";
                            continue;
                        }

                        if (img->getDataType() != tmp->getDataType())
                        {
                            msg_error() << "Inconsistent cubemap datatype in '" << filename[i] << "'";
                            continue;
                        }

                        if (img->getChannelFormat() != tmp->getChannelFormat())
                        {
                            msg_error() << "Inconsistent cubemap channel format in '" << filename[i] << "'";
                            continue;
                        }
                    }

                    memcpy(img->getCubeMipmapPixels(i, 0), tmp->getPixels(), tmp->getImageSize());
                }

            if (tmp) delete tmp;
        }
    }
    else
    {
        std::string filename = d_textureFilename.getFullPath();
        if(sofa::helper::system::DataRepository.findFile(filename))
        {
            // Ordinary texture.
            img = helper::io::Image::Create(d_textureFilename.getFullPath().c_str());
        }
        else
        {
            serr << "OglTexture file " << d_textureFilename.getFullPath() << " not found." << sendl;
            //create dummy texture
            if (img) { delete img; img=nullptr; }
            img = new helper::io::Image();
            unsigned int dummyWidth = 128;
            unsigned int dummyHeight = 128;
            unsigned int dummyNbb = 8;

            img->init(dummyWidth, dummyHeight, dummyNbb);

            //Make texture
            unsigned char* data = img->getPixels();

            for(std::size_t i=0 ; i < dummyHeight*dummyWidth*(dummyNbb/8); i++)
                data[i] = (unsigned char)128;
        }
    }

    OglShaderElement::init();
}

void OglTexture::initVisual()
{
#ifdef GL_MAX_TEXTURE_IMAGE_UNITS_ARB //http://developer.nvidia.com/object/General_FAQ.html#t6
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS_ARB, &MAX_NUMBER_OF_TEXTURE_UNIT);
#else
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &MAX_NUMBER_OF_TEXTURE_UNIT);
#endif

    if (d_textureUnit.getValue() > MAX_NUMBER_OF_TEXTURE_UNIT)
    {
        serr << "Unit Texture too high ; set it at the unit texture n°1 (MAX_NUMBER_OF_TEXTURE_UNIT=" << MAX_NUMBER_OF_TEXTURE_UNIT << ")" << sendl;
        d_textureUnit.setValue(1);
    }

    if (texture) delete texture;
    texture = new helper::gl::Texture(img, d_repeat.getValue(), d_linearInterpolation.getValue(),
            d_generateMipmaps.getValue(), d_srgbColorspace.getValue(),
            d_minLod.getValue(), d_maxLod.getValue());
    texture->init();

    setActiveTexture(d_textureUnit.getValue());
    for(std::set<OglShader*>::iterator it = d_shaders.begin(), iend = d_shaders.end(); it!=iend; ++it)
    {
        (*it)->setTexture(d_indexShader.getValue(), d_id.getValue().c_str(), d_textureUnit.getValue());
        //serr << "OGLTextureDEBUG: shader textured:" << (*it)->getName() << sendl;
    }
    setActiveTexture(0);
}

void OglTexture::reinit()
{
    if (d_textureUnit.getValue() > MAX_NUMBER_OF_TEXTURE_UNIT)
    {
        serr << "Unit Texture too high ; set it at the unit texture n°1" << sendl;
        d_textureUnit.setValue(1);
    }
}

void OglTexture::fwdDraw(core::visual::VisualParams*)
{
    if (d_enabled.getValue())
    {
        setActiveTexture(d_textureUnit.getValue());
        bind();
        setActiveTexture(0);
    }
}

void OglTexture::bwdDraw(core::visual::VisualParams*)
{
    if (d_enabled.getValue())
    {
        setActiveTexture(d_textureUnit.getValue());
        unbind();
        setActiveTexture(0);
    }
}

void OglTexture::bind()
{
    if (!texture) initVisual();
    texture->bind();
    glEnable(texture->getTarget());
}

void OglTexture::unbind()
{
    texture->unbind();
    glDisable(texture->getTarget());
}

///////////////////////////////////////////////////////////////////////////////

OglTexture2D::OglTexture2D()
    :texture2DFilename(initData(&texture2DFilename, (std::string) "", "texture2DFilename", "Texture2D Filename"))
{
    serr << helper::logging::Message::Deprecated << "OglTexture2D is deprecated. Please use OglTexture instead." << sendl;
}

OglTexture2D::~OglTexture2D()
{
}

void OglTexture2D::init()
{
    d_textureFilename.setValue(texture2DFilename.getValue());
    OglTexture::init();
}


}//end of namespaces
}
}
