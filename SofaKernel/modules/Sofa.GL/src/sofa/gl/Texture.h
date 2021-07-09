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
#include <sofa/gl/gl.h>
#include <sofa/gl/config.h>
#include <sofa/helper/io/Image.h>

namespace sofa::gl
{

class SOFA_GL_API Texture
{
private:
    helper::io::Image *image;
    GLuint id, target;
    bool repeat, linearInterpolation, generateMipmaps, srgbColorspace;
    float minLod, maxLod;
public:

    Texture()
        :image(nullptr),id(0),repeat(true), linearInterpolation(true), generateMipmaps(true),
         srgbColorspace(false), minLod(-1000), maxLod(1000)
    {
    }

    Texture (helper::io::Image *img, bool repeat = true, bool linearInterpolation = true, bool generateMipmaps = true,
            bool srgbColorspace = false, float minLod = -1000, float maxLod = 1000)
        :image(img),id(0),repeat(repeat), linearInterpolation(linearInterpolation), generateMipmaps(generateMipmaps),
         srgbColorspace(srgbColorspace), minLod(minLod), maxLod(maxLod)
    {}

    helper::io::Image* getImage(void);
    GLuint getTarget() const { return target; }
    void   bind(void);
    void   unbind(void);
    void   init ();
    void   update ();	// to use for dynamic change of the texture image (no memory overhead due to multiple texture creation)
    ~Texture();

    GLuint getId() const { return id; }

private:
    Texture(const Texture& ) {}
    Texture operator=(const Texture& ) { return Texture(); }
};

} // namespace sofa::gl
