/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_GL_TEXTURE_H
#define SOFA_HELPER_GL_TEXTURE_H

#include <sofa/helper/system/gl.h>
#include <sofa/helper/helper.h>
#include <sofa/helper/io/Image.h>

namespace sofa
{

namespace helper
{

namespace gl
{

//using namespace sofa::defaulttype;

class SOFA_HELPER_API Texture
{
private:
    io::Image *image;
    GLuint id;
    bool repeat, linearInterpolation;
public:
    Texture (io::Image *img, bool repeat = false, bool linearInterpolation = true)
        :image(img),id(0),repeat(repeat), linearInterpolation(linearInterpolation)
    {};

    io::Image* getImage(void);
    void   bind(void);
    void   unbind(void);
    void   init (void);
    ~Texture();
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
