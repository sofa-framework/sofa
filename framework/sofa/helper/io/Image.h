/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_HELPER_IO_IMAGE_H
#define SOFA_HELPER_IO_IMAGE_H

#include <stdlib.h>
#include <sofa/helper/Factory.h>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace io
{

class SOFA_HELPER_API Image
{
protected:
    int width, height, nbBits;
    unsigned char *data;
public:
    Image();

    virtual ~Image();

    void init(int w, int h, int nbb);
    void clear();

    int getWidth() const                  { return width; }
    int getHeight() const                 { return height; }
    int getNbBits() const                 { return nbBits; }
    int getLineSize() const               { return ((nbBits+7)/8)*width; }
    int getImageSize() const              { return getLineSize()*height; }
    unsigned char * getData()             { return data; }
    const unsigned char * getData() const { return data; }

    typedef Factory<std::string, Image, std::string> FactoryImage;

    static Image* Create(std::string filename);

    template<class Object>
    static void create(Object*& obj, std::string arg)
    {
        obj = new Object(arg);
    }
};

} // namespace io

#if defined(WIN32) && !defined(SOFA_BUILD_HELPER)
extern template class SOFA_HELPER_API Factory<std::string, io::Image, std::string>;
#endif

} // namespace helper

} // namespace sofa

#endif
