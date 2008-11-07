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
#ifndef SOFA_HELPER_IO_IMAGERAW_H
#define SOFA_HELPER_IO_IMAGERAW_H

#include <sofa/helper/io/Image.h>
#include <string>

#include <sofa/helper/system/config.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace io
{

class SOFA_HELPER_API ImageRAW : public Image
{
public:
    ImageRAW ();
    virtual ~ImageRAW() {}

    void init(int w, int h, int d, int nbb, int hsize);

    // header size in Bytes
    int getHeaderSize() const             { return headerSize; }

    int getDataSize() const               { return getLineSize()*height*depth; }

    // number of slices of a 3D image
    int getDepth() const                  { return depth; }

    unsigned char * getHeader()           { return header; }
    const unsigned char * getHeader() const { return header; }

    bool load(std::string filename);
    bool save(std::string filename, int compression_level = -1);

private:
    int depth;
    int headerSize;

    unsigned char *header;
};


} // namespace io

} // namespace helper

} // namespace sofa

#endif
