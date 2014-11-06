/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_IO_IMAGERAW_H
#define SOFA_HELPER_IO_IMAGERAW_H

#include <sofa/helper/io/Image.h>
#include <string>

#include <sofa/helper/system/config.h>
#include <sofa/SofaFramework.h>

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

    void initHeader(unsigned hsize);

    // header size in Bytes
    unsigned getHeaderSize() const { return headerSize; }

    unsigned char * getHeader()           { return header; }
    const unsigned char * getHeader() const { return header; }

    bool load(std::string filename);
    bool save(std::string filename, int compression_level = -1);

private:
    unsigned headerSize;
    unsigned char *header;
};


} // namespace io

} // namespace helper

} // namespace sofa

#endif
