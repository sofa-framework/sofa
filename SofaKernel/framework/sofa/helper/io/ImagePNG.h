/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_HELPER_IO_IMAGEPNG_H
#define SOFA_HELPER_IO_IMAGEPNG_H
#include <sofa/helper/helper.h>

#include <sofa/helper/io/Image.h>
#include <string>
#include <cassert>

#include <sofa/helper/system/config.h>

namespace sofa
{

namespace helper
{

namespace io
{

#ifdef SOFA_HAVE_PNG

class SOFA_HELPER_API ImagePNG : public Image
{
public:
    ImagePNG ()
    {
    }

    ImagePNG (const std::string &filename)
    {
        load(filename);
    }

    bool load(std::string filename);
    bool save(std::string filename, int compression_level = -1);
};

#else
// #warning not supported by MSVC
//#warning PNG not supported. Define SOFA_HAVE_PNG in sofa.cfg to activate.
#endif

} // namespace io

} // namespace helper

} // namespace sofa

#endif
