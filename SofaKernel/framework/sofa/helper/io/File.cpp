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

#include <sofa/helper/system/config.h>
#include <sofa/helper/logging/Messaging.h>

#include "File.h"

#include <iostream>

namespace sofa
{

namespace helper
{

namespace io
{

File::File() :
    myFileAccess(BaseFileAccess::Create())
{

}

File::File(const std::string& filename, std::ios_base::openmode openMode) :
    myFileAccess(BaseFileAccess::Create())
{
    open(filename, openMode);
}

File::~File()
{
    close();

    delete myFileAccess;
}

bool File::open(const std::string& filename, std::ios_base::openmode openMode)
{
    if(!checkFileAccess())
    {
        msg_error("File") << "While trying to read file: " + filename;
        return false;
    }

    return myFileAccess->open(filename, openMode);
}

void File::close()
{
    if(!checkFileAccess())
        return;

    myFileAccess->close();
}

std::streambuf* File::streambuf() const
{
    if(!checkFileAccess())
        return NULL;

    return myFileAccess->streambuf();
}

std::string File::readAll()
{
    if(!checkFileAccess())
        return "";

    return myFileAccess->readAll();
}

bool File::checkFileAccess() const
{
    if(!myFileAccess)
    {
        msg_error("File") << "File cannot be accessed without a FileAccess object";
        return false;
    }

    return true;
}

} // namespace io

} // namespace helper

} // namespace sofa
