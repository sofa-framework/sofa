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

#ifndef SOFA_HELPER_IO_FILEACCESS_H
#define SOFA_HELPER_IO_FILEACCESS_H

#include <sofa/helper/helper.h>

#include "BaseFileAccess.h"

#include <fstream>
#include <string>

namespace sofa
{

namespace helper
{

namespace io
{

// \brief Allow reading and writing into a file.
class SOFA_HELPER_API FileAccess : public BaseFileAccess
{
    friend class FileAccessCreator<FileAccess>;

protected:
    FileAccess();

public:
    ~FileAccess();

    virtual bool open(const std::string& filename, std::ios_base::openmode openMode);
    virtual void close();

    virtual std::streambuf* streambuf() const;
    virtual std::string readAll();
    virtual void write(const std::string& data);

private:
    std::fstream myFile;

};

} // namespace io

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_IO_FILEACCESS_H
