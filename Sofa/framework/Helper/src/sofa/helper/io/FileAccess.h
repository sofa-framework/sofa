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

#ifndef SOFA_HELPER_IO_FILEACCESS_H
#define SOFA_HELPER_IO_FILEACCESS_H

#include <sofa/helper/config.h>

#include <sofa/helper/io/BaseFileAccess.h>

#include <fstream>
#include <string>

namespace sofa::helper::io
{

// \brief Allow reading and writing into a file.
class SOFA_HELPER_API FileAccess : public BaseFileAccess
{
    friend class FileAccessCreator<FileAccess>;

protected:
    FileAccess();

public:
    ~FileAccess() override;

    virtual bool open(const std::string& filename, std::ios_base::openmode openMode) override;
    void close() override;

    virtual std::streambuf* streambuf() const override;
    virtual std::string readAll() override;
    virtual void write(const std::string& data) override;

private:
    std::fstream myFile;

};
} // namespace sofa::helper::io

#endif // SOFA_HELPER_IO_FILEACCESS_H
