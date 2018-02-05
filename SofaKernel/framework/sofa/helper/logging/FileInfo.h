/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
/*****************************************************************************
* User of this library should read the documentation
* in the TextMessaging.h file.
*****************************************************************************/
#ifndef FILEINFO_H
#define FILEINFO_H

#include <iostream>
#include <string>
#include <cstring>
#include <sofa/helper/helper.h>
#include <sstream>
#include <set>
#include <boost/shared_ptr.hpp>


namespace sofa
{

namespace helper
{

namespace logging
{

static const char * s_unknownFile = "unknown-file";

/// To keep a trace (file,line) from where the message have been created
/// The filename must be a valid pointer throughoug the message processing
/// If this cannot be guaranteed then use the FileInfoOwningFilename class
/// instead.
struct FileInfo
{
    typedef boost::shared_ptr<FileInfo> SPtr;

    const char *filename {nullptr};
    int line             {0};
    FileInfo(const char *f, int l): filename(f), line(l) {}

protected:
    FileInfo(): filename(s_unknownFile), line(0) {}
};

/// To keep a trace (file,line) from where the message have been created
struct FileInfoOwningFilename : public FileInfo
{
    FileInfoOwningFilename(const char *f, int l) {
        char *tmp  = new char[strlen(f)+1] ;
        strcpy(tmp, f) ;
        filename = tmp ;
        line = l ;
    }

    FileInfoOwningFilename(const std::string& f, int l) {
        char *tmp  = new char[f.size()+1] ;
        strcpy(tmp, f.c_str()) ;
        filename = tmp ;
        line = l ;
    }

    ~FileInfoOwningFilename(){
        if(filename)
            delete filename ;
    }
};

static FileInfo::SPtr EmptyFileInfo(new FileInfo(s_unknownFile, 0)) ;

#define SOFA_FILE_INFO sofa::helper::logging::FileInfo::SPtr(new sofa::helper::logging::FileInfo(__FILE__, __LINE__))
#define SOFA_FILE_INFO_COPIED_FROM(file,line) sofa::helper::logging::FileInfo::SPtr(new sofa::helper::logging::FileInfoOwningFilename(file,line))

} // logging
} // helper
} // sofa


#endif // MESSAGE_H
