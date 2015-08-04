/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* This component is open-source                                               *
*                                                                             *
* Authors: Damien Marchal                                                     *
*          Bruno Carrez                                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the TextMessaging.h file.
*****************************************************************************/
#ifndef MESSAGE_H
#define MESSAGE_H
#include <iostream>
#include <string>
#include <sofa/helper/helper.h>

namespace sofa {
namespace core {
namespace objectmodel{
class Base ;
}
}
}

namespace sofa
{

namespace helper
{

namespace logging
{


static const char * s_unknownFile = "unknown-file";

struct FileInfo
{
    const char *filename;
    int line;
    FileInfo(const char *f, int l): filename(f), line(l) {}
    FileInfo(): filename(s_unknownFile), line(0) {}
};

#define SOFA_FILE_INFO sofa::helper::logging::FileInfo(__FILE__, __LINE__)

using std::ostream ;
using std::string ;

class SOFA_HELPER_API Message
{
public:
    Message() {};
    Message(const string& mclass, const string& type, const string& message,
            const string& sender = "", const FileInfo& fileInfo = FileInfo());

    const FileInfo& fileInfo() const ;
    const string& message() const ;
    const string& context() const ;
    const string& type() const ;
    const string& sender() const ;
    int           id() const ;
    void          setId(int id) ;

private:
    string  m_sender;
    FileInfo m_fileInfo;
    string  m_message;
    string  m_class;
    string  m_type;
    int     m_id;
};

ostream& operator<< (ostream&, const Message&) ;

} // logging
} // helper
} // sofa


#endif // MESSAGE_H
