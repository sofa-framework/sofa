/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sstream>


namespace sofa
{

namespace helper
{

namespace logging
{


static const char * s_unknownFile = "unknown-file";

/// To keep a trace (file,line) from where the message have been created
struct FileInfo
{
    const char *filename;
    int line;
    FileInfo(const char *f, int l): filename(f), line(l) {}
    FileInfo(): filename(s_unknownFile), line(0) {}
};

#define SOFA_FILE_INFO sofa::helper::logging::FileInfo(__FILE__, __LINE__)


class SOFA_HELPER_API Message
{
public:

    /// possible levels of messages (ordered)
    enum Type {Info=0, Deprecated, Warning, Error, Fatal, TEmpty, TypeCount};

    /// class of messages
    enum Class {Dev, Runtime, CEmpty, ClassCount};

    Message() {}
    Message( const Message& msg );
    Message(Class mclass, Type type,
            const std::string& sender = "", const FileInfo& fileInfo = FileInfo());

    Message& operator=( const Message& msg );

    const FileInfo&          fileInfo() const { return m_fileInfo; }
    const std::stringstream& message() const  { return m_stream; }
    Class                    context() const  { return m_class; }
    Type                     type() const     { return m_type; }
    const std::string&       sender() const   { return m_sender; }
//    int                      id() const       { return m_id; }
//    void                     setId(int id)    { m_id=id; }

    bool empty() const;

    template<class T>
    Message& operator<<(const T &x)
    {
        m_stream << x;
        return *this;
    }


    static Message emptyMsg ;

protected:
    std::string       m_sender; ///< who send the message (component or module)
    FileInfo          m_fileInfo; ///< a trace (file,line) from where the message have been created
    std::stringstream m_stream; ///< the actual message
    Class             m_class; ///< who is the attender of the message (developers or users)?
    Type              m_type; ///< the message level
    int               m_id; ///< should it be stored here or in the handler that needs it?

};


SOFA_HELPER_API std::ostream& operator<< (std::ostream&, const Message&) ;

} // logging
} // helper
} // sofa


#endif // MESSAGE_H
