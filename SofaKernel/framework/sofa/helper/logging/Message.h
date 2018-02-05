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
#ifndef MESSAGE_H
#define MESSAGE_H

#include <iostream>
#include <string>
#include <cstring>
#include <sofa/helper/helper.h>
#include <sstream>
#include <set>
#include <boost/shared_ptr.hpp>

#include "ComponentInfo.h"
#include "FileInfo.h"

namespace sofa
{

namespace helper
{

namespace logging
{

/// A message is the core object of the msg_* API.
/// A message contains a textual description provided by the user as well as
/// the location in source file (or in a separated file) from where the message have been
/// emitted.
/// A message also contains a ComponentInfo which can be used to provide additional details
/// about the emitter's context.
class SOFA_HELPER_API Message
{
public:
    /// possible levels of messages (ordered)
    enum Type {Info=0, Advice, Deprecated, Warning, Error, Fatal, TEmpty, TypeCount};
    typedef std::set<Type> TypeSet;
    static TypeSet AnyTypes ;

    /// class of messages
    enum Class {Dev, Runtime, Log, CEmpty, ClassCount};

    Message() {}
    Message( const Message& msg );
    Message(Class mclass, Type type,
            const ComponentInfo::SPtr& = ComponentInfo::SPtr(),
            const FileInfo::SPtr& fileInfo = EmptyFileInfo) ;

    Message& operator=( const Message& msg );

    const FileInfo::SPtr&       fileInfo() const { return m_fileInfo; }
    const ComponentInfo::SPtr&  componentInfo() const { return m_componentinfo ; }
    const std::stringstream& message() const  { return m_stream; }
    Class                    context() const  { return m_class; }
    Type                     type() const     { return m_type; }
    const std::string&       sender() const   { return m_componentinfo->sender(); }

    const std::string messageAsString() const  { return m_stream.str(); }
    bool empty() const;

    template<class T>
    Message& operator<<(const T &x)
    {
        m_stream << x;
        return *this;
    }

    static Message emptyMsg ;

protected:
    ComponentInfo::SPtr m_componentinfo; /// a trace (name, path) from whom has emitted this message.
    FileInfo::SPtr      m_fileInfo; ///< a trace (file,line) from where the message have been created
    std::stringstream   m_stream; ///< the actual message
    Class               m_class; ///< who is the attender of the message (developers or users)?
    Type                m_type; ///< the message level
    int                 m_id; ///< should it be stored here or in the handler that needs it?

};


template<> SOFA_HELPER_API Message& Message::operator<<(const FileInfo::SPtr &fi) ;

SOFA_HELPER_API std::ostream& operator<< (std::ostream&, const Message&) ;
SOFA_HELPER_API const std::string toString(const Message::Type type) ;

} // logging
} // helper
} // sofa


#endif // MESSAGE_H
