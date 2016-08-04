/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v16.08                  *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "Message.h"

using std::endl ;
using std::string ;

namespace sofa
{

namespace helper
{

namespace logging
{

Message Message::emptyMsg(CEmpty, TEmpty, "", FileInfo()) ;

Message::Message(Class mclass, Type type,
                 const string& sender, const FileInfo& fileInfo):
    m_sender(sender),
    m_fileInfo(fileInfo),
    m_class(mclass),
    m_type(type),
    m_id(-1)
{
}

Message::Message( const Message& msg )
    : m_sender(msg.sender())
    , m_fileInfo(msg.fileInfo())
    , m_class(msg.context())
    , m_type(msg.type())
//    , m_id(msg.id())
{
    m_stream << msg.message().str();
}

Message& Message::operator=( const Message& msg )
{
    m_sender = msg.sender();
    m_fileInfo = msg.fileInfo();
    m_class = msg.context();
    m_type = msg.type();
//    m_id = msg.id();
    m_stream << msg.message().str();
    return *this;
}


std::ostream& operator<< (std::ostream& s, const Message& m){
    s << "[" << m.sender() << "]: " << endl ;
//    s << "         Message id: " << m.id() << endl ;
    s << "    Message type   : " << m.type() << endl ;
    s << "    Message content: " << m.message().str() << endl ;
    s << "    source code loc: " << m.fileInfo().filename << ":" << m.fileInfo().line << endl ;
    return s;
}

bool Message::empty() const
{
    // getting the size without creating a copy like m_stream.str().size()

    std::streambuf* buf = m_stream.rdbuf();

    // the current position to restore it after
    std::stringstream::pos_type cur = buf->pubseekoff(0, std::ios_base::cur);

    // go to the end
    std::stringstream::pos_type end = buf->pubseekoff(0, std::ios_base::end);

    // restore initial position
    buf->pubseekpos(cur, m_stream.out);

    return end <= 0;
}

} // logging
} // helper
} // sofa
