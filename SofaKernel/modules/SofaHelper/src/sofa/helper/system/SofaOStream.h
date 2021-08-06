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
#ifndef SOFAOSTREAM_H_
#define SOFAOSTREAM_H_

#include <sofa/helper/config.h>
#include <sstream>
#include <sofa/helper/logging/Message.h>

// SOFA_DEPRECATED_HEADER_NOT_REPLACED("v21.12 (PR#XXXX)", "v22.06")

// The methods in SofaOStream have been deprecated, but we still want it to work.
// Its usage in Base.h would generate warnings.
// That is why the deprecation is set between SOFA_BUILD_SOFA_CORE macro, to disable the warnings only when building Sofa.Core.
// Warning messages will appear when used outside of Sofa.Core (logically in the components)

namespace sofa
{

namespace helper
{

namespace system
{


/// SofaEndl asks its eventual container to process the stream
template<class Container>
class SofaEndl
{

protected:

    Container* parent;

public:

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    friend inline std::ostream &operator << (std::ostream& out, const SofaEndl<Container> & s)
    {
        if (s.parent)
            s.parent->processStream(out);
        else
            out << std::endl;
        return out;
    }

    SofaEndl(): parent(nullptr)
    {
    }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    void setParent(Container* p)
    {
        parent = p;
    }
};




/// a SofaOStream is a std::ostringstream encapsulation that can stream a logging::FileInfo and a logging::Message::Type
template< int DefaultMessageType = logging::Message::Info >
class SofaOStream
{

public:

    SofaOStream(std::ostringstream& os) : m_ostream(os), m_messageType((logging::Message::Type)DefaultMessageType) {}

    bool operator==(const std::ostream& os) { return &os == &m_ostream; }
    
#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    friend inline SofaOStream& operator<<( SofaOStream& out, const logging::FileInfo::SPtr& fi )
    {
        out.m_fileInfo = fi;
        return out;
    }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    friend inline SofaOStream& operator<<( SofaOStream& out, const logging::Message::Type& mt )
    {
        out.m_messageType = mt;
        return out;
    }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    friend inline SofaOStream& operator<<( SofaOStream& out, const logging::Message::Class& mc )
    {
        out.m_messageClass = mc;
        return out;
    }

    template<class T>
#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    friend inline std::ostringstream& operator<<( SofaOStream& out, const T& t )
    {
        out.m_ostream << t;
        return out.m_ostream;
    }

    // a few useful functions on ostringstream, for a complete API, convert this in a ostringstream
#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    std::string str() const { return m_ostream.str(); }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    void str(const std::string& s) { m_ostream.str(s); }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    std::streamsize precision() const { return m_ostream.precision(); }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    std::streamsize precision( std::streamsize p ) { return m_ostream.precision(p); }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    std::ostringstream& ostringstream() const { return m_ostream; }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    const logging::FileInfo::SPtr& fileInfo() const { return m_fileInfo; }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    const logging::Message::Type& messageType() const { return m_messageType; }

#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    const logging::Message::Class& messageClass() const { return m_messageClass; }

    /// clearing the SofaOStream (set empty string, empty FileInfo, default Message type)
#ifndef SOFA_BUILD_SOFA_CORE
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // SOFA_BUILD_SOFA_CORE
    void clear()
    {
        str("");
        m_fileInfo = helper::logging::EmptyFileInfo;
        m_messageType = (logging::Message::Type)DefaultMessageType;
        m_messageClass = logging::Message::Runtime;
    }

protected:

    /// the effective ostringstream
    std::ostringstream& m_ostream;

    /// the current FileInfo
    logging::FileInfo::SPtr m_fileInfo {helper::logging::EmptyFileInfo};
    /// the current Message type
    logging::Message::Type m_messageType;
    /// the current Message class
    logging::Message::Class m_messageClass;
};



}

}

}


#endif
