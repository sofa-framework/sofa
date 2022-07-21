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

SOFA_DISABLED_HEADER_NOT_REPLACED("v21.12 (PR#2292)", "v22.06")

// The methods in SofaOStream have been deprecated, but we still want it to work.
// Its usage in Base.h would generate warnings.
// That is why the deprecation is set between SOFA_BUILD_SOFA_CORE macro, to disable the warnings only when building Sofa.Core.
// Warning messages will appear when used outside of Sofa.Core (logically in the components)

// There is also a check on the version of MSVC2017
// as there is a bug when applying [[deprecated]] on a friend method.
// (and does not choke when using the msvc keyword __declspec(deprecated("xxx"))

#ifndef SOFA_BUILD_SOFA_CORE
#if defined(_MSC_VER) && (_MSC_VER < 1920) // if less than VS 2019
#define SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE() \
    __declspec(deprecated(\
        "It is still usable but has been DEPRECATED since v21.12 (PR#2292). " \
        "You have until v22.06 to fix your code. " \
        "Use the Messaging API instead of using SofaOStream and sout/serr/sendl."))
#else // _MSC_VER
#define SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE() \
        SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM()
#endif // _MSC_VER
#else // SOFA_BUILD_SOFA_CORE
#define SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE() // dont do anything if we are building Sofa.Core
#endif // SOFA_BUILD_SOFA_CORE
        
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

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    friend inline
    std::ostream &operator << (std::ostream& out, const SofaEndl<Container> & s)
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

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
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
    
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    friend inline SofaOStream& operator<<( SofaOStream& out, const logging::FileInfo::SPtr& fi )
    {
        out.m_fileInfo = fi;
        return out;
    }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    friend inline SofaOStream& operator<<( SofaOStream& out, const logging::Message::Type& mt )
    {
        out.m_messageType = mt;
        return out;
    }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    friend inline SofaOStream& operator<<( SofaOStream& out, const logging::Message::Class& mc )
    {
        out.m_messageClass = mc;
        return out;
    }

    template<class T>
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    friend inline std::ostringstream& operator<<( SofaOStream& out, const T& t )
    {
        out.m_ostream << t;
        return out.m_ostream;
    }

    // a few useful functions on ostringstream, for a complete API, convert this in a ostringstream
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    std::string str() const { return m_ostream.str(); }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    void str(const std::string& s) { m_ostream.str(s); }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    std::streamsize precision() const { return m_ostream.precision(); }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    std::streamsize precision( std::streamsize p ) { return m_ostream.precision(p); }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    std::ostringstream& ostringstream() const { return m_ostream; }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    const logging::FileInfo::SPtr& fileInfo() const { return m_fileInfo; }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    const logging::Message::Type& messageType() const { return m_messageType; }

    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
    const logging::Message::Class& messageClass() const { return m_messageClass; }

    /// clearing the SofaOStream (set empty string, empty FileInfo, default Message type)
    SOFA_ATTRIBUTE_DEPRECATED__SOFAOSTREAM__OVERRIDE()
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
