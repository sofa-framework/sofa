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
/*****************************************************************************
* User of this library should read the documentation
* in the TextMessaging.h file.
*****************************************************************************/
#ifndef COMPONENTINFO_H
#define COMPONENTINFO_H

#include <iostream>
#include <string>
#include <cstring>
#include <sofa/helper/config.h>
#include <sstream>
#include <set>
#include <memory>

namespace sofa::helper::logging
{

/// The base class to keep track of informations associated with a message.
/// A component info object have a sender method to return the name string identifying the
/// sender of a message.
///
struct SOFA_HELPER_API ComponentInfo
{
    ComponentInfo() ;
    ComponentInfo(const std::string& name) ;
    virtual ~ComponentInfo() ;

    /// Returns a string identifying the sender of a message.
    const std::string& sender() const ;

    /// Write a textual version of the content of the ComponentInfo. You should
    /// override this function when inheriting from the ComponentInfo base class.
    virtual std::ostream& toStream(std::ostream& out) const ;

    friend std::ostream& operator<<(std::ostream& out, const ComponentInfo& nfo) ;
    friend std::ostream& operator<<(std::ostream& out, const ComponentInfo* nfo) ;

    typedef std::shared_ptr<ComponentInfo> SPtr;
protected:
    std::string m_sender ;
};

/// This function is used in the msg_* macro to handle emitting case based on string.
inline const ComponentInfo::SPtr getComponentInfo(const std::string& s)
{
    return ComponentInfo::SPtr( new ComponentInfo(s) );
}

/// This function is used in the msg_* macro to handle emitting case based on string.
inline const ComponentInfo::SPtr getComponentInfo(const ComponentInfo::SPtr& s)
{
    return s;
}

inline bool notMuted(const ComponentInfo::SPtr&){ return true; }

/// This function is used in the msg_* macro to handle string based on string.
inline bool notMuted(const std::string&){ return true; }

} // namespace sofa::helper::logging


#endif // COMPONENTINFO_H
