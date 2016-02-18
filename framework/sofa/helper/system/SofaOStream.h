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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFAOSTREAM_H_
#define SOFAOSTREAM_H_

#include <sofa/helper/helper.h>
#include <sstream>
#include <iostream>
#include <sofa/helper/logging/Message.h>

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

    friend inline std::ostream &operator << (std::ostream& out, const SofaEndl<Container> & s)
    {
        if (s.parent)
            s.parent->processStream(out);
        else
            out << std::endl;
        return out;
    }

    SofaEndl(): parent(NULL)
    {
    }

    void setParent(Container* p)
    {
        parent = p;
    }
};



/// a SofaOStream is a simple std::ostringstream that can stream a logging::FileInfo
class SOFA_HELPER_API SofaOStream : public std::ostringstream
{
protected:
    logging::FileInfo m_fileInfo;

public:

    void clear()
    {
        this->str("");
        m_fileInfo = helper::logging::FileInfo();
    }

    const logging::FileInfo& fileInfo() const
    {
        return m_fileInfo;
    }

    friend inline SofaOStream& operator<<( SofaOStream& out, const logging::FileInfo& fi )
    {
        out.m_fileInfo = fi;
        return out;
    }

};



}

}

}


#endif
