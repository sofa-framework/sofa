/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/SofaFramework.h>
#include <sstream>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace system
{

//class SOFA_HELPER_API SofaOStreamContainer
//{
//public:
//    virtual ~SofaOStreamContainer();
//    virtual void processStream(std::ostream& out) = 0;
//};

template<class Container>
class SofaOStream
{
protected:
    Container* parent;
public:

    friend inline std::ostream &operator << (std::ostream& out, SofaOStream<Container> & s)
    {
        if (s.parent)
            s.parent->processStream(out);
        else out << std::endl;
        return out;
    }

    SofaOStream()
        : parent(NULL)
    {
    }

    ~SofaOStream()
    {
    }

    void setParent(Container* p)
    {
        parent = p;
    }

};

}

}

}


#endif
