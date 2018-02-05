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
* in the messaging.h file.
******************************************************************************/
#ifndef MESSAGEFORMATTER_H
#define MESSAGEFORMATTER_H

#include <sstream>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace logging
{

class Message;

class SOFA_HELPER_API MessageFormatter
{
public:
    virtual void formatMessage(const Message& m,std::ostream& out) = 0 ;

protected:
    MessageFormatter(){} // no public default constructor, it should be enough to have singleton for MessageFormatters

};

} // logging
} // helper
} // sofa


#endif // MESSAGEFORMATTER_H
