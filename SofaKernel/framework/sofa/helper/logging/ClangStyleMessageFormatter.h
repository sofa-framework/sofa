/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef CLANGSTYLEMESSAGEFORMATTER_H
#define CLANGSTYLEMESSAGEFORMATTER_H
#include <sstream>
#include "Message.h"
#include "MessageFormatter.h"
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace logging
{


class SOFA_HELPER_API ClangStyleMessageFormatter : public MessageFormatter
{
public:
    virtual void formatMessage(const Message& m,std::ostream& out);
    static ClangStyleMessageFormatter& getInstance() { return s_instance; }

private:
    // singleton API
    ClangStyleMessageFormatter();
    ClangStyleMessageFormatter(const ClangStyleMessageFormatter&);
    void operator=(const ClangStyleMessageFormatter&);
    static ClangStyleMessageFormatter s_instance;
};


} // logging
} // helper
} // sofa

#endif // CLANGSTYLEMESSAGEFORMATTER_H
