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
* in the messaging.h file.
******************************************************************************/
#ifndef DEFAULTSTYLEMESSAGEFORMATTER_H
#define DEFAULTSTYLEMESSAGEFORMATTER_H
#include <sstream>
#include <string>
#include <sofa/helper/logging/MessageFormatter.h>


namespace sofa::helper::logging
{

class Message;

/// Format the message using a very simple sofa style. For more advanced formatting style
/// have a look at RichStyleMessageFormatter.
/// Example:
///           [ERROR] ClassName(instanceName): this is a message printed somewhere.
class SOFA_HELPER_API DefaultStyleMessageFormatter : public MessageFormatter
{
public:
    static inline MessageFormatter &getInstance ()
    {
        static DefaultStyleMessageFormatter instance;
        return instance;
    }

    void formatMessage (const Message &m, std::ostream &out) override;

private:
    // singleton API
    DefaultStyleMessageFormatter () {}
    DefaultStyleMessageFormatter (const DefaultStyleMessageFormatter &);
    void operator= (const DefaultStyleMessageFormatter &);
};
} // namespace sofa::helper::logging

#endif // DEFAULTSTYLEMESSAGEFORMATTER_H
