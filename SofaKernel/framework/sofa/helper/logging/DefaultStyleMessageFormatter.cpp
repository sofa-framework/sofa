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


#include "DefaultStyleMessageFormatter.h"
#include "Message.h"

#include <sofa/helper/system/console.h>
#include <sofa/helper/fixed_array.h>


namespace sofa
{

namespace helper
{

namespace logging
{


helper::fixed_array<std::string,Message::TypeCount> s_messageTypePrefixes;
helper::fixed_array<Console::ColorType,Message::TypeCount> s_messageTypeColors;
DefaultStyleMessageFormatter DefaultStyleMessageFormatter::s_instance;




DefaultStyleMessageFormatter::DefaultStyleMessageFormatter()
{
    s_messageTypePrefixes[Message::Advice]      = "[SUGGESTION] ";
    s_messageTypePrefixes[Message::Info]        = "[INFO]    ";
    s_messageTypePrefixes[Message::Deprecated]  = "[DEPRECATED] ";
    s_messageTypePrefixes[Message::Warning]     = "[WARNING] ";
    s_messageTypePrefixes[Message::Error]       = "[ERROR]   ";
    s_messageTypePrefixes[Message::Fatal]       = "[FATAL]   ";
    s_messageTypePrefixes[Message::TEmpty]      = "[EMPTY]   ";

    s_messageTypeColors[Message::Advice]     = Console::BRIGHT_GREEN;
    s_messageTypeColors[Message::Info]       = Console::BRIGHT_GREEN;
    s_messageTypeColors[Message::Deprecated] = Console::BRIGHT_YELLOW;
    s_messageTypeColors[Message::Warning]    = Console::BRIGHT_CYAN;
    s_messageTypeColors[Message::Error]      = Console::BRIGHT_RED;
    s_messageTypeColors[Message::Fatal]      = Console::BRIGHT_PURPLE;
    s_messageTypeColors[Message::TEmpty]     = Console::DEFAULT_COLOR;
}

void DefaultStyleMessageFormatter::formatMessage(const Message& m,std::ostream& out)
{
    out << s_messageTypeColors[m.type()] << s_messageTypePrefixes[m.type()];

    if (!m.sender().empty())
        out << Console::BLUE << "[" << m.componentInfo() << "] ";

    out << Console::DEFAULT_COLOR << m.message().str() << std::endl;
}


} // logging
} // helper
} // sofa
