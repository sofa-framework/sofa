/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/helper/system/console.h>
#include "Message.h"
#include "MessageFormatter.h"

namespace sofa
{

namespace helper
{

namespace logging
{

std::string MessageFormatter::getPrefixText(unsigned int type) const {
    switch (type) {
        case Message::Advice     : return "[SUGGESTION] ";
        case Message::Deprecated : return "[DEPRECATED] ";
        case Message::Warning    : return "[WARNING] ";
        case Message::Info       : return "[INFO]    ";
        case Message::Error      : return "[ERROR]   ";
        case Message::Fatal      : return "[FATAL]   ";
        case Message::TEmpty     : return "[EMPTY]   ";

        default:
            return "";
    }
}

std::string MessageFormatter::getPrefixCode(unsigned int type) const {
    switch (type) {
        case Message::Advice     : return Console::Code(Console::BRIGHT_GREEN);
        case Message::Info       : return Console::Code(Console::BRIGHT_GREEN);
        case Message::Deprecated : return Console::Code(Console::BRIGHT_YELLOW);
        case Message::Warning    : return Console::Code(Console::BRIGHT_CYAN);
        case Message::Error      : return Console::Code(Console::BRIGHT_RED);
        case Message::Fatal      : return Console::Code(Console::BRIGHT_PURPLE);

        case Message::TEmpty:
        default:
            return Console::Code(Console::DEFAULT);
    }
}

} // namespace logging

} // namespace helper

} // namespace sofa