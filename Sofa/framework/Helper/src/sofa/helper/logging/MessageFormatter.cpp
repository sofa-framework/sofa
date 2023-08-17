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

#include <sofa/helper/system/console.h>
#include <sofa/helper/logging/Message.h>
#include <sofa/helper/logging/MessageFormatter.h>

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

std::ostream & MessageFormatter::setColor(std::ostream &os, unsigned int type) const {
    switch (type) {
        case Message::Advice     : return os << console::Foreground::Bright::Green;
        case Message::Info       : return os << console::Foreground::Bright::Green;
        case Message::Warning    : return os << console::Foreground::Bright::Yellow;
        case Message::Deprecated : return os << console::Foreground::Normal::Yellow;
        case Message::Error      : return os << console::Foreground::Bright::Red;
        case Message::Fatal      : return os << console::Foreground::Bright::Magenta;

        case Message::TEmpty:
        default:
            return os << console::Foreground::Normal::Reset;
    }
}

} // namespace logging

} // namespace helper

} // namespace sofa
