/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* This component is open-source                                               *
*                                                                             *
* Authors: Bruno Carrez                                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/

#include <sstream>
using std::ostringstream ;

#include <iostream>
using std::endl ;
using std::cout ;
using std::cerr ;

#include "DefaultStyleMessageFormatter.h"
#include "Message.h"


namespace sofa
{

namespace helper
{

namespace logging
{

static DefaultStyleMessageFormatter s_DefaultStyleMessageFormatter;


static helper::fixed_array<std::string,Message::TypeCount> setPrefixes()
{
    helper::fixed_array<std::string,Message::TypeCount> prefixes;

    prefixes[Message::Info]    = "[INFO]    ";
    prefixes[Message::Warning] = "[WARNING] ";
    prefixes[Message::Error]   = "[ERROR]   ";
    prefixes[Message::Fatal]   = "[FATAL]   ";

    return prefixes;
}
const helper::fixed_array<std::string,Message::TypeCount> DefaultStyleMessageFormatter::s_MessageTypePrefixes = setPrefixes();


static helper::fixed_array<Console::ColorType,Message::TypeCount> setColors()
{
    helper::fixed_array<Console::ColorType,Message::TypeCount> colors;

    colors[Message::Info]    = Console::BRIGHT_GREEN;
    colors[Message::Warning] = Console::BRIGHT_CYAN;
    colors[Message::Error]   = Console::BRIGHT_RED;
    colors[Message::Fatal]   = Console::BRIGHT_PURPLE;

    return colors;
}
const helper::fixed_array<Console::ColorType,Message::TypeCount> DefaultStyleMessageFormatter::s_MessageTypeColors = setColors();






MessageFormatter* DefaultStyleMessageFormatter::getInstance()
{
    return &s_DefaultStyleMessageFormatter;
}

void DefaultStyleMessageFormatter::formatMessage(const Message& m,std::ostream& out)
{
    out << s_MessageTypeColors[m.type()] << s_MessageTypePrefixes[m.type()] << Console::DEFAULT_COLOR;

    if (!m.sender().empty())
        out << Console::BLUE << "[" << m.sender() << "] " << Console::DEFAULT_COLOR;

    out << m.message();
}


} // logging
} // helper
} // sofa
