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
#include "console.h"
#include <sofa/helper/Utils.h>
#include <sofa/helper/logging/Messaging.h>
#include <cstdlib>

#ifdef __linux__

#include <sys/ioctl.h>

#endif


#ifndef WIN32

#  include <unistd.h>           // for isatty()

#endif


namespace sofa {
namespace helper {

using SystemCodeType = Console::SystemCodeType;
Console::ColorsStatus Console::s_colorsStatus = Console::ColorsAuto;

#ifdef WIN32
static SystemCodeType getDefaultColor();
#endif

void Console::init()
{
    // Change s_colorsStatus based on the SOFA_COLOR_TERMINAL environnement variable.
    const char *sofa_color_terminal = std::getenv("SOFA_COLOR_TERMINAL");
    if (sofa_color_terminal != nullptr) {
        const std::string colors(sofa_color_terminal);
        if (colors == "yes" || colors == "on" || colors == "always")
            s_colorsStatus = Console::ColorsEnabled;
        else if (colors == "no" || colors == "off" || colors == "never")
            s_colorsStatus = Console::ColorsDisabled;
        else if (colors == "auto")
            s_colorsStatus = Console::ColorsAuto;
        else
            msg_warning("Console::init()") << "Bad value for environnement variable SOFA_COLOR_TERMINAL (" << colors
                                           << ")";
    }
}

SystemCodeType Console::Code(Style s)
{
#ifdef WIN32
    static const SystemCodeType default_style_code = getDefaultColor();
    switch (s) {
        case BLACK         : return 0;
        case BLUE          : return 1;
        case GREEN         : return 2;
        case CYAN          : return 3;
        case RED           : return 4;
        case PURPLE        : return 5;
        case YELLOW        : return 6;
        case WHITE         : return 7;
        case BRIGHT_BLACK  : return 8;
        case BRIGHT_BLUE   : return 9;
        case BRIGHT_GREEN  : return 10;
        case BRIGHT_CYAN   : return 11;
        case BRIGHT_RED    : return 12;
        case BRIGHT_PURPLE : return 13;
        case BRIGHT_YELLOW : return 14;
        case BRIGHT_WHITE  : return 15;

        //TODO(dmarchal): Implement the rich text on windows...
        case ITALIC:
        case UNDERLINE:
        case DEFAULT:
        default:
        return default_style_code;;
    }

#else
    switch (s) {
        case BLACK         : return "\033[0;30m";
        case RED           : return "\033[0;31m";
        case GREEN         : return "\033[0;32m";
        case YELLOW        : return "\033[0;33m";
        case BLUE          : return "\033[0;34m";
        case PURPLE        : return "\033[0;35m";
        case CYAN          : return "\033[0;36m";
        case WHITE         : return "\033[0;37m";
        case BRIGHT_BLACK  : return "\033[1;30m";
        case BRIGHT_RED    : return "\033[1;31m";
        case BRIGHT_GREEN  : return "\033[1;32m";
        case BRIGHT_YELLOW : return "\033[1;33m";
        case BRIGHT_BLUE   : return "\033[1;34m";
        case BRIGHT_PURPLE : return "\033[1;35m";
        case BRIGHT_CYAN   : return "\033[1;36m";
        case BRIGHT_WHITE  : return "\033[1;37m";
        case ITALIC        : return "\033[3m";
        case UNDERLINE     : return "\033[4m";

        case DEFAULT:
        default:
            return "\033[0m";
    }
#endif
}

#ifdef WIN32

static HANDLE getOutputHandle()
{
    static bool first = true;
    static HANDLE s_console = NULL;
    if (first)
    {
        s_console = GetStdHandle(STD_OUTPUT_HANDLE);
        if (s_console == INVALID_HANDLE_VALUE)
        {
            std::cerr << "Console::getOutputHandle(): " << Utils::GetLastError() << std::endl;
        }
        if (s_console == NULL)
        {
            std::cerr << "Console::getOutputHandle(): no stdout handle!" << std::endl;
        }
        first = false;
    }
    return s_console;
}

static SystemCodeType getDefaultColor()
{
    CONSOLE_SCREEN_BUFFER_INFO currentInfo;
    GetConsoleScreenBufferInfo(getOutputHandle(), &currentInfo);
    return currentInfo.wAttributes;
}

void Console::setColorsStatus(ColorsStatus status)
{
    s_colorsStatus = status;
}

bool Console::shouldUseColors(std::ostream& stream)
{
    // On Windows, colors are not handled with control characters, so we can
    // probably always use them unless explicitely disabled.
    return getColorsStatus() != ColorsDisabled;
}

#else

static bool s_stdoutIsRedirected = false;
static bool s_stderrIsRedirected = false;

void Console::setColorsStatus(ColorsStatus status)
{
    s_colorsStatus = status;
    // Check for redirections and save the result.  (This assumes that
    // stdout and stderr won't be redirected in the middle of the execution
    // of the program, which seems reasonable to me.)
    s_stdoutIsRedirected = isatty(STDOUT_FILENO) == 0;
    s_stderrIsRedirected = isatty(STDERR_FILENO) == 0;
}

bool Console::shouldUseColors(std::ostream &stream)
{
    if (s_colorsStatus == Console::ColorsAuto)
        return (&stream == &std::cout && !s_stdoutIsRedirected)
               || (&stream == &std::cerr && !s_stderrIsRedirected);
    else
        return s_colorsStatus == Console::ColorsEnabled;
}

#endif

std::ostream &operator<<(std::ostream &stream, const SystemCodeType & code)
{
    if (Console::shouldUseColors(stream))
#ifdef WIN32
        SetConsoleTextAttribute(getOutputHandle(), code);
#else
        return stream << code;
    else
#endif
        return stream;
}

size_t Console::getColumnCount()
{
#ifdef __linux__
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col;
#else
    //TODO(dmarchal): implement macOS and Windows or a portable version of this function.
    return 80;
#endif
}

Console::ColorsStatus Console::getColorsStatus()
{
    return s_colorsStatus;
}

}
}
