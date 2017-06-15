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
#include "console.h"
#include <sofa/helper/Utils.h>
#include <sofa/helper/logging/Messaging.h>
#include <stdlib.h>             // For getenv()

#ifdef __linux__
#include <sys/ioctl.h>
#include <unistd.h>
#endif


#ifndef WIN32
#  include <unistd.h>           // for isatty()
#endif


namespace sofa {
namespace helper {

    Console::ColorsStatus Console::s_colorsStatus = Console::ColorsAuto;

    void Console::init()
    {
        // Change s_colorsStatus based on the SOFA_COLOR_TERMINAL environnement variable.
        const char *sofa_color_terminal = getenv("SOFA_COLOR_TERMINAL");
        if (sofa_color_terminal != NULL)
        {
            const std::string colors(sofa_color_terminal);
            if (colors == "yes" || colors == "on" || colors == "always")
                s_colorsStatus = Console::ColorsEnabled;
            else if (colors == "no" || colors == "off" || colors == "never")
                s_colorsStatus = Console::ColorsDisabled;
            else if (colors == "auto")
                s_colorsStatus = Console::ColorsAuto;
            else
                msg_warning("Console::init()") << "Bad value for environnement variable SOFA_COLOR_TERMINAL (" << colors << ")";
        }
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

    static Console::ColorType getDefaultColor()
    {
        CONSOLE_SCREEN_BUFFER_INFO currentInfo;
        GetConsoleScreenBufferInfo(getOutputHandle(), &currentInfo);
        return currentInfo.wAttributes;
    }
    static Console::CodeType getDefaultCode()
    {
        CONSOLE_SCREEN_BUFFER_INFO currentInfo;
        GetConsoleScreenBufferInfo(getOutputHandle(), &currentInfo);
        return currentInfo.wAttributes;
    }



    const Console::ColorType Console::BLACK         = Console::ColorType(0);
    const Console::ColorType Console::BLUE          = Console::ColorType(1);
    const Console::ColorType Console::GREEN         = Console::ColorType(2);
    const Console::ColorType Console::CYAN          = Console::ColorType(3);
    const Console::ColorType Console::RED           = Console::ColorType(4);
    const Console::ColorType Console::PURPLE        = Console::ColorType(5);
    const Console::ColorType Console::YELLOW        = Console::ColorType(6);
    const Console::ColorType Console::WHITE         = Console::ColorType(7);
    const Console::ColorType Console::BRIGHT_BLACK  = Console::ColorType(8);
    const Console::ColorType Console::BRIGHT_BLUE   = Console::ColorType(9);
    const Console::ColorType Console::BRIGHT_GREEN  = Console::ColorType(10);
    const Console::ColorType Console::BRIGHT_CYAN   = Console::ColorType(11);
    const Console::ColorType Console::BRIGHT_RED    = Console::ColorType(12);
    const Console::ColorType Console::BRIGHT_PURPLE = Console::ColorType(13);
    const Console::ColorType Console::BRIGHT_YELLOW = Console::ColorType(14);
    const Console::ColorType Console::BRIGHT_WHITE  = Console::ColorType(15);
    const Console::ColorType Console::DEFAULT_COLOR = getDefaultColor();

    //TODO(dmarchal): Implement the rich text on windows...
    const Console::CodeType Console::ITALIC = getDefaultCode();
    const Console::CodeType Console::UNDERLINE = getDefaultCode();
    const Console::CodeType Console::DEFAULT_CODE = getDefaultCode();

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

    SOFA_HELPER_API std::ostream& operator<<( std::ostream& stream, Console::ColorType color )
    {
        if (Console::shouldUseColors(stream))
            SetConsoleTextAttribute(getOutputHandle(), color.value);
        return stream;
    }

    SOFA_HELPER_API std::ostream& operator<<( std::ostream& stream, Console::CodeType code )
    {
        if (Console::shouldUseColors(stream))
            SetConsoleTextAttribute(getOutputHandle(), code.value);
        return stream;
    }

#else

    const Console::ColorType Console::BLACK         = Console::ColorType("\033[0;30m");
    const Console::ColorType Console::RED           = Console::ColorType("\033[0;31m");
    const Console::ColorType Console::GREEN         = Console::ColorType("\033[0;32m");
    const Console::ColorType Console::YELLOW        = Console::ColorType("\033[0;33m");
    const Console::ColorType Console::BLUE          = Console::ColorType("\033[0;34m");
    const Console::ColorType Console::PURPLE        = Console::ColorType("\033[0;35m");
    const Console::ColorType Console::CYAN          = Console::ColorType("\033[0;36m");
    const Console::ColorType Console::WHITE         = Console::ColorType("\033[0;37m");
    const Console::ColorType Console::BRIGHT_BLACK  = Console::ColorType("\033[1;30m");
    const Console::ColorType Console::BRIGHT_RED    = Console::ColorType("\033[1;31m");
    const Console::ColorType Console::BRIGHT_GREEN  = Console::ColorType("\033[1;32m");
    const Console::ColorType Console::BRIGHT_YELLOW = Console::ColorType("\033[1;33m");
    const Console::ColorType Console::BRIGHT_BLUE   = Console::ColorType("\033[1;34m");
    const Console::ColorType Console::BRIGHT_PURPLE = Console::ColorType("\033[1;35m");
    const Console::ColorType Console::BRIGHT_CYAN   = Console::ColorType("\033[1;36m");
    const Console::ColorType Console::BRIGHT_WHITE  = Console::ColorType("\033[1;37m");
    const Console::ColorType Console::DEFAULT_COLOR = Console::ColorType("\033[0m");

    const Console::CodeType Console::UNDERLINE = Console::CodeType("\033[4m");
    const Console::CodeType Console::ITALIC = Console::CodeType("\033[3m");
    const Console::CodeType Console::DEFAULT_CODE = Console::CodeType("\033[0m");


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

    bool Console::shouldUseColors(std::ostream& stream)
    {
        if (s_colorsStatus == Console::ColorsAuto)
            return (&stream == &std::cout && !s_stdoutIsRedirected)
                || (&stream == &std::cerr && !s_stderrIsRedirected);
        else
            return s_colorsStatus == Console::ColorsEnabled;
    }

    std::ostream& operator<<( std::ostream& stream, Console::ColorType color )
    {
        if (Console::shouldUseColors(stream))
            return stream << color.value;
        else
            return stream;
    }

    std::ostream& operator<<( std::ostream& stream, Console::CodeType code )
    {
        if (Console::shouldUseColors(stream))
            return stream << code.value;
        else
            return stream;
    }


#endif

    size_t Console::getColumnCount(){
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
