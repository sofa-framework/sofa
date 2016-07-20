#include "console.h"
#include <sofa/helper/Utils.h>
#include <sofa/helper/logging/Messaging.h>
#include <stdlib.h>             // For getenv()

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

#endif

    Console::ColorsStatus Console::getColorsStatus()
    {
        return s_colorsStatus;
    }

}
}
