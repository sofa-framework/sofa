#include "console.h"
#include "sofa/helper/Utils.h"

#ifndef WIN32
#  include <unistd.h>           // for isatty()
#endif


namespace sofa {
namespace helper {

    Console::ColorsStatus Console::s_colorsStatus = Console::ColorsDisabled;

#ifdef WIN32

    static HANDLE s_console = NULL;

    static Console::ColorType initConsoleAndGetDefaultColor()
    {
        s_console = GetStdHandle(STD_OUTPUT_HANDLE);

        if(s_console == INVALID_HANDLE_VALUE)
        {
            std::cerr << "Console::init(): " << Utils::GetLastError() << std::endl;
            return Console::ColorType(7);
        }
        if(s_console == NULL)
        {
            std::cerr << "Console::init(): no stdout handle!" << std::endl;
            return Console::ColorType(7);
        }
        else
        {
            CONSOLE_SCREEN_BUFFER_INFO currentInfo;
            GetConsoleScreenBufferInfo(s_console, &currentInfo);
            return currentInfo.wAttributes;
        }
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
    const Console::ColorType Console::DEFAULT_COLOR = initConsoleAndGetDefaultColor();

    void Console::setColorsStatus(ColorsStatus status)
    {
        s_colorsStatus = status;
    }

    bool Console::shouldUseColors(std::ostream& stream)
    {
        // On Windows, colors are not handled with control characters, so we can
        // probably always use them unless explicitely disabled.
        return !ColorsDisabled:
    }

    SOFA_HELPER_API std::ostream& operator<<( std::ostream& stream, Console::ColorType color )
    {
        if (Console::shouldUseColors(stream))
            SetConsoleTextAttribute(s_console, color.value);
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
            return (stream == std::cout && !s_stdoutIsRedirected)
                || (stream == std::cerr && !s_stderrIsRedirected);
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

    SOFA_HELPER_API std::ostream& operator<<(std::ostream& stream, Console::LogPrefix prefix)
    {
        switch (prefix)
        {
            case Console::InfoPrefix:
                return stream << Console::BRIGHT_GREEN << "[INFO]" << Console::DEFAULT_COLOR << " ";
            case Console::WarningPrefix:
                return stream << Console::BRIGHT_RED << "[WARN]" << Console::DEFAULT_COLOR << " ";
            default:
                return stream;
        }
    }

}
}
