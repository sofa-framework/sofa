#include "console.h"
#include "sofa/helper/Utils.h"



namespace sofa {
namespace helper {


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

    SOFA_HELPER_API std::ostream& operator<<( std::ostream& stream, Console::ColorType color )
    {
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


    std::ostream& operator<<( std::ostream& stream, Console::ColorType color )
    {
        return ( stream << color.value );
    }

#endif






}
}
