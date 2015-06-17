#include "console.h"



namespace sofa {
namespace helper {


#ifdef WIN32

    typedef unsigned ColorType;

    const Console::ColorType Console::BLUE = 9;
    const Console::ColorType Console::GREEN = 10;
    const Console::ColorType Console::CYAN = 11;
    const Console::ColorType Console::RED = 12;
    const Console::ColorType Console::PURPLE = 13;
    const Console::ColorType Console::YELLOW = 14;
    const Console::ColorType Console::WHITE = 15;
    const Console::ColorType Console::BLACK = 0;

#else

    typedef std::string ColorType;

    const Console::ColorType Console::BLUE = "\033[1;34m ";
    const Console::ColorType Console::GREEN = "\033[1;32m ";
    const Console::ColorType Console::CYAN = "\033[1;36m ";
    const Console::ColorType Console::RED = "\033[1;31m ";
    const Console::ColorType Console::PURPLE = "\033[1;35m ";
    const Console::ColorType Console::YELLOW = "\033[1;33m ";
    const Console::ColorType Console::WHITE = "\033[1;37m ";
    const Console::ColorType Console::BLACK = " \033[0m";

#endif


    Console::Console()
    {
#ifdef WIN32
        s_console = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO currentInfo
        GetConsoleScreenBufferInfo(s_console, &currentInfo);
        s_defaultColor = currentInfo.wAttributes;
#endif
    }

    Console& Console::getInstance()
    {
        static Console instance;
        return instance;
    }



}
}
