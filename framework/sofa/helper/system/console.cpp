#include "console.h"



namespace sofa {
namespace helper {


#ifdef WIN32

    const Console::ColorType Console::BLUE   = Console::ColorType(9);
    const Console::ColorType Console::GREEN  = Console::ColorType(10);
    const Console::ColorType Console::CYAN   = Console::ColorType(11);
    const Console::ColorType Console::RED    = Console::ColorType(12);
    const Console::ColorType Console::PURPLE = Console::ColorType(13);
    const Console::ColorType Console::YELLOW = Console::ColorType(14);
    const Console::ColorType Console::WHITE  = Console::ColorType(15);
    const Console::ColorType Console::BLACK  = Console::ColorType(0);
    Console::ColorType Console::DEFAULT_COLOR = Console::BLACK;

    HANDLE Console::s_console;

    std::ostream& operator<<( std::ostream& stream, Console::ColorType color )
    {
        Console::init();

        SetConsoleTextAttribute( Console::s_console, color.value );
        return stream;
    }

#else

    const Console::ColorType Console::BLUE   = Console::ColorType("\033[1;34m ");
    const Console::ColorType Console::GREEN  = Console::ColorType("\033[1;32m ");
    const Console::ColorType Console::CYAN   = Console::ColorType("\033[1;36m ");
    const Console::ColorType Console::RED    = Console::ColorType("\033[1;31m ");
    const Console::ColorType Console::PURPLE = Console::ColorType("\033[1;35m ");
    const Console::ColorType Console::YELLOW = Console::ColorType("\033[1;33m ");
    const Console::ColorType Console::WHITE  = Console::ColorType("\033[1;37m ");
    const Console::ColorType Console::BLACK  = Console::ColorType(" \033[0m");
    Console::ColorType Console::DEFAULT_COLOR = Console::BLACK;


    std::ostream& operator<<( std::ostream& stream, Console::ColorType color )
    {
        return ( stream << color.value );
    }

#endif






}
}
