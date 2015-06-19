#ifndef __HELPER_SYSTEM_console_H_
#define __HELPER_SYSTEM_console_H_

#include <string.h>
#include <iostream>
#include <sofa/helper/system/config.h>

namespace sofa {
namespace helper {




class Console
{

    Console() {} // private constructor


#ifdef WIN32

    /// this color type can be used with stream operator on windows
    typedef unsigned ColorType;

    /// windows console HANDLE
    static HANDLE s_console;

    /// @internal windows needs to get HANDLES
    static void init()
    {
        if( s_console == INVALID_HANDLE_VALUE )
        {
            s_console = GetStdHandle(STD_OUTPUT_HANDLE);
            CONSOLE_SCREEN_BUFFER_INFO currentInfo
            GetConsoleScreenBufferInfo(s_console, &currentInfo);
            DEFAULT_COLOR = currentInfo.wAttributes;
        }
    }

#else

    /// this color type can be used with stream operator on POSIX
    typedef std::string ColorType;

#endif

    /// to use stream operator with a color on any system
    friend std::ostream& operator<<(std::ostream &stream, ColorType color);

public:

    static const ColorType BLUE;
    static const ColorType GREEN;
    static const ColorType CYAN;
    static const ColorType RED;
    static const ColorType PURPLE;
    static const ColorType YELLOW;
    static const ColorType WHITE;
    static const ColorType BLACK;
    static ColorType DEFAULT_COLOR;

    /// standard [INFO] prefix
    static std::ostream& infoPrefix() { return ( std::cout << GREEN << "[INFO]" << DEFAULT_COLOR ); }
    /// standard [WARN] prefix
    static std::ostream& warningPrefix() { return ( std::cerr << RED << "[WARN]" << DEFAULT_COLOR );  }

};




}
}


#endif
