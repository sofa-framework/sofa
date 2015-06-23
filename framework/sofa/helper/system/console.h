#ifndef __HELPER_SYSTEM_console_H_
#define __HELPER_SYSTEM_console_H_


#include <sofa/helper/helper.h>
#include <string.h>
#include <iostream>
#include <sofa/helper/system/config.h>

namespace sofa {
namespace helper {




class SOFA_HELPER_API Console
{

    Console() {} // private constructor

public:

#ifdef WIN32
    typedef unsigned SystemColorType;
#else
    typedef std::string SystemColorType;
#endif

    /// this color type can be used with stream operator on any system
    struct ColorType
    {
        Console::SystemColorType value;
        ColorType( const ColorType& c ) : value(c.value) {}
        ColorType( const Console::SystemColorType& v ) : value(v) {}
        void operator= ( const ColorType& c ) { value=c.value; }
        void operator= ( const Console::SystemColorType& v ) { value=v; }
    };

    /// to use stream operator with a color on any system
    SOFA_HELPER_API friend std::ostream& operator<<(std::ostream &stream, ColorType color);

    static const ColorType BLUE;
    static const ColorType GREEN;
    static const ColorType CYAN;
    static const ColorType RED;
    static const ColorType PURPLE;
    static const ColorType YELLOW;
    static const ColorType WHITE;
    static const ColorType BLACK;
    static const ColorType BRIGHT_BLUE;
    static const ColorType BRIGHT_GREEN;
    static const ColorType BRIGHT_CYAN;
    static const ColorType BRIGHT_RED;
    static const ColorType BRIGHT_PURPLE;
    static const ColorType BRIGHT_YELLOW;
    static const ColorType BRIGHT_WHITE;
    static const ColorType BRIGHT_BLACK;
    static const ColorType DEFAULT_COLOR;

    /// standard [INFO] prefix
    static std::ostream& infoPrefix() { return ( std::cout << GREEN << "[INFO]" << DEFAULT_COLOR ); }
    /// standard [WARN] prefix
    static std::ostream& warningPrefix() { return ( std::cerr << RED << "[WARN]" << DEFAULT_COLOR );  }

};




}
}


#endif
