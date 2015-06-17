#ifndef __HELPER_SYSTEM_console_H_
#define __HELPER_SYSTEM_console_H_

#include <string.h>
#include <iostream>
#include <sofa/helper/system/config.h>

namespace sofa {
namespace helper {




class Console
{

    Console(); // private constructor for singleton

    static Console& getInstance(); // private singleton

#ifdef WIN32

    typedef unsigned ColorType;

    static const HANDLE s_console;
    static const ColorType s_defaultColor;

    std::ostream& coloredMessageImpl( std::ostream& stream, const std::string& msg, ColorType color )
    {
        SetConsoleTextAttribute( s_console, color );
        stream << prefix;
        SetConsoleTextAttribute( s_console, s_defaultColor );
    }


#else

    typedef std::string ColorType;

    inline std::ostream& coloredMessageImpl( std::ostream& stream, const std::string& msg, ColorType color )
    {
        return ( stream << color << msg << BLACK );
    }

#endif



public:

    static const ColorType BLUE;
    static const ColorType GREEN;
    static const ColorType CYAN;
    static const ColorType RED;
    static const ColorType PURPLE;
    static const ColorType YELLOW;
    static const ColorType WHITE;
    static const ColorType BLACK;


    static std::ostream& coloredMessage( std::ostream& stream, const std::string& msg, ColorType color )
    {
        return getInstance().coloredMessageImpl( stream, msg, color );
    }

    static std::ostream& infoPrefix() { return coloredMessage( std::cout, "[INFO]", GREEN ); }
    static std::ostream& warningPrefix() { return coloredMessage( std::cout, "[WARN]", RED ); }


};





}
}


#endif
