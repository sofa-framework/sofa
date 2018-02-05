/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

    /// @brief Initialize Console.
    ///
    /// Enable or disable colors based on the value of the SOFA_COLOR_TERMINAL
    /// environnement variable (possible values: yes, no, auto).
    static void init();

#ifdef WIN32
    typedef unsigned SystemColorType;
    typedef unsigned SystemCodeType;
#else
    typedef std::string SystemColorType;
    typedef std::string SystemCodeType;
#endif

    /// this color type can be used with stream operator on any system
    struct ColorType
    {
        Console::SystemColorType value;
        ColorType() : value(DEFAULT_COLOR.value) {}
        ColorType( const ColorType& c ) : value(c.value) {}
        ColorType( const Console::SystemColorType& v ) : value(v) {}
        void operator= ( const ColorType& c ) { value=c.value; }
        void operator= ( const Console::SystemColorType& v ) { value=v; }
    };

    struct CodeType
    {
        Console::SystemCodeType value;
        CodeType() : value(DEFAULT_CODE.value) {}
        CodeType( const CodeType& c ) : value(c.value) {}
        CodeType( const Console::SystemCodeType& v ) : value(v) {}
        void operator= ( const CodeType& c ) { value=c.value; }
        void operator= ( const Console::SystemCodeType& v ) { value=v; }
    };

    /// to use stream operator with a color on any system
    SOFA_HELPER_API friend std::ostream& operator<<(std::ostream &stream, ColorType color);
    SOFA_HELPER_API friend std::ostream& operator<<(std::ostream &stream, CodeType color);

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

    static const CodeType ITALIC;
    static const CodeType UNDERLINE;
    static const CodeType DEFAULT_CODE;

    enum ColorsStatus {ColorsEnabled, ColorsDisabled, ColorsAuto};
    /// Enable or disable colors in stdout / stderr.
    ///
    /// This controls whether using ColorType values in streams will actually do
    /// anything.  Passing ColorsAuto means that colors will be used for stdout
    /// only if it hasn't been redirected (on Unix only). Same thing for stderr.
    /// By default, colors are disabled.
    static void setColorsStatus(ColorsStatus status);
    static ColorsStatus getColorsStatus();

    static size_t getColumnCount() ;

private:
    static ColorsStatus s_colorsStatus;
    /// Internal helper function that determines whether colors should be used.
    static bool shouldUseColors(std::ostream& stream);
};

}
}


#endif
