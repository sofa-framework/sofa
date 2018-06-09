/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifdef WIN32
    typedef unsigned SystemCodeType;
#else
    typedef std::string SystemCodeType;
#endif

    enum Style {
        DEFAULT,
        BLUE,
        GREEN,
        CYAN,
        RED,
        PURPLE,
        YELLOW,
        WHITE,
        BLACK,
        BRIGHT_BLUE,
        BRIGHT_GREEN,
        BRIGHT_CYAN,
        BRIGHT_RED,
        BRIGHT_PURPLE,
        BRIGHT_YELLOW,
        BRIGHT_WHITE,
        BRIGHT_BLACK,
        ITALIC,
        UNDERLINE
    };

    enum ColorsStatus {ColorsEnabled, ColorsDisabled, ColorsAuto};

    /// @brief Initialize Console.
    ///
    /// Enable or disable colors based on the value of the SOFA_COLOR_TERMINAL
    /// environnement variable (possible values: yes, no, auto).
    static void init();

    /// @brief Get the console code for a given style format
    static SystemCodeType Code(Style s);

    /// to use stream operator with a color on any system
    SOFA_HELPER_API friend std::ostream& operator<<(std::ostream &stream, const SystemCodeType & color);

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
