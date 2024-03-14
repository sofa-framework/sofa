/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>

#ifndef WIN32

#include <unistd.h>

#ifdef __linux__
#include <sys/ioctl.h>
#endif

#else

#if defined(_WIN32_WINNT) && (_WIN32_WINNT < 0x0600)
#error                                                                         \
  "Please include this before any windows system headers or set _WIN32_WINNT at least to _WIN32_WINNT_VISTA"
#elif !defined(_WIN32_WINNT)
#define _WIN32_WINNT _WIN32_WINNT_VISTA
#endif

#include <windows.h>
#include <io.h>
#include <memory>

// Only defined in windows 10 onwards, redefining in lower windows since it
// doesn't gets used in lower versions
// https://docs.microsoft.com/en-us/windows/console/getconsolemode
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

#endif

#include <sofa/helper/config.h>

namespace sofa {
namespace helper {

/**
 * Utility that manages the output style of a stream into a terminal. It is based heavily on the work
 * of https://github.com/agauniyal/rang
 */
namespace console {

enum class Status {  // Toggle the status of the output style
    Auto = 0, // (Default) automatically detects whether the terminal supports styled output
    On   = 1,
    Off  = 2
};

#ifdef WIN32
enum class Mode {  // Windows Terminal Mode
    Auto   = 0, // (Default) automatically detects whether Ansi or Native API
    Ansi   = 1, // Force use Ansi API
    Native = 2  // Force use Native API
};
#endif

enum class Style {
    Reset     = 0,
    Bold      = 1,
    Dim       = 2,
    Italic    = 3,
    Underline = 4,
    Blink     = 5,
    Rblink    = 6,
    Reversed  = 7,
    Conceal   = 8,
    Crossed   = 9
};

struct Foreground {
    enum class Normal {
        Black   = 30,
        Red     = 31,
        Green   = 32,
        Yellow  = 33,
        Blue    = 34,
        Magenta = 35,
        Cyan    = 36,
        Gray    = 37,
        Reset   = 39
    };

    enum class Bright {
        Black   = 90,
        Red     = 91,
        Green   = 92,
        Yellow  = 93,
        Blue    = 94,
        Magenta = 95,
        Cyan    = 96,
        Gray    = 97
    };
};

struct Background {
    enum class Normal {
        Black   = 40,
        Red     = 41,
        Green   = 42,
        Yellow  = 43,
        Blue    = 44,
        Magenta = 45,
        Cyan    = 46,
        Gray    = 47,
        Reset   = 49
    };

    enum class Bright {
        Black   = 100,
        Red     = 101,
        Green   = 102,
        Yellow  = 103,
        Blue    = 104,
        Magenta = 105,
        Cyan    = 106,
        Gray    = 107
    };
};

template<typename T>
using enableStd = typename std::enable_if<
    std::is_same<T, sofa::helper::console::Style>::value ||
    std::is_same<T, sofa::helper::console::Foreground::Normal>::value ||
    std::is_same<T, sofa::helper::console::Foreground::Bright>::value ||
    std::is_same<T, sofa::helper::console::Background::Normal>::value ||
    std::is_same<T, sofa::helper::console::Background::Bright>::value,
    std::ostream &>::type;

#define __CONSOLE_INTERNAL__
#include <sofa/helper/system/console_internal.h>
#undef __CONSOLE_INTERNAL__

/// Enable or disable colors in stdout / stderr.
///
/// This controls whether using styled values in streams will actually do
/// anything.  Passing Auto means that styled output will be used for the stream
/// only if it hasn't been redirected (on Unix only).
/// By default, colors are enabled if supported (auto).
SOFA_HELPER_API void setStatus(Status status) noexcept ;
SOFA_HELPER_API Status getStatus() noexcept ;

SOFA_HELPER_API size_t getColumnCount() ;

/// to use stream operator with a styled output on any system
template <typename T>
inline enableStd<T> operator<<(std::ostream &os, const T & value)
{
    const Status status = internal::get_status();
    switch (status) {
        case sofa::helper::console::Status ::Auto:
            return sofa::helper::console::internal::supportsColor()
                   && sofa::helper::console::internal::isTerminal(os.rdbuf())
                   ? sofa::helper::console::internal::setColor(os, value)
                   : os;
        case sofa::helper::console::Status::On : return sofa::helper::console::internal::setColor(os, value);
        default: return os;
    }
}

} // namespace console

} // namespace helper
} // namespace sofa

#endif
