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

#ifndef __HELPER_SYSTEM_console_internal_H_
#define __HELPER_SYSTEM_console_internal_H_

#ifndef __CONSOLE_INTERNAL__
#error The console internal header must be exclusively included inside console.h and nowhere else.
#endif

namespace internal {

#ifdef WIN32

inline bool isMsysPty(int fd) noexcept
{
    // Dynamic load for binary compability with old Windows
    const auto ptrGetFileInformationByHandleEx
      = reinterpret_cast<decltype(&GetFileInformationByHandleEx)>(
        GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")),
                       "GetFileInformationByHandleEx"));
    if (!ptrGetFileInformationByHandleEx) {
        return false;
    }

    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }

    // Check that it's a pipe:
    if (GetFileType(h) != FILE_TYPE_PIPE) {
        return false;
    }

    // POD type is binary compatible with FILE_NAME_INFO from WinBase.h
    // It have the same alignment and used to avoid UB in caller code
    struct MY_FILE_NAME_INFO {
        DWORD FileNameLength;
        WCHAR FileName[MAX_PATH];
    };

    const auto pNameInfo = std::unique_ptr<MY_FILE_NAME_INFO>(
      new (std::nothrow) MY_FILE_NAME_INFO());
    if (!pNameInfo) {
        return false;
    }

    // Check pipe name is template of
    // {"cygwin-","msys-"}XXXXXXXXXXXXXXX-ptyX-XX
    if (!ptrGetFileInformationByHandleEx(h, FileNameInfo, pNameInfo.get(),
                                         sizeof(MY_FILE_NAME_INFO))) {
        return false;
    }
    const std::wstring name(pNameInfo->FileName, pNameInfo->FileNameLength / sizeof(WCHAR));
    if ((name.find(L"msys-") == std::wstring::npos
         && name.find(L"cygwin-") == std::wstring::npos)
        || name.find(L"-pty") == std::wstring::npos) {
        return false;
    }

    return true;
}

struct SGR {  // Select Graphic Rendition parameters for Windows console
    BYTE fgColor;  // foreground color (0-15) lower 3 rgb bits + intense bit
    BYTE bgColor;  // background color (0-15) lower 3 rgb bits + intense bit
    BYTE bold;  // emulated as FOREGROUND_INTENSITY bit
    BYTE underline;  // emulated as BACKGROUND_INTENSITY bit
    BOOLEAN inverse;  // swap foreground/bold & background/underline
    BOOLEAN conceal;  // set foreground/bold to background/underline
};

enum class AttrColor : BYTE {  // Color attributes for console screen buffer
    black   = 0,
    red     = 4,
    green   = 2,
    yellow  = 6,
    blue    = 1,
    magenta = 5,
    cyan    = 3,
    gray    = 7
};

inline HANDLE getConsoleHandle(const std::streambuf *osbuf) noexcept
{
    if (osbuf == std::cout.rdbuf()) {
        static const HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        return hStdout;
    } else if (osbuf == std::cerr.rdbuf() || osbuf == std::clog.rdbuf()) {
        static const HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
        return hStderr;
    }
    return INVALID_HANDLE_VALUE;
}

inline bool setWinTermAnsiColors(const std::streambuf *osbuf) noexcept
{
    const HANDLE h = getConsoleHandle(osbuf);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    DWORD dwMode = 0;
    if (!GetConsoleMode(h, &dwMode)) {
        return false;
    }
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (!SetConsoleMode(h, dwMode)) {
        return false;
    }
    return true;
}

inline bool supportsAnsi(const std::streambuf *osbuf) noexcept
{
    using std::cerr;
    using std::clog;
    using std::cout;
    if (osbuf == cout.rdbuf()) {
        static const bool cout_ansi
          = (isMsysPty(_fileno(stdout)) || setWinTermAnsiColors(osbuf));
        return cout_ansi;
    } else if (osbuf == cerr.rdbuf() || osbuf == clog.rdbuf()) {
        static const bool cerr_ansi
          = (isMsysPty(_fileno(stderr)) || setWinTermAnsiColors(osbuf));
        return cerr_ansi;
    }
    return false;
}

inline const SGR &defaultState() noexcept
{
    static const SGR defaultSgr = []() -> SGR {
        CONSOLE_SCREEN_BUFFER_INFO info;
        WORD attrib = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),
                                       &info)
            || GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE),
                                          &info)) {
            attrib = info.wAttributes;
        }
        SGR sgr     = { 0, 0, 0, 0, FALSE, FALSE };
        sgr.fgColor = attrib & 0x0F;
        sgr.bgColor = (attrib & 0xF0) >> 4;
        return sgr;
    }();
    return defaultSgr;
}

inline BYTE ansi2attr(BYTE rgb) noexcept
{
    static const AttrColor rev[8]
      = { AttrColor::black,  AttrColor::red,  AttrColor::green,
          AttrColor::yellow, AttrColor::blue, AttrColor::magenta,
          AttrColor::cyan,   AttrColor::gray };
    return static_cast<BYTE>(rev[rgb]);
}

inline void setWinSGR(Background::Normal color, SGR &state) noexcept
{
    if (color != Background::Normal::Reset) {
        state.bgColor = ansi2attr(static_cast<BYTE>(color) - 40);
    } else {
        state.bgColor = defaultState().bgColor;
    }
}

inline void setWinSGR(Foreground::Normal color, SGR &state) noexcept
{
    if (color != Foreground::Normal::Reset) {
        state.fgColor = ansi2attr(static_cast<BYTE>(color) - 30);
    } else {
        state.fgColor = defaultState().fgColor;
    }
}

inline void setWinSGR(Background::Bright color, SGR &state) noexcept
{
    state.bgColor = (BACKGROUND_INTENSITY >> 4)
      | ansi2attr(static_cast<BYTE>(color) - 100);
}

inline void setWinSGR(Foreground::Bright color, SGR &state) noexcept
{
    state.fgColor
      = FOREGROUND_INTENSITY | ansi2attr(static_cast<BYTE>(color) - 90);
}

inline void setWinSGR(Style style, SGR &state) noexcept
{
    switch (style) {
        case Style::Reset: state = defaultState(); break;
        case Style::Bold: state.bold = FOREGROUND_INTENSITY; break;
        case Style::Underline:
        case Style::Blink:
            state.underline = BACKGROUND_INTENSITY;
            break;
        case Style::Reversed: state.inverse = TRUE; break;
        case Style::Conceal: state.conceal = TRUE; break;
        default: break;
    }
}

inline SGR &current_state() noexcept
{
    static SGR state = defaultState();
    return state;
}

inline WORD SGR2Attr(const SGR &state) noexcept
{
    WORD attrib = 0;
    if (state.conceal) {
        if (state.inverse) {
            attrib = (state.fgColor << 4) | state.fgColor;
            if (state.bold)
                attrib |= FOREGROUND_INTENSITY | BACKGROUND_INTENSITY;
        } else {
            attrib = (state.bgColor << 4) | state.bgColor;
            if (state.underline)
                attrib |= FOREGROUND_INTENSITY | BACKGROUND_INTENSITY;
        }
    } else if (state.inverse) {
        attrib = (state.fgColor << 4) | state.bgColor;
        if (state.bold) attrib |= BACKGROUND_INTENSITY;
        if (state.underline) attrib |= FOREGROUND_INTENSITY;
    } else {
        attrib = state.fgColor | (state.bgColor << 4) | state.bold
          | state.underline;
    }
    return attrib;
}

template <typename T>
inline void setWinColorAnsi(std::ostream &os, T const value)
{
    os << "\033[" << static_cast<int>(value) << "m";
}

template <typename T>
inline void setWinColorNative(std::ostream &os, T const value)
{
    const HANDLE h = getConsoleHandle(os.rdbuf());
    if (h != INVALID_HANDLE_VALUE) {
        setWinSGR(value, current_state());
        // Out all buffered text to console with previous settings:
        os.flush();
        SetConsoleTextAttribute(h, SGR2Attr(current_state()));
    }
}

#endif

inline std::atomic<Status> &get_status() noexcept
{
    static std::atomic<Status > status(Status ::Auto);
    return status;
}

#ifdef WIN32
inline std::atomic<Mode> &get_mode() noexcept
{
    static std::atomic<Mode> mode(Mode::Auto);
    return mode;
}
#endif

inline bool supportsColor() noexcept
{
#ifndef WIN32

    static const bool result = [] {
        const char *Terms[]
            = { "ansi",    "color",  "console", "cygwin", "gnome",
                "konsole", "kterm",  "linux",   "msys",   "putty",
                "rxvt",    "screen", "vt100",   "xterm" };

        const char *env_p = std::getenv("TERM");
        if (env_p == nullptr) {
            return false;
        }
        return std::any_of(std::begin(Terms), std::end(Terms),
                           [&](const char *term) {
                               return std::strstr(env_p, term) != nullptr;
                           });
    }();

#else
    // All windows versions support colors through native console methods
        static constexpr bool result = true;
#endif
    return result;
}

inline bool isTerminal(const std::streambuf *osbuf) noexcept
{
    using std::cerr;
    using std::clog;
    using std::cout;

#ifndef WIN32
    if (osbuf == cout.rdbuf()) {
        static const bool cout_term = isatty(fileno(stdout)) != 0;
        return cout_term;
    } else if (osbuf == cerr.rdbuf() || osbuf == clog.rdbuf()) {
        static const bool cerr_term = isatty(fileno(stderr)) != 0;
        return cerr_term;
    }
#else
    if (osbuf == cout.rdbuf()) {
            static const bool cout_term
              = (_isatty(_fileno(stdout)) || isMsysPty(_fileno(stdout)));
            return cout_term;
        } else if (osbuf == cerr.rdbuf() || osbuf == clog.rdbuf()) {
            static const bool cerr_term
              = (_isatty(_fileno(stderr)) || isMsysPty(_fileno(stderr)));
            return cerr_term;
        }
#endif
    return false;
}

#ifndef WIN32

template <typename T>
inline enableStd<T> setColor(std::ostream &os, T const value)
{
    return os << "\033[" << static_cast<int>(value) << "m";
}

#else

template <typename T>
inline enableStd<T> setColor(std::ostream &os, T const value)
{
    if (get_mode() == Mode::Auto) {
        if (supportsAnsi(os.rdbuf())) {
            setWinColorAnsi(os, value);
        } else {
            setWinColorNative(os, value);
        }
    } else if (get_mode() == Mode::Ansi) {
        setWinColorAnsi(os, value);
    } else {
        setWinColorNative(os, value);
    }
    return os;
}

#endif

} // namespace internal

#endif //__HELPER_SYSTEM_console_internal_H_
