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
#include <sofa/helper/BackTrace.h>

#include <csignal>
#if !defined(WIN32) && !defined(__APPLE__)
#include <execinfo.h>
#include <unistd.h>
#endif
#if defined(WIN32)
#include "windows.h"
#include "DbgHelp.h"
#pragma comment(lib, "Dbghelp.lib")
#endif
#if defined(__GNUC__)
#include <cxxabi.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <string>


namespace sofa::helper
{

BackTrace::StackTrace BackTrace::getTrace(size_t maxEntries)
{
    BackTrace::StackTrace result;

#if defined(__GNUC__) && !defined(__APPLE__) && !defined(WIN32)
    void *array[128];
    int size = backtrace(array, sizeof(array) / sizeof(array[0]));
    if (size > 0)
    {
        char** symbols = backtrace_symbols(array, size);
        if (symbols != nullptr)
        {
            for (size_t i = 1; i < std::min(size_t(size), maxEntries); ++i)
            {
                char* symbol = symbols[i];
                // Decode the method's name to a more readable form if possible
                char *beginmangled = strrchr(symbol,'(');
                if (beginmangled != nullptr)
                {
                    beginmangled++;
                    char *endmangled = strrchr(beginmangled ,')');
                    if (endmangled != nullptr)
                    {
                        // remove +0x[0-9a-fA-f]* suffix
                        char* savedend = endmangled;
                        while((endmangled[-1]>='0' && endmangled[-1]<='9') ||
                              (endmangled[-1]>='a' && endmangled[-1]<='f') ||
                              (endmangled[-1]>='A' && endmangled[-1]<='F'))
                            --endmangled;
                        if (endmangled[-1]=='x' && endmangled[-2]=='0' && endmangled[-3]=='+')
                            endmangled -= 3;
                        else
                            endmangled = savedend; // suffix not found
                        char* name = (char*)malloc(endmangled-beginmangled+1);
                        memcpy(name, beginmangled, endmangled-beginmangled);
                        name[endmangled-beginmangled] = '\0';
                        int status;
                        char* realname = abi::__cxa_demangle(name, 0, 0, &status);
                        if (realname != nullptr)
                        {
                            free(name);
                            name = realname;
                        }
                        if(strlen(name) != 0)
                            result.push_back(std::string(name));
                        free(name);
                    }
                }
            }
            free(symbols);
        }
        else
        {
            backtrace_symbols_fd(array, size, STDERR_FILENO);
        }
    }

#elif !defined(__GNUC__) && !defined(__APPLE__) && defined(WIN32)
    unsigned int   i;
    void         * stack[100];
    unsigned short frames;
    SYMBOL_INFO  * symbol;
    HANDLE         process;

    process = GetCurrentProcess();

    SymInitialize(process, nullptr, TRUE);

    frames = CaptureStackBackTrace(0, 100, stack, nullptr);
    symbol = (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (i = 0; i < frames; i++)
    {
        SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
        result.push_back(symbol->Name);
    }

    free(symbol);
#endif

    return result;
}

/// Dump current backtrace to stderr.
/// Currently only works on Linux. NOOP on other architectures.
void BackTrace::dump()
{
    auto stacktrace = getTrace();
    if(stacktrace.empty())
        return;

    // iterates over all the traces skipping the first...which is containing the call to "dump()"
    for(auto it = stacktrace.begin()+1; it != stacktrace.end(); ++it)
    {
        std::cerr << "  " << *it << std::endl;
    }
}

/// Enable dump of backtrace when a signal is received.
/// Useful to have information about crashes without starting a debugger (as it is not always easy to do, i.e. for parallel/distributed applications).
/// Currently only works on Linux. NOOP on other architectures
void BackTrace::autodump()
{
    signal(SIGABRT, BackTrace::sig);
    signal(SIGSEGV, BackTrace::sig);
    signal(SIGILL, BackTrace::sig);
    signal(SIGFPE, BackTrace::sig);
    signal(SIGINT, BackTrace::sig);
    signal(SIGTERM, BackTrace::sig);
#if !defined(WIN32)
    signal(SIGPIPE, BackTrace::sig);
#endif
}

static std::string SigDescription(int sig)
{
    switch (sig)
    {
    case SIGABRT:
        return "SIGABRT: usually caused by an abort() or assert()";
        break;
    case SIGFPE:
        return "SIGFPE: arithmetic exception, such as divide by zero";
        break;
    case SIGILL:
        return "SIGILL: illegal instruction";
        break;
    case SIGINT:
        return "SIGINT: interactive attention signal, probably a ctrl+c";
        break;
    case SIGSEGV:
        return "SIGSEGV: segfault";
        break;
    case SIGTERM:
    default:
        return "SIGTERM: a termination request was sent to the program";
        break;
    }

    return "Unknown signal";
}

void BackTrace::sig(int sig)
{
    std::cerr << std::endl << "########## SIG " << sig << " - " << SigDescription(sig) << " ##########" << std::endl;
    dump();
    signal(sig,SIG_DFL);
    raise(sig);
}

} // namespace sofa::helper



