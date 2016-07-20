/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/BackTrace.h>

#if !defined(WIN32) && !defined(_XBOX) && !defined(PS3)
#include <signal.h>
#endif
#if !defined(WIN32) && !defined(_XBOX) && !defined(__APPLE__) && !defined(PS3)
#include <execinfo.h>
#include <unistd.h>
#endif
#if defined(__GNUC__) && !defined(PS3)
#include <cxxabi.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sofa
{

namespace helper
{

/// Dump current backtrace to stderr.
/// Currently only works on Linux. NOOP on other architectures.
void BackTrace::dump()
{
#if defined(__GNUC__) && !defined(__APPLE__) && !defined(WIN32) && !defined(_XBOX) && !defined(PS3)
    void *array[128];
    int size = backtrace(array, sizeof(array) / sizeof(array[0]));
    if (size > 0)
    {
        char** symbols = backtrace_symbols(array, size);
        if (symbols != NULL)
        {
            for (int i = 0; i < size; ++i)
            {
                char* symbol = symbols[i];

                // Decode the method's name to a more readable form if possible
                char *beginmangled = strrchr(symbol,'(');
                if (beginmangled != NULL)
                {
                    ++beginmangled;
                    char *endmangled = strrchr(beginmangled ,')');
                    if (endmangled != NULL)
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
                        if (realname != NULL)
                        {
                            free(name);
                            name = realname;
                        }
                        fprintf(stderr,"-> %.*s%s%s\n",(int)(beginmangled-symbol),symbol,name,endmangled);
                        free(name);
                    }
                    else
                        fprintf(stderr,"-> %s\n",symbol);
                }
                else
                    fprintf(stderr,"-> %s\n",symbol);
            }
            free(symbols);
        }
        else
        {
            backtrace_symbols_fd(array, size, STDERR_FILENO);
        }
    }
#endif
}

/// Enable dump of backtrace when a signal is received.
/// Useful to have information about crashes without starting a debugger (as it is not always easy to do, i.e. for parallel/distributed applications).
/// Currently only works on Linux. NOOP on other architectures
void BackTrace::autodump()
{
#if !defined(WIN32) && !defined(_XBOX) && !defined(PS3)
    signal(SIGSEGV, BackTrace::sig);
    signal(SIGILL, BackTrace::sig);
    signal(SIGFPE, BackTrace::sig);
    signal(SIGPIPE, BackTrace::sig);
    signal(SIGINT, BackTrace::sig);
    signal(SIGTERM, BackTrace::sig);
#endif
}

void BackTrace::sig(int sig)
{
#if !defined(WIN32) && !defined(_XBOX) && !defined(PS3)
    fprintf(stderr,"\n########## SIG %d ##########\n",sig);
    dump();
    signal(sig,SIG_DFL);
    raise(sig);
#else
    fprintf(stderr,"\nERROR: BackTrace::sig(%d) not supported.\n",sig);
#endif
}

} // namespace helper

} // namespace sofa

