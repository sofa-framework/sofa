#include "BackTrace.h"

#if !defined(WIN32)
#include <signal.h>
#endif
#if !defined(WIN32) && !defined(__APPLE__)
#include <execinfo.h>
#include <unistd.h>
#endif
#ifdef __GNUC__
#include <cxxabi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace Sofa
{

namespace Components
{

namespace Common
{

/// Dump current backtrace to stderr.
/// Currently only works on Linux. NOOP on other architectures.
void BackTrace::dump()
{
#if defined(__GNUC__) && !defined(__APPLE__) && !defined(WIN32)
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
                        fprintf(stderr,"-> %.*s%s%s\n",beginmangled-symbol,symbol,name,endmangled);
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
#if !defined(WIN32)
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
#if !defined(WIN32)
    fprintf(stderr,"\n########## SIG %d ##########\n",sig);
    dump();
    signal(sig,SIG_DFL);
    raise(sig);
#endif
}

} // namespace Common

} // namespace Components

} // namespace Sofa
