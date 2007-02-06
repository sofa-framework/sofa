#ifndef SOFA_COMPONENTS_COMMON_BACKTRACE_H
#define SOFA_COMPONENTS_COMMON_BACKTRACE_H

namespace Sofa
{

namespace Components
{

namespace Common
{

class BackTrace
{
public:
    /// Dump current backtrace to stderr.
    /// Currently only works on Linux. NOOP on other architectures.
    static void dump();

    /// Enable dump of backtrace when a signal is received.
    /// Useful to have information about crashes without starting a debugger (as it is not always easy to do, i.e. for parallel/distributed applications).
    /// Currently only works on Linux. NOOP on other architectures
    static void autodump();

protected:

    /// Callback for signals
    static void sig(int sig);
};

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
