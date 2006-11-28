#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>

#include "mycuda.h"

namespace Sofa
{
namespace Contrib
{
namespace CUDA
{

void mycudaLogError(int err, const char* src)
{
    std::cerr << "CUDA: Error "<<err<<" returned from "<<src<<".\n";
}

int myprintf(const char* fmt, ...)
{
    va_list args;
    va_start( args, fmt );
    int r = vprintf( fmt, args );
    va_end( args );
    return r;
}

} // namespace CUDA
} // namespace Contrib
} // namespace Sofa
