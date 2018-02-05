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
#include <sofa/helper/system/config.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/BackTrace.h>
#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>

#include "mycuda.h"

#if defined(__GNUC__) && !defined(__APPLE__) && !defined(WIN32)

#define SOFA_GPU_DEBUG_STACK_TRACE 10

#ifdef SOFA_GPU_DEBUG_STACK_TRACE
#include <stdexcept>
#include <string>
#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>

class rtstack: public std::runtime_error
{
public:
    explicit rtstack(const std::string & __arg) : std::runtime_error(__arg + '\n' + __getStack()) {};

private:
    std::string __getStack() const throw()
    {
        // 100 = profondeur max de la pile qui sera affichee (arbitraire)
        // On recupere la pile d'appels et les symboles associes
        // ATTENTION Utiliser le switch -rdynamic sinon on n'a pas les symboles

        void *traces[100];
        size_t size    = backtrace (traces, 100);
        char** symbols = backtrace_symbols (traces, size);

        std::string stacktrace;
        int nbdisp = 0;
        bool startSofa = false;
        for (size_t i = 0; i < size; i++)
        {
//            stacktrace += symbols[i];

            // On a typiquemenet des chaines de cette sorte: ./a.out(_Z1ev+0x180) [0x401704]
            // On recherche les caracteres ( et +
            std::string fct_name;
            char *begin=NULL, *end=NULL;

            for (char *j=symbols[i]; *j; ++j)
            {
                if (*j=='(') begin = j;
                else if (*j=='+') end = j;
            }

            // si ( et + ont ete trouves
            if (begin && end)
            {
                *begin++ = 0;
                *end     = 0;

                // on a trouve le nom "mangled", on va maintenant le demangler
                int status;
                char *ret = abi::__cxa_demangle(begin,NULL,NULL,&status);

                // Si ret != 0, on a demangle qq chose
                if (ret)
                {
                    fct_name = ret;
                    free(ret);
                    ret = NULL;
                }
                else     // demangling failed, just pretend it's a C function with no args
                {
                    fct_name = begin;
                    fct_name += "::()";
                }

                if (!startSofa && std::string::npos != fct_name.find(std::string("displayStack")))
                {
                    startSofa = true; // wait to see displayStack() before printing the stack
                }
                else if (startSofa && std::string::npos != fct_name.find(std::string("::")))
                {
                    stacktrace += fct_name;
                    stacktrace += '\n';
                    nbdisp++;
                    if (nbdisp>=SOFA_GPU_DEBUG_STACK_TRACE)
                    {
                        free (symbols);
                        return stacktrace;
                    }
                }
            }
        }
        free (symbols);
        return stacktrace;
    }
};
#endif
#endif

namespace sofa
{
namespace gpu
{
namespace cuda
{

SOFA_LINK_CLASS(CudaBoxROI)
SOFA_LINK_CLASS(CudaFixedConstraint)
SOFA_LINK_CLASS(CudaMechanicalObject)
//SOFA_LINK_CLASS(CudaSpringForceField)
SOFA_LINK_CLASS(CudaUniformMass)
SOFA_LINK_CLASS(CudaDiagonalMass)
SOFA_LINK_CLASS(CudaPlaneForceField)
SOFA_LINK_CLASS(CudaSphereForceField)
SOFA_LINK_CLASS(CudaEllipsoidForceField)
SOFA_LINK_CLASS(CudaIdentityMapping)
SOFA_LINK_CLASS(CudaBarycentricMapping)
SOFA_LINK_CLASS(CudaRigidMapping)
SOFA_LINK_CLASS(CudaSubsetMapping)
#ifdef SOFA_HAVE_MINIFLOWVR
SOFA_LINK_CLASS(CudaDistanceGridCollisionModel)
SOFA_LINK_CLASS(CudaCollisionDetection)
#endif
SOFA_LINK_CLASS(CudaTetrahedronFEMForceField)
SOFA_LINK_CLASS(CudaMouseInteractor)
//SOFA_LINK_CLASS(CudaCollision)
SOFA_LINK_CLASS(CudaPointModel)
SOFA_LINK_CLASS(CudaSphereModel)
SOFA_LINK_CLASS(CudaSetTopology)
SOFA_LINK_CLASS(CudaVisualModel)

#ifdef SOFACUDA_ENABLE_VOLUMETRICRENDERING
SOFA_LINK_CLASS(CudaOglTetrahedralModel)
#endif // SOFACUDA_ENABLE_VOLUMETRICRENDERING

extern "C"
{
    MycudaVerboseLevel mycudaVerboseLevel = LOG_ERR;
//MycudaVerboseLevel mycudaVerboseLevel = LOG_INFO;
//MycudaVerboseLevel mycudaVerboseLevel = LOG_TRACE;
}

static void timerSyncCB(void*)
{
    mycudaThreadSynchronize();
}

void mycudaPrivateInit(int /*device*/)
{
    const char* sync = getenv("CUDA_TIMER_SYNC");
    if (!sync || !*sync || atoi(sync))
        sofa::helper::AdvancedTimer::setSyncCallBack(timerSyncCB, NULL);
    const char* verbose = getenv("CUDA_VERBOSE");
    if (verbose && *verbose)
        mycudaVerboseLevel = (MycudaVerboseLevel) atoi(verbose);
}

void mycudaLogError(const char* err, const char* src)
{
    std::cerr << "CUDA error: "<< err <<" returned from "<< src <<".\n";
    sofa::helper::BackTrace::dump();
    assert(0);
    exit(1);
}

int mycudaPrintf(const char* fmt, ...)
{
    va_list args;
    va_start( args, fmt );
    int r = vfprintf( stderr, fmt, args );
    va_end( args );
    return r;
}

const char* mygetenv(const char* name)
{
    return getenv(name);
}

#if defined(SOFA_GPU_DEBUG_STACK_TRACE) && defined(__GNUC__) && !defined(__APPLE__) && !defined(WIN32)
void displayStack(const char * name)
{
    try
    {
        throw (rtstack(name));
    }
    catch (std::exception &exc)
    {
        std::cerr << exc.what();
    };
}
#else
void displayStack(const char * /*name*/)
{
}
#endif


} // namespace cuda
} // namespace gpu
} // namespace sofa
