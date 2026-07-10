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
#define SOFACHOLMOD_EIGENCHOLMODSUPERNODALLLT_CPP

#include <SofaCHOLMOD/config.h>

#include <Eigen/CholmodSupport>
#include <SofaCHOLMOD/EigenCholmodSupernodalLLT.h>
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.inl>

#include <sofa/core/ObjectFactory.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace sofacholmod
{

// CHOLMOD only supports double precision
static_assert(std::is_same_v<SReal, double>, "EigenCholmodSupernodalLLT requires double precision (SReal must be double)");

namespace
{

/// Try to set the number of threads used by the BLAS backend at runtime.
///
/// CHOLMOD's supernodal factorization delegates to dense BLAS3 kernels, so the
/// thread count that matters is the BLAS one. It cannot be changed through
/// environment variables once the process is running (OpenBLAS reads them when
/// the library is loaded, before this component is created), hence we resolve
/// the backend's runtime setter dynamically and call it.
///
/// Returns true if a supported BLAS thread-control entry point was found.
bool trySetBlasNumThreads(int numThreads)
{
    using SetNumThreadsFn = void (*)(int);

    // Value-based setters exported by the common optimized BLAS backends.
    static const char* const candidates[] = {
        "openblas_set_num_threads", // OpenBLAS
        "goto_set_num_threads",     // GotoBLAS / older OpenBLAS
        "MKL_Set_Num_Threads",      // Intel MKL
    };

#if defined(_WIN32)
    // The BLAS DLL is already loaded (pulled in by CHOLMOD); look it up by name.
    static const char* const moduleNames[] = {
        "libopenblas", "openblas", "libblas", "mkl_rt"
    };
    for (const char* moduleName : moduleNames)
    {
        HMODULE module = GetModuleHandleA(moduleName);
        if (!module)
        {
            continue;
        }
        for (const char* name : candidates)
        {
            if (auto fn = reinterpret_cast<SetNumThreadsFn>(
                    reinterpret_cast<void*>(GetProcAddress(module, name))))
            {
                fn(numThreads);
                return true;
            }
        }
    }
    return false;
#else
    // The BLAS symbols are already loaded in the process (via CHOLMOD),
    // so search the global symbol table.
    for (const char* name : candidates)
    {
        if (auto fn = reinterpret_cast<SetNumThreadsFn>(dlsym(RTLD_DEFAULT, name)))
        {
            fn(numThreads);
            return true;
        }
    }
    return false;
#endif
}

} // namespace

template<class TBlockType>
EigenCholmodSupernodalLLT<TBlockType>::EigenCholmodSupernodalLLT()
    : d_numThreads(initData(&d_numThreads, 0,
        "numThreads",
        "Number of threads used by the BLAS backend of CHOLMOD's supernodal "
        "factorization. A value <= 0 (default) keeps the BLAS default "
        "(OPENBLAS_NUM_THREADS / OMP_NUM_THREADS). A small value (1-4) is often "
        "faster on medium-sized systems by avoiding thread oversubscription. "
        "Only effective with OpenBLAS or MKL."))
{
}

template<class TBlockType>
void EigenCholmodSupernodalLLT<TBlockType>::init()
{
    Inherit1::init();
    applyBlasNumThreads();
}

template<class TBlockType>
void EigenCholmodSupernodalLLT<TBlockType>::reinit()
{
    Inherit1::reinit();
    applyBlasNumThreads();
}

template<class TBlockType>
void EigenCholmodSupernodalLLT<TBlockType>::applyBlasNumThreads()
{
    const int numThreads = d_numThreads.getValue();
    if (numThreads <= 0)
    {
        // Keep the BLAS default (controlled by the environment).
        return;
    }

    if (trySetBlasNumThreads(numThreads))
    {
        msg_info() << "BLAS backend set to use " << numThreads
                   << " thread(s) for CHOLMOD factorization.";
    }
    else
    {
        msg_warning() << "Could not set the number of BLAS threads at runtime: no supported "
                         "BLAS thread-control API (OpenBLAS/MKL) was found. The value of '"
                      << d_numThreads.getName() << "' is ignored on this platform. Control the "
                         "thread count via environment variables instead (e.g. OPENBLAS_NUM_THREADS).";
    }
}

template class SOFACHOLMOD_API EigenCholmodSupernodalLLT< SReal >;
template class SOFACHOLMOD_API EigenCholmodSupernodalLLT< sofa::type::Mat<3,3,SReal> >;

MainCholmodSupernodalLLTFactory::~MainCholmodSupernodalLLTFactory() = default;

void registerEigenCholmodSupernodalLLT(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Direct linear solver using a supernodal sparse LL^T Cholesky factorization from CHOLMOD (SuiteSparse).")
        .add< EigenCholmodSupernodalLLT< SReal > >()
        .add< EigenCholmodSupernodalLLT< sofa::type::Mat<3, 3, SReal> > >());
}

} // namespace sofacholmod
