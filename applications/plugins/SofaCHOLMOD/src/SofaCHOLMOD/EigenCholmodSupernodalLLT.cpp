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
#include <SofaCHOLMOD/CholmodSolverProxy.h>
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.inl>

#include <sofa/core/ObjectFactory.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace sofacholmod
{

// CHOLMOD only supports double precision. This is enforced at configure time by
// the plugin's CMakeLists.txt, which requires SOFA_FLOATING_POINT_TYPE=double.

namespace
{

/// Resolve a symbol from the BLAS backend already loaded in the process (the
/// same one CHOLMOD uses). Returns nullptr if not found. No link-time dependency
/// on BLAS is introduced.
void* resolveBlasSymbol(const char* name)
{
#if defined(_WIN32)
    // The BLAS DLL is already loaded (pulled in by CHOLMOD); look it up by name.
    static const char* const moduleNames[] = {
        "libopenblas", "openblas", "libblas", "mkl_rt"
    };
    for (const char* moduleName : moduleNames)
    {
        if (HMODULE module = GetModuleHandleA(moduleName))
        {
            if (auto* p = reinterpret_cast<void*>(GetProcAddress(module, name)))
            {
                return p;
            }
        }
    }
    return nullptr;
#else
    // The BLAS symbols are already loaded in the process (via CHOLMOD),
    // so search the global symbol table.
    return dlsym(RTLD_DEFAULT, name);
#endif
}

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

    for (const char* name : candidates)
    {
        if (auto fn = reinterpret_cast<SetNumThreadsFn>(resolveBlasSymbol(name)))
        {
            fn(numThreads);
            return true;
        }
    }
    return false;
}

/// Compute the lower triangle of W = fact * Z^T * Z, where Z is an n x m
/// column-major matrix with leading dimension ldZ (>= n). Uses the BLAS
/// symmetric rank-k update (dsyrk): it computes only the triangle we need (half
/// the flops of a full product) and is multi-threaded on OpenBLAS/MKL according
/// to the BLAS thread count (see the numThreads Data). Falls back to an Eigen
/// rank update if dsyrk is unavailable.
void computeLowerZtZ(const double* Z, Eigen::Index n, Eigen::Index m, Eigen::Index ldZ,
                     double fact, Eigen::MatrixXd& W)
{
    // CBLAS enum values (avoids depending on a cblas.h header being present).
    constexpr int CblasColMajor = 102;
    constexpr int CblasLower    = 122;
    constexpr int CblasTrans    = 112;

    using DsyrkFn = void (*)(int, int, int, int, int, double,
                             const double*, int, double, double*, int);
    static DsyrkFn dsyrk = reinterpret_cast<DsyrkFn>(resolveBlasSymbol("cblas_dsyrk"));

    if (dsyrk)
    {
        // W := fact * Z^T * Z (lower triangle), with Z treated as A (n x m).
        dsyrk(CblasColMajor, CblasLower, CblasTrans,
              static_cast<int>(m), static_cast<int>(n), fact,
              Z, static_cast<int>(ldZ), 0.0, W.data(), static_cast<int>(m));
    }
    else
    {
        Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>> Zmap(Z, n, m, Eigen::OuterStride<>(ldZ));
        W.setZero();
        W.selfadjointView<Eigen::Lower>().rankUpdate(Zmap.transpose(), fact);
    }
}

} // namespace

template<class TBlockType>
EigenCholmodSupernodalLLT<TBlockType>::EigenCholmodSupernodalLLT()
    : d_numThreads(initData(&d_numThreads, 1,
        "numThreads",
        "Number of threads used by the BLAS backend of CHOLMOD's supernodal "
        "factorization. Default is 1: this is the fastest setting for the vast "
        "majority of SOFA scenes, and it avoids catastrophic thread "
        "oversubscription when several solvers are factorized in parallel (e.g. "
        "with parallelODESolving). Increase it only for a single, very large "
        "standalone system. A value <= 0 keeps the BLAS default "
        "(OPENBLAS_NUM_THREADS / OMP_NUM_THREADS). Only effective with OpenBLAS "
        "or MKL."))
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
void EigenCholmodSupernodalLLT<TBlockType>::invert(Matrix& A)
{
    if (const int nthreads = d_numThreads.getValue(); nthreads > 0)
    {
        if (auto* proxy = dynamic_cast<CholmodSolverProxy*>(this->m_solver.get()))
        {
            proxy->cholmodCommon().nthreads_max = nthreads;
        }
    }

    Inherit1::invert(A);
}


// Adds the compliance block W = fact * J * A^{-1} * J^T into 'result'.
//
// CHOLMOD factorizes P*A*P^T = L*L^T, so A^{-1} = P^T * L^{-T} * L^{-1} * P and
//     J * A^{-1} * J^T = (L^{-1} * P * J^T)^T * (L^{-1} * P * J^T) = Z^T * Z.
// The whole block is therefore one triangular forward-solve (with the m columns
// of J^T as right-hand sides) followed by one symmetric product Z^T*Z, avoiding
// a full solve per constraint row.
template<class TBlockType>
bool EigenCholmodSupernodalLLT<TBlockType>::addJMInvJtLocal(
    Matrix* M, ResMatrixType* result, const JMatrixType* J, SReal fact)
{
    SOFA_UNUSED(M);

    if (!this->isComponentStateValid())
    {
        return true;
    }

    // Access the raw CHOLMOD factor via our proxy. The generic EigenSolverWrapper
    // does not expose it, so fall back to the base implementation if, for any
    // reason, the registered solver is not our CholmodSolverProxy.
    auto* proxy = dynamic_cast<CholmodSolverProxy*>(this->m_solver.get());
    if (!proxy)
    {
        return Inherit1::addJMInvJtLocal(M, result, J, fact);
    }

    if (J->rowSize() == 0)
    {
        return true;
    }

    // Global row index of each (sparse) constraint row, used to scatter W back.
    std::vector<sofa::SignedIndex> jLocal2Global;
    jLocal2Global.reserve(J->rowSize());
    for (auto jit = J->begin(), jitend = J->end(); jit != jitend; ++jit)
    {
        jLocal2Global.push_back(jit->first);
    }

    if (jLocal2Global.empty())
    {
        return true;
    }

    const auto m = static_cast<Eigen::Index>(jLocal2Global.size()); // constraint rows
    const auto n = static_cast<Eigen::Index>(J->colSize());         // system size

    // 1. Expand the sparse J into a dense J^T (n x m, column-major as CHOLMOD expects).
    //    m_Jt is a reused buffer: setZero resizes only when the shape changes.
    m_Jt.setZero(n, m);
    {
        Eigen::Index col = 0;
        for (auto jit = J->begin(), jitend = J->end(); jit != jitend; ++jit, ++col)
        {
            for (auto it = jit->second.begin(), itend = jit->second.end(); it != itend; ++it)
            {
                m_Jt(it->first, col) = it->second;
            }
        }
    }

    // 2. + 3. Z = L^{-1} * P * J^T, all m columns at once (permutation then supernodal
    //    BLAS3 forward-solve). The proxy reuses its workspace across calls, so Z_cd is
    //    owned by the proxy and must not be freed here.
    cholmod_dense Jt_cd = Eigen::viewAsCholmod(m_Jt);
    cholmod_dense* Z_cd = proxy->forwardSolveLP(&Jt_cd);
    if (!Z_cd)
    {
        msg_error() << "CHOLMOD forward solve failed";
        return false;
    }

    // 4. W = fact * Z^T * Z. W is symmetric, so only its lower triangle is computed.
    Eigen::MatrixXd W(m, m);
    computeLowerZtZ(static_cast<const double*>(Z_cd->x), n, m,
                    static_cast<Eigen::Index>(Z_cd->d), fact, W);

    // 5. Scatter the lower triangle (j >= i) into 'result', mirroring across the diagonal.
    for (Eigen::Index i = 0; i < m; ++i)
    {
        for (Eigen::Index j = i; j < m; ++j)
        {
            const double value = W(j, i);
            result->add(jLocal2Global[j], jLocal2Global[i], value);
            if (i != j)
            {
                result->add(jLocal2Global[i], jLocal2Global[j], value);
            }
        }
    }

    return true;
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
                         "thread count via environment variables instead (e.g. OPENBLAS_NUM_THREADS or MKL_NUM_THREADS).";
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
