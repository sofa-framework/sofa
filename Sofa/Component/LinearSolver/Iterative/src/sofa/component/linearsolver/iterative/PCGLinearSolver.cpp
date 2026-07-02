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
#define SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_PCGLINEARSOLVER_CPP
#include <sofa/component/linearsolver/iterative/PCGLinearSolver.inl>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/linearsystem/MatrixFreeSystem.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>

namespace sofa::component::linearsolver::iterative
{

using CRSd  = linearalgebra::CompressedRowSparseMatrix<SReal>;
using CRS3d = linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,SReal>>;
using FVd   = linearalgebra::FullVector<SReal>;

// Preconditioned CG on an assembled CRS matrix (both CRS<SReal> and CRS<Mat3x3>).
//
// Mirrors the generic PCGLinearSolver::solve() in the .inl, but drives the
// preconditioner directly on the assembled matrix (precond->invert(M) once, then
// precond->solve(M, z, r) per iteration) instead of the GraphScattered
// setRHS/solveSystem/dispatch plumbing, which does not exist for a CRS
// preconditioner such as SSORPreconditioner. The stopping test follows the
// generic version: r_norm = <r, M^{-1} r> compared against tolerance * <b, b>.
//
// The body is inlined into each specialisation of solve() below: it relies on the
// protected members (cgstep_*, d_graph, newton_iter, TempVectorContainer) of the
// solver, so it cannot live in a free helper, and a member template would leak
// into the GraphScattered specialisation which must keep the .inl implementation.

// ---------------------------------------------------------------------------
// CompressedRowSparseMatrix<SReal> specialisations
// ---------------------------------------------------------------------------
template<>
void PCGLinearSolver<CRSd, FVd>::checkLinearSystem()
{
    this->template doCheckLinearSystem<
        sofa::component::linearsystem::MatrixLinearSystem<CRSd, FVd>>();
}

template<>
void PCGLinearSolver<CRSd, FVd>::ensureRequiredLinearSystemType()
{
    // MatrixLinearSystem<CRSd,FVd> is the expected assembled system — nothing to reject.
}

template<>
void PCGLinearSolver<CRSd, FVd>::bwdInit()
{
    if (this->isComponentStateInvalid())
        return;
    // No PreconditionedMatrixFreeSystem linking is needed for the CRS pipeline.
}

template<>
void PCGLinearSolver<CRSd, FVd>::solve(CRSd& M, FVd& x, FVd& b)
{
    using MLSolver = MatrixLinearSolver<CRSd, FVd>;

    // Resolve the preconditioner (may be null -> plain CG).
    MLSolver* precond = nullptr;
    if (l_preconditioner.get() != nullptr && d_use_precond.getValue())
    {
        precond = dynamic_cast<MLSolver*>(l_preconditioner.get());
        if (!precond)
            msg_warning() << "Preconditioner '" << l_preconditioner->getName()
                          << "' is not a MatrixLinearSolver<CRS,FullVector> "
                          << "— running unpreconditioned.";
        else
            precond->invert(M); // factorise/refresh against the current matrix
    }

    std::map<std::string, sofa::type::vector<SReal>>& graph = *d_graph.beginEdit();
    newton_iter++;
    char name[256];
    snprintf(name, sizeof(name), "Error %d", newton_iter);
    sofa::type::vector<SReal>& graph_error = graph[std::string(name)];

    const core::ExecParams* params = core::execparams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    FVd& r = *vtmp.createTempVector();
    FVd& w = *vtmp.createTempVector();
    FVd& s = *vtmp.createTempVector();
    // createTempVector() default-constructs an empty FullVector; size it to the
    // system before use, otherwise the preconditioner writes out of bounds.
    r.resize(M.rowSize());
    w.resize(M.rowSize());
    s.resize(M.rowSize());

    const SReal b_norm = b.dot(b);
    const SReal tol    = d_tolerance.getValue() * b_norm;

    r = M * x;
    cgstep_beta(r, b, SReal(-1));   // r = b - M*x

    if (precond) precond->solve(M, w, r);   // w = M^{-1} r
    else         w = r;

    SReal r_norm = r.dot(w);        // r . M^{-1} r
    graph_error.push_back(r_norm / b_norm);

    unsigned iter = 1;
    while ((iter <= d_maxIter.getValue()) && (r_norm > tol))
    {
        s = M * w;
        const SReal dtq   = w.dot(s);
        const SReal alpha = r_norm / dtq;
        cgstep_alpha(x, w,  alpha);   // x += alpha w
        cgstep_alpha(r, s, -alpha);   // r -= alpha (M w)

        if (precond) precond->solve(M, s, r);
        else         s = r;

        const SReal deltaOld = r_norm;
        r_norm = r.dot(s);
        graph_error.push_back(r_norm / b_norm);
        const SReal beta = r_norm / deltaOld;
        cgstep_beta(w, s, beta);      // w = s + beta w
        iter++;
    }

    d_graph.endEdit();
    vtmp.deleteTempVector(&r);
    vtmp.deleteTempVector(&w);
    vtmp.deleteTempVector(&s);
    sofa::helper::AdvancedTimer::valSet("PCG iterations", iter);
}

// ---------------------------------------------------------------------------
// CompressedRowSparseMatrix<Mat<3,3>> specialisations
// ---------------------------------------------------------------------------
template<>
void PCGLinearSolver<CRS3d, FVd>::checkLinearSystem()
{
    this->template doCheckLinearSystem<
        sofa::component::linearsystem::MatrixLinearSystem<CRS3d, FVd>>();
}

template<>
void PCGLinearSolver<CRS3d, FVd>::ensureRequiredLinearSystemType()
{
    // MatrixLinearSystem<CRS3d,FVd> is the expected assembled system — nothing to reject.
}

template<>
void PCGLinearSolver<CRS3d, FVd>::bwdInit()
{
    if (this->isComponentStateInvalid())
        return;
    // No PreconditionedMatrixFreeSystem linking is needed for the CRS pipeline.
}

template<>
void PCGLinearSolver<CRS3d, FVd>::solve(CRS3d& M, FVd& x, FVd& b)
{
    using MLSolver = MatrixLinearSolver<CRS3d, FVd>;

    // Resolve the preconditioner (may be null -> plain CG).
    MLSolver* precond = nullptr;
    if (l_preconditioner.get() != nullptr && d_use_precond.getValue())
    {
        precond = dynamic_cast<MLSolver*>(l_preconditioner.get());
        if (!precond)
            msg_warning() << "Preconditioner '" << l_preconditioner->getName()
                          << "' is not a MatrixLinearSolver<CRS3,FullVector> "
                          << "— running unpreconditioned.";
        else
            precond->invert(M); // factorise/refresh against the current matrix
    }

    std::map<std::string, sofa::type::vector<SReal>>& graph = *d_graph.beginEdit();
    newton_iter++;
    char name[256];
    snprintf(name, sizeof(name), "Error %d", newton_iter);
    sofa::type::vector<SReal>& graph_error = graph[std::string(name)];

    const core::ExecParams* params = core::execparams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    FVd& r = *vtmp.createTempVector();
    FVd& w = *vtmp.createTempVector();
    FVd& s = *vtmp.createTempVector();
    // createTempVector() default-constructs an empty FullVector; size it to the
    // system before use, otherwise the preconditioner writes out of bounds.
    r.resize(M.rowSize());
    w.resize(M.rowSize());
    s.resize(M.rowSize());

    const SReal b_norm = b.dot(b);
    const SReal tol    = d_tolerance.getValue() * b_norm;

    r = M * x;
    cgstep_beta(r, b, SReal(-1));   // r = b - M*x

    if (precond) precond->solve(M, w, r);   // w = M^{-1} r
    else         w = r;

    SReal r_norm = r.dot(w);        // r . M^{-1} r
    graph_error.push_back(r_norm / b_norm);

    unsigned iter = 1;
    while ((iter <= d_maxIter.getValue()) && (r_norm > tol))
    {
        s = M * w;
        const SReal dtq   = w.dot(s);
        const SReal alpha = r_norm / dtq;
        cgstep_alpha(x, w,  alpha);   // x += alpha w
        cgstep_alpha(r, s, -alpha);   // r -= alpha (M w)

        if (precond) precond->solve(M, s, r);
        else         s = r;

        const SReal deltaOld = r_norm;
        r_norm = r.dot(s);
        graph_error.push_back(r_norm / b_norm);
        const SReal beta = r_norm / deltaOld;
        cgstep_beta(w, s, beta);      // w = s + beta w
        iter++;
    }

    d_graph.endEdit();
    vtmp.deleteTempVector(&r);
    vtmp.deleteTempVector(&w);
    vtmp.deleteTempVector(&s);
    sofa::helper::AdvancedTimer::valSet("PCG iterations", iter);
}

// ---------------------------------------------------------------------------
// Factory registration + explicit instantiations
// ---------------------------------------------------------------------------
void registerPCGLinearSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData(
        "Linear solver using the preconditioned conjugate gradient iterative algorithm.")
        .add< PCGLinearSolver<GraphScatteredMatrix, GraphScatteredVector> >()
        .add< PCGLinearSolver<CRSd, FVd> >()
        .add< PCGLinearSolver<CRS3d, FVd> >());
}

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PCGLinearSolver<GraphScatteredMatrix, GraphScatteredVector>;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PCGLinearSolver<CRSd, FVd>;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PCGLinearSolver<CRS3d, FVd>;

} // namespace sofa::component::linearsolver::iterative
