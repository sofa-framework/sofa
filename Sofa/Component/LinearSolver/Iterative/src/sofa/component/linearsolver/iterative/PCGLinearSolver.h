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
#pragma once
#include <sofa/component/linearsolver/iterative/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/helper/map.h>

#include <cmath>

namespace sofa::component::linearsolver::iterative
{

/// Linear system solver using the preconditioned conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class PCGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{

public:

    SOFA_CLASS(
        SOFA_TEMPLATE2(PCGLinearSolver,TMatrix,TVector),
        SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, TMatrix, TVector));

    using Matrix = TMatrix;
    using Vector = TVector;
    using Real = typename Matrix::Real;
    using Inherit = sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>;

    Data<unsigned> d_maxIter; ///< Maximum number of iterations after which the iterative descent of the Conjugate Gradient must stop
    Data<Real> d_tolerance; ///< Desired accuracy of the Conjugate Gradient solution evaluating: |r|²/|b|² (ratio of current residual norm over initial residual norm)
    Data<bool> d_use_precond; ///< Use a preconditioner
    SingleLink<PCGLinearSolver, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_preconditioner; ///< Link towards the linear solver used to precondition the conjugate gradient
    core::objectmodel::lifecycle::DeprecatedData d_update_step; ///< Number of steps before the next refresh of preconditioners
    Data<std::map < std::string, sofa::type::vector<Real> > > d_graph; ///< Graph of residuals at each iteration

    void solve (Matrix& M, Vector& x, Vector& b) override;
    void init() override;
    void bwdInit() override;

private:
    unsigned next_refresh_step;
    bool first;
    int newton_iter;

protected:
    PCGLinearSolver();

    void ensureRequiredLinearSystemType();

    /// This method is separated from the rest to be able to use custom/optimized versions depending on
    /// the types of vectors. It computes: p = p*beta + r
    inline void cgstep_beta(Vector& p, Vector& r, Real beta);

    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: x += p*alpha, r -= q*alpha
    inline void cgstep_alpha(Vector& x,Vector& p, Real alpha);

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void checkLinearSystem() override;
};

template<class TMatrix, class TVector>
inline void PCGLinearSolver<TMatrix,TVector>::cgstep_beta(Vector& p, Vector& r, Real beta)
{
    p *= beta;
    p += r; //z;
}

template<class TMatrix, class TVector>
inline void PCGLinearSolver<TMatrix,TVector>::cgstep_alpha(Vector& x,Vector& p, Real alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
}

template<>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, Real beta);

template<>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x,Vector& p, Real alpha);

#if !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_PCGLINEARSOLVER_CPP)
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PCGLinearSolver<GraphScatteredMatrix, GraphScatteredVector>;
#endif // !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_PCGLINEARSOLVER_CPP)

} // namespace sofa::component::linearsolver::iterative
