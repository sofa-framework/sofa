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
#include <sofa/component/linearsolver/direct/config.h>

#ifdef SOFA_COMPONENT_LINEARSOLVER_DIRECT_HAVE_CHOLMOD

#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.h>
#include <sofa/component/linearsolver/direct/EigenSolverFactory.h>

namespace sofa::component::linearsolver::direct
{

/**
 * Direct linear solver based on a supernodal sparse LL^T Cholesky factorization
 * using CHOLMOD from the SuiteSparse library (via Eigen's CholmodSupport).
 *
 * This solver is significantly faster than simplicial solvers (SparseLDLSolver,
 * EigenSimplicialLDLT) for medium-to-large systems (n > ~2000) thanks to
 * supernodal BLAS3-based factorization. CHOLMOD automatically selects the best
 * fill-reducing ordering (AMD, METIS, or NESDIS) regardless of the linked
 * OrderingMethod component.
 */
template<class TBlockType>
class EigenCholmodSupernodalLLT
    : public EigenDirectSparseSolver<
        TBlockType,
        MainCholmodSupernodalLLTFactory
    >
{
public:
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType> Matrix;
    using Real = typename Matrix::Real;
    typedef sofa::linearalgebra::FullVector<Real> Vector;

    SOFA_CLASS(SOFA_TEMPLATE(EigenCholmodSupernodalLLT, TBlockType), SOFA_TEMPLATE2(EigenDirectSparseSolver, TBlockType, MainCholmodSupernodalLLTFactory));
};

#ifndef SOFA_COMPONENT_LINEARSOLVER_DIRECT_EIGENCHOLMODSUPERNODALLLT_CPP
    extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API EigenCholmodSupernodalLLT< SReal >;
    extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API EigenCholmodSupernodalLLT< sofa::type::Mat<3,3,SReal> >;
#endif

} // namespace sofa::component::linearsolver::direct

#endif // SOFA_COMPONENT_LINEARSOLVER_DIRECT_HAVE_CHOLMOD
