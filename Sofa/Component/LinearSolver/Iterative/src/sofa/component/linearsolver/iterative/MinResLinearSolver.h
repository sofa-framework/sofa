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

#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/helper/map.h>

#include <cmath>


namespace sofa::component::linearsolver::iterative
{

/// Linear system solver using the MINRES iterative algorithm
/// @author Matthieu Nesme
/// @date 2013
template<class TMatrix, class TVector>
class MinResLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MinResLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    Data<unsigned> f_maxIter; ///< maximum number of iterations of the Conjugate Gradient solution
    Data<double> f_tolerance; ///< desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)

    Data<std::map < std::string, sofa::type::vector<SReal> > > f_graph; ///< Graph of residuals at each iteration

protected:
    MinResLinearSolver();

public:
    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b) override;

    void parse(core::objectmodel::BaseObjectDescription* arg) override;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_MINRESLINEARSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< FullMatrix<SRreal>, FullVector<SRreal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< SparseMatrix<SRreal>, FullVector<SRreal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<SRreal>, FullVector<SRreal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<type::Mat<2,2,SRreal> >, FullVector<SRreal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<type::Mat<3,3,SRreal> >, FullVector<SRreal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<type::Mat<4,4,SRreal> >, FullVector<SRreal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<type::Mat<6,6,SRreal> >, FullVector<SRreal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<type::Mat<8,8,SRreal> >, FullVector<SRreal> >;
#endif

} //namespace sofa::component::linearsolver::iterative
