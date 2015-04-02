/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_MinResLinearSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_MinResLinearSolver_H

#include <SofaBaseLinearSolver/MatrixLinearSolver.h>

#include <sofa/helper/map.h>

#include <math.h>


namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define DISPLAY_TIME

/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class MinResLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MinResLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    Data<unsigned> f_maxIter;
    Data<double> f_tolerance;
    Data<bool> f_verbose;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graph;
#ifdef DISPLAY_TIME
    double time1;
    double time2;
    double timeStamp;
#endif
protected:

    MinResLinearSolver();

public:
    void resetSystem();

    void setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams);

    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b);

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BASE_LINEAR_SOLVER)

extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< FullMatrix<double>, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< SparseMatrix<double>, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,float> >, FullVector<float> >;

#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
