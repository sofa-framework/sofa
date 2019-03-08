/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <sofa/helper/map.h>
#include <cmath>
#include <SofaSparseSolver/SparseLDLSolverImpl.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Direct linear solver based on Sparse LDL^T factorization, implemented with the CSPARSE library
template<class TMatrix, class TVector, class TThreadManager = NoThreadManager>
class SparseLDLSolver : public sofa::component::linearsolver::SparseLDLSolverImpl<TMatrix,TVector, TThreadManager>
{
public :
    SOFA_CLASS(SOFA_TEMPLATE3(SparseLDLSolver,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::SparseLDLSolverImpl,TMatrix,TVector,TThreadManager));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::SparseLDLSolverImpl<TMatrix,TVector,TThreadManager> Inherit;
    typedef typename Inherit::ResMatrixType ResMatrixType;
    typedef typename Inherit::JMatrixType JMatrixType;
    typedef SparseLDLImplInvertData<helper::vector<int>, helper::vector<Real> > InvertData;

    void solve (Matrix& M, Vector& x, Vector& b) override ;
    void invert(Matrix& M) override;
    bool addJMInvJtLocal(TMatrix * M, ResMatrixType * result,const JMatrixType * J, double fact) override;
    int numStep;

    Data<bool> f_saveMatrixToFile;      ///< save matrix to a text file (can be very slow, as full matrix is stored)
    sofa::core::objectmodel::DataFileName d_filename;   ///< file where this matrix will be saved
    Data<int> d_precision;      ///< number of digits used to save system's matrix, default is 6

    MatrixInvertData * createInvertData() override {
        return new InvertData();
    }

protected :
    SparseLDLSolver();

    FullMatrix<Real> Jminv,Jdense;
    sofa::component::linearsolver::CompressedRowSparseMatrix<Real> Mfiltered;
};

#if  !defined(SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_CPP)
extern template class SOFA_SOFASPARSESOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< double>,FullVector<double> >;
extern template class SOFA_SOFASPARSESOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< defaulttype::Mat<3,3,double> >,FullVector<double> >;

#endif


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
