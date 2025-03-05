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
/******************************************************************************
* Contributors:
*   - jeremie.allard@insimo.fr (InSimo)
*******************************************************************************/

#pragma once
#include <sofa/component/linearsolver/direct/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/helper/map.h>
#include <cmath>
#include <sofa/component/linearsolver/direct/SparseLDLSolverImpl.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa::component::linearsolver::direct
{

// Direct linear solver based on Sparse LDL^T factorization, implemented with the CSPARSE library
template<class TMatrix, class TVector, class TThreadManager = NoThreadManager>
class SparseLDLSolver : public SparseLDLSolverImpl<TMatrix,TVector, TThreadManager>
{
public :
    SOFA_CLASS(SOFA_TEMPLATE3(SparseLDLSolver,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(SparseLDLSolverImpl,TMatrix,TVector,TThreadManager));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef SparseLDLSolverImpl<TMatrix,TVector,TThreadManager> Inherit;
    typedef typename Inherit::ResMatrixType ResMatrixType;
    typedef typename Inherit::JMatrixType JMatrixType;
    typedef SparseLDLImplInvertData<type::vector<int>, type::vector<Real> > InvertData;

    void init() override;
    void parse( sofa::core::objectmodel::BaseObjectDescription* arg ) override;
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;
    bool doAddJMInvJtLocal(ResMatrixType* result, const JMatrixType* J, SReal fact, InvertData* data);
    bool addJMInvJtLocal(TMatrix * M, ResMatrixType * result,const JMatrixType * J, SReal fact) override;
    int numStep;

    MatrixInvertData * createInvertData() override {
        return new InvertData();
    }

protected :
    SparseLDLSolver();

    type::vector<sofa::SignedIndex> Jlocal2global;
    sofa::linearalgebra::FullMatrix<Real> JLinvDinv, JLinv;
    sofa::linearalgebra::CompressedRowSparseMatrix<Real> Mfiltered;

    bool factorize(Matrix& M, InvertData * invertData);

    void showInvalidSystemMessage(const std::string& reason) const;

    using Triplet = std::tuple<sofa::SignedIndex, sofa::SignedIndex, Real>;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API SparseLDLSolver< sofa::linearalgebra::CompressedRowSparseMatrix< SReal>, sofa::linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API SparseLDLSolver< sofa::linearalgebra::CompressedRowSparseMatrix< type::Mat<3,3,SReal> >, sofa::linearalgebra::FullVector<SReal> >;
#endif

} // namespace sofa::component::linearsolver::direct
