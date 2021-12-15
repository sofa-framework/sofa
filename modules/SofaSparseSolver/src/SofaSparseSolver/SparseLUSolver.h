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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_H
#include <SofaSparseSolver/config.h>

#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <csparse.h>

namespace sofa::component::linearsolver
{

//defaut structure for a LU factorization
template<class Real>
class SparseLUInvertData : public MatrixInvertData {
public :

    css *S;
    csn *N;
    cs A;
    type::vector<sofa::Index> A_i, A_p;
    type::vector<Real> A_x;
    Real * tmp;
    SparseLUInvertData()
    {
        S=nullptr; N=nullptr; tmp=nullptr;
    }

    ~SparseLUInvertData()
    {
        if (S) cs_sfree (S);
        if (N) cs_nfree (N);
        if (tmp) cs_free (tmp);
    }
};

/// Direct linear solver based on Sparse LU factorization, implemented with the CSPARSE library
template<class TMatrix, class TVector, class TThreadManager= NoThreadManager>
class SparseLUSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(SparseLUSolver,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,TThreadManager));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;

    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager> Inherit;

    Data<bool> f_verbose; ///< Dump system state at each iteration
    Data<double> f_tol; ///< tolerance of factorization

    SparseLUSolver();
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;

protected :

    MatrixInvertData * createInvertData() override {
        return new SparseLUInvertData<Real>();
    }

};

#if  !defined(SOFA_COMPONENT_LINEARSOLVER_SPARSELUSOLVER_CPP)
    extern template class SOFA_SOFASPARSESOLVER_API SparseLUSolver< sofa::linearalgebra::CompressedRowSparseMatrix< double>, sofa::linearalgebra::FullVector<double> >;
#endif

} // namespace sofa::component::linearsolver

#endif
