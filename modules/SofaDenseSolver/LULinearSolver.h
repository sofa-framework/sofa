/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_LULINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_LULINEARSOLVER_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Linear system solver using the default (LU factorization) algorithm
template<class Matrix, class Vector>
class LULinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<Matrix,Vector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(LULinearSolver,Matrix,Vector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,Matrix,Vector));

    Data<bool> f_verbose;
    typename Matrix::LUSolver* solver;
    typename Matrix::InvMatrixType Minv;
    bool computedMinv;
protected:
    LULinearSolver()
        : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
        , solver(NULL), computedMinv(false)
    {
    }

    ~LULinearSolver()
    {
        if (solver != NULL)
            delete solver;
    }
public:
    /// Invert M
    void invert (Matrix& M) override
    {
        if (solver != NULL)
            delete solver;
        solver = M.makeLUSolver();
        computedMinv = false;
    }

    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b) override
    {



        const bool verbose  = f_verbose.getValue();

        if( verbose )
        {
            serr<<"LULinearSolver, b = "<< b <<sendl;
            serr<<"LULinearSolver, M = "<< M <<sendl;
        }
        if (solver)
            M.solve(&x,&b, solver);
        else
            M.solve(&x,&b);

        // x is the solution of the system
        if( verbose )
        {
            serr<<"LULinearSolver::solve, solution = "<<x<<sendl;
        }
    }

    void computeMinv()
    {
        if (!computedMinv)
        {
            if (solver)
                Minv = solver->i();
            else
                Minv = this->currentGroup->systemMatrix->i();
            computedMinv = true;
        }
        /*typename Matrix::InvMatrixType I;
        I = ((*this->currentGroup->systemMatrix)*Minv);
        for (int i=0;i<I.rowSize();++i)
            for (int j=0;j<I.rowSize();++j)
            {
                double err = I.element(i,j)-((i==j)?1.0:0.0);
                if (fabs(err) > 1.0e-6)
                    serr << "ERROR: I("<<i<<","<<j<<") error "<<err<<sendl;
            }*/
    }

    double getMinvElement(int i, int j)
    {
        return Minv.element(i,j);
    }

    template<class RMatrix, class JMatrix>
    bool addJMInvJt(RMatrix& result, JMatrix& J, double fact)
    {
        const unsigned int Jrows = J.rowSize();
        const unsigned int Jcols = J.colSize();
        if (Jcols != (unsigned int)this->currentGroup->systemMatrix->rowSize())
        {
            serr << "LULinearSolver::addJMInvJt ERROR: incompatible J matrix size." << sendl;
            return false;
        }

        if (!Jrows) return false;
        computeMinv();

        const typename JMatrix::LineConstIterator jitend = J.end();
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != jitend; ++jit1)
        {
            int row1 = jit1->first;
            for (typename JMatrix::LineConstIterator jit2 = jit1; jit2 != jitend; ++jit2)
            {
                int row2 = jit2->first;
                double acc = 0.0;
                for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(), i1end = jit1->second.end(); i1 != i1end; ++i1)
                {
                    int col1 = i1->first;
                    double val1 = i1->second;
                    for (typename JMatrix::LElementConstIterator i2 = jit2->second.begin(), i2end = jit2->second.end(); i2 != i2end; ++i2)
                    {
                        int col2 = i2->first;
                        double val2 = i2->second;
                        acc += val1 * getMinvElement(col1,col2) * val2;
                    }
                }
                acc *= fact;
                //sout << "W("<<row1<<","<<row2<<") += "<<acc<<" * "<<fact<<sendl;
                result.add(row1,row2,acc);
                if (row1!=row2)
                    result.add(row2,row1,acc);
            }
        }
        return true;
    }

    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact) override
    {
        if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
        {
            if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
            else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
        }
        else if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
        {
            if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
            else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
        }
        else if (defaulttype::BaseMatrix* r = result)
        {
            if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
            else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
        }
        return false;
    }

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
