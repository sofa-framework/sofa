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

#ifndef SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_H
#include <SofaSparseSolver/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/helper/map.h>
#include <cmath>
#include <SofaSparseSolver/SparseLDLSolverImpl.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa::component::linearsolver
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
    typedef SparseLDLImplInvertData<type::vector<int>, type::vector<Real> > InvertData;

    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;
    bool addJMInvJtLocal(TMatrix * M, ResMatrixType * result,const JMatrixType * J, SReal fact) override;
    int numStep;

    Data<bool> f_saveMatrixToFile;      ///< save matrix to a text file (can be very slow, as full matrix is stored)
    sofa::core::objectmodel::DataFileName d_filename;   ///< file where this matrix will be saved
    Data<int> d_precision;      ///< number of digits used to save system's matrix, default is 6

    MatrixInvertData * createInvertData() override {
        return new InvertData();
    }

    // Override canCreate in order to analyze if template has been set or not.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        const std::string_view templateString = arg->getAttribute("template", "");

        if (templateString.empty())
        {
            const std::string header = "SparseLDLSolver(" + std::string(arg->getAttribute("name", "")) + ")";
            msg_warning(header) << "Template is empty\n"
                                << "By default SparseLDLSolver uses blocks with a single double (to handle all cases of simulations).\n"
                                << "If you are using only 3D DOFs, you may consider using blocks of Matrix3 to speedup the calculations.\n"
                                << "If it is the case, add " << "template=\"CompressedRowSparseMatrixMat3x3d\" " << "to this object in your scene\n"
                                << "Otherwise, if you want to disable this message, add " << "template=\"CompressedRowSparseMatrixd\" " << ".";
        }

        return Inherit::canCreate(obj, context, arg);
    }

protected :
    SparseLDLSolver();

    type::vector<int> Jlocal2global;
    sofa::linearalgebra::FullMatrix<Real> JLinvDinv, JLinv;
    sofa::linearalgebra::CompressedRowSparseMatrix<Real> Mfiltered;
};

#if  !defined(SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_CPP)
extern template class SOFA_SOFASPARSESOLVER_API SparseLDLSolver< sofa::linearalgebra::CompressedRowSparseMatrix< double>, sofa::linearalgebra::FullVector<double> >;
extern template class SOFA_SOFASPARSESOLVER_API SparseLDLSolver< sofa::linearalgebra::CompressedRowSparseMatrix< type::Mat<3,3,double> >, sofa::linearalgebra::FullVector<double> >;

#endif


} // namespace sofa::component::linearsolver

#endif
