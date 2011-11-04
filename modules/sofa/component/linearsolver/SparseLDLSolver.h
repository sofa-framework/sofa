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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_H

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/helper/map.h>
#include <math.h>

#include <sofa/component/linearsolver/ParallelMatrixLinearSolver.inl>
#include <sofa/defaulttype/BaseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{



/// Direct linear solver based on Sparse LDL^T factorization, implemented with the CSPARSE library
template<class TMatrix, class TVector>
class SparseLDLSolver : public sofa::component::linearsolver::ParallelMatrixLinearSolver<TMatrix,TVector>
{
public :
    SOFA_CLASS(SOFA_TEMPLATE2(SparseLDLSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::ParallelMatrixLinearSolver,TMatrix,TVector));

public:
    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<bool> f_verbose;

    SparseLDLSolver();
    ~SparseLDLSolver();
    void solve (Matrix& M, Vector& x, Vector& b);
    void invert(Matrix& M);

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const SparseLDLSolver<TMatrix,TVector>* = NULL)
    {
        return TMatrix::Name();
    }

    MatrixInvertData * createInvertData()
    {
        return new SparseLDLSolverInvertData();
    }

protected :

    void LDL_ordering(Matrix& M);
    void LDL_symbolic(Matrix& M);
    void LDL_numeric(Matrix& M);

    helper::vector<int> xadj,adj;
    helper::vector<Real> Y,B;
    helper::vector<int> Parent,Lnz,Flag,Pattern;
    helper::vector<int> perm, invperm; //premutation inverse


    class SparseLDLSolverInvertData : public MatrixInvertData
    {
    public :
        sofa::component::linearsolver::CompressedRowSparseMatrix<Real> Mfiltered;
        int n;

        int * Mcolptr;
        int * Mrowind;
        Real * Mvalues;
        helper::vector<Real> values,D;
        helper::vector<int> rowind,colptr;
        helper::vector<int> perm, invperm; //premutation inverse
    };

};

#if defined(WIN32) && !defined(SOFA_BUILD_SPARSE_SOLVER)
extern template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< double>,FullVector<double> >;
extern template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< defaulttype::Mat<3,3,double> >,FullVector<double> >;
extern template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< float>,FullVector<float> >;
extern template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< defaulttype::Mat<3,3,float> >,FullVector<float> >;
#endif


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
