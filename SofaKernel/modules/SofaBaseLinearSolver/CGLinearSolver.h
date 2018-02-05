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
#ifndef SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_H
#include "config.h"

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
class CGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CGLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    Data<unsigned> f_maxIter;
    Data<SReal> f_tolerance;
    Data<SReal> f_smallDenominatorThreshold;
    Data<bool> f_warmStart;
    Data<bool> f_verbose;
    Data<std::map < std::string, sofa::helper::vector<SReal> > > f_graph;
#ifdef DISPLAY_TIME
    SReal time1;
    SReal time2;
    SReal timeStamp;
#endif
protected:

    CGLinearSolver();

    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: p = p*beta + r
    inline void cgstep_beta(const core::ExecParams* params, Vector& p, Vector& r, SReal beta);
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: x += p*alpha, r -= q*alpha
    inline void cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, SReal alpha);

public:
    virtual void init() override;
    virtual void reinit() override;

    void resetSystem() override;

    void setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams) override;

    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b) override;

};

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(const core::ExecParams* /*params*/, Vector& p, Vector& r, SReal beta);

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, SReal alpha);

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_CPP)
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< FullMatrix<double>, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< SparseMatrix<double>, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,double> >, FullVector<double> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,double> >, FullVector<double> >;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,float> >, FullVector<float> >;
extern template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,float> >, FullVector<float> >;
#endif
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
