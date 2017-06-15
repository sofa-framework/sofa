/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
// Author: François Faure, INRIA-UJF, (C) 2006
#define SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_CPP
#include <SofaBaseLinearSolver/CGLinearSolver.inl>

#include <sofa/core/ObjectFactory.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using sofa::core::MultiVecDerivId;

template<> SOFA_BASE_LINEAR_SOLVER_API
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(const core::ExecParams* /*params*/, Vector& p, Vector& r, SReal beta)
{
    p.eq(r,p,beta); // p = p*beta + r
}

template<> SOFA_BASE_LINEAR_SOLVER_API
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, SReal alpha)
{
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
#else // single-operation optimization
    typedef sofa::core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = (MultiVecDerivId)x;
    ops[0].second.push_back(std::make_pair((MultiVecDerivId)x,1.0));
    ops[0].second.push_back(std::make_pair((MultiVecDerivId)p,alpha));
    ops[1].first = (MultiVecDerivId)r;
    ops[1].second.push_back(std::make_pair((MultiVecDerivId)r,1.0));
    ops[1].second.push_back(std::make_pair((MultiVecDerivId)q,-alpha));
    this->executeVisitor(simulation::MechanicalVMultiOpVisitor(params, ops));
#endif
}

SOFA_DECL_CLASS(CGLinearSolver)

int CGLinearSolverClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
        .add< CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector > >(true)
#ifndef SOFA_FLOAT
        .add< CGLinearSolver< FullMatrix<double>, FullVector<double> > >()
        .add< CGLinearSolver< SparseMatrix<double>, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<2,2,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<3,3,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<4,4,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<6,6,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<8,8,double> >, FullVector<double> > >()
#endif
#ifndef SOFA_DOUBLE
        .add< CGLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<2,2,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<3,3,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<4,4,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<6,6,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<8,8,float> >, FullVector<float> > >()
#endif
        .addAlias("CGSolver")
        .addAlias("ConjugateGradient")
        ;

template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
#ifndef SOFA_FLOAT
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< FullMatrix<double>, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< SparseMatrix<double>, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,double> >, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,double> >, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,double> >, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,double> >, FullVector<double> >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,float> >, FullVector<float> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >, FullVector<float> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,float> >, FullVector<float> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,float> >, FullVector<float> >;
template class SOFA_BASE_LINEAR_SOLVER_API CGLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,float> >, FullVector<float> >;
#endif
} // namespace linearsolver

} // namespace component

} // namespace sofa

