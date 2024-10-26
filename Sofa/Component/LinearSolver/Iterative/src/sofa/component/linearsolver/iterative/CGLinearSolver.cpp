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
#define SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_CPP
#include <sofa/component/linearsolver/iterative/CGLinearSolver.inl>

#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalVMultiOpVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVMultiOpVisitor;

namespace sofa::component::linearsolver::iterative
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using sofa::core::MultiVecDerivId;

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(const core::ExecParams* /*params*/, Vector& p, Vector& r, Real beta)
{
    p.eq(r,p,beta); // p = p*beta + r
}

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, Real alpha)
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
    this->executeVisitor(MechanicalVMultiOpVisitor(params, ops));
#endif
}
using namespace sofa::linearalgebra;

void registerCGLinearSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Linear system solver using the conjugate gradient iterative algorithm")
        .add< CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector > >(true)
        .add< CGLinearSolver< FullMatrix<SReal>, FullVector<SReal> > >()
        .add< CGLinearSolver< SparseMatrix<SReal>, FullVector<SReal> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<2, 2, SReal> >, FullVector<SReal> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<3, 3, SReal> >, FullVector<SReal> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<4, 4, SReal> >, FullVector<SReal> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<6, 6, SReal> >, FullVector<SReal> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<8, 8, SReal> >, FullVector<SReal> > >());
}

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< FullMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< SparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< CompressedRowSparseMatrix<type::Mat<2,2,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< CompressedRowSparseMatrix<type::Mat<3,3,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< CompressedRowSparseMatrix<type::Mat<4,4,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< CompressedRowSparseMatrix<type::Mat<6,6,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< CompressedRowSparseMatrix<type::Mat<8,8,SReal> >, FullVector<SReal> >;


} // namespace sofa::component::linearsolver::iterative
