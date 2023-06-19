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
#pragma once
#include <sofa/component/linearsolver/preconditioner/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/DiagonalMatrix.h>
#include <sofa/helper/map.h>

#include <cmath>

namespace sofa::component::linearsolver::preconditioner
{

template<class TVector>
class BlockJacobiPreconditionerInternalData
{
};


/// Linear solver based on a NxN bloc diagonal matrix (i.e. block Jacobi preconditioner)
template<class TMatrix, class TVector>
class BlockJacobiPreconditioner : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BlockJacobiPreconditioner,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    typedef typename TMatrix::Block SubMatrix;

    SOFA_ATTRIBUTE_DEPRECATED__PRECONDITIONER_VERBOSEDATA()
    Data<bool> f_verbose; ///< Dump system state at each iteration

protected:
    BlockJacobiPreconditioner();
public:
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;

    BlockJacobiPreconditionerInternalData<TVector> internalData; ///< not use in CPU

    /// Returns the sofa template name. By default the name of the c++ class signature is exposed...
    /// so we need to override that by implementing GetCustomTemplateName() function
    /// More details on the name customization infrastructure is in NameDecoder.h
    static const std::string GetCustomTemplateName()
    {
        return TVector::Name();
    }

    void parse(core::objectmodel::BaseObjectDescription *arg) override;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_BLOCKJACOBIPRECONDITIONER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API BlockJacobiPreconditioner<sofa::linearalgebra::BlockDiagonalMatrix<3, SReal>, sofa::linearalgebra::FullVector<SReal> >;
#endif // !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_BLOCKJACOBIPRECONDITIONER_CPP)

} // namespace sofa::component::linearsolver::preconditioner

