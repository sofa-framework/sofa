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
#include <LMConstraint/config.h>

#include <LMConstraint/LMConstraintSolver.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/core/MultiVecId.h>

namespace sofa::component::constraintset
{

class LMCONSTRAINT_API LMConstraintDirectSolver : public LMConstraintSolver
{
    typedef Eigen::SparseMatrix<SReal,Eigen::ColMajor>    SparseColMajorMatrixEigen;
    typedef helper::vector<linearsolver::LLineManipulator> JacobianRows;
    using MultiVecId = sofa::core::MultiVecId;

public:
    SOFA_CLASS(LMConstraintDirectSolver, LMConstraintSolver);
protected:
    LMConstraintDirectSolver();
public:
    bool buildSystem(const core::ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool solveSystem(const core::ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;

protected:

    void analyseConstraints(const helper::vector< sofa::core::behavior::BaseLMConstraint* > &LMConstraints, core::ConstraintParams::ConstOrder order,
            JacobianRows &rowsL,JacobianRows &rowsLT, helper::vector< unsigned int > &rightHandElements) const;

    void buildLeftRectangularMatrix(const DofToMatrix& invMassMatrix,
            DofToMatrix& LMatrix, DofToMatrix& LTMatrix,
            SparseColMajorMatrixEigen &LeftMatrix, DofToMatrix &invMass_Ltrans) const;

    Data<sofa::helper::OptionsGroup> solverAlgorithm; ///< Algorithm used to solve the system W.Lambda=c
};

} //namespace sofa::component::constraintset
