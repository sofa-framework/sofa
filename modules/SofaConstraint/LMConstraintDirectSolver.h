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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LMCONSTRAINTDIRECTSOLVER_H
#define SOFA_COMPONENT_CONSTRAINTSET_LMCONSTRAINTDIRECTSOLVER_H

#include <SofaConstraint/LMConstraintSolver.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

class SOFA_CONSTRAINT_API LMConstraintDirectSolver : public LMConstraintSolver
{
//	typedef Eigen::DynamicSparseMatrix<SReal,Eigen::ColMajor>    SparseColMajorMatrixEigen;
    typedef Eigen::SparseMatrix<SReal,Eigen::ColMajor>    SparseColMajorMatrixEigen;

    typedef helper::vector<linearsolver::LLineManipulator> JacobianRows;

public:
    SOFA_CLASS(LMConstraintDirectSolver, LMConstraintSolver);
protected:
    LMConstraintDirectSolver();
public:
    virtual bool buildSystem(const core::ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null());
    virtual bool solveSystem(const core::ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null());

protected:

    void analyseConstraints(const helper::vector< sofa::core::behavior::BaseLMConstraint* > &LMConstraints, core::ConstraintParams::ConstOrder order,
            JacobianRows &rowsL,JacobianRows &rowsLT, helper::vector< unsigned int > &rightHandElements) const;

    void buildLeftRectangularMatrix(const DofToMatrix& invMassMatrix,
            DofToMatrix& LMatrix, DofToMatrix& LTMatrix,
            SparseColMajorMatrixEigen &LeftMatrix, DofToMatrix &invMass_Ltrans) const;

    Data<sofa::helper::OptionsGroup> solverAlgorithm;
};

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
