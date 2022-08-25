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
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalGetConstraintJacobianVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalGetConstraintJacobianVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorToBaseVectorVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalMultiVectorToBaseVectorVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorFromBaseVectorVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalMultiVectorFromBaseVectorVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorPeqBaseVectorVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalMultiVectorPeqBaseVectorVisitor;

namespace sofa::component::linearsolver
{

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::MatrixLinearSolver()
    : Inherit()
    , invertData()
    , linearSystem()
    , currentMFactor(), currentBFactor(), currentKFactor()
    , d_multithreading(initData(&d_multithreading, true, "Multithreading", "Enable multithreading for the assembly of the compliance matrix. Sparse solver only."))
{
}

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::~MatrixLinearSolver() = default;

template<class Matrix, class Vector>
MatrixInvertData * MatrixLinearSolver<Matrix,Vector>::getMatrixInvertData(linearalgebra::BaseMatrix * /*m*/)
{
    if (invertData==nullptr)
    {
        invertData = std::unique_ptr<MatrixInvertData>(createInvertData());
    }
    return invertData.get();
}

template<class Matrix, class Vector>
MatrixInvertData * MatrixLinearSolver<Matrix,Vector>::createInvertData()
{
    msg_error("MatrixLinearSolver") << "The solver didn't implement MatrixLinearSolver::getMatrixInvertData this function is not available in MatrixLinearSolver, nullptr is return." ;
    return nullptr;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resetSystem()
{
    if (!this->frozen)
    {
        if (linearSystem.systemMatrix) linearSystem.systemMatrix->clear();
        linearSystem.needInvert = true;
    }
    if (linearSystem.systemRHVector) linearSystem.systemRHVector->clear();
    if (linearSystem.systemLHVector) linearSystem.systemLHVector->clear();
    linearSystem.solutionVecId = core::MultiVecDerivId::null();
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resizeSystem(Size n)
{
    if (!this->frozen)
    {
        if (!linearSystem.systemMatrix) linearSystem.systemMatrix = createMatrix();
        linearSystem.systemMatrix->resize(n, n);
    }

    if (!linearSystem.systemRHVector) linearSystem.systemRHVector = createPersistentVector();
    linearSystem.systemRHVector->resize(n);

    if (!linearSystem.systemLHVector) linearSystem.systemLHVector = createPersistentVector();
    linearSystem.systemLHVector->resize(n);

    linearSystem.needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemMatrix(Matrix * matrix)
{
    linearSystem.systemMatrix = matrix;
    if (matrix!=nullptr) {
        if (!linearSystem.systemRHVector) linearSystem.systemRHVector = createPersistentVector();
        linearSystem.systemRHVector->resize(matrix->colSize());
        if (!linearSystem.systemLHVector) linearSystem.systemLHVector = createPersistentVector();
        linearSystem.systemLHVector->resize(matrix->colSize());
    }
    linearSystem.needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    this->currentMFactor = mparams->mFactor();
    this->currentBFactor = sofa::core::mechanicalparams::bFactor(mparams);
    this->currentKFactor = mparams->kFactor();

    if (!this->frozen)
    {
        linearSystem.matrixAccessor.setDoPrintInfo(this->f_printLog.getValue() ) ;

        simulation::common::MechanicalOperations mops(mparams, this->getContext());

        // Create the matrix if not yet done
        if (!linearSystem.systemMatrix) linearSystem.systemMatrix = createMatrix();

        linearSystem.matrixAccessor.setGlobalMatrix(linearSystem.systemMatrix);
        linearSystem.matrixAccessor.clear();

        // The following operation traverses the BaseMechanicalState of the current context tree,
        // and accumulate the number of degrees of freedom to get the total number of degrees of
        // freedom, which is the size of the linear system.
        // During the accumulation, it also prepares the indices to parts of the matrix associated
        // to each BaseMechanicalState. Each BaseMechanicalState will then write to this submatrix
        // based on the provided index.
        mops.getMatrixDimension(&linearSystem.matrixAccessor);

        linearSystem.matrixAccessor.setupMatrices();
        resizeSystem(linearSystem.matrixAccessor.getGlobalDimension());
        linearSystem.systemMatrix->clear();
        mops.addMBK_ToMatrix(&(linearSystem.matrixAccessor), mparams->mFactor(), sofa::core::mechanicalparams::bFactor(mparams), mparams->kFactor());
        linearSystem.matrixAccessor.computeGlobalMatrix();
    }

}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::rebuildSystem(SReal massFactor, SReal forceFactor)
{
    sofa::core::MechanicalParams mparams;
    mparams.setMFactor(this->currentMFactor*massFactor);
    mparams.setBFactor(this->currentBFactor*forceFactor);
    mparams.setKFactor(this->currentKFactor*forceFactor);
    if (!this->frozen)
    {
        simulation::common::MechanicalOperations mops(&mparams, this->getContext());
        if (!linearSystem.systemMatrix) linearSystem.systemMatrix = createMatrix();
        linearSystem.matrixAccessor.setGlobalMatrix(linearSystem.systemMatrix);
        linearSystem.matrixAccessor.clear();
        mops.getMatrixDimension(&(linearSystem.matrixAccessor));
        linearSystem.matrixAccessor.setupMatrices();
        resizeSystem(linearSystem.matrixAccessor.getGlobalDimension());
        linearSystem.systemMatrix->clear();
        mops.addMBK_ToMatrix(&(linearSystem.matrixAccessor), mparams.mFactor(), mparams.bFactor(), mparams.kFactor());
        linearSystem.matrixAccessor.computeGlobalMatrix();
    }

    this->invertSystem();
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemRHVector(core::MultiVecDerivId v)
{
    executeVisitor( MechanicalMultiVectorToBaseVectorVisitor(core::execparams::defaultInstance(), v,
                                                             linearSystem.systemRHVector,
                                                             &(linearSystem.matrixAccessor)) );
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemLHVector(core::MultiVecDerivId v)
{
    linearSystem.solutionVecId = v;
    if (!linearSystem.solutionVecId.isNull())
    {
        executeVisitor( MechanicalMultiVectorToBaseVectorVisitor(core::execparams::defaultInstance(), v, linearSystem.systemLHVector, &(linearSystem.matrixAccessor)) );
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::solveSystem()
{
    if (linearSystem.needInvert)
    {
        this->invert(*linearSystem.systemMatrix);
        linearSystem.needInvert = false;
    }
    this->solve(*linearSystem.systemMatrix, *linearSystem.systemLHVector, *linearSystem.systemRHVector);
    if (!linearSystem.solutionVecId.isNull())
    {
        executeVisitor( MechanicalMultiVectorFromBaseVectorVisitor(core::execparams::defaultInstance(), linearSystem.solutionVecId, linearSystem.systemLHVector, &(linearSystem.matrixAccessor)) );
    }
}

template<class Matrix, class Vector>
Vector* MatrixLinearSolver<Matrix,Vector>::createPersistentVector()
{
    return new Vector;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::deletePersistentVector(Vector* v)
{
    delete v;
}

template<class Matrix, class Vector>
Matrix* MatrixLinearSolver<Matrix,Vector>::createMatrix()
{
    return new Matrix;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::deleteMatrix(Matrix* v)
{
    delete v;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::invertSystem()
{
    if (linearSystem.needInvert)
    {
        this->invert(*linearSystem.systemMatrix);
        linearSystem.needInvert = false;
    }
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addJMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, SReal fact)
{
    for (typename JMatrixType::Index row=0; row<J->rowSize(); row++)
    {
        // STEP 1 : put each line of matrix Jt in the right hand term of the system
        for (typename JMatrixType::Index i=0; i<J->colSize(); i++) linearSystem.systemRHVector->set(i, J->element(row, i)); // linearSystem.systemMatrix->rowSize()

        // STEP 2 : solve the system :
        solveSystem();

        // STEP 3 : project the result using matrix J
        if (const linearalgebra::SparseMatrix<Real> * j = dynamic_cast<const linearalgebra::SparseMatrix<Real> * >(J))   // optimization for sparse matrix
        {
            const typename linearalgebra::SparseMatrix<Real>::LineConstIterator jitend = j->end();
            for (typename linearalgebra::SparseMatrix<Real>::LineConstIterator jit = j->begin(); jit != jitend; ++jit)
            {
                auto row2 = jit->first;
                double acc = 0.0;
                for (typename linearalgebra::SparseMatrix<Real>::LElementConstIterator i2 = jit->second.begin(), i2end = jit->second.end(); i2 != i2end; ++i2)
                {
                    auto col2 = i2->first;
                    double val2 = i2->second;
                    acc += val2 * linearSystem.systemLHVector->element(col2);
                }
                acc *= fact;
                result->add(row2,row,acc);
            }
        }
        else
        {
            dmsg_error() << "addJMInvJt is only implemented for linearalgebra::SparseMatrix<Real>" ;
            return false;
        }
    }

    return true;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, SReal fact)
{
    for (typename JMatrixType::Index row=0; row<J->rowSize(); row++)
    {
        // STEP 1 : put each line of matrix Jt in the right hand term of the system
        for (typename JMatrixType::Index i=0; i<J->colSize(); i++) 
            linearSystem.systemRHVector->set(i, J->element(row, i)); // linearSystem.systemMatrix->rowSize()

        // STEP 2 : solve the system :
        solveSystem();

        // STEP 3 : project the result using matrix J
        for (typename JMatrixType::Index i=0; i<J->colSize(); i++) result->add(row, i, linearSystem.systemRHVector->element(i) * fact);
    }

    return true;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    if (J->rowSize()==0) return true;

    JMatrixType * j_local = internalData.getLocalJ(J);
    ResMatrixType * res_local = internalData.getLocalRes(result);
    bool res = addJMInvJtLocal(linearSystem.systemMatrix, res_local, j_local, fact);
    internalData.addLocalRes(result);
    return res;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    if (J->rowSize()==0) return true;

    JMatrixType * j_local = internalData.getLocalJ(J);
    ResMatrixType * res_local = internalData.getLocalRes(result);
    bool res = addMInvJtLocal(linearSystem.systemMatrix, res_local, j_local, fact);
    internalData.addLocalRes(result);
    return res;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::buildComplianceMatrix(const sofa::core::ConstraintParams* cparams, linearalgebra::BaseMatrix* result, SReal fact)
{
    JMatrixType * j_local = internalData.getLocalJ();
    j_local->clear();
    j_local->resize(result->rowSize(), linearSystem.systemMatrix->colSize());

    if (result->rowSize() == 0)
    {
        return true;
    }

    executeVisitor(MechanicalGetConstraintJacobianVisitor(cparams,j_local));

    return addJMInvJt(result,j_local,fact);
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::applyConstraintForce(const sofa::core::ConstraintParams* cparams, sofa::core::MultiVecDerivId dx, const linearalgebra::BaseVector* f)
{
    linearSystem.systemRHVector->clear();
    linearSystem.systemRHVector->resize(linearSystem.systemMatrix->colSize());
    /// rhs = J^t * f
    internalData.projectForceInConstraintSpace(linearSystem.systemRHVector, f);
    /// lhs = M^-1 * rhs
    this->solve(*linearSystem.systemMatrix, *linearSystem.systemLHVector, *linearSystem.systemRHVector);

    executeVisitor(MechanicalMultiVectorFromBaseVectorVisitor(cparams, dx, linearSystem.systemLHVector, &(linearSystem.matrixAccessor)) );
    executeVisitor(MechanicalMultiVectorFromBaseVectorVisitor(cparams, cparams->lambda(), linearSystem.systemRHVector, &(linearSystem.matrixAccessor)));
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::computeResidual(const core::ExecParams* params,linearalgebra::BaseVector* f) {
    linearSystem.systemRHVector->clear();
    linearSystem.systemRHVector->resize(linearSystem.systemMatrix->colSize());

    internalData.projectForceInConstraintSpace(linearSystem.systemRHVector, f);

    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::core::behavior::MultiVecDeriv force(&vop, core::VecDerivId::force() );

    executeVisitor( MechanicalMultiVectorPeqBaseVectorVisitor(core::execparams::defaultInstance(), force, linearSystem.systemRHVector, &(linearSystem.matrixAccessor)) );
}



} // namespace sofa::component::linearsolver
