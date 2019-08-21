/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_INL
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_INL

#include <SofaBaseLinearSolver/MatrixLinearSolver.h>

#include <sofa/helper/BackTrace.h>

namespace sofa {

namespace component {

namespace linearsolver {


template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::MatrixLinearSolver()
    : Inherit()
    , currentGroup(&defaultGroup)
{
    invertData = nullptr;
}

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::~MatrixLinearSolver()
{
    if (invertData) delete invertData;
    invertData = nullptr;
}

template<class Matrix, class Vector>
MatrixInvertData * MatrixLinearSolver<Matrix,Vector>::getMatrixInvertData(defaulttype::BaseMatrix * /*m*/)
{
    if (invertData==nullptr) invertData=createInvertData();
    return invertData;
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
        if (currentGroup->systemMatrix) currentGroup->systemMatrix->clear();
        currentGroup->needInvert = true;
    }
    if (currentGroup->systemRHVector) currentGroup->systemRHVector->clear();
    if (currentGroup->systemLHVector) currentGroup->systemLHVector->clear();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resizeSystem(int n)
{
    if (!this->frozen)
    {
        if (!currentGroup->systemMatrix) currentGroup->systemMatrix = createMatrix();
        currentGroup->systemMatrix->resize(n,n);
    }
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = createPersistentVector();
    currentGroup->systemRHVector->resize(n);

    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = createPersistentVector();
    currentGroup->systemLHVector->resize(n);
    currentGroup->needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemMatrix(Matrix * matrix)
{
    if (currentGroup==nullptr) currentGroup=&defaultGroup;
    currentGroup->systemMatrix = matrix;
    if (matrix!=nullptr) {
        if (!currentGroup->systemRHVector) currentGroup->systemRHVector = createPersistentVector();
        currentGroup->systemRHVector->resize(matrix->colSize());
        if (!currentGroup->systemLHVector) currentGroup->systemLHVector = createPersistentVector();
        currentGroup->systemLHVector->resize(matrix->colSize());
    }
    currentGroup->needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    this->currentMFactor = mparams->mFactor();
    this->currentBFactor = mparams->bFactor();
    this->currentKFactor = mparams->kFactor();

    if (!this->frozen)
    {
        simulation::Node* root = dynamic_cast<simulation::Node*>(this->getContext());
        SReal dim = 0;
        simulation::MechanicalGetDimensionVisitor(mparams, &dim).execute(root);
        currentGroup->systemSize = dim;
        currentGroup->matrixAccessor.setDoPrintInfo( this->f_printLog.getValue() ) ;

        simulation::common::MechanicalOperations mops(mparams, this->getContext());
        if (!currentGroup->systemMatrix) currentGroup->systemMatrix = createMatrix();
        currentGroup->matrixAccessor.setGlobalMatrix(currentGroup->systemMatrix);
        currentGroup->matrixAccessor.clear();

        mops.getMatrixDimension(&(currentGroup->matrixAccessor));
        currentGroup->matrixAccessor.setupMatrices();
        resizeSystem(currentGroup->matrixAccessor.getGlobalDimension());
        currentGroup->systemMatrix->clear();
        mops.addMBK_ToMatrix(&(currentGroup->matrixAccessor), mparams->mFactor(), mparams->bFactor(), mparams->kFactor());
        currentGroup->matrixAccessor.computeGlobalMatrix();
    }

}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::rebuildSystem(double massFactor, double forceFactor)
{
    sofa::core::MechanicalParams mparams;
    mparams.setMFactor(this->currentMFactor*massFactor);
    mparams.setBFactor(this->currentBFactor*forceFactor);
    mparams.setKFactor(this->currentKFactor*forceFactor);
    if (!this->frozen)
    {
        simulation::common::MechanicalOperations mops(&mparams, this->getContext());
        if (!currentGroup->systemMatrix) currentGroup->systemMatrix = createMatrix();
        currentGroup->matrixAccessor.setGlobalMatrix(currentGroup->systemMatrix);
        currentGroup->matrixAccessor.clear();
        mops.getMatrixDimension(&(currentGroup->matrixAccessor));
        currentGroup->matrixAccessor.setupMatrices();
        resizeSystem(currentGroup->matrixAccessor.getGlobalDimension());
        currentGroup->systemMatrix->clear();
        mops.addMBK_ToMatrix(&(currentGroup->matrixAccessor), mparams.mFactor(), mparams.bFactor(), mparams.kFactor());
        currentGroup->matrixAccessor.computeGlobalMatrix();
    }

    this->invertSystem();
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemRHVector(core::MultiVecDerivId v)
{
    executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor(core::ExecParams::defaultInstance(), v,
                                                                         currentGroup->systemRHVector,
                                                                         &(currentGroup->matrixAccessor)) );
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemLHVector(core::MultiVecDerivId v)
{
    currentGroup->solutionVecId = v;
    if (!currentGroup->solutionVecId.isNull())
    {
        executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor( core::ExecParams::defaultInstance(), v, currentGroup->systemLHVector, &(currentGroup->matrixAccessor)) );
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::solveSystem()
{
    if (currentGroup->needInvert)
    {
        this->invert(*currentGroup->systemMatrix);
        currentGroup->needInvert = false;
    }
    this->solve(*currentGroup->systemMatrix, *currentGroup->systemLHVector, *currentGroup->systemRHVector);
    if (!currentGroup->solutionVecId.isNull())
    {
        executeVisitor( simulation::MechanicalMultiVectorFromBaseVectorVisitor(core::ExecParams::defaultInstance(), currentGroup->solutionVecId, currentGroup->systemLHVector, &(currentGroup->matrixAccessor)) );
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
    if (currentGroup->needInvert)
    {
        this->invert(*currentGroup->systemMatrix);
        currentGroup->needInvert = false;
    }
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addJMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, double fact)
{
    for (typename JMatrixType::Index row=0; row<J->rowSize(); row++)
    {
        // STEP 1 : put each line of matrix Jt in the right hand term of the system
        for (typename JMatrixType::Index i=0; i<J->colSize(); i++) currentGroup->systemRHVector->set(i,J->element(row,i)); // currentGroup->systemMatrix->rowSize()

        // STEP 2 : solve the system :
        solveSystem();

        // STEP 3 : project the result using matrix J
        if (const SparseMatrix<Real> * j = dynamic_cast<const SparseMatrix<Real> * >(J))   // optimization for sparse matrix
        {
            const typename SparseMatrix<Real>::LineConstIterator jitend = j->end();
            for (typename SparseMatrix<Real>::LineConstIterator jit = j->begin(); jit != jitend; ++jit)
            {
                int row2 = jit->first;
                double acc = 0.0;
                for (typename SparseMatrix<Real>::LElementConstIterator i2 = jit->second.begin(), i2end = jit->second.end(); i2 != i2end; ++i2)
                {
                    int col2 = i2->first;
                    double val2 = i2->second;
                    acc += val2 * currentGroup->systemLHVector->element(col2);
                }
                acc *= fact;
                result->add(row2,row,acc);
            }
        }
        else
        {
            dmsg_error("MatrixLinearSolver") << "AsyncMatrixLinearSolver::addJMInvJt is only implemented for SparseMatrix<Real>" ;
            return false;
        }
    }

    return true;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, double fact)
{
    for (typename JMatrixType::Index row=0; row<J->rowSize(); row++)
    {
        // STEP 1 : put each line of matrix Jt in the right hand term of the system
        for (typename JMatrixType::Index i=0; i<J->colSize(); i++) currentGroup->systemRHVector->set(i,J->element(row,i)); // currentGroup->systemMatrix->rowSize()

        // STEP 2 : solve the system :
        solveSystem();

        // STEP 3 : project the result using matrix J
        for (typename JMatrixType::Index i=0; i<J->colSize(); i++) result->add(row,i,currentGroup->systemRHVector->element(i) * fact);
    }

    return true;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (J->rowSize()==0) return true;

    JMatrixType * j_local = internalData.getLocalJ(J);
    ResMatrixType * res_local = internalData.getLocalRes(result);
    bool res = addJMInvJtLocal(currentGroup->systemMatrix,res_local,j_local,fact);
    internalData.addLocalRes(result);
    return res;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (J->rowSize()==0) return true;

    JMatrixType * j_local = internalData.getLocalJ(J);
    ResMatrixType * res_local = internalData.getLocalRes(result);
    bool res = addMInvJtLocal(currentGroup->systemMatrix,res_local,j_local,fact);
    internalData.addLocalRes(result);
    return res;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::buildComplianceMatrix(const sofa::core::ConstraintParams* cparams, defaulttype::BaseMatrix* result, double fact)
{
    JMatrixType * j_local = internalData.getLocalJ();
    j_local->clear();
    j_local->resize(result->rowSize(), currentGroup->systemMatrix->colSize());

    if (result->rowSize() == 0)
    {
        return true;
    }

    executeVisitor(simulation::MechanicalGetConstraintJacobianVisitor(cparams,j_local));

    return addJMInvJt(result,j_local,fact);
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::applyConstraintForce(const sofa::core::ConstraintParams* cparams, sofa::core::MultiVecDerivId dx, const defaulttype::BaseVector* f)
{
    currentGroup->systemRHVector->clear();
    currentGroup->systemRHVector->resize(currentGroup->systemMatrix->colSize());
    /// rhs = J^t * f
    internalData.projectForceInConstraintSpace(currentGroup->systemRHVector,f);
    /// lhs = M^-1 * rhs
    this->solve(*currentGroup->systemMatrix,*currentGroup->systemLHVector,*currentGroup->systemRHVector);

    executeVisitor(simulation::MechanicalMultiVectorFromBaseVectorVisitor(cparams, dx, currentGroup->systemLHVector, &(currentGroup->matrixAccessor)) );
    executeVisitor(simulation::MechanicalMultiVectorFromBaseVectorVisitor(cparams, cparams->lambda(), currentGroup->systemRHVector, &(currentGroup->matrixAccessor)));
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::computeResidual(const core::ExecParams* params,defaulttype::BaseVector* f) {
    currentGroup->systemRHVector->clear();
    currentGroup->systemRHVector->resize(currentGroup->systemMatrix->colSize());

    internalData.projectForceInConstraintSpace(currentGroup->systemRHVector,f);

    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::core::behavior::MultiVecDeriv force(&vop, core::VecDerivId::force() );

    executeVisitor( simulation::MechanicalMultiVectorPeqBaseVectorVisitor(core::ExecParams::defaultInstance(), force, currentGroup->systemRHVector, &(currentGroup->matrixAccessor)) );
}



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
