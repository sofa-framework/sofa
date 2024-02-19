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

#include <sofa/core/ObjectFactory.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.inl>

namespace sofa::component::linearsolver
{

using namespace sofa::component::linearsystem;

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::MatrixLinearSolver()
    : Inherit()
    , d_parallelInverseProduct(initData(&d_parallelInverseProduct, false,
                                        "parallelInverseProduct", "Parallelize the computation of the product J*M^{-1}*J^T "
                                                                  "where M is the matrix of the linear system and J is any "
                                                                  "matrix with compatible dimensions"))
    , invertData()
    , linearSystem()
    , currentMFactor(), currentBFactor(), currentKFactor()
    , l_linearSystem(initLink("linearSystem", "The linear system to solve"))
{
    this->addUpdateCallback("parallelInverseProduct", {&d_parallelInverseProduct},
    [this](const core::DataTracker& tracker) -> sofa::core::objectmodel::ComponentState
    {
        SOFA_UNUSED(tracker);
        if (d_parallelInverseProduct.getValue())
        {
            simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
            assert(taskScheduler);

            if (taskScheduler->getThreadCount() < 1)
            {
                taskScheduler->init(0);
                msg_info() << "Task scheduler initialized on " << taskScheduler->getThreadCount() << " threads";
            }
            else
            {
                msg_info() << "Task scheduler already initialized on " << taskScheduler->getThreadCount() << " threads";
            }
        }
        return this->d_componentState.getValue();
    },
    {});
}

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::~MatrixLinearSolver() = default;

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::init()
{
    Inherit1::init();

    checkLinearSystem();

    this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
}

template <class Matrix, class Vector>
void MatrixLinearSolver<Matrix, Vector, NoThreadManager>::checkLinearSystem()
{
    doCheckLinearSystem<MatrixLinearSystem<Matrix, Vector> >();
}

template <class Matrix, class Vector>
template<class TLinearSystemType>
void MatrixLinearSolver<Matrix, Vector, NoThreadManager>::doCheckLinearSystem()
{
    if (!l_linearSystem)
    {
        sofa::type::vector<TypedMatrixLinearSystem<Matrix, Vector>* > listLinearSystems;
        this->getContext()->getObjects(listLinearSystems);

        listLinearSystems.erase(std::remove(listLinearSystems.begin(), listLinearSystems.end(), nullptr),
                                listLinearSystems.end());

        if (listLinearSystems.empty())
        {
            sofa::type::vector<sofa::core::behavior::BaseMatrixLinearSystem* > listBaseLinearSystems;
            this->getContext()->getObjects(listBaseLinearSystems);

            if (listBaseLinearSystems.empty())
            {
                msg_info() << "A linear system is required, but has not been found. Add a linear system to your scene to "
                    "remove this warning. The list of available linear system components is: ["
                    << sofa::core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::BaseMatrixLinearSystem>() << "].\n"
                    << "A component of type " << TLinearSystemType::GetClass()->className << " (template "
                    << TLinearSystemType::GetClass()->templateName << ") will be automatically added for you in Node "
                    << this->getContext()->getPathName() << ".";
            }
            else
            {
                msg_warning() << "A linear system has been found, but not the expected type."
                    << "Add a linear system with a compatible type to your scene to remove this warning.\n"
                    << "A component of type " << TLinearSystemType::GetClass()->className << " (template "
                    << TLinearSystemType::GetClass()->templateName << ") will be automatically added for you in Node "
                    << this->getContext()->getPathName() << ".";
            }
            createDefaultLinearSystem<TLinearSystemType>();
        }
        else
        {
            sofa::type::vector<MatrixLinearSolver<Matrix, Vector>* > listSolvers;
            this->getContext()->getObjects(listSolvers);

            listSolvers.erase(std::remove(listSolvers.begin(), listSolvers.end(), nullptr),
                              listSolvers.end());

            sofa::type::vector<TypedMatrixLinearSystem<Matrix, Vector>* > notAlreadyAssociated;
            for (auto system : listLinearSystems)
            {
                if (std::none_of(listSolvers.begin(), listSolvers.end(),
                    [system](auto solver)
                    {
                        return solver && solver->l_linearSystem.get() == system;
                    }))
                {
                    notAlreadyAssociated.push_back(system);
                }
            }

            if (notAlreadyAssociated.empty())
            {
                msg_warning() << "A linear system has been found, but it is already associated to another linear solver. "
                    << "A component of type " << TLinearSystemType::GetClass()->className << " (template "
                    << TLinearSystemType::GetClass()->templateName << ") will be automatically added for you in Node "
                    << this->getContext()->getPathName() << ".";
                createDefaultLinearSystem<TLinearSystemType>();
            }
            else
            {
                if (notAlreadyAssociated.size() == 1)
                {
                    msg_info() << "Linear system found: " << l_linearSystem->getPathName();
                }
                else
                {
                    msg_warning() << "Several linear systems have been found and are candidates to be associated "
                        << "to this linear solver. The first one in the list is selected. Set the link " << l_linearSystem.getLinkedPath()
                        << " properly to remove this warning.";
                }
                l_linearSystem.set(*notAlreadyAssociated.begin());
            }
        }
    }
}


template<class Matrix, class Vector>
template<class TLinearSystemType>
void MatrixLinearSolver<Matrix,Vector>::createDefaultLinearSystem()
{
    if (auto system = sofa::core::objectmodel::New<TLinearSystemType>())
    {
        this->addSlave(system);
        system->setName( this->getContext()->getNameHelper().resolveName(system->getClassName(), {}));
        system->f_printLog.setValue(this->f_printLog.getValue());
        l_linearSystem.set(system);
    }
    else
    {
        msg_error() << TLinearSystemType::GetClass()->className << " failed to be instantiated";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

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
        if (auto* systemMatrix = this->getSystemMatrix())
        {
            systemMatrix->clear();
        }
        linearSystem.needInvert = true;
    }
    if (auto* rhs = this->getSystemRHVector())
    {
        rhs->clear();
    }
    if (auto* solution = this->getSystemLHVector())
    {
        solution->clear();
    }
    linearSystem.solutionVecId = core::MultiVecDerivId::null();
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resizeSystem(Size n)
{
    if (this->getSystemRHVector())
    {
        this->getSystemRHVector()->resize(n);
    }

    if (this->getSystemLHVector())
    {
        this->getSystemLHVector()->resize(n);
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
        if (l_linearSystem)
        {
            l_linearSystem->buildSystemMatrix(mparams);
            resizeSystem(l_linearSystem->getMatrixSize()[0]);
        }
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
    if (l_linearSystem)
    {
        l_linearSystem->setRHS(v);
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemLHVector(core::MultiVecDerivId v)
{
    linearSystem.solutionVecId = v;
    if (l_linearSystem)
    {
        l_linearSystem->setSystemSolution(v);
    }
}

template <class Matrix, class Vector>
Matrix* MatrixLinearSolver<Matrix, Vector, NoThreadManager>::getSystemMatrix()
{
    return l_linearSystem ? l_linearSystem->getSystemMatrix() : nullptr;
}

template <class Matrix, class Vector>
linearalgebra::BaseMatrix* MatrixLinearSolver<Matrix, Vector, NoThreadManager>::getSystemBaseMatrix()
{
    if (!l_linearSystem)
    {
        return nullptr;
    }
    return l_linearSystem->getSystemMatrix();
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::solveSystem()
{
    auto* systemMatrix = getSystemMatrix();
    if (!systemMatrix)
    {
        msg_error() << "System matrix is not setup properly";
        return;
    }

    // Step 1: Invert the system, e.g. factorization of the matrix
    if (linearSystem.needInvert)
    {
        this->invert(*systemMatrix);
        linearSystem.needInvert = false;
    }

    // Step 2: Solve the system based on the system inversion
    this->solve(*systemMatrix, *this->getSystemLHVector(), *this->getSystemRHVector());

    // Step 3: Apply the solution
    applySystemSolution();
}

template <class Matrix, class Vector>
void MatrixLinearSolver<Matrix, Vector, NoThreadManager>::applySystemSolution()
{
    if (!linearSystem.solutionVecId.isNull())
    {
        if (l_linearSystem)
        {
            l_linearSystem->dispatchSystemSolution(linearSystem.solutionVecId);
        }
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
    if (linearSystem.needInvert && l_linearSystem)
    {
        this->invert(*l_linearSystem->getSystemMatrix());
        linearSystem.needInvert = false;
    }
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix, Vector>::addJMInvJtLocal(Matrix* M, ResMatrixType* result, const JMatrixType* J, const SReal fact)
{
    if (!this->isComponentStateValid())
    {
        return true;
    }

    if (!d_parallelInverseProduct.getValue())
    {
        return singleThreadAddJMInvJtLocal(M, result, J, fact);
    }

    static_assert(std::is_same_v<JMatrixType, linearalgebra::SparseMatrix<Real>>, "This function supposes a SparseMatrix");

    auto* systemMatrix = getSystemMatrix();
    if (!systemMatrix)
    {
        msg_error() << "System matrix is not setup properly";
        return false;
    }

    if (linearSystem.needInvert)
    {
        this->invert(*systemMatrix);
        linearSystem.needInvert = false;
    }

    simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler);

    sofa::type::vector<Vector> rhsVector(J->rowSize());
    sofa::type::vector<Vector> lhsVector(J->rowSize());
    sofa::type::vector<Vector> columnResult(J->rowSize());

    std::mutex mutex;

    simulation::parallelForEach(*taskScheduler, 0, J->rowSize(),
        [&](const typename JMatrixType::Index row)
        {
            rhsVector[row].resize(J->colSize());
            lhsVector[row].resize(J->colSize());
            columnResult[row].resize(J->colSize());

            // STEP 1 : put each line of matrix Jt in the right hand term of the system
            for (typename JMatrixType::Index i = 0; i < J->colSize(); ++i)
            {
                rhsVector[row].set(i, J->element(row, i));
            }

            // STEP 2 : solve the system :
            this->solve(*systemMatrix, lhsVector[row], rhsVector[row]);

            // STEP 3 : project the result using matrix J
            for (const auto& [row2, line] : *J)
            {
                Real acc = 0;
                for (const auto& [col2, val2] : line)
                {
                    acc += val2 * lhsVector[row].element(col2);
                }
                acc *= fact;
                columnResult[row][row2] += acc;
            }

            // STEP 4 : assembly of the result
            std::lock_guard lock(mutex);

            for (const auto& [row2, line] : *J)
            {
                result->add(row2, row, columnResult[row][row2]);
            }
        }
    );

    return true;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix, Vector>::singleThreadAddJMInvJtLocal(Matrix* M, ResMatrixType* result, const JMatrixType* J, const SReal fact)
{
    SOFA_UNUSED(M);
    static_assert(std::is_same_v<JMatrixType, linearalgebra::SparseMatrix<Real>>, "This function supposes a SparseMatrix");

    auto* systemMatrix = getSystemMatrix();
    if (!systemMatrix)
    {
        msg_error() << "System matrix is not setup properly";
        return false;
    }

    if (linearSystem.needInvert)
    {
        this->invert(*systemMatrix);
        linearSystem.needInvert = false;
    }

    for (typename JMatrixType::Index row = 0; row < J->rowSize(); ++row)
    {
        // STEP 1 : put each line of matrix Jt in the right hand term of the system
        for (typename JMatrixType::Index i = 0; i < J->colSize(); ++i)
        {
            this->getSystemRHVector()->set(i, J->element(row, i)); // linearSystem.systemMatrix->rowSize()
        }

        // STEP 2 : solve the system :
        this->solve(*systemMatrix, *this->getSystemLHVector(), *this->getSystemRHVector());

        // STEP 3 : project the result using matrix J
        for (const auto& [row2, line] : *J)
        {
            Real acc = 0;
            for (const auto& [col2, val2] : line)
            {
                acc += val2 * getSystemLHVector()->element(col2);
            }
            acc *= fact;
            result->add(row2, row, acc);
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
        {
            getSystemRHVector()->set(i, J->element(row, i)); // linearSystem.systemMatrix->rowSize()
        }

        // STEP 2 : solve the system :
        solveSystem();

        // STEP 3 : project the result using matrix J
        for (typename JMatrixType::Index i=0; i<J->colSize(); i++)
        {
            result->add(row, i, getSystemRHVector()->element(i) * fact);
        }
    }

    return true;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    if (J->rowSize()==0) return true;

    const JMatrixType * j_local = internalData.getLocalJ(J);
    ResMatrixType * res_local = internalData.getLocalRes(result);
    const bool res = addJMInvJtLocal(getSystemMatrix(), res_local, j_local, fact);
    internalData.addLocalRes(result);
    return res;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    if (J->rowSize()==0) return true;

    const JMatrixType * j_local = internalData.getLocalJ(J);
    ResMatrixType * res_local = internalData.getLocalRes(result);
    const bool res = addMInvJtLocal(getSystemMatrix(), res_local, j_local, fact);
    internalData.addLocalRes(result);
    return res;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::buildComplianceMatrix(const sofa::core::ConstraintParams* cparams, linearalgebra::BaseMatrix* result, SReal fact)
{
    JMatrixType * j_local = internalData.getLocalJ();
    j_local->clear();
    j_local->resize(result->rowSize(), getSystemMatrix()->colSize());

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
    getSystemRHVector()->clear();
    getSystemRHVector()->resize(getSystemMatrix()->colSize());
    /// rhs = J^t * f
    internalData.projectForceInConstraintSpace(getSystemRHVector(), f);
    /// lhs = M^-1 * rhs
    this->solve(*getSystemMatrix(), *getSystemLHVector(), *getSystemRHVector());

    l_linearSystem->dispatchSystemSolution(dx);
    l_linearSystem->dispatchSystemRHS(cparams->lambda());
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::computeResidual(const core::ExecParams* params,linearalgebra::BaseVector* f) {
    getSystemRHVector()->clear();
    getSystemRHVector()->resize(getSystemMatrix()->colSize());

    internalData.projectForceInConstraintSpace(getSystemRHVector(), f);

    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::core::behavior::MultiVecDeriv force(&vop, core::VecDerivId::force() );

    executeVisitor( MechanicalMultiVectorPeqBaseVectorVisitor(core::execparams::defaultInstance(), force, getSystemRHVector(), &(linearSystem.matrixAccessor)) );
}



} // namespace sofa::component::linearsolver
