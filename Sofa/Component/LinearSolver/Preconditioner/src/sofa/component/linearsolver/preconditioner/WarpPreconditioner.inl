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

#include <sofa/component/linearsolver/preconditioner/WarpPreconditioner.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/component/linearsolver/preconditioner/RotationMatrixSystem.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <iostream>
#include <cmath>

#include <sofa/type/Quat.h>

namespace sofa::component::linearsolver::preconditioner
{

template<class TMatrix, class TVector,class ThreadManager>
WarpPreconditioner<TMatrix,TVector,ThreadManager >::WarpPreconditioner()
    : l_linearSolver(initLink("linearSolver", "Link towards the linear solver used to build the warp conditioner"))
    , f_useRotationFinder(initData(&f_useRotationFinder, (unsigned)0, "useRotationFinder", "Which rotation Finder to use" ) )
    , d_updateStep(initData(&d_updateStep, 1u, "update_step", "Number of steps before the next refresh of the system matrix in the main solver" ) )
{
    rotationWork[0] = nullptr;
    rotationWork[1] = nullptr;

    first = true;
    indexwork = 0;
}

template<class TMatrix, class TVector,class ThreadManager>
WarpPreconditioner<TMatrix,TVector,ThreadManager >::~WarpPreconditioner()
{
    if (rotationWork[0]) delete rotationWork[0];
    if (rotationWork[1]) delete rotationWork[1];

    rotationWork[0] = nullptr;
    rotationWork[1] = nullptr;
}

template <class TMatrix, class TVector, class ThreadManager>
void WarpPreconditioner<TMatrix, TVector, ThreadManager>::init()
{
    Inherit1::init();
    first = true;
}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::bwdInit()
{
    if (l_linearSolver.empty())
    {
        msg_info() << "Link \"linearSolver\" to the desired linear solver should be set to ensure right behavior." << msgendl
                   << "First LinearSolver found in current context will be used.";
        l_linearSolver.set( this->getContext()->template get<sofa::core::behavior::LinearSolver>(sofa::core::objectmodel::BaseContext::Local) );
    }

    if (l_linearSolver.get() == nullptr)
    {
        msg_error() << "No LinearSolver component found at path: " << l_linearSolver.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    else
    {
        if (l_linearSolver.get()->getTemplateName() == "GraphScattered")
        {
            msg_error() << "Can not use the solver " << l_linearSolver.get()->getName() << " because it is templated on GraphScatteredType";
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        else
        {
            msg_info() << "LinearSolver path used: '" << l_linearSolver.getLinkedPath() << "'";
        }
    }

    const sofa::core::objectmodel::BaseContext * c = this->getContext();
    c->get<sofa::core::behavior::BaseRotationFinder >(&rotationFinders, sofa::core::objectmodel::BaseContext::Local);

    std::stringstream tmpStr;
    tmpStr << "Found " << rotationFinders.size() << " Rotation finders" << msgendl;
    for (unsigned i=0; i<rotationFinders.size(); i++) {
        tmpStr << i << " : " << rotationFinders[i]->getName() << msgendl;
    }
    msg_info() << tmpStr.str();

    first = true;
    indexwork = 0;
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class TMatrix, class TVector,class ThreadManager>
typename  WarpPreconditioner<TMatrix, TVector, ThreadManager >::Index
WarpPreconditioner<TMatrix,TVector,ThreadManager >::getSystemDimention(const sofa::core::MechanicalParams* mparams) {
    simulation::common::MechanicalOperations mops(mparams, this->getContext());

    this->linearSystem.matrixAccessor.setGlobalMatrix(this->getSystemMatrix());
    this->linearSystem.matrixAccessor.clear();
    mops.getMatrixDimension(&(this->linearSystem.matrixAccessor));
    this->linearSystem.matrixAccessor.setupMatrices();
    return this->linearSystem.matrixAccessor.getGlobalDimension();
}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams)
{
    this->currentMFactor = mparams->mFactor();
    this->currentBFactor = sofa::core::mechanicalparams::bFactor(mparams);
    this->currentKFactor = mparams->kFactor();
    if (!this->frozen)
    {
        simulation::common::MechanicalOperations mops(mparams, this->getContext());
        if (!this->l_linearSystem)
        {
            msg_error() << "No linear system associated to this component. Cannot assemble the matrix.";
            return;
        }
    }

    if (first || ( d_updateStep.getValue() > 0 && nextRefreshStep >= d_updateStep.getValue()) || (d_updateStep.getValue() == 0))
    {
        l_linearSolver.get()->setSystemMBKMatrix(mparams);
        nextRefreshStep = 1;
    }

    if (first)
    {
        updateSystemSize = getSystemDimention(mparams);
        this->resizeSystem(updateSystemSize);

        first = false;

        if (!rotationWork[indexwork]) rotationWork[indexwork] = new TRotationMatrix();

        rotationWork[indexwork]->resize(updateSystemSize,updateSystemSize);
        rotationFinders[f_useRotationFinder.getValue()]->getRotations(rotationWork[indexwork]);

        if (l_linearSolver.get()->isAsyncSolver()) indexwork = (indexwork==0) ? 1 : 0;

        if (!rotationWork[indexwork]) rotationWork[indexwork] = new TRotationMatrix();

        rotationWork[indexwork]->resize(updateSystemSize,updateSystemSize);
        rotationFinders[f_useRotationFinder.getValue()]->getRotations(rotationWork[indexwork]);

        this->l_linearSystem->resizeSystem(updateSystemSize);
        this->l_linearSystem->getSystemMatrix()->setIdentity(); // identity rotationa after update

    }
    else if (l_linearSolver.get()->hasUpdatedMatrix())
    {
        updateSystemSize = getSystemDimention(mparams);
        this->resizeSystem(updateSystemSize);

        if (!rotationWork[indexwork]) rotationWork[indexwork] = new TRotationMatrix();

        rotationWork[indexwork]->resize(updateSystemSize,updateSystemSize);
        rotationFinders[f_useRotationFinder.getValue()]->getRotations(rotationWork[indexwork]);

        if (l_linearSolver.get()->isAsyncSolver()) indexwork = (indexwork==0) ? 1 : 0;

        this->l_linearSystem->getSystemMatrix()->setIdentity(); // identity rotationa after update
    }
    else
    {
        currentSystemSize = getSystemDimention(sofa::core::mechanicalparams::defaultInstance());

        this->l_linearSystem->resizeSystem(currentSystemSize);
        this->l_linearSystem->getSystemMatrix()->clear();

        rotationFinders[f_useRotationFinder.getValue()]->getRotations(this->l_linearSystem->getSystemMatrix());

        this->l_linearSystem->getSystemMatrix()->opMulTM(this->l_linearSystem->getSystemMatrix(),rotationWork[indexwork]);
    }
}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::invert(Matrix& /*Rcur*/) {}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::updateSystemMatrix() {
    ++nextRefreshStep;
    l_linearSolver.get()->updateSystemMatrix();
}

template <class TMatrix, class TVector, class ThreadManager>
void WarpPreconditioner<TMatrix, TVector, ThreadManager>::checkLinearSystem()
{
    if (!this->l_linearSystem)
    {
        auto* matrixLinearSystem = this->getContext()->template get<RotationMatrixSystem<Matrix, Vector> >();
        if (!matrixLinearSystem)
        {
            this->template createDefaultLinearSystem<RotationMatrixSystem<Matrix, Vector> >();
        }
    }
}


/// Solve the system as constructed using the previous methods
template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::solve(Matrix& Rcur, Vector& solution, Vector& rh) {
    Rcur.opMulTV(l_linearSolver.get()->getSystemRHBaseVector(),&rh);

    l_linearSolver.get()->solveSystem();

    Rcur.opMulV(&solution,l_linearSolver.get()->getSystemLHBaseVector());
}

/// Solve the system as constructed using the previous methods
template<class TMatrix, class TVector,class ThreadManager>
bool WarpPreconditioner<TMatrix,TVector,ThreadManager >::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) {
    if (J->rowSize()==0 || !l_linearSolver.get()) return true;

    this->l_linearSystem->getSystemMatrix()->rotateMatrix(&j_local,J);

    return l_linearSolver.get()->addJMInvJt(result,&j_local,fact);
}

template<class TMatrix, class TVector,class ThreadManager>
bool WarpPreconditioner<TMatrix,TVector,ThreadManager >::addMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) {
    this->l_linearSystem->getSystemMatrix()->rotateMatrix(&j_local,J);
    return l_linearSolver.get()->addMInvJt(result,&j_local,fact);
}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::computeResidual(const core::ExecParams* params, linearalgebra::BaseVector* f) {
    l_linearSolver.get()->computeResidual(params,f);
}

} // namespace sofa::component::linearsolver::preconditioner
