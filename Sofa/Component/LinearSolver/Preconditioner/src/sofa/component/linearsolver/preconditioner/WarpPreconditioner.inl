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
    , d_useRotationFinder(this, "v25.12", "v26.06", "useRotationFinder", "This Data has been replaced with the Link 'rotationFinder' in RotationMatrixSystem")
    , d_updateStep(this, "v25.12", "v26.06", "update_step", "Instead, use the Data 'assemblingRate' in the RotationMatrixSystem" )
{
}

template <class TMatrix, class TVector, class ThreadManager>
void WarpPreconditioner<TMatrix, TVector, ThreadManager>::init()
{
    Inherit1::init();

    ensureRequiredLinearSystemType();
}

template <class TMatrix, class TVector, class ThreadManager>
void WarpPreconditioner<TMatrix, TVector, ThreadManager>::ensureRequiredLinearSystemType()
{
    if (this->l_linearSystem)
    {
        auto* preconditionedMatrix =
            dynamic_cast<RotationMatrixSystem<TMatrix, TVector>*>(this->l_linearSystem.get());
        if (!preconditionedMatrix)
        {
            msg_error() << "This linear solver is designed to work with a "
                        << RotationMatrixSystem<TMatrix, TVector>::GetClass()->className
                        << " linear system, but a " << this->l_linearSystem->getClassName()
                        << " was found";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }
}

template <class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::bwdInit()
{
    if (l_linearSolver.empty())
    {
        msg_info() << "Link \"" << l_linearSolver.getName() << "\" to the desired linear solver should be set to ensure right behavior." << msgendl
                   << "First LinearSolver found in current context will be used.";
        l_linearSolver.set( this->getContext()->template get<sofa::core::behavior::LinearSolver>(sofa::core::objectmodel::BaseContext::Local) );
    }

    if (l_linearSolver.get() == nullptr)
    {
        msg_error() << "No LinearSolver component found at path: " << l_linearSolver.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (l_linearSolver->getTemplateName() == "GraphScattered")
    {
        msg_error() << "Cannot use the solver " << l_linearSolver->getName()
                    << " because it is templated on GraphScatteredType";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "LinearSolver path used: '" << l_linearSolver.getLinkedPath() << "'";

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::invert(Matrix& /*Rcur*/) {}

template <class TMatrix, class TVector, class ThreadManager>
void WarpPreconditioner<TMatrix, TVector, ThreadManager>::checkLinearSystem()
{
    // a RotationMatrixSystem component is created in the absence of a linear system
    this->template doCheckLinearSystem<RotationMatrixSystem<Matrix, Vector>>();
}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::solve(Matrix& R, Vector& solution, Vector& rhs)
{
    // The matrix A in l_linearSolver is rotated such as R * A * R^T
    // The new linear system to solve is then R * A * R^T * x = b
    // This is solved in 3 steps:

    // Step 1:
    //   R * A * R^T * x = b <=> A * R^T * x = R^T * b
    //   R^T * b is computed in this step:
    R.opMulTV(l_linearSolver->getLinearSystem()->getSystemRHSBaseVector(), &rhs);

    // Step 2:
    //   Let's define y = R^T * x, then the linear system is A * y = R^T * b.
    //   This step solves A * y = R^T * b using the linear solver where y is the unknown.
    l_linearSolver->solveSystem();

    // Step 3:
    //   Since y = R^T * x, x is deduced: x = R * y
    R.opMulV(&solution, l_linearSolver->getLinearSystem()->getSystemSolutionBaseVector());
}

/// Solve the system as constructed using the previous methods
template<class TMatrix, class TVector,class ThreadManager>
bool WarpPreconditioner<TMatrix,TVector,ThreadManager >::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    if (J->rowSize()==0 || !l_linearSolver.get()) return true;

    this->l_linearSystem->getSystemMatrix()->rotateMatrix(&j_local,J);

    return l_linearSolver->addJMInvJt(result,&j_local,fact);
}

template<class TMatrix, class TVector,class ThreadManager>
bool WarpPreconditioner<TMatrix,TVector,ThreadManager >::addMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    this->l_linearSystem->getSystemMatrix()->rotateMatrix(&j_local,J);
    return l_linearSolver->addMInvJt(result,&j_local,fact);
}

template<class TMatrix, class TVector,class ThreadManager>
void WarpPreconditioner<TMatrix,TVector,ThreadManager >::computeResidual(const core::ExecParams* params, linearalgebra::BaseVector* f)
{
    l_linearSolver->computeResidual(params,f);
}

} // namespace sofa::component::linearsolver::preconditioner
