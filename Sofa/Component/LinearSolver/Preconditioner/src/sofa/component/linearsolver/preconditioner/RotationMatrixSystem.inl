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
#include <sofa/component/linearsolver/preconditioner/RotationMatrixSystem.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa::component::linearsolver::preconditioner
{

template <class TMatrix, class TVector>
RotationMatrixSystem<TMatrix, TVector>::RotationMatrixSystem()
    : d_assemblingRate(initData(&d_assemblingRate, 1u, "assemblingRate",
        "Rate of update of the preconditioner matrix"))
    , l_mainAssembledSystem(initLink("mainSystem", "Main assembled linear system that will be warped"))
    , l_rotationFinder(initLink("rotationFinder", "Link toward the rotation finder used to compute the rotation matrix"))
{}

template <class TMatrix, class TVector>
void RotationMatrixSystem<TMatrix, TVector>::init()
{
    linearsystem::TypedMatrixLinearSystem<TMatrix, TVector>::init();

    if (!l_rotationFinder)
    {
        if (auto* rotationFinder = this->getContext()->get<sofa::core::behavior::BaseRotationFinder>())
        {
            l_rotationFinder.set(rotationFinder);
            msg_info() << "Rotation finder found: '" << l_rotationFinder->getPathName() << "'";
        }
        else
        {
            msg_error() << "A RotationFinder is required by " << this->getClassName() << " but has not "
                          "been found. The list of available rotation finders is: "
                        << core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::BaseRotationFinder>();
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    reinitAssemblyCounter();

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class TMatrix, class TVector>
void RotationMatrixSystem<TMatrix, TVector>::reset()
{
    reinitAssemblyCounter();
}

template <class TMatrix, class TVector>
void RotationMatrixSystem<TMatrix, TVector>::buildSystemMatrix(
    const core::MechanicalParams* mparams)
{
    bool isAssembled = false;

    if (l_mainAssembledSystem)
    {
        if (++m_assemblyCounter >= d_assemblingRate.getValue())
        {
            l_mainAssembledSystem->buildSystemMatrix(mparams);
            m_assemblyCounter = 0;
            isAssembled = true;
        }
    }

    if (isAssembled)
    {
        this->getSystemMatrix()->setIdentity();
    }
    else
    {
        updateMatrixWithRotations();
    }
}

template <class TMatrix, class TVector>
void RotationMatrixSystem<TMatrix, TVector>::reinitAssemblyCounter()
{
    m_assemblyCounter = d_assemblingRate.getValue();  // to assemble the first time
}

template <class TMatrix, class TVector>
void RotationMatrixSystem<TMatrix, TVector>::updateMatrixWithRotations()
{
    if (l_rotationFinder)
    {
        ensureValidRotationWork();
        l_rotationFinder->getRotations(rotationWork[indexwork].get());
    }
}

template <class TMatrix, class TVector>
void RotationMatrixSystem<TMatrix, TVector>::ensureValidRotationWork()
{
    if (indexwork < rotationWork.size())
    {
        rotationWork[indexwork] = std::make_unique<TMatrix>();
    }
    else
    {
        msg_error() << "Wrong index";
    }
}

}  // namespace sofa::component::linearsolver::preconditioner
