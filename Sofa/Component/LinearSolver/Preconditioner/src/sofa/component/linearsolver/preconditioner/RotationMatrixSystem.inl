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
        if (auto* rotationFinder = this->getContext()->template get<sofa::core::behavior::BaseRotationFinder>())
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

    if (l_mainAssembledSystem.empty())
    {
        msg_info() << "Link \"" << l_mainAssembledSystem.getName() << "\" to the desired linear system should be set to ensure right behavior." << msgendl
                   << "First assembled linear system found in current context will be used (if any).";

        const auto listSystems = this->getContext()->template getObjects<sofa::core::behavior::BaseMatrixLinearSystem>(sofa::core::objectmodel::BaseContext::Local);
        for (const auto& system : listSystems)
        {
            if (system->getTemplateName() != "GraphScattered" && system != this)
            {
                l_mainAssembledSystem.set(system);
                break;
            }
        }
    }

    if (l_mainAssembledSystem.get() == nullptr)
    {
        msg_error() << "No linear system component found at path: " << l_mainAssembledSystem.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (l_mainAssembledSystem->getTemplateName() == "GraphScattered")
    {
        msg_error() << "Cannot use the solver " << l_mainAssembledSystem->getName()
                    << " because it is templated on GraphScatteredType";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "Linear system path used: '" << l_mainAssembledSystem->getPathName() << "'";

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
    bool isMainSystemAssembled = false;

    if (l_mainAssembledSystem)
    {
        if (++m_assemblyCounter >= d_assemblingRate.getValue())
        {
            l_mainAssembledSystem->buildSystemMatrix(mparams);
            m_assemblyCounter = 0;
            isMainSystemAssembled = true;
        }
    }

    if (isMainSystemAssembled)
    {
        const auto matrixSize = l_mainAssembledSystem->getMatrixSize();
        this->resizeSystem(matrixSize[0]);
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
        l_rotationFinder->getRotations(this->getSystemBaseMatrix());
    }
}

}  // namespace sofa::component::linearsolver::preconditioner
