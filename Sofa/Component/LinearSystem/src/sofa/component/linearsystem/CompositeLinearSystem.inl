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

#include <sofa/component/linearsystem/CompositeLinearSystem.h>
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.inl>

namespace sofa::component::linearsystem
{

template <class TMatrix, class TVector>
CompositeLinearSystem<TMatrix, TVector>::CompositeLinearSystem()
    : Inherit1()
    , l_linearSystems(initLink("linearSystems", "List of linear systems to assemble"))
    , l_solverLinearSystem(initLink("solverLinearSystem", "Among the list of linear systems, which one is to be used by the linear solver"))
{
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::init()
{
    TypedMatrixLinearSystem<TMatrix, TVector>::init();

    if (l_linearSystems.empty())
    {
        if (l_solverLinearSystem)
        {
            l_linearSystems.add(l_solverLinearSystem);
        }
        else
        {
            msg_error() << "At least one linear system must be provided";
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }
    else
    {
        if (l_solverLinearSystem)
        {
            if (std::none_of(l_linearSystems.begin(), l_linearSystems.end(), [&](auto linearSystem){return linearSystem == l_solverLinearSystem;}))
            {
                l_linearSystems.add(l_solverLinearSystem);
            }
        }
        else
        {
            bool found = false;
            for (const auto& linearSystem : l_linearSystems)
            {
                if (auto* typedLinearSystem = dynamic_cast<TypedMatrixLinearSystem<TMatrix, TVector>*>(linearSystem))
                {
                    l_solverLinearSystem.set(typedLinearSystem);
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                msg_error() << "At least one linear system with a compatible type must be provided";
                sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            }
        }
    }
}

template <class TMatrix, class TVector>
TMatrix* CompositeLinearSystem<TMatrix, TVector>::getSystemMatrix() const
{
    return l_solverLinearSystem ? l_solverLinearSystem->getSystemMatrix() : nullptr;
}

template <class TMatrix, class TVector>
TVector* CompositeLinearSystem<TMatrix, TVector>::getRHSVector() const
{
    return l_solverLinearSystem ? l_solverLinearSystem->getRHSVector() : nullptr;
}

template <class TMatrix, class TVector>
TVector* CompositeLinearSystem<TMatrix, TVector>::getSolutionVector() const
{
    return l_solverLinearSystem ? l_solverLinearSystem->getSolutionVector() : nullptr;
}

template <class TMatrix, class TVector>
linearalgebra::BaseMatrix* CompositeLinearSystem<TMatrix, TVector>::getSystemBaseMatrix() const
{
    return l_solverLinearSystem ? l_solverLinearSystem->getSystemBaseMatrix() : nullptr;
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::buildSystemMatrix(
    const core::MechanicalParams* mparams)
{
    for (unsigned int i = 0 ; i < l_linearSystems.size(); ++i)
    {
        if (l_linearSystems[i])
        {
            l_linearSystems[i]->buildSystemMatrix(mparams);
        }
    }
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::resizeSystem(sofa::Size n)
{
    for (unsigned int i = 0 ; i < l_linearSystems.size(); ++i)
    {
        if (l_linearSystems[i])
        {
            l_linearSystems[i]->resizeSystem(n);
        }
    }
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::clearSystem()
{
    for (unsigned int i = 0 ; i < l_linearSystems.size(); ++i)
    {
        if (l_linearSystems[i])
        {
            l_linearSystems[i]->clearSystem();
        }
    }
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::setRHS(core::MultiVecDerivId v)
{
    for (unsigned int i = 0 ; i < l_linearSystems.size(); ++i)
    {
        if (l_linearSystems[i])
        {
            l_linearSystems[i]->setRHS(v);
        }
    }
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::setSystemSolution(core::MultiVecDerivId v)
{
    for (unsigned int i = 0 ; i < l_linearSystems.size(); ++i)
    {
        if (l_linearSystems[i])
        {
            l_linearSystems[i]->setSystemSolution(v);
        }
    }
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::dispatchSystemSolution(core::MultiVecDerivId v)
{
    if (l_solverLinearSystem)
    {
        l_solverLinearSystem->dispatchSystemSolution(v);
    }
}

template <class TMatrix, class TVector>
void CompositeLinearSystem<TMatrix, TVector>::dispatchSystemRHS(core::MultiVecDerivId v)
{
    if (l_solverLinearSystem)
    {
        l_solverLinearSystem->dispatchSystemRHS(v);
    }
}

}
