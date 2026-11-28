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

#include <sofa/component/playback/config.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/simulation/Node.h>
#include <sofa/type/vector.h>

namespace sofa::component::playback
{

/// Return all the components of type T stored in the given node (no search in the other nodes),
/// having all the given tags.
template<class T>
sofa::type::vector<T*> getLocalComponents(const simulation::Node* node, const core::objectmodel::TagSet& tags = {})
{
    sofa::type::vector<T*> components;
    node->get<T>(&components, tags, core::objectmodel::BaseContext::Local);
    return components;
}

/// Among the given components, return the one whose link l_state points to the given mechanical
/// state, or nullptr if there is none.
/// A component with an empty link is accepted only if the given mechanical state is the only one
/// in the node, i.e. if there is no ambiguity. In that case, the link is set explicitly.
template<class T>
T* findComponentLinkedToState(const sofa::type::vector<T*>& components,
                              core::behavior::BaseMechanicalState* mstate,
                              const simulation::Node* node)
{
    for (T* component : components)
    {
        if (component->l_state.get() == mstate)
        {
            return component;
        }
    }

    if (node->mechanicalState.size() == 1)
    {
        for (T* component : components)
        {
            if (component->l_state.empty())
            {
                component->l_state.set(mstate);
                return component;
            }
        }
    }

    return nullptr;
}

} // namespace sofa::component::playback
