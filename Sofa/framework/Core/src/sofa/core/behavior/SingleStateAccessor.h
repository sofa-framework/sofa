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

#include <sofa/core/behavior/StateAccessor.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::core::behavior
{

/**
 * Base class for components having access to one mechanical state with a specific template parameter, in order to read
 * and/or write state variables.
 */
template<class DataTypes>
class SingleStateAccessor : public virtual StateAccessor
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(SingleStateAccessor, DataTypes), StateAccessor);

    void init() override
    {
        Inherit1::init();

        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);

        if (!mstate.get())
        {
            mstate.set(dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState()));

            msg_error_when(!mstate) << "No compatible MechanicalState found in the current context. "
                "This may be because there is no MechanicalState in the local context, "
                "or because the type is not compatible.";
            d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }

        l_mechanicalStates.clear();
        l_mechanicalStates.add(mstate);
    }

    MechanicalState<DataTypes>* getMState() { return mstate.get(); }
    const MechanicalState<DataTypes>* getMState() const { return mstate.get(); }

protected:

    explicit SingleStateAccessor(MechanicalState<DataTypes> *mm = nullptr)
        : Inherit1()
        , mstate(initLink("mstate", "MechanicalState used by this component"), mm)
    {}

    ~SingleStateAccessor() override = default;

    SingleLink<SingleStateAccessor<DataTypes>, MechanicalState<DataTypes>, BaseLink::FLAG_STRONGLINK> mstate;
};

}
