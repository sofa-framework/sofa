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
 * Base class for components having access to a pair of mechanical states with a specific template parameter, in order
 * to read and/or write state variables.
 */
template<class DataTypes1, class DataTypes2 = DataTypes1>
class PairStateAccessor : public virtual StateAccessor
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(PairStateAccessor, DataTypes1, DataTypes2), StateAccessor);

    void init() override;

    /// Retrieve the associated MechanicalState #1
    MechanicalState<DataTypes1>* getMState1() { return mstate1; }

    /// Retrieve the associated MechanicalState #1
    const MechanicalState<DataTypes1>* getMState1() const { return mstate1; }

    /// Retrieve the associated MechanicalState #1 as a BaseMechanicalState
    BaseMechanicalState* getMechModel1() { return mstate1; }

    /// Retrieve the associated MechanicalState #1 as a BaseMechanicalState
    const BaseMechanicalState* getMechModel1() const { return mstate1; }

    /// Retrieve the associated MechanicalState #2
    MechanicalState<DataTypes2>* getMState2() { return mstate2; }

    /// Retrieve the associated MechanicalState #2
    const MechanicalState<DataTypes2>* getMState2() const { return mstate2; }

    /// Retrieve the associated MechanicalState #2 as a BaseMechanicalState
    BaseMechanicalState* getMechModel2() { return mstate2; }

    /// Retrieve the associated MechanicalState #2 as a BaseMechanicalState
    const BaseMechanicalState* getMechModel2() const { return mstate2; }

protected:

    PairStateAccessor(MechanicalState<DataTypes1> *mm1, MechanicalState<DataTypes2> *mm2);

    ~PairStateAccessor() override = default;

    SingleLink<PairStateAccessor<DataTypes1, DataTypes2>, MechanicalState<DataTypes1>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate1;
    SingleLink<PairStateAccessor<DataTypes1, DataTypes2>, MechanicalState<DataTypes2>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate2;
};

template <class DataTypes1, class DataTypes2>
void PairStateAccessor<DataTypes1, DataTypes2>::init()
{
    Inherit1::init();

    if (!mstate1.get())
    {
        mstate1.set(dynamic_cast< MechanicalState<DataTypes1>* >(getContext()->getMechanicalState()));

        msg_error_when(!mstate1) << "MechanicalState #1 (" << mstate1.getName() << "): No compatible MechanicalState "
                "found in the current context. "
                "This may be because there is no MechanicalState in the local context, "
                "or because the type is not compatible.";
    }

    if (!mstate2.get())
    {
        mstate2.set(dynamic_cast< MechanicalState<DataTypes2>* >(getContext()->getMechanicalState()));

        msg_error_when(!mstate2) << "MechanicalState #2 (" << mstate2.getName() << "): No compatible MechanicalState "
                "found in the current context. "
                "This may be because there is no MechanicalState in the local context, "
                "or because the type is not compatible.";
    }

    l_mechanicalStates.clear();
    l_mechanicalStates.add(mstate1);
    l_mechanicalStates.add(mstate2);
}

template <class DataTypes1, class DataTypes2>
PairStateAccessor<DataTypes1, DataTypes2>::PairStateAccessor(
    MechanicalState<DataTypes1>* mm1,
    MechanicalState<DataTypes2>* mm2)
    : Inherit1()
    , mstate1(initLink("object1", "First object associated to this component"), mm1)
    , mstate2(initLink("object2", "Second object associated to this component"), mm2)
{}
}
