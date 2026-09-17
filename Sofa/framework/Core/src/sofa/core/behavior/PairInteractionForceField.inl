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

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <iostream>

namespace sofa::core::behavior
{

template<class DataTypes>
PairInteractionForceField<DataTypes>::PairInteractionForceField(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : Inherit1(), Inherit2(mm1, mm2)
{
    if (!mm1)
        this->mstate1.setPath("@./"); // default to state of the current node
    if (!mm2)
        this->mstate2.setPath("@./"); // default to state of the current node
}

template<class DataTypes>
PairInteractionForceField<DataTypes>::~PairInteractionForceField()
{
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{
    auto state1 = this->mstate1.get();
    auto state2 = this->mstate2.get();
    if (state1 && state2)
    {
        addForce(mparams, *fId[state1].write(), *fId[state2].write(),
            *mparams->readX(state1), *mparams->readX(state2),
            *mparams->readV(state1), *mparams->readV(state2));
    }
    else
        msg_error() << "PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* /*mparams*/, MultiVecDerivId /*fId*/ ), mstate missing";
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId,
    ConstMultiVecDerivId dxId, ConstMultiVecCoordId xId, ConstMultiVecDerivId vId)
{
    auto state1 = this->mstate1.get();
    auto state2 = this->mstate2.get();
    if (state1 && state2)
    {
        Data<VecDeriv>* df1 = dfId[state1].write(); assert(df1);
        Data<VecDeriv>* df2 = dfId[state2].write(); assert(df2);

        const Data<VecDeriv>* dx1 = dxId[state1].read(); assert(dx1);
        const Data<VecDeriv>* dx2 = dxId[state2].read(); assert(dx2);

        const Data<VecCoord>* x1 = xId[state1].read(); assert(x1);
        const Data<VecCoord>* x2 = xId[state2].read(); assert(x2);

        const Data<VecDeriv>* v1 = vId[state1].read(); assert(v1);
        const Data<VecDeriv>* v2 = vId[state2].read(); assert(v2);

        const AddDForceVectors vectors1 {
            .df = *df1,
            .dx = *dx1,
            .x = *x1,
            .v = *v1
        };

        const AddDForceVectors vectors2 {
            .df = *df2,
            .dx = *dx2,
            .x = *x2,
            .v = *v2
        };

        doAddDForce(mparams, vectors1, vectors2);
    }
    else
        msg_error() << "PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* /*mparams*/, MultiVecDerivId /*fId*/ ), mstate missing";
}

template <class TDataTypes>
void PairInteractionForceField<TDataTypes>::doAddDForce(const MechanicalParams* mparams,
                                                        const AddDForceVectors& vectors1,
                                                        const AddDForceVectors& vectors2)
{
    // compatibility with legacy `addDForce`.
    addDForce(mparams, vectors1.df, vectors2.df, vectors1.dx, vectors2.dx);
}

template<class DataTypes>
SReal PairInteractionForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    auto state1 = this->mstate1.get();
    auto state2 = this->mstate2.get();
    if (state1 && state2)
        return getPotentialEnergy(mparams, *mparams->readX(state1),*mparams->readX(state2));
    else return 0.0;
}

} // namespace sofa::core::behavior
