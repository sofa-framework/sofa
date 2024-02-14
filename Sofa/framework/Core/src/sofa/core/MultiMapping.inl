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

#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::core
{

template< class In, class Out>
MultiMapping<In,Out>::MultiMapping()
    : fromModels(initLink("input", "Input Object(s)"))
    , toModels(initLink("output", "Output Object(s)"))
    , f_applyRestPosition( initData( &f_applyRestPosition, false, "applyRestPosition", "set to true to apply this mapping to restPosition at init"))
{

}

template < class In, class Out >
void MultiMapping<In,Out>::addInputModel(BaseState* fromModel, const std::string& path)
{
    State<In>* from = dynamic_cast<State<In>*>(fromModel);
    assert(from && "MultiMapping needs a State of the appropriate type to add as input model");
    this->fromModels.add(from, path);
}

template< class In, class Out >
void MultiMapping<In,Out>::addOutputModel(BaseState* toModel, const std::string& path)
{
    State<Out>* to = dynamic_cast<State<Out>*>(toModel);
    assert(to);
    this->toModels.add(to, path);
    if (isMechanical())
    {
        if(to != nullptr && !testMechanicalState(to))
            setNonMechanical();
    }
}

template< class In, class Out>
const typename MultiMapping<In,Out>::VecFromModels& MultiMapping<In,Out>::getFromModels()
{
    return this->fromModels.getValue();
}

template< class In, class Out>
const typename MultiMapping<In,Out>::VecToModels& MultiMapping<In,Out>::getToModels()
{
    return this->toModels.getValue();
}

template< class In, class Out >
type::vector<BaseState*> MultiMapping<In,Out>::getFrom()
{
    const VecFromModels& models = getFromModels();
    const size_t size = models.size();
    type::vector<BaseState*> baseModels(size);
    for (size_t i=0; i<size; ++i) baseModels[i] = models[i].ptr.get();
    return baseModels;
}

template< class In, class Out >
type::vector<BaseState* > MultiMapping<In,Out>::getTo()
{
    const VecToModels& models = getToModels();
    const size_t size = models.size();
    type::vector<BaseState*> baseModels(size);
    for (size_t i=0; i<size; ++i) baseModels[i] = models[i].ptr.get();
    return baseModels;
}

template <class In, class Out>
type::vector<behavior::BaseMechanicalState*> MultiMapping<In,Out>::getMechFrom()
{
    type::vector<behavior::BaseMechanicalState*> mechFromVec;
    for (size_t i=0 ; i<this->fromModels.size() ; i++)
    {
        behavior::BaseMechanicalState* meshFrom = this->fromModels.get((unsigned)i)->toBaseMechanicalState();
        if(meshFrom)
            mechFromVec.push_back(meshFrom);
    }
    return mechFromVec;
}

template <class In, class Out>
type::vector<behavior::BaseMechanicalState*> MultiMapping<In,Out>::getMechTo()
{
    type::vector<behavior::BaseMechanicalState*> mechToVec;
    for (size_t i=0 ; i<this->toModels.size() ; i++)
    {
        behavior::BaseMechanicalState* meshTo = this->toModels.get((unsigned)i)->toBaseMechanicalState();
        if(meshTo)
            mechToVec.push_back(meshTo);
    }
    return mechToVec;
}

template <class In, class Out>
void MultiMapping<In,Out>::init()
{
    Inherit1::init();

    for (auto toModel : this->toModels)
    {
        if (!toModel->toBaseMechanicalState())
        {
            this->setNonMechanical();
        }
    }

    apply(mechanicalparams::defaultInstance() , VecCoordId::position(), ConstVecCoordId::position());
    applyJ(mechanicalparams::defaultInstance() , VecDerivId::velocity(), ConstVecDerivId::velocity());
    if (f_applyRestPosition.getValue())
        apply(mechanicalparams::defaultInstance(), VecCoordId::restPosition(), ConstVecCoordId::restPosition());
}

template <class In, class Out>
void MultiMapping<In,Out>::apply(const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos)
{
    type::vector<OutDataVecCoord*> vecOutPos;
    getVecOutCoord(outPos, vecOutPos);
    type::vector<const InDataVecCoord*> vecInPos;
    getConstVecInCoord(inPos, vecInPos);
    this->apply(mparams, vecOutPos, vecInPos);
}// MultiMapping::apply

template <class In, class Out>
void MultiMapping<In,Out>::applyJ(const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel)
{
    type::vector<OutDataVecDeriv*> vecOutVel;
    getVecOutDeriv(outVel, vecOutVel);
    type::vector<const InDataVecDeriv*> vecInVel;
    getConstVecInDeriv(inVel, vecInVel);
    this->applyJ(mparams, vecOutVel, vecInVel);
}// MultiMapping::applyJ

template <class In, class Out>
void MultiMapping<In,Out>::applyJT(const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce)
{
    type::vector<InDataVecDeriv*> vecOutForce;
    getVecInDeriv(inForce, vecOutForce);
    type::vector<const OutDataVecDeriv*> vecInForce;
    getConstVecOutDeriv(outForce, vecInForce);

    this->applyJT(mparams, vecOutForce, vecInForce);
}// MultiMapping::applyJT

template <class In, class Out>
void MultiMapping<In,Out>::disable()
{
}
} // namespace sofa::core
