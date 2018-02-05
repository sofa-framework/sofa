/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_MULTIMAPPING_INL
#define SOFA_CORE_MULTIMAPPING_INL

#include <sofa/core/MultiMapping.h>

namespace sofa
{

namespace core
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
        if(to != NULL && !testMechanicalState(to))
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
helper::vector<BaseState*> MultiMapping<In,Out>::getFrom()
{
    const VecFromModels& models = getFromModels();
    size_t size = models.size();
    helper::vector<BaseState*> baseModels(size);
    for (size_t i=0; i<size; ++i) baseModels[i] = models[i].ptr.get();
    return baseModels;
}

template< class In, class Out >
helper::vector<BaseState* > MultiMapping<In,Out>::getTo()
{
    const VecToModels& models = getToModels();
    size_t size = models.size();
    helper::vector<BaseState*> baseModels(size);
    for (size_t i=0; i<size; ++i) baseModels[i] = models[i].ptr.get();
    return baseModels;
}

template <class In, class Out>
helper::vector<behavior::BaseMechanicalState*> MultiMapping<In,Out>::getMechFrom()
{
    helper::vector<behavior::BaseMechanicalState*> mechFromVec;
    for (size_t i=0 ; i<this->fromModels.size() ; i++)
    {
        behavior::BaseMechanicalState* meshFrom = this->fromModels.get((unsigned)i)->toBaseMechanicalState();
        if(meshFrom)
            mechFromVec.push_back(meshFrom);
    }
    return mechFromVec;
}

template <class In, class Out>
helper::vector<behavior::BaseMechanicalState*> MultiMapping<In,Out>::getMechTo()
{
    helper::vector<behavior::BaseMechanicalState*> mechToVec;
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
    maskFrom.resize( this->fromModels.size() );
    for( unsigned i=0 ; i<this->fromModels.size() ; ++i )
        if( core::behavior::BaseMechanicalState* stateFrom = this->fromModels[i]->toBaseMechanicalState() ) maskFrom[i] = &stateFrom->forceMask;
    maskTo.resize( this->toModels.size() );
    for( unsigned i=0 ; i<this->toModels.size() ; ++i )
        if( core::behavior::BaseMechanicalState* stateTo = this->toModels[i]->toBaseMechanicalState() ) maskTo[i] = &stateTo->forceMask;
        else this->setNonMechanical();

    apply(MechanicalParams::defaultInstance() , VecCoordId::position(), ConstVecCoordId::position());
    applyJ(MechanicalParams::defaultInstance() , VecDerivId::velocity(), ConstVecDerivId::velocity());
    if (f_applyRestPosition.getValue())
        apply(MechanicalParams::defaultInstance(), VecCoordId::restPosition(), ConstVecCoordId::restPosition());
}

template <class In, class Out>
void MultiMapping<In,Out>::apply(const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos)
{
    helper::vector<OutDataVecCoord*> vecOutPos;
    getVecOutCoord(outPos, vecOutPos);
    helper::vector<const InDataVecCoord*> vecInPos;
    getConstVecInCoord(inPos, vecInPos);
    this->apply(mparams, vecOutPos, vecInPos);

#ifdef SOFA_USE_MASK
    this->m_forceMaskNewStep = true;
#endif
}// MultiMapping::apply

template <class In, class Out>
void MultiMapping<In,Out>::applyJ(const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel)
{
    helper::vector<OutDataVecDeriv*> vecOutVel;
    getVecOutDeriv(outVel, vecOutVel);
    helper::vector<const InDataVecDeriv*> vecInVel;
    getConstVecInDeriv(inVel, vecInVel);
    this->applyJ(mparams, vecOutVel, vecInVel);
}// MultiMapping::applyJ

template <class In, class Out>
void MultiMapping<In,Out>::applyJT(const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce)
{
    helper::vector<InDataVecDeriv*> vecOutForce;
    getVecInDeriv(inForce, vecOutForce);
    helper::vector<const OutDataVecDeriv*> vecInForce;
    getConstVecOutDeriv(outForce, vecInForce);

    this->applyJT(mparams, vecOutForce, vecInForce);

#ifdef SOFA_USE_MASK
    if( this->m_forceMaskNewStep )
    {
        this->m_forceMaskNewStep = false;
        updateForceMask();
    }
#endif
}// MultiMapping::applyJT

template <class In, class Out>
std::string MultiMapping<In,Out>::templateName(const MultiMapping<In, Out>* /*mapping*/)
{
    //return std::string("MultiMapping<") + In::Name() + std::string(",") + Out::Name() + std::string(">");
    return In::Name() + std::string(",") + Out::Name();
}

template <class In, class Out>
void MultiMapping<In,Out>::disable()
{
}

template < class In, class Out >
void MultiMapping<In,Out>::updateForceMask()
{
    helper::vector<behavior::BaseMechanicalState*> fromModels = getMechFrom();
    for (size_t i=0 ; i<fromModels.size() ; i++)
        fromModels[i]->forceMask.assign(fromModels[i]->getSize(),true);
}

} // namespace core

} // namespace sofa

#endif
