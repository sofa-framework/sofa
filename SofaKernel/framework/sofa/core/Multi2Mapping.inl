/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_MULTI2MAPPING_INL
#define SOFA_CORE_MULTI2MAPPING_INL

#include <sofa/core/Multi2Mapping.h>

namespace sofa
{

namespace core
{

template < class In1, class In2, class Out >
Multi2Mapping<In1,In2,Out>::Multi2Mapping()
    : fromModels1(initLink("input1", "Input Object(s) (1st Data type)"))
    , fromModels2(initLink("input2", "Input Object(s) (2st Data type)"))
    , toModels(initLink("output", "Output Object(s)"))
    , f_applyRestPosition( initData( &f_applyRestPosition, false, "applyRestPosition", "set to true to apply this mapping to restPosition at init"))
{
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addInputModel1(State<In1>* from, const std::string& path)
{
    if (from)
        this->fromModels1.add(from,path);
    else if (!path.empty())
        this->fromModels1.addPath(path);
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addInputModel2(State<In2>* from, const std::string& path)
{
    if (from)
        this->fromModels2.add(from,path);
    else if (!path.empty())
        this->fromModels2.addPath(path);
}

template< class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addOutputModel(State<Out>* to, const std::string& path)
{
    if (to)
        this->toModels.add(to,path);
    else if (!path.empty())
        this->toModels.addPath(path);
    if (to && isMechanical() && !testMechanicalState(to))
        setNonMechanical();
}

///<TO REMOVE>
//cannot compile
//template< class In1, class In2, class Out > template <class In>
//helper::vector<State<In>*>&  Multi2Mapping<In1,In2,Out>::getFromModels()
//{
//  if (!fromModels1.empty() && dynamic_cast< State<In>* >(fromModels1[0]))
//  {
//
//  }
//  else if (!fromModels2.empty() && dynamic_cast< State<In>* >(fromModels2[0]))
//  {
//    return fromModels2;
//  }
//}

template< class In1, class In2, class Out >
const typename Multi2Mapping<In1,In2,Out>::VecFromModels1& Multi2Mapping<In1,In2,Out>::getFromModels1()
{
    return fromModels1.getValue();
}

template< class In1, class In2, class Out >
const typename Multi2Mapping<In1,In2,Out>::VecFromModels2& Multi2Mapping<In1,In2,Out>::getFromModels2()
{
    return fromModels2.getValue();
}

template< class In1, class In2, class Out >
const typename Multi2Mapping<In1,In2,Out>::VecToModels& Multi2Mapping<In1,In2,Out>::getToModels()
{
    return toModels.getValue();
}

template< class In1, class In2, class Out >
helper::vector<BaseState*> Multi2Mapping<In1,In2,Out>::getFrom()
{
    const VecFromModels1& models1 = getFromModels1();
    const VecFromModels2& models2 = getFromModels2();
    size_t size1 = models1.size();
    size_t size2 = models2.size();
    helper::vector<BaseState*> baseModels(size1+size2);
    for (size_t i=0; i<size1; ++i) baseModels[      i] = models1[i].ptr.get();
    for (size_t i=0; i<size2; ++i) baseModels[size1+i] = models2[i].ptr.get();
    return baseModels;
}

template< class In1, class In2, class Out >
helper::vector<BaseState* > Multi2Mapping<In1,In2,Out>::getTo()
{
    const VecToModels& models = getToModels();
    size_t size = models.size();
    helper::vector<BaseState*> baseModels(size);
    for (size_t i=0; i<size; ++i) baseModels[i] = models[i].ptr.get();
    return baseModels;
}

template < class In1, class In2,class Out>
helper::vector<behavior::BaseMechanicalState*> Multi2Mapping<In1,In2,Out>::getMechFrom()
{
    helper::vector<behavior::BaseMechanicalState*> mechFromVec;
    for (size_t i=0 ; i<this->fromModels1.size() ; i++)
    {
        behavior::BaseMechanicalState* meshFrom = this->fromModels1.get((unsigned)i)->toBaseMechanicalState();
        if(meshFrom)
            mechFromVec.push_back(meshFrom);
    }
    for (size_t i=0 ; i<this->fromModels2.size() ; i++)
    {
        behavior::BaseMechanicalState* meshFrom = this->fromModels2.get((unsigned)i)->toBaseMechanicalState();
        if(meshFrom)
            mechFromVec.push_back(meshFrom);
    }
    return mechFromVec;
}

template < class In1, class In2,class Out>
helper::vector<behavior::BaseMechanicalState*> Multi2Mapping<In1,In2,Out>::getMechTo()
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

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::init()
{
    maskFrom1.resize( this->fromModels1.size() );
    for( unsigned i=0 ; i<this->fromModels1.size() ; ++i )
        if( core::behavior::BaseMechanicalState* stateFrom = this->fromModels1[i]->toBaseMechanicalState() ) maskFrom1[i] = &stateFrom->forceMask;
    maskFrom2.resize( this->fromModels2.size() );
    for( unsigned i=0 ; i<this->fromModels2.size() ; ++i )
        if( core::behavior::BaseMechanicalState* stateFrom = this->fromModels2[i]->toBaseMechanicalState() ) maskFrom2[i] = &stateFrom->forceMask;
    maskTo.resize( this->toModels.size() );
    for( unsigned i=0 ; i<this->toModels.size() ; ++i )
        if (core::behavior::BaseMechanicalState* stateTo = this->toModels[i]->toBaseMechanicalState()) maskTo[i] = &stateTo->forceMask;
        else this->setNonMechanical();

    apply(MechanicalParams::defaultInstance() , VecCoordId::position(), ConstVecCoordId::position());
    applyJ(MechanicalParams::defaultInstance() , VecDerivId::velocity(), ConstVecDerivId::velocity());
    if (f_applyRestPosition.getValue())
        apply(MechanicalParams::defaultInstance(), VecCoordId::restPosition(), ConstVecCoordId::restPosition());
}

///<TO REMOVE>
/*
template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::updateMapping()
{
  if( (this->fromModels1.empty() && this->fromModels2.empty() ) || this->toModels.empty() )
    return;

  helper::vector<OutDataVecCoord*> vecOutPos;
  getVecOutCoord(VecId::position(), vecOutPos);

  const ConstVecId constIdPos = ConstVecId::position();
  helper::vector<const In1DataVecCoord*> vecIn1Pos;
  getConstVecIn1Coord(constIdPos, vecIn1Pos);
  helper::vector<const In2DataVecCoord*> vecIn2Pos;
  getConstVecIn2Coord(constIdPos, vecIn2Pos);
  apply ( vecOutPos, vecIn1Pos, vecIn2Pos);

  helper::vector<OutDataVecDeriv*> vecOutVel;
  getVecOutDeriv(VecId::velocity(), vecOutVel);
  const ConstVecId constIdVel = ConstVecId::velocity();
  helper::vector<const In1DataVecDeriv*> vecIn1Vel;
  getConstVecIn1Deriv(constIdVel, vecIn1Vel);
  helper::vector<const In2DataVecDeriv*> vecIn2Vel;
  getConstVecIn2Deriv(constIdVel, vecIn2Vel);
  applyJ( vecOutVel, vecIn1Vel, vecIn2Vel);
}
*/

template < class In1, class In2, class Out >
std::string Multi2Mapping<In1,In2,Out>::templateName(const Multi2Mapping<In1, In2, Out>* /*mapping*/)
{
    return std::string(In1::Name()) + "," + In2::Name() + "," + Out::Name();
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::disable()
{
}


template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::updateForceMask()
{
    helper::vector<behavior::BaseMechanicalState*> fromModels = getMechFrom();
    for (size_t i=0 ; i<fromModels.size() ; i++)
        fromModels[i]->forceMask.assign(fromModels[i]->getSize(),true);
}

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_MULTI2MAPPING_INL
