/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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
    : m_inputObjects1(initData(&m_inputObjects1, "input1", "Input Object(s) (1st Data type)"))
    , m_inputObjects2(initData(&m_inputObjects2, "input2", "Input Object(s) (2nd Data type)"))
    , m_outputObjects(initData(&m_outputObjects, "output", "Output Object(s)"))
{

}


template < class In1, class In2, class Out >
Multi2Mapping<In1,In2,Out>::Multi2Mapping(helper::vector< State<In1>* > in1, helper::vector< State<In2>* > in2, helper::vector< State<Out>* > out)
    : fromModels1(in1), fromModels2(in2), toModels(out)
    , m_inputObjects1(initData(&m_inputObjects1, "input1", "Input Object(s) (1st Data type)"))
    , m_inputObjects2(initData(&m_inputObjects2, "input2", "Input Object(s) (2nd Data type)"))
    , m_outputObjects(initData(&m_outputObjects, "output", "Output Object(s)"))
{

}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addInputModel(State<In1>* from)
{
    if (!from) return;
    this->fromModels1.push_back(from);
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addInputModel(State<In2>* from)
{
    if (!from) return;
    this->fromModels2.push_back(from);
}

template< class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addOutputModel(State<Out>* to)
{
    this->toModels.insert( toModels.end(), to);
    if (!isMechanical())
    {
        if(to != NULL && !testMechanicalState(to))
            setNonMechanical();
    }
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
helper::vector<State<In1>*>&  Multi2Mapping<In1,In2,Out>::getFromModels1()
{
    return fromModels1;
}

template< class In1, class In2, class Out >
helper::vector<State<In2>*>&  Multi2Mapping<In1,In2,Out>::getFromModels2()
{
    return fromModels2;
}

template< class In1, class In2, class Out >
helper::vector< State<Out>* >& Multi2Mapping<In1,In2,Out>::getToModels()
{
    return toModels;
}

template< class In1, class In2, class Out >
helper::vector<BaseState*> Multi2Mapping<In1,In2,Out>::getFrom()
{
    helper::vector<BaseState*> base_fromModels;
    std::copy(fromModels1.begin(), fromModels1.end(), std::back_inserter(base_fromModels));
    std::copy(fromModels2.begin(), fromModels2.end(), std::back_inserter(base_fromModels));
    return base_fromModels;
}

template< class In1, class In2, class Out >
helper::vector<BaseState* > Multi2Mapping<In1,In2,Out>::getTo()
{
    helper::vector<BaseState*> base_toModels;
    std::copy(toModels.begin(), toModels.end(), std::back_inserter(base_toModels));
    return base_toModels;
}

template < class In1, class In2,class Out>
helper::vector<behavior::BaseMechanicalState*> Multi2Mapping<In1,In2,Out>::getMechFrom()
{
    helper::vector<behavior::BaseMechanicalState*> mechFromVec;
    for (unsigned int i=0 ; i<this->fromModels1.size() ; i++)
    {
        behavior::BaseMechanicalState* meshFrom = dynamic_cast<behavior::BaseMechanicalState*> (this->fromModels1[i]);
        if(meshFrom)
            mechFromVec.push_back(meshFrom);
    }
    for (unsigned int i=0 ; i<this->fromModels2.size() ; i++)
    {
        behavior::BaseMechanicalState* meshFrom = dynamic_cast<behavior::BaseMechanicalState*> (this->fromModels2[i]);
        if(meshFrom)
            mechFromVec.push_back(meshFrom);
    }

    return mechFromVec;
}

template < class In1, class In2,class Out>
helper::vector<behavior::BaseMechanicalState*> Multi2Mapping<In1,In2,Out>::getMechTo()
{
    helper::vector<behavior::BaseMechanicalState*> mechToVec;
    for (unsigned int i=0 ; i<this->toModels.size() ; i++)
    {
        behavior::BaseMechanicalState* meshTo = dynamic_cast<behavior::BaseMechanicalState*> (this->toModels[i]);
        if(meshTo)
            mechToVec.push_back(meshTo);
    }
    return mechToVec;
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::init()
{
    ///<TO REMOVE>
    //this->updateMapping();
    apply(MechanicalParams::defaultInstance()  /* PARAMS FIRST */, VecId::position(), ConstVecId::position());
    applyJ(MechanicalParams::defaultInstance()  /* PARAMS FIRST */, VecId::velocity(), ConstVecId::velocity());
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
    return std::string("Multi2Mapping<[") + In1::Name() + std::string(",") + In2::Name() + std::string("],")+ Out::Name() + std::string(">");
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::disable()
{
}

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_MULTI2MAPPING_INL
