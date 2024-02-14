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

#include <sofa/core/Multi2Mapping.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::core
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
type::vector<BaseState*> Multi2Mapping<In1,In2,Out>::getFrom()
{
    const VecFromModels1& models1 = getFromModels1();
    const VecFromModels2& models2 = getFromModels2();
    const size_t size1 = models1.size();
    const size_t size2 = models2.size();
    type::vector<BaseState*> baseModels(size1+size2);
    for (size_t i=0; i<size1; ++i) baseModels[      i] = models1[i].ptr.get();
    for (size_t i=0; i<size2; ++i) baseModels[size1+i] = models2[i].ptr.get();
    return baseModels;
}

template< class In1, class In2, class Out >
type::vector<BaseState* > Multi2Mapping<In1,In2,Out>::getTo()
{
    const VecToModels& models = getToModels();
    const size_t size = models.size();
    type::vector<BaseState*> baseModels(size);
    for (size_t i=0; i<size; ++i) baseModels[i] = models[i].ptr.get();
    return baseModels;
}

template < class In1, class In2,class Out>
type::vector<behavior::BaseMechanicalState*> Multi2Mapping<In1,In2,Out>::getMechFrom()
{
    type::vector<behavior::BaseMechanicalState*> mechFromVec;
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
type::vector<behavior::BaseMechanicalState*> Multi2Mapping<In1,In2,Out>::getMechTo()
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

template < class In1, class In2,class Out>
void Multi2Mapping<In1,In2,Out>::apply (const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos )
{
    type::vector<OutDataVecCoord*> vecOutPos;
    getVecOutCoord(outPos, vecOutPos);
    type::vector<const In1DataVecCoord*> vecIn1Pos;
    getConstVecIn1Coord(inPos, vecIn1Pos);
    type::vector<const In2DataVecCoord*> vecIn2Pos;
    getConstVecIn2Coord(inPos, vecIn2Pos);

    this->apply(mparams, vecOutPos, vecIn1Pos, vecIn2Pos);
}

template < class In1, class In2,class Out>
void Multi2Mapping<In1,In2,Out>::applyJ (const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel )
{
    type::vector<OutDataVecDeriv*> vecOutVel;
    getVecOutDeriv(outVel, vecOutVel);
    type::vector<const In1DataVecDeriv*> vecIn1Vel;
    getConstVecIn1Deriv(inVel, vecIn1Vel);
    type::vector<const In2DataVecDeriv*> vecIn2Vel;
    getConstVecIn2Deriv(inVel, vecIn2Vel);
    this->applyJ(mparams, vecOutVel, vecIn1Vel, vecIn2Vel);
}

template < class In1, class In2,class Out>
void Multi2Mapping<In1,In2,Out>::applyJT (const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce )
{
    type::vector<In1DataVecDeriv*> vecOut1Force;
    getVecIn1Deriv(inForce, vecOut1Force);
    type::vector<In2DataVecDeriv*> vecOut2Force;
    getVecIn2Deriv(inForce, vecOut2Force);

    type::vector<const OutDataVecDeriv*> vecInForce;
    getConstVecOutDeriv(outForce, vecInForce);
    this->applyJT(mparams, vecOut1Force, vecOut2Force, vecInForce);
}

template < class In1, class In2,class Out>
void Multi2Mapping<In1,In2,Out>::applyJT(const ConstraintParams* cparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst )
{
    type::vector<In1DataMatrixDeriv*> matOut1Const;
    getMatIn1Deriv(inConst, matOut1Const);
    type::vector<In2DataMatrixDeriv*> matOut2Const;
    getMatIn2Deriv(inConst, matOut2Const);

    type::vector<const OutDataMatrixDeriv*> matInConst;
    getConstMatOutDeriv(outConst, matInConst);
    this->applyJT(cparams, matOut1Const, matOut2Const, matInConst);
}

template < class In1, class In2,class Out>
void Multi2Mapping<In1,In2,Out>::computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc )
{
    type::vector<OutDataVecDeriv*> vecOutAcc;
    getVecOutDeriv(outAcc, vecOutAcc);

    type::vector<const In1DataVecDeriv*> vecIn1Vel;
    getConstVecIn1Deriv(inVel, vecIn1Vel);
    type::vector<const In1DataVecDeriv*> vecIn1Acc;
    getConstVecIn1Deriv(inAcc, vecIn1Acc);

    type::vector<const In2DataVecDeriv*> vecIn2Vel;
    getConstVecIn2Deriv(inVel, vecIn2Vel);
    type::vector<const In2DataVecDeriv*> vecIn2Acc;
    getConstVecIn2Deriv(inAcc, vecIn2Acc);

    this->computeAccFromMapping(mparams, vecOutAcc, vecIn1Vel, vecIn2Vel,vecIn1Acc, vecIn2Acc);
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1, In2, Out>::init()
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

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::disable()
{
}
} // namespace sofa
