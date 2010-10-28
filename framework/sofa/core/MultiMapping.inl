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
#ifndef SOFA_CORE_MULTIMAPPING_INL
#define SOFA_CORE_MULTIMAPPING_INL

#include <sofa/core/MultiMapping.h>

namespace sofa
{

namespace core
{

template < class In, class Out >
MultiMapping<In,Out>::MultiMapping(helper::vector< State<In>* > in, helper::vector< State<Out>* > out)
    : fromModels(in), toModels(out)
    , m_inputObjects(initData(&m_inputObjects, "input", "Input Object(s)"))
    , m_outputObjects(initData(&m_outputObjects, "output", "Output Object(s)"))
{

}

template < class In, class Out >
void MultiMapping<In,Out>::addInputModel(State<In>* from)
{
    this->fromModels.push_back(from);
}

template< class In, class Out >
void MultiMapping<In,Out>::addOutputModel(State<Out>* to)
{
    this->toModels.push_back(to);
    if (!isMechanical())
    {
        if(to != NULL && !testMechanicalState(to))
            setNonMechanical();
    }
}

template< class In, class Out>
helper::vector< State<In>* >&  MultiMapping<In,Out>::getFromModels()
{
    return this->fromModels;
}

template< class In, class Out>
helper::vector< State<Out>* >& MultiMapping<In,Out>::getToModels()
{
    return this->toModels;
}
template< class In, class Out >
helper::vector<BaseState*> MultiMapping<In,Out>::getFrom()
{
    helper::vector<BaseState*> base_fromModels;
    std::copy(fromModels.begin(), fromModels.end(), std::back_inserter(base_fromModels));
    return base_fromModels;
}

template< class In, class Out >
helper::vector<BaseState* > MultiMapping<In,Out>::getTo()
{
    helper::vector<BaseState*> base_toModels;
    std::copy(toModels.begin(), toModels.end(), std::back_inserter(base_toModels));
    return base_toModels;
}

template <class In, class Out>
helper::vector<behavior::BaseMechanicalState*> MultiMapping<In,Out>::getMechFrom()
{
    helper::vector<behavior::BaseMechanicalState*> mechFromVec;
    for (unsigned int i=0 ; i<this->fromModels.size() ; i++)
    {
        behavior::BaseMechanicalState* meshFrom = dynamic_cast<behavior::BaseMechanicalState*> (this->fromModels[i]);
        if(meshFrom)
            mechFromVec.push_back(meshFrom);
    }
    return mechFromVec;
}

template <class In, class Out>
helper::vector<behavior::BaseMechanicalState*> MultiMapping<In,Out>::getMechTo()
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

template <class In, class Out>
void MultiMapping<In,Out>::init()
{
    ///<TO REMOVE>
    //this->updateMapping();
    ///???
    apply(VecId::position(), ConstVecId::position(), MechanicalParams::defaultInstance());
    applyJ(VecId::velocity(), ConstVecId::velocity(), MechanicalParams::defaultInstance());

}

#ifdef SOFA_SMP
template<class T>
struct ParallelMultiMappingApply
{
    void operator()(void *m, Shared_rw<defaulttype::SharedVector<typename T::Out::VecCoord*> > out, Shared_r<defaulttype::SharedVector<const typename T::In::VecCoord*> > in, const MechanicalParams* mparams)
    {
        ((T *)m)->apply(out.access(), in.read(), mparams);
    }
};

template<class T>
struct ParallelMultiMappingApplyJ
{
    void operator()(void *m, Shared_rw<defaulttype::SharedVector<typename T::Out::VecDeriv*> > out, Shared_r<defaulttype::SharedVector<const typename T::In::VecDeriv*> > in)
    {
        ((T *)m)->applyJ(out.access(), in.read());
    }
};

template<class T>
struct accessOutPos
{
    void operator()(void *m, Shared_rw<typename T::Out::VecCoord> out)
    {
        out.access();
    }
};

template<class T>
struct ParallelMultiMappingApply3
{
    void operator()(void *m, Shared_rw<typename T::Out::VecCoord> out, Shared_r<typename T::In::VecCoord> in1, Shared_r<typename T::In::VecCoord> in2)
    {
        out.access();
        in1.read();
        in2.read();
        ((T *)m)->apply(((T *)m)->VecOutPos,((T *)m)->VecInPos);
    }
};

template<class T>
struct ParallelMultiMappingApplyJ3
{
    void operator()(void *m, Shared_rw<typename T::Out::VecDeriv> out, Shared_r<typename T::In::VecDeriv> in1,Shared_r<typename T::In::VecDeriv> in2)
    {
        out.access();
        in1.read();
        in2.read();
        ((T *)m)->applyJ(((T *)m)->VecOutVel,((T *)m)->VecInVel);
    }
};
#endif /* SOFA_SMP */

template <class In, class Out>
void MultiMapping<In,Out>::apply(MultiVecCoordId outPos, ConstMultiVecCoordId inPos, const MechanicalParams* mparams)
{
    helper::vector<OutDataVecCoord*> vecOutPos;
    getVecOutCoord(outPos, vecOutPos);
    helper::vector<const InDataVecCoord*> vecInPos;
    getConstVecInCoord(inPos, vecInPos);

#ifdef SOFA_SMP
//		if (mparams->execMode() == ExecParams::EXEC_KAAPI)
//			Task<ParallelMultiMappingApply< MultiMapping<In,Out> > >(this,
//					**defaulttype::getShared(*out), **defaulttype::getShared(*in), mparams);
//		else
#endif /* SOFA_SMP */
    this->apply(vecOutPos, vecInPos, mparams);
}// MultiMapping::apply

template <class In, class Out>
void MultiMapping<In,Out>::applyJ(MultiVecDerivId outVel, ConstMultiVecDerivId inVel, const MechanicalParams* mparams)
{
    helper::vector<OutDataVecDeriv*> vecOutVel;
    getVecOutDeriv(outVel, vecOutVel);
    helper::vector<const InDataVecDeriv*> vecInVel;
    getConstVecInDeriv(inVel, vecInVel);

#ifdef SOFA_SMP
//		if (mparams->execMode() == ExecParams::EXEC_KAAPI)
//			Task<ParallelMultiMappingApplyJ< MultiMapping<In,Out> > >(this,
//					**defaulttype::getShared(*out), **defaulttype::getShared(*in), mparams);
//		else
#endif /* SOFA_SMP */
    this->applyJ(vecOutVel, vecInVel, mparams);
}// MultiMapping::applyJ

template <class In, class Out>
void MultiMapping<In,Out>::applyJT(MultiVecDerivId inForce, ConstMultiVecDerivId outForce, const MechanicalParams* mparams)
{
    helper::vector<InDataVecDeriv*> vecOutForce;
    getVecInDeriv(inForce, vecOutForce);
    helper::vector<const OutDataVecDeriv*> vecInForce;
    getConstVecOutDeriv(outForce, vecInForce);

    this->applyJT(vecOutForce, vecInForce, mparams);
}// MultiMapping::applyJT

template <class In, class Out>
std::string MultiMapping<In,Out>::templateName(const MultiMapping<In, Out>* /*mapping*/)
{
    return std::string("MultiMapping<") + In::Name() + std::string(",") + Out::Name() + std::string(">");
}

template <class In, class Out>
void MultiMapping<In,Out>::disable()
{
}


} // namespace core

} // namespace sofa

#endif
