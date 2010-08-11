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
void MultiMapping<In,Out>::addInputModel(In* from)
{
    this->fromModels.push_back(from);
}

template< class In, class Out >
void MultiMapping<In,Out>::addOutputModel(Out* to)
{
    this->toModels.push_back(to);
}

template< class In, class Out>
helper::vector<In*>&  MultiMapping<In,Out>::getFromModels()
{
    return this->fromModels;
}

template< class In, class Out>
helper::vector<Out*>& MultiMapping<In,Out>::getToModels()
{
    return this->toModels;
}
template< class In, class Out >
helper::vector<objectmodel::BaseObject*> MultiMapping<In,Out>::getFrom()
{
    helper::vector<objectmodel::BaseObject*> base_fromModels;
    std::copy(fromModels.begin(), fromModels.end(), std::back_inserter(base_fromModels));
    return base_fromModels;
}

template< class In, class Out >
helper::vector<objectmodel::BaseObject* > MultiMapping<In,Out>::getTo()
{
    helper::vector<objectmodel::BaseObject*> base_toModels;
    std::copy(toModels.begin(), toModels.end(), std::back_inserter(base_toModels));
    return base_toModels;
}
template <class In, class Out>
void MultiMapping<In,Out>::init()
{
    this->updateMapping();
}

#ifndef SOFA_SMP
template <class In, class Out>
void MultiMapping<In,Out>::updateMapping()
{
    if( this->fromModels.empty() || this->toModels.empty() )
    {
        return;
    }

    const VecId &idCoord = VecId::position();
    helper::vector<OutVecCoord*> vecOutPos;
    getVecOutCoord(idCoord, vecOutPos);
    helper::vector<const InVecCoord*> vecInPos;
    getConstVecInCoord(idCoord, vecInPos);
    apply ( vecOutPos, vecInPos);


    const VecId &idDeriv = VecId::velocity();
    helper::vector<OutVecDeriv*> vecOutVel;
    getVecOutDeriv(idDeriv, vecOutVel);
    helper::vector<const InVecDeriv*> vecInVel;
    getConstVecInDeriv(idDeriv, vecInVel);
    applyJ( vecOutVel, vecInVel);
}

#else

template<class T>
struct ParallelMultiMappingApply
{
    void operator()(void *m, Shared_rw<defaulttype::SharedVector<typename T::Out::VecCoord*> > out, Shared_r<defaulttype::SharedVector<const typename T::In::VecCoord*> > in)
    {
        ((T *)m)->apply(out.access(), in.read());
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
struct ParallelMultiMappingApplyCPU
{
    void operator()(void *m, Shared_rw<defaulttype::SharedVector<typename T::Out::VecCoord*> > out, Shared_r<defaulttype::SharedVector<const typename T::In::VecCoord*> > in)
    {
        ((T *)m)->apply(out.access(), in.read());
    }
};

template<class T>
struct ParallelMultiMappingApplyJCPU
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
template<class T>
struct ParallelMultiMappingApplyCPU3
{
    void operator()(void *m, Shared_rw<typename T::Out::VecCoord> out, Shared_r<typename T::In::VecCoord> in1,Shared_r<typename T::In::VecCoord> in2)
    {
        out.access();
        in1.read();
        in2.read();
        ((T *)m)->apply(((T *)m)->VecOutPos,((T *)m)->VecInPos);
    }
};

template<class T>
struct ParallelMultiMappingApplyJCPU3
{
    void operator()(void *m, Shared_rw<typename T::Out::VecDeriv> out, Shared_r<typename T::In::VecDeriv> in1,Shared_r<typename T::In::VecDeriv> in2)
    {
        out.access();
        in1.read();
        in2.read();

        ((T *)m)->applyJ(((T *)m)->VecOutVel,((T *)m)->VecInVel);
    }
};


template <class In, class Out>
void MultiMapping<In,Out>::updateMapping()
{
    if( this->fromModels.empty() || this->toModels.empty() )
        return;

    std::cout << "update mapping" << std::endl;

    const VecId &idCoord = VecId::position();

    VecOutPos.resize(0);
    getVecOutCoord(idCoord, VecOutPos);
    std::cout << "VecOutPos size" << VecOutPos.size() << std::endl;
    VecInPos.resize(0);
    getConstVecInCoord(idCoord, VecInPos);

    const VecId &idDeriv = VecId::velocity();

    VecOutVel.resize(0);
    getVecOutDeriv(idDeriv, VecOutVel);
    VecInVel.resize(0);
    getConstVecInDeriv(idDeriv, VecInVel);

    //Task<ParallelMultiMappingApplyCPU3<MultiMapping<In,Out> >, ParallelMultiMappingApply3<MultiMapping<In,Out> > >(this,**this->toModels[0]->getX(),**this->fromModels[0]->getX(),**this->fromModels[1]->getX());
    Task<ParallelMultiMappingApplyCPU3<MultiMapping<In,Out> >, ParallelMultiMappingApply3<MultiMapping<In,Out> > >(this,**(VecOutPos[0]),**(VecInPos[0]),**(VecInPos[1]));
    Task<ParallelMultiMappingApplyJCPU3<MultiMapping<In,Out> >, ParallelMultiMappingApplyJ3<MultiMapping<In,Out> > >(this,**(VecOutVel[0]),**(VecInVel[0]),**(VecInVel[1]));
    //Task<ParallelMultiMappingApplyJCPU< MultiMapping< In,Out > >,  ParallelMultiMappingApplyJ< MultiMapping< In,Out > > >(this,**(VecOutVel[0]), **(VecInVel[0]),**(VecInVel[1]));
}

#endif



template <class In, class Out>
std::string MultiMapping<In,Out>::templateName(const MultiMapping<In, Out>* /*mapping*/)
{
    return std::string("MultiMapping<")+In::DataTypes::Name() + std::string(",") + Out::DataTypes::Name() + std::string(">");
}

template <class In, class Out>
void MultiMapping<In,Out>::disable()
{
}


} // namespace core

} // namespace sofa

#endif
