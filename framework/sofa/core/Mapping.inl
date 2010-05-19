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
#ifndef SOFA_CORE_MAPPING_INL
#define SOFA_CORE_MAPPING_INL

#include <sofa/core/Mapping.h>
#include <iostream>

namespace sofa
{

namespace core
{

template <class In, class Out>
Mapping<In,Out>::Mapping(In* from, Out* to)
    : fromModel(from), toModel(to),
      object1(initData(&object1, std::string("../.."), "object1", "First object to map")),
      object2(initData(&object2, std::string(".."), "object2", "Second object to map"))
{
}

template <class In, class Out>
Mapping<In,Out>::~Mapping()
{
}

template <class In, class Out>
In* Mapping<In,Out>::getFromModel()
{
    return this->fromModel;
}

template <class In, class Out>
Out* Mapping<In,Out>::getToModel()
{
    return this->toModel;
}

template <class In, class Out>
helper::vector<objectmodel::BaseObject*> Mapping<In,Out>::getFrom()
{
    helper::vector<objectmodel::BaseObject*> vec(1,this->fromModel);
    return  vec;
}

template <class In, class Out>
helper::vector<objectmodel::BaseObject*> Mapping<In,Out>::getTo()
{
    helper::vector<objectmodel::BaseObject*> vec(1,this->toModel);
    return vec;
}

template <class In, class Out>
void Mapping<In,Out>::init()
{
    if (!object2.isSet() && toModel)
    {
        this->object2.setValue( toModel->getName() );
    }
    this->updateMapping();
}

template <class In, class Out>
void Mapping<In,Out>::disable()
{
}

template <class In, class Out>
void Mapping<In,Out>::setModels(In* from, Out* to)
{
    this->fromModel = from;
    this->toModel = to;
}
template <class In, class Out>
std::string Mapping<In,Out>::templateName(const Mapping<In, Out>* /*mapping*/)
{
    return std::string("Mapping<")+In::DataTypes::Name() + std::string(",") + Out::DataTypes::Name() + std::string(">");
}

#ifndef SOFA_SMP
template <class In, class Out>
void Mapping<In,Out>::updateMapping()
{
    if (this->toModel == NULL || this->fromModel == NULL)
        return;

    if (this->toModel->getX()!=NULL && this->fromModel->getX()!=NULL)
    {
        apply(*this->toModel->getX(), *this->fromModel->getX());
        //serr<<"Mapping<In,Out>::updateMapping(), *this->fromModel->getX() = "<<*this->fromModel->getX()<<sendl;
        //serr<<"Mapping<In,Out>::updateMapping(), *this->toModel->getX() = "<<*this->toModel->getX()<<sendl;
    }
    if (this->toModel->getV()!=NULL && this->fromModel->getV()!=NULL)
    {
        applyJ(*this->toModel->getV(), *this->fromModel->getV());
    }
}
#else


template<class T>
struct ParallelMappingApply
{
    void operator()(void *m, Shared_rw< typename T::Out::VecCoord > out, Shared_r< typename T::In::VecCoord > in)
    {
        ((T *)m)->apply(out.access(), in.read());
    }
};

template<class T>
struct ParallelMappingApplyJ
{
    void operator()(void *m, Shared_rw< typename T::Out::VecDeriv> out, Shared_r<typename T::In::VecDeriv> in)
    {
        ((T *)m)->applyJ(out.access(), in.read());
    }
};
template<class T>
struct ParallelMappingApplyCPU
{
    void operator()(void *m, Shared_rw< typename T::Out::VecCoord > out, Shared_r< typename T::In::VecCoord > in)
    {
        ((T *)m)->apply(out.access(), in.read());
    }
};

template<class T>
struct ParallelMappingApplyJCPU
{
    void operator()(void *m, Shared_rw< typename T::Out::VecDeriv> out, Shared_r<typename T::In::VecDeriv> in)
    {
        ((T *)m)->applyJ(out.access(), in.read());
    }
};


template<class In, class Out>
void sofa::core::Mapping< In,Out >::updateMapping()
{
    if (this->toModel == NULL || this->fromModel == NULL)
        return;
    if (this->toModel->getX()!=NULL && this->fromModel->getX()!=NULL)
    {
        Task<ParallelMappingApplyCPU< Mapping<In,Out > >,  ParallelMappingApply< Mapping<In,Out > > >(this,**(this->toModel->getX()), **(this->fromModel->getX()));
    }
    if (this->toModel->getV()!=NULL && this->fromModel->getV()!=NULL)
    {
        Task<ParallelMappingApplyJCPU< Mapping < In,Out > >,  ParallelMappingApplyJ< Mapping < In,Out > >  >(this,**this->toModel->getV(), **this->fromModel->getV());
    }
}

#endif
} // namespace core

} // namespace sofa

#endif
