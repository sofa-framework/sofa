/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_MAPPING_INL
#define SOFA_CORE_MAPPING_INL

#include <sofa/core/Mapping.h>
#include <iostream>
using std::cerr;
using std::endl;

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
objectmodel::BaseObject* Mapping<In,Out>::getFrom()
{
    return this->fromModel;
}

template <class In, class Out>
objectmodel::BaseObject* Mapping<In,Out>::getTo()
{
    return this->toModel;
}

template <class In, class Out>
void Mapping<In,Out>::init()
{
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
void Mapping<In,Out>::updateMapping()
{
    if (this->toModel == NULL || this->fromModel == NULL)
        return;

    if (this->toModel->getX()!=NULL && this->fromModel->getX()!=NULL)
    {
        apply(*this->toModel->getX(), *this->fromModel->getX());
        //cerr<<"Mapping<In,Out>::updateMapping(), *this->fromModel->getX() = "<<*this->fromModel->getX()<<endl;
        //cerr<<"Mapping<In,Out>::updateMapping(), *this->toModel->getX() = "<<*this->toModel->getX()<<endl;
    }
    if (this->toModel->getV()!=NULL && this->fromModel->getV()!=NULL)
    {
        applyJ(*this->toModel->getV(), *this->fromModel->getV());
    }
}

} // namespace core

} // namespace sofa

#endif
