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
#define TESTPLUGINA_COMPONENT_B_CPP

#include "ComponentB.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::test
{

template<class T>
ComponentB<T>::ComponentB()
{
}


template<class T>
ComponentB<T>::~ComponentB()
{
}

int ComponentBClass = sofa::core::RegisterObject("Component B")
    .add< ComponentB<double> >()
    .add< ComponentB<sofa::defaulttype::Vec2Types> >()
    .add< ComponentB<sofa::defaulttype::Rigid3Types> >()

;

template class SOFA_TESTPLUGINA_API ComponentB<double>; 
template class SOFA_TESTPLUGINA_API ComponentB<sofa::defaulttype::Vec2Types>;
template class SOFA_TESTPLUGINA_API ComponentB<sofa::defaulttype::Rigid3Types>;


} // namespace sofa::test


