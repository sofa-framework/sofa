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
#define TESTPLUGIN_COMPONENT_B_CPP

#include "ComponentB.h"

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace test
{

template<class T>
ComponentB<T>::ComponentB()
{
}


template<class T>
ComponentB<T>::~ComponentB()
{
}

SOFA_DECL_CLASS(ComponentB)

int ComponentBClass = sofa::core::RegisterObject("Component B")
#ifndef SOFA_FLOAT
    .add< ComponentB<double> >()
    .add< ComponentB<sofa::defaulttype::Vec2dTypes> >()
    .add< ComponentB<sofa::defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
    .add< ComponentB<float> >()
    .add< ComponentB<sofa::defaulttype::Vec2fTypes> >()
    .add< ComponentB<sofa::defaulttype::Rigid3fTypes> >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_TESTPLUGIN_API ComponentB<double>; 
template class SOFA_TESTPLUGIN_API ComponentB<sofa::defaulttype::Vec2dTypes>;
template class SOFA_TESTPLUGIN_API ComponentB<sofa::defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_TESTPLUGIN_API ComponentB<float>;
template class SOFA_TESTPLUGIN_API ComponentB<sofa::defaulttype::Vec2fTypes>;
template class SOFA_TESTPLUGIN_API ComponentB<sofa::defaulttype::Rigid3fTypes>;
#endif



} // namespace test

} // namespace sofa
