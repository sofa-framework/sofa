/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <Communication/config.h>
#include <Communication/components/communicationBinder.h>
#include <sofa/core/ObjectFactory.h>

using sofa::core::RegisterObject ;

#include <iostream>

namespace sofa
{

namespace component
{

namespace communication
{

template <class DataTypes>
CommunicationBinder<DataTypes>::CommunicationBinder()
        : m_data(initData(&m_data, "data", "position coordinates of the degrees of freedom"))
{}

template <class DataTypes>
CommunicationBinder<DataTypes>::~CommunicationBinder(){}

template <class DataTypes>
void CommunicationBinder<DataTypes>::init()
{
    f_listening = true;
}

template <class DataTypes>
void CommunicationBinder<DataTypes>::handleEvent(Event* event)
{
    std::cout << getTemplateName() << std::endl;
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        // TODO update data
    }
}

int CommunicationBinderClass = RegisterObject("Binding device communication.")
        #ifndef SOFA_FLOAT
                .add< CommunicationBinder<float> >()
        #endif
        #ifndef SOFA_DOUBLE
                .add< CommunicationBinder<double> >()
        #endif
                .add< CommunicationBinder<std::string> >(true)
                .add< CommunicationBinder<int> >()
                .add< CommunicationBinder<unsigned int> >()
        ;

} /// communication

} /// component

} /// sofa
