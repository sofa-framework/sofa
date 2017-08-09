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
#include "serverCommunication.inl"

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{


template<>
std::string ServerCommunication<double>::templateName(const ServerCommunication<double>* object)
{
    SOFA_UNUSED(object);
    return "double";
}

template<>
std::string ServerCommunication<float>::templateName(const ServerCommunication<float>* object)
{
    SOFA_UNUSED(object);
    return "float";
}

template<>
std::string ServerCommunication<int>::templateName(const ServerCommunication<int>* object)
{
    SOFA_UNUSED(object);
    return "int";
}

template<>
std::string ServerCommunication<std::string>::templateName(const ServerCommunication<std::string>* object)
{
    SOFA_UNUSED(object);
    return "string";
}

#ifndef SOFA_FLOAT
template class SOFA_CORE_API ServerCommunication<float>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CORE_API ServerCommunication<double>;
#endif
template class SOFA_CORE_API ServerCommunication<int>;
template class SOFA_CORE_API ServerCommunication<std::string>;

} /// communication

} /// component

} /// sofa
