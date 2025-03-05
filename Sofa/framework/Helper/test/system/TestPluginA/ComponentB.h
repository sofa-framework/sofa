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
#ifndef TESTPLUGINA_COMPONENT_B_H
#define TESTPLUGINA_COMPONENT_B_H

#include <TestPluginA/TestPluginA.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::test
{

template<class T>
class SOFA_TESTPLUGINA_API ComponentB : public sofa::core::objectmodel::BaseObject
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(ComponentB, T), sofa::core::objectmodel::BaseObject);

protected:
    ComponentB();
    ~ComponentB() override;

};

#if !defined(TESTPLUGINA_COMPONENT_B_CPP)
extern template class SOFA_TESTPLUGINA_API ComponentB<double>;
extern template class SOFA_TESTPLUGINA_API ComponentB<defaulttype::Vec2Types>;
extern template class SOFA_TESTPLUGINA_API ComponentB<defaulttype::Rigid3Types>;

#endif //  !defined(TESTPLUGINA_COMPONENT_B_CPP)

} // namespace sofa::test


#endif // TESTPLUGINA_COMPONENT_B_H
