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
#include <sofa/gl/component/shader/OglAttribute.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gl::component::shader
{

using namespace sofa::defaulttype;

void registerOglAttribute(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("OglFloatAttribute").add< OglFloatAttribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloat2Attribute").add< OglFloat2Attribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloat3Attribute").add< OglFloat3Attribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloat4Attribute").add< OglFloat4Attribute >());

    factory->registerObjects(core::ObjectRegistrationData("OglIntAttribute").add< OglIntAttribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglInt2Attribute").add< OglInt2Attribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglInt3Attribute").add< OglInt3Attribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglInt4Attribute").add< OglInt4Attribute >());

    factory->registerObjects(core::ObjectRegistrationData("OglUIntAttribute").add< OglUIntAttribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglUInt2Attribute").add< OglUInt2Attribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglUInt3Attribute").add< OglUInt3Attribute >());
    factory->registerObjects(core::ObjectRegistrationData("OglUInt4Attribute").add< OglUInt4Attribute >());
}

} // namespace sofa::gl::component::shader
