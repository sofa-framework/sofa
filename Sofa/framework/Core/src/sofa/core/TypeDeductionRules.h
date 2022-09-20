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
#pragma once
#include <sofa/core/config.h>

namespace sofa::core
{
SOFA_CORE_API std::string CopyTypeFromMechanicalState(sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription*);
SOFA_CORE_API std::string CopyTypeFromMeshTopology(sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription*);

/// Deduce the object template from the object pointed by the linkName, if not set, use the provided defaultValue
template<class TargetObject>
SOFA_CORE_API std::string DeducedFromLink(const std::string& attributeName, const std::string defaultValue, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
{
    // get the template type from the object pointed by the linkPath
    const std::string linkedPath = arg->getAttribute(attributeName, defaultValue.c_str());
    auto base = sofa::core::PathResolver::FindBaseFromClassAndPath(context, TargetObject::GetClass(), linkedPath);
    if(base!=nullptr)
        return base->getTemplateName();

    return "";
}

/// Deduce the object template from the object pointed by the linkName, if not set, use the provided defaultValue
SOFA_CORE_API std::string DeducedFromLink(const std::string& attributeName, const std::string defaultValue, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg);

/// Deduce the object template from the object pointed by the linkName, then search in the current context for a MechanicalState
SOFA_CORE_API std::string DeducedFromLinkedMechanicalState(const std::string& attributeName, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription*);

/// Deduce the object template from the object pointed by the linkName, then search in the current context for a BaseMeshTopology
SOFA_CORE_API std::string DeducedFromLinkedBaseMeshTopology(const std::string& attributeName, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription*);
}
