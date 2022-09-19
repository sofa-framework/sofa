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
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/TypeDeductionRules.h>

namespace sofa::core
{

std::string CopyTypeFromMechanicalState(sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription*)
{
    if(context->getMechanicalState())
        return context->getMechanicalState()->getTemplateName();
    return "";
}

std::string CopyTypeFromMeshTopology(sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription*)
{
    if(context->getMeshTopology())
        return context->getMeshTopology()->getTemplateName();
    return "";
}


std::string DeducedFromLinkedMechanicalState(const std::string& attributeName, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
{
    // get the template type from the object pointed by the linkPath
    std::string linkedPath = arg->getAttribute(attributeName,"");
    if (!linkedPath.empty() )
    {
        auto base = sofa::core::PathResolver::FindBaseFromPath(context, linkedPath);
        if(base!=nullptr)
            return base->getTemplateName();
    }

    // if we were not able to get it, then deduce it from the context's
    if(context->getMechanicalState())
        return context->getMechanicalState()->getTemplateName();

    return "";
}

std::string DeducedFromLinkedBaseMeshTopology(const std::string& attributeName, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
{
    // get the template type from the object pointed by the linkPath
    std::string linkedPath = arg->getAttribute(attributeName,"");
    if (!linkedPath.empty() )
    {
        auto base = sofa::core::PathResolver::FindBaseFromPath(context, linkedPath);
        if(base!=nullptr)
            return base->getTemplateName();
    }

    // if we were not able to get it, then deduce it from the context's
    if(context->getMeshTopology())
        return context->getMeshTopology()->getTemplateName();

    return "";
}

}
