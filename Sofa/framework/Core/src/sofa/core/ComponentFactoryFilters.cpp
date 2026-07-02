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
#include <sofa/core/ComponentFactoryFilters.h>
#include <sofa/defaulttype/TemplatesAliases.h>

namespace sofa::core
{

std::vector<ComponentRegistrationData::SPtr> ExactTemplateMatchFilter::filter(
    const std::vector<ComponentRegistrationData::SPtr>& candidates,
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) const
{
    SOFA_UNUSED(context);

    std::vector<ComponentRegistrationData::SPtr> exactlyMatchingCandidates;

    for (const auto& candidate : candidates)
    {
        bool matchAllTemplateParameters = true;
        for (const auto& [attribute, value] : candidate->templateAttributes)
        {
            const auto resolvedValue = defaulttype::TemplateAliases::resolveAlias(value);
            const char* attr = arg->getAttribute(attribute, nullptr);
            if (attr == nullptr)
            {
                matchAllTemplateParameters = false;
            }
            else
            {
                const std::string attrStr{attr};
                const auto resolvedAlias = defaulttype::TemplateAliases::resolveAlias(attrStr);
                if (resolvedAlias != resolvedValue)
                {
                    matchAllTemplateParameters = false;
                }
            }
        }

        if (matchAllTemplateParameters)
        {
            exactlyMatchingCandidates.push_back(candidate);
        }
    }

    return exactlyMatchingCandidates;
}

std::vector<ComponentRegistrationData::SPtr> LegacyTemplateKeywordFilter::filter(
    const std::vector<ComponentRegistrationData::SPtr>& candidates,
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) const
{
    SOFA_UNUSED(context);

    const char* templateAttr = arg->getAttribute("template", nullptr);
    if (!templateAttr) return {};

    std::string templateAttrStr{templateAttr};
    templateAttrStr = defaulttype::TemplateAliases::resolveAlias(templateAttrStr);

    std::vector<ComponentRegistrationData::SPtr> matchingCandidates;

    for (const auto& candidate : candidates)
    {
        const auto templateList = sofa::helper::join(
            candidate->templateAttributes.begin(), candidate->templateAttributes.end(),
            [](const auto& attr)
            { return defaulttype::TemplateAliases::resolveAlias(attr.second); }, ',');
        if (templateAttrStr == templateList)
        {
            matchingCandidates.push_back(candidate);
        }
    }

    return matchingCandidates;
}

std::vector<ComponentRegistrationData::SPtr> PartialTemplateMatchFilter::filter(
    const std::vector<ComponentRegistrationData::SPtr>& candidates,
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) const
{
    SOFA_UNUSED(context);

    std::vector<ComponentRegistrationData::SPtr> partiallyMatchingCandidates;

    for (const auto& candidate : candidates)
    {
        for (const auto& [attribute, value] : candidate->templateAttributes)
        {
            const char* attr = arg->getAttribute(attribute, nullptr);
            if (attr != nullptr)
            {
                const std::string attrStr{attr};
                if (defaulttype::TemplateAliases::resolveAlias(attrStr) == value)
                {
                    partiallyMatchingCandidates.push_back(candidate);
                    break;
                }
            }
        }
    }

    return partiallyMatchingCandidates;
}

std::vector<ComponentRegistrationData::SPtr> NoFilter::filter(
    const std::vector<ComponentRegistrationData::SPtr>& candidates,
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) const
{
    SOFA_UNUSED(context);
    SOFA_UNUSED(arg);
    return candidates;
}
}  // namespace sofa::core
