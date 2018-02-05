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

#include "ComponentLibrary.h"
#include "CategoryLibrary.h"

namespace sofa
{

namespace core
{

std::string caseInsensitive(const std::string &text)
{
    std::string result; result.resize(text.size());
    for (unsigned int i=0; i<text.size(); ++i) result[i] = toupper(text[i]);
    return result;
}

//-------------------------------------------------------------------------------------------------------
ComponentLibrary::ComponentLibrary( const std::string &componentN, const std::string &categoryN, ClassEntry::SPtr e, const std::vector< std::string > &exampleFiles):  name(componentN), categoryName(categoryN),entry(e)
{

    description  = std::string("<H2>")  + entry->className + std::string(": ");

    std::vector< std::string > possiblePaths;

    std::vector<std::string> categories;
    const objectmodel::BaseClass* entryClass = entry->creatorMap.begin()->second->getClass();
    CategoryLibrary::getCategories(entryClass, categories);
    for (std::vector< std::string >::iterator it=categories.begin(); it!=categories.end() ; ++it)
    {
        if (it != categories.begin()) description += std::string(", ");
        description += (*it);
    }

    //Find a scene
    std::string nameComponentCaseInsensitive = caseInsensitive(entry->className);

    for (unsigned int i=0; i<exampleFiles.size(); ++i)
    {
        std::string exampleCaseInsensitive = caseInsensitive(exampleFiles[i]);
//             if (exampleFiles[i].findRev(entry->className.c_str()) >= 0 )
        if (exampleCaseInsensitive.find(nameComponentCaseInsensitive) != std::string::npos)
            possiblePaths.push_back(exampleFiles[i]);
    }

    std::string nameSpace = sofa::core::objectmodel::BaseClass::decodeNamespaceName(entry->creatorMap.begin()->second->type());


    description += std::string("</H2>");

    description += std::string("<ul>");

    description += std::string("<li><b>Description: </b>") + entry->description + std::string("</li>");


    if (!nameSpace.empty())
        description += std::string("<li><b>NameSpace: </b>")+nameSpace +std::string("</li>");
    if (!entry->authors.empty())
        description += std::string("<li><b>Authors: </b>")+entry->authors +std::string("</li>");
    if (!entry->license.empty())
        description += std::string("<li><b>License: </b>") + entry->license + std::string("</li>");

    if (possiblePaths.size() != 0)
    {
        description += std::string("<li><b>Example: </b><ul>");
        for (unsigned int i=0; i<possiblePaths.size(); ++i)
        {
            description += std::string("<li><a href=\"")+possiblePaths[i]+std::string("\">") + possiblePaths[i] + std::string("</a></li>");
        }
        description += std::string("</ul>");
    }

    description += std::string("</ul>");
}





void ComponentLibrary::addTemplate( const std::string &nameT)
{
    if (nameT.empty()) return;
    templateName.push_back(nameT);
}


void ComponentLibrary::endConstruction()
{
}

}
}
