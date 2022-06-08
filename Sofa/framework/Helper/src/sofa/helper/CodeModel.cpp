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
#include <sofa/helper/CodeModel.h>
#include <vector>
#include <iostream>
#include <map>

namespace sofa::helper
{

class CodeModelEntry
{
public:
    std::map<std::string, std::string> rdf;

    const std::string& get(const std::string& key) const
    {
        return rdf.at(key);
    }
};

class CodeModel
{
    std::map<std::string, int> nameToEntryIndex;
    std::vector<CodeModelEntry> entries;

public:
    const CodeModelEntry* getEntry(const std::string& s)
    {
        auto r = nameToEntryIndex.find(s);
        if(r==nameToEntryIndex.end())
        {
            return nullptr;
        }
        return &entries[r->second];
    }

    const CodeModelEntry* loadCodeEntry(const std::string& componentFullName)
    {
        std::cout << "Loading documentation for " << componentFullName << std::endl;

        return nullptr;
    }
};

CodeModel& getCodeModel()
{
    static CodeModel model;
    return model;
}

const std::string CodeModelInstance::getDescription(const std::string& componentFullName)
{
    const CodeModelEntry* entry = getCodeModel().getEntry(componentFullName);
    if(!entry)
        entry = getCodeModel().loadCodeEntry(componentFullName);

    if(!entry)
    {
        std::cout << "There is really no code model for this component. " << componentFullName << " .. use the object factory ? " << std::endl;
        return "";
    }

    return entry->get("description");
}


} /// namespace sofa::helper
