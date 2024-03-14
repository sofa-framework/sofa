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
#include <sofa/defaulttype/AbstractTypeInfo.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>
namespace sofa::defaulttype
{

void TypeInfoRegistryTools::dumpRegistryContentToStream(std::ostream& out,
                                                        TypeInfoType type,
                                                        const std::string& target)
{
    const auto types = sofa::defaulttype::TypeInfoRegistry::GetRegisteredTypes(target);

    int selected=0;
    for(const auto& info :types)
    {
        if(type==TypeInfoType::MISSING && info)
            selected++;
        else if(type==TypeInfoType::NAMEONLY && info && !info->ValidInfo())
            selected++;
        else if(type==TypeInfoType::COMPLETE && info && info->ValidInfo())
            selected++;
    }

    out << "Target '" << target << "' has " << selected << "/" << types.size()  <<  " types." << std::endl;
    for(const auto& info :types)
    {
        if(type==TypeInfoType::MISSING && info)
        {            
            out << " ? " << info->name() << std::endl;
        }else if(type==TypeInfoType::NAMEONLY && info && !info->ValidInfo())
        {
            out << " N " << info->name() << std::endl;
        }else if(type==TypeInfoType::COMPLETE && info && info->ValidInfo())
        {
            out << " C " << info->name() << std::endl;
        }
    }
}

}
