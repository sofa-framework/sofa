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
#include "AbstractTypeInfo.h"
#include "TypeInfoRegistryTools.h"
namespace sofa::defaulttype
{

void TypeInfoRegistryTools::dumpRegistryContentToStream(std::ostream& out,
                                                        TypeInfoType type,
                                                        const std::string& target)
{
    auto types = sofa::defaulttype::TypeInfoRegistry::GetRegisteredTypes(target);
    out << "Module '" << target << "' has " << types.size()  <<  " types defined:" << std::endl;
    for(auto& info :types)
    {
        std::string prefix = (info->ValidInfo())?" - ":" ! ";
        if(type==TypeInfoType::NONE && info)
        {            
            out << prefix << info->name() << std::endl;
        }else if(type==TypeInfoType::PARTIAL && info && !info->ValidInfo())
        {
            out << prefix << info->name() << std::endl;
        }else if(type==TypeInfoType::COMPLETE && info && info->ValidInfo())
        {
            out << prefix << info->name() << std::endl;
        }else if(type==TypeInfoType::ALL)
        {
            out << prefix << info->name() << std::endl;
        }
    }
}

}
