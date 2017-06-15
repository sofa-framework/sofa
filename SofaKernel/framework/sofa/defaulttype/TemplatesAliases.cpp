/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "TemplatesAliases.h"

#include <iostream>
#include <map>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace defaulttype
{

typedef std::map<std::string, std::string> TemplateAliasesMap;
typedef TemplateAliasesMap::const_iterator TemplateAliasesMapIterator;
TemplateAliasesMap& getTemplateAliasesMap()
{
	static TemplateAliasesMap theMap;
	return theMap;
}

bool TemplateAliases::addAlias(const std::string& name, const std::string& result)
{
	TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
	if (templateAliases.find(name) != templateAliases.end())
    {
        msg_warning("ObjectFactory") << "cannot create template alias " << name << " as it already exists";
		return false;
	}
	else
	{
		templateAliases[name] = result;
		return true;
	}
}

std::string TemplateAliases::resolveAlias(const std::string& name)
{
	TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
	TemplateAliasesMapIterator it = templateAliases.find(name);
	if (it != templateAliases.end())
		return it->second;
	else if (name.find(",") != std::string::npos) // Multiple templates, resolve each one
	{
		std::string resolved = name;
		std::string::size_type first = 0;
		while (true)
		{
			std::string::size_type last = resolved.find_first_of(",", first);
			if (last == std::string::npos) // Take until the end of the string if there is no more comma
				last = resolved.size();
			std::string token = resolved.substr(first, last-first);

			// Replace the token with the alias (if there is one)
			it = templateAliases.find(token);
			if (it != templateAliases.end())
				resolved.replace(first, last-first, it->second);

			// Recompute the start of next token as we can have changed the length of the string
			first = resolved.find_first_of(",", first);
			if (first == std::string::npos)
				break;
			++first;
		}

		return resolved;
	}
	else
		return name;
}
	
RegisterTemplateAlias::RegisterTemplateAlias(const std::string& alias, const std::string& result)
{
	TemplateAliases::addAlias(alias, result);
}

#ifndef SOFA_FLOAT
RegisterTemplateAlias Vec1Alias("Vec1", "Vec1d");
RegisterTemplateAlias Vec2Alias("Vec2", "Vec2d");
RegisterTemplateAlias Vec3Alias("Vec3", "Vec3d");
RegisterTemplateAlias Vec4Alias("Vec4", "Vec4d");
RegisterTemplateAlias Vec6Alias("Vec6", "Vec6d");
RegisterTemplateAlias Rigid2Alias("Rigid2", "Rigid2d");
RegisterTemplateAlias Rigid3Alias("Rigid3", "Rigid3d");
RegisterTemplateAlias RigidAlias("Rigid", "Rigid3d");
#else
RegisterTemplateAlias Vec1Alias("Vec1", "Vec1f");
RegisterTemplateAlias Vec2Alias("Vec2", "Vec2f");
RegisterTemplateAlias Vec3Alias("Vec3", "Vec3f");
RegisterTemplateAlias Vec4Alias("Vec4", "Vec4f");
RegisterTemplateAlias Vec6Alias("Vec6", "Vec6f");
RegisterTemplateAlias Rigid2Alias("Rigid2", "Rigid2f");
RegisterTemplateAlias Rigid3Alias("Rigid3", "Rigid3f");
RegisterTemplateAlias RigidAlias("Rigid", "Rigid3f");
#endif

}// defaulttype

}// sofa
