/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "VecTypes.h"
#include "RigidTypes.h"
namespace sofa
{

namespace defaulttype
{

typedef std::map<std::string, TemplateAlias> TemplateAliasesMap;
typedef TemplateAliasesMap::const_iterator TemplateAliasesMapIterator;
TemplateAliasesMap& getTemplateAliasesMap()
{
	static TemplateAliasesMap theMap;
	return theMap;
}

bool TemplateAliases::addAlias(const std::string& name, const std::string& result, const bool doWarnUser)
{
	TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
	if (templateAliases.find(name) != templateAliases.end())
    {
        msg_warning("ObjectFactory") << "cannot create template alias " << name << " as it already exists";
		return false;
	}
	else
	{
        templateAliases[name] = std::make_pair(result, doWarnUser);
		return true;
	}
}

const TemplateAlias* TemplateAliases::getTemplateAlias(const std::string &name)
{
    TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
    TemplateAliasesMapIterator it = templateAliases.find(name);
    if (it != templateAliases.end())
        return  &(it->second);
    return nullptr;
}

std::string TemplateAliases::resolveAlias(const std::string& name)
{
	TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
	TemplateAliasesMapIterator it = templateAliases.find(name);
	if (it != templateAliases.end())
        return it->second.first;
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
                resolved.replace(first, last-first, it->second.first);

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
	
RegisterTemplateAlias::RegisterTemplateAlias(const std::string& alias, const std::string& result, const bool doWarnUser)
{
    TemplateAliases::addAlias(alias, result, doWarnUser);
}


/// The following types are the generic 'precision'
static RegisterTemplateAlias Vec1Alias("Vec1", sofa::defaulttype::Vec1Types::Name());
static RegisterTemplateAlias Vec2Alias("Vec2", sofa::defaulttype::Vec2Types::Name());
static RegisterTemplateAlias Vec3Alias("Vec3", sofa::defaulttype::Vec3Types::Name());
static RegisterTemplateAlias Vec6Alias("Vec6", sofa::defaulttype::Vec6Types::Name());
static RegisterTemplateAlias Rigid2Alias("Rigid2", sofa::defaulttype::Rigid2Types::Name());
static RegisterTemplateAlias Rigid3Alias("Rigid3", sofa::defaulttype::Rigid3Types::Name());

/// Compatibility aliases for niceness.
static RegisterTemplateAlias RigidAlias("Rigid", sofa::defaulttype::Rigid3Types::Name(), true);

static RegisterTemplateAlias Rigid2fAlias("Rigid2f", sofa::defaulttype::Rigid2Types::Name(), isSRealDouble());
static RegisterTemplateAlias Rigid3fAlias("Rigid3f", sofa::defaulttype::Rigid3Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec1fAlias("Vec1f", sofa::defaulttype::Vec1Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec2fAlias("Vec2f", sofa::defaulttype::Vec2Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec3fAlias("Vec3f", sofa::defaulttype::Vec3Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec6fAlias("Vec6f", sofa::defaulttype::Vec6Types::Name(), isSRealDouble());

static RegisterTemplateAlias Vec1dAlias("Vec1d", sofa::defaulttype::Vec1Types::Name(), isSRealFloat());
static RegisterTemplateAlias Vec2dAlias("Vec2d", sofa::defaulttype::Vec2Types::Name(), isSRealFloat());
static RegisterTemplateAlias Vec3dAlias("Vec3d", sofa::defaulttype::Vec3Types::Name(), isSRealFloat());
static RegisterTemplateAlias Vec6dAlias("Vec6d", sofa::defaulttype::Vec6Types::Name(), isSRealFloat());
static RegisterTemplateAlias Rigid2dAlias("Rigid2d", sofa::defaulttype::Rigid2Types::Name(), isSRealFloat());
static RegisterTemplateAlias Rigid3dAlias("Rigid3d", sofa::defaulttype::Rigid3Types::Name(), isSRealFloat());

// deprecated template names

[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec1fAlias("ExtVec1f", sofa::defaulttype::Vec1Types::Name(), isSRealDouble());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec2fAlias("ExtVec2f", sofa::defaulttype::Vec2Types::Name(), isSRealDouble());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec3fAlias("ExtVec3f", sofa::defaulttype::Vec3Types::Name(), isSRealDouble());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec6fAlias("ExtVec6f", sofa::defaulttype::Vec6Types::Name(), isSRealDouble());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec1dAlias("ExtVec1d", sofa::defaulttype::Vec1Types::Name(), isSRealFloat());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec2dAlias("ExtVec2d", sofa::defaulttype::Vec2Types::Name(), isSRealFloat());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec3dAlias("ExtVec3d", sofa::defaulttype::Vec3Types::Name(), isSRealFloat());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec6dAlias("ExtVec6d", sofa::defaulttype::Vec6Types::Name(), isSRealFloat());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec1Alias("ExtVec1", sofa::defaulttype::Vec1Types::Name());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec2Alias("ExtVec2", sofa::defaulttype::Vec2Types::Name());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec3Alias("ExtVec3", sofa::defaulttype::Vec3Types::Name());
[[deprecated("since 19.06, ExtVecTypes are deprecated. Use VecTypes instead. Aliases will be removed in 19.12")]]
static RegisterTemplateAlias ExtVec6Alias("ExtVec6", sofa::defaulttype::Vec6Types::Name());


}// defaulttype

}// sofa
