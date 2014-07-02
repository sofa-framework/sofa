#include "TemplatesAliases.h"

#include <iostream>
#include <map>

namespace sofa
{

namespace defaulttype
{

typedef std::map<std::string, std::string> TemplateAliasesMap;
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
		std::cerr << "ERROR: ObjectFactory: cannot create template alias " << name << " as it already exists.\n";
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
	if (templateAliases.find(name) != templateAliases.end())
		return templateAliases[name];
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
RegisterTemplateAlias Vec6Alias("Vec6", "Vec6d");
RegisterTemplateAlias Rigid2Alias("Rigid2", "Rigid2d");
RegisterTemplateAlias Rigid3Alias("Rigid", "Rigid3d");
#else
RegisterTemplateAlias Vec1Alias("Vec1", "Vec1f");
RegisterTemplateAlias Vec2Alias("Vec2", "Vec2f");
RegisterTemplateAlias Vec3Alias("Vec3", "Vec3f");
RegisterTemplateAlias Vec6Alias("Vec6", "Vec6f");
RegisterTemplateAlias Rigid2Alias("Rigid2", "Rigid2f");
RegisterTemplateAlias Rigid3Alias("Rigid", "Rigid3f");
#endif

}// defaulttype

}// sofa

