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
#include <sofa/defaulttype/defaulttype.h>

#include <string>

namespace sofa
{

namespace defaulttype
{

/**
 *  \brief Class used to store and resolve template aliases.
 *
 *  \see RegisterTemplateAlias for how new aliases should be registered.
 *
 */
class SOFA_DEFAULTTYPE_API TemplateAliases
{
public:
	/// Add an alias for a template
    ///
    /// \param name     name of the new alias
    /// \param result   real template pointed to
    static bool addAlias(const std::string& name, const std::string& result);

	/// Get the template pointed to by the alias. Returns the input if there is no alias.
	static std::string resolveAlias(const std::string& name);
};

/**
 *  \brief Helper class used to register a template alias in the TemplateAliases class.
 *
 *  It should be used as a temporary object. For example :
 *  \code
 *    core::RegisterTemplateAlias Vec3Alias("Vec3", "Vec3d");
 *  \endcode
 *
 */
class SOFA_DEFAULTTYPE_API RegisterTemplateAlias
{
public:
    /// Register an alias
    RegisterTemplateAlias(const std::string& alias, const std::string& result);
};

}// defaulttype

}// sofa
