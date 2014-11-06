#include <sofa/SofaFramework.h>

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

