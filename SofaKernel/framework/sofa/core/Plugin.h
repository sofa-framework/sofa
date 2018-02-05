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
#ifndef SOFA_CORE_PLUGIN_H
#define SOFA_CORE_PLUGIN_H

#include <sofa/helper/system/config.h>

#define SOFA_PLUGIN(PluginClass)				\
public: 							\
    static sofa::core::Plugin::RegisterObject			\
    registerObject(const std::string& description) {		\
      return RegisterObject(getInstance(), description);	\
    }								\
    static sofa::core::Plugin& getInstance() {			\
        static PluginClass instance;				\
	return instance;					\
    }

#define SOFA_DECL_PLUGIN(PluginClass)				\
    class PluginClass: public sofa::core::Plugin {		\
        SOFA_PLUGIN(PluginClass);				\
    public:							\
	PluginClass();						\
    };

#define SOFA_PLUGIN_ENTRY_POINT(PluginClass)                        \
    extern "C" SOFA_EXPORT_DYNAMIC_LIBRARY void * get_plugin()	    \
    {                                                               \
        return &PluginClass::getInstance();                         \
    }


#include <sofa/core/core.h>
#include <sofa/core/ObjectFactory.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace sofa
{
namespace core
{


class SOFA_CORE_API Plugin
{
public:

    struct SOFA_CORE_API ComponentEntry
    {
        /// The name of the class or class template.
        std::string name;
        /// True iff the component is a class template.
        bool isATemplate;
        /// If the component is a template, the parameters used in the
        /// default instanciation of the template, if any.
        ///
        /// If the component is a not template, or if it has no
        /// default instanciation in this plugin, this string is empty.
        std::string defaultTemplateParameters;
        /// A description of the component.
        std::string description;
        /// Aliases (different names) for the component.
        std::set<std::string> aliases;
        /// Creators for the component.
        sofa::core::ObjectFactory::CreatorMap creators;

        ComponentEntry() {};
        ComponentEntry(const std::string& name, bool isATemplate):
            name(name), isATemplate(isATemplate) {};
    };
    typedef std::map<std::string, ComponentEntry> ComponentEntryMap;


    class SOFA_CORE_API RegisterObject
    {
    private:
        sofa::core::Plugin& m_plugin;
        std::string m_description;
        bool m_firstAddDone;
        ComponentEntry *m_entry;
    public:

        /// Start the registration by giving the description of this class.
        RegisterObject(sofa::core::Plugin& plugin,
                       const std::string& description):
            m_plugin(plugin), m_description(description),
            m_firstAddDone(false), m_entry(NULL) {
        }

        /// Add an alias name for this class
        RegisterObject& addAlias(std::string val)
        {
            if (m_firstAddDone) {
                m_entry->aliases.insert(val);
            }
            return *this;
        }

        /// Add more descriptive text about this class
        RegisterObject& addDescription(std::string val)
        {
            (void)(val);
            return *this;
        }

        /// Specify a list of authors (separated with spaces)
        RegisterObject& addAuthor(std::string val)
        {
            (void)(val);
            return *this;
        }

        /// Specify a license (LGPL, GPL, ...)
        RegisterObject& addLicense(std::string val)
        {
            (void)(val);
            return *this;
        }

        /// Add a template instanciation of this class.
        ///
        /// \param isDefault    set to true if this should be the default instance when no template name is given.
        template<class Component>
            RegisterObject& add(bool isDefault=false)
        {
            Component* p = NULL;
            const std::string name = Component::className(p);
            const std::string templateParameters = Component::templateName(p);

            if (!m_firstAddDone) {
                m_entry = &m_plugin.m_components[name];
                *m_entry = ComponentEntry(name, !templateParameters.empty());
                m_entry->description = m_description;
		m_firstAddDone = true;
            }

            m_plugin.m_components[name].creators[templateParameters] =
                ObjectFactory::Creator::SPtr(new ObjectCreator<Component>);

            if (isDefault)
                m_entry->defaultTemplateParameters =
                    templateParameters;

            return *this;
        }

        operator int() { return 0; }
    };
    // friend class RegisterObject;

    Plugin(std::string name, bool isLegacy=false):
        m_name(name), m_isLegacy(isLegacy) {}

    Plugin(const std::string& name, const std::string& description,
           const std::string& version, const std::string& license,
           const std::string& authors, bool isLegacy=false):
        m_name(name), m_description(description), m_version(version),
        m_license(license), m_authors(authors), m_isLegacy(isLegacy) {}

    virtual ~Plugin() {}

    virtual bool init() { return true; }
    virtual bool exit() { return true; }
    virtual bool canBeUnloaded() { return true; }

    /// Get the ComponentEntry for a component already added to the plugin
    const ComponentEntry& getComponentEntry(std::string name);

    /// Get the map containing the information about the plugin's components.
    const ComponentEntryMap& getComponentEntries() const { return m_components; }

    const std::string& getName() const { return m_name; }
    const std::string& getDescription() const { return m_description; }
    const std::string& getVersion() const { return m_version; }
    const std::string& getLicense() const { return m_license; }
    const std::string& getAuthors() const { return m_authors; }
    bool isLegacy() const { return m_isLegacy; }

protected:

    /// Add a component
    template <class Component> ComponentEntry& addComponent(const std::string& description = "") {
        Component* p = NULL;
        const std::string name = Component::className(p);
        const std::string templateParameters = Component::templateName(p);

        if (m_components.find(name) != m_components.end())
            throw std::runtime_error("Nope.");

        ComponentEntry& entry = m_components[name];
        entry = ComponentEntry(name, !templateParameters.empty());
        entry.description = description;
        entry.defaultTemplateParameters = templateParameters;
        entry.creators[templateParameters] =
            ObjectFactory::Creator::SPtr(new ObjectCreator<Component>);
        return entry;
    }

    /// Add an instanciation of a class template.
    template<class Component> void addTemplateInstance()
    {
        Component* p = NULL;
        const std::string name = Component::className(p);
        const std::string templateParameters = Component::templateName(p);

        if (m_components.find(name) == m_components.end())
            m_components[name] = ComponentEntry(name, true);

        ComponentEntry& entry = m_components[name];
        entry.creators[templateParameters] =
            ObjectFactory::Creator::SPtr(new ObjectCreator<Component>);
    }

    /// Set the description of a component that already has an entry.
    void setDescription(std::string componentName, std::string description);

    /// Add an alias for a component that already has an entry.
    void addAlias(std::string componentName, std::string alias);

    void setDescription(const std::string& description) { m_description = description; };
    void setVersion(const std::string& version) { m_version = version; };
    void setLicense(const std::string& license) { m_license = license; };
    void setAuthors(const std::string& authors) { m_authors = authors; };

private:
    std::string m_name;
    std::string m_description;
    std::string m_version;
    std::string m_license;
    std::string m_authors;
    bool m_isLegacy;
    ComponentEntryMap m_components;
};


}

}

#endif
