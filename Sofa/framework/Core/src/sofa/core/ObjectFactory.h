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
#pragma once

#include <sofa/core/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseClassNameHelper.h>
#include <numeric>
#include <sofa/helper/Utils.h>
#include <sofa/url.h>


namespace sofa::helper::system
{
    class Plugin;
}

namespace sofa::core
{

class ObjectRegistrationData;

typedef std::function<void(sofa::core::objectmodel::Base*, sofa::core::objectmodel::BaseObjectDescription*)> OnCreateCallback ;

/**
 *  \brief Main class used to register and dynamically create objects
 *
 *  It uses the Factory design pattern, where each class is registered in a map
 *  and dynamically retrieved given the type name.
 *
 *  It also stores metainformation on each class, such as description,
 *  authors, license, and available template types.
 */
class SOFA_CORE_API ObjectFactory
{
public:

    /**
     * Abstract interface used to create instances (object) of a given type
     * See the derived class @ref ObjectCreator.
     */
    class SOFA_CORE_API BaseObjectCreator
    {
    public:
        using SPtr = std::shared_ptr<BaseObjectCreator>;

        virtual ~BaseObjectCreator() = default;

        /// Pre-construction check.
        ///
        /// \return true if the object can be created successfully.
        virtual bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) = 0;

        /// Construction method called by the factory.
        ///
        /// \pre canCreate(context, arg) == true.
        virtual objectmodel::BaseObject::SPtr createInstance(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) = 0;

        /// type_info structure associated with the type of instantiated objects.
        virtual const std::type_info& type() = 0;

        /// BaseClass structure associated with the type of instantiated objects.
        virtual const objectmodel::BaseClass* getClass() = 0;

        /// The name of the library or executable containing the binary code for this component
        virtual const char* getTarget() = 0;

        virtual const char* getHeaderFileLocation() = 0;
    };
    using Creator SOFA_CORE_DEPRECATED_RENAME_CREATOR_BASEOBJECTCREATOR() = BaseObjectCreator;

    using TemplateName = std::string;

    /// For a given templated class, the map stores all creators and the key is the template name.
    using ObjectTemplateCreatorMap = std::map<TemplateName, BaseObjectCreator::SPtr>;

    using CreatorMap SOFA_CORE_DEPRECATED_RENAME_CREATORMAP_OBJECTTEMPLATECREATORMAP() = ObjectTemplateCreatorMap;

    /// Record storing information about a class
    class ClassEntry
    {
    public:
        using SPtr = std::shared_ptr<ClassEntry>;

        std::string className;
        std::set<std::string> aliases;
        std::string description;
        std::string authors;
        std::string license;
        std::string documentationURL;
        std::string defaultTemplate;
        ObjectTemplateCreatorMap creatorMap; // to create instances of the class for different templates
        std::map<std::string, std::vector<std::string>> m_dataAlias ;
    };

    using ClassName = std::string;

    /// Map to store all class entries, key is the class name.
    using ClassEntryMap = std::map<ClassName, ClassEntry::SPtr>;

protected:

    /// Main registry of all classes
    ClassEntryMap registry;

    OnCreateCallback m_callbackOnCreate ;

    /// Keep track of plugins who already registered
    using RegisteredPluginSet = std::set<std::string>;
    RegisteredPluginSet m_registeredPluginSet;

public:

    ~ObjectFactory();

    /// Get an entry given a class name (or alias)
    ClassEntry& getEntry(std::string classname);

    /// Test if a creator exists for a given classname
    bool hasCreator(std::string classname);

    /// Return the shortname for this classname. Empty string if
    /// no creator exists for this classname.
    std::string shortName(std::string classname);

    /// Fill the given vector with all the registered classes
    void getAllEntries(std::vector<ClassEntry::SPtr>& result, bool filterUnloadedPlugins = true);

    /// Fill the given vector with the registered classes from a given target
    void getEntriesFromTarget(std::vector<ClassEntry::SPtr>& result, std::string target);

    /// Return the list of classes from a given target
    std::string listClassesFromTarget(std::string target, std::string separator = ", ");

    /// Fill the given vector with all the registered classes derived from BaseClass
    template<class BaseClass>
    void getEntriesDerivedFrom(std::vector<ClassEntry::SPtr>& result) const;

    /// Return the list of classes derived from BaseClass as a string
    template<class BaseClass>
    std::string listClassesDerivedFrom(const std::string& separator = ", ") const;

    /// Add an alias name for an already registered class
    ///
    /// \param name     name of the new alias
    /// \param target   class pointed to by the new alias
    /// \param force    set to true if this method should override any entry already registered for this name
    /// \param previous (output) previous ClassEntry registered for this name
    bool addAlias(std::string name, std::string target, bool force=false,
                  ClassEntry::SPtr* previous = nullptr);

    /// Reset an alias to a previous state
    ///
    /// \param name     name of the new alias
    /// \param previous previous ClassEntry that need to be registered back for this name
    void resetAlias(std::string name, ClassEntry::SPtr previous);

    /// Create an object given a context and a description.
    objectmodel::BaseObject::SPtr createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    /// Get the ObjectFactory singleton instance
    static ObjectFactory* getInstance();

    /// \copydoc createObject
    static objectmodel::BaseObject::SPtr CreateObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        return getInstance()->createObject(context, arg);
    }

    /// \copydoc addAlias
    static bool AddAlias(std::string name, std::string result, bool force=false,
                         ClassEntry::SPtr* previous = nullptr)
    {
        return getInstance()->addAlias(name, result, force, previous);
    }

    /// \copydoc resetAlias
    static void ResetAlias(std::string name, ClassEntry::SPtr previous)
    {
        getInstance()->resetAlias(name, previous);
    }

    /// \copydoc hasCreator
    static bool HasCreator(std::string classname)
    {
        return getInstance()->hasCreator(classname);
    }

    static std::string ShortName(std::string classname)
    {
        return getInstance()->shortName(classname);
    }

    /// Dump the content of the factory to a text stream.
    void dump(std::ostream& out = std::cout);

    /// Dump the content of the factory to a XML stream.
    void dumpXML(std::ostream& out = std::cout);

    /// Dump the content of the factory to a HTML stream.
    void dumpHTML(std::ostream& out = std::cout);

    void setCallback(OnCreateCallback cb) { m_callbackOnCreate = cb ; }

    bool registerObjectsFromPlugin(const std::string& pluginName);
    bool registerObjects(ObjectRegistrationData& ro);

};

template<class BaseClass>
void ObjectFactory::getEntriesDerivedFrom(std::vector<ClassEntry::SPtr>& result) const
{
    result.clear();
    for (const auto& [mapKeyClassName, entryInRegistry] : registry)
    {
        // Discard the entry if its class name is not consistent with its key in the map.
        // Differences happen for class aliases.
        if (entryInRegistry->className == mapKeyClassName)
        {
            const auto& templateCreators = entryInRegistry->creatorMap;
            const auto isAnyInstantiationDerived = std::any_of(templateCreators.begin(), templateCreators.end(),
                [](const auto& it)
                {
                    const auto& templateInstantiation = it.second;
                    const auto* instantiationClass = templateInstantiation->getClass();
                    return instantiationClass
                        && instantiationClass->hasParent(BaseClass::GetClass());
                });
            if (isAnyInstantiationDerived) //at least one template instantiation of the class is derived from BaseClass
            {
                result.push_back(entryInRegistry);
            }
        }
    }
}

template<class BaseClass>
std::string ObjectFactory::listClassesDerivedFrom(const std::string& separator) const
{
    std::vector<ClassEntry::SPtr> entries;
    getEntriesDerivedFrom<BaseClass>(entries);

    return sofa::helper::join(entries.begin(), entries.end(),
        [](const ClassEntry::SPtr& entry){ return entry->className;}, separator);
}

/**
 *  \brief Typed Creator class used to create instances of object type RealObject
 */
template<class RealObject>
class ObjectCreator : public ObjectFactory::BaseObjectCreator
{
public:
    bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) override
    {
        RealObject* instance = nullptr;
        return RealObject::canCreate(instance, context, arg);
    }
    objectmodel::BaseObject::SPtr createInstance(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) override
    {
        RealObject* instance = nullptr;
        return RealObject::create(instance, context, arg);
    }
    const std::type_info& type() override
    {
        return typeid(RealObject);
    }
    const objectmodel::BaseClass* getClass() override
    {
        return RealObject::GetClass();
    }
    /// The name of the library or executable containing the binary code for this component
    const char* getTarget() override
    {
#ifdef SOFA_TARGET
        return sofa_tostring(SOFA_TARGET);
#else
        return "";
#endif
    }

    const char* getHeaderFileLocation() override
    {
        return RealObject::HeaderFileLocation();
    }
};

/**
 *  \brief Helper class used to register a class in the ObjectFactory.
 *
 *  This class accumulate information about a given class, as well as creators
 *  for each supported template instantiation, to register a new entry in
 *  the ObjectFactory.
 *
 *  It should be used as a temporary object, finalized when used to initialize
 *  an int static variable. For example :
 *  \code
 *    int Fluid3DClass = core::RegisterObject("Eulerian 3D fluid")
 *    .add\< Fluid3D \>()
 *    .addLicense("LGPL")
 *    ;
 *  \endcode
 *
 */
class SOFA_CORE_API ObjectRegistrationData
{
protected:
    /// Class entry being constructed
    ObjectFactory::ClassEntry entry;

public:

    /// Start the registration by giving the description of this class.
    explicit ObjectRegistrationData(const std::string& description);

    /// Add an alias name for this class
    ObjectRegistrationData& addAlias(std::string val);

    /// Add more descriptive text about this class
    ObjectRegistrationData& addDescription(std::string val);

    /// Specify a list of authors (separated with spaces)
    ObjectRegistrationData& addAuthor(std::string val);

    /// Specify a license (LGPL, GPL, ...)
    ObjectRegistrationData& addLicense(std::string val);

    /// Specify a documentation URL
    ObjectRegistrationData& addDocumentationURL(std::string url);

    /// Add a creator able to instance this class with the given templatename.
    ///
    /// See the add<RealObject>() method for an easy way to add a Creator.
    ObjectRegistrationData& addCreator(std::string classname, std::string templatename,
                               ObjectFactory::BaseObjectCreator::SPtr creator);

    /// Add a template instantiation of this class.
    ///
    /// \param defaultTemplate    set to true if this should be the default instance when no template name is given.
    template<class RealObject>
    ObjectRegistrationData& add(bool defaultTemplate=false)
    {
        const std::string classname = sofa::core::objectmodel::BaseClassNameHelper::getClassName<RealObject>();
        const std::string templatename = sofa::core::objectmodel::BaseClassNameHelper::getTemplateName<RealObject>();

        if (defaultTemplate)
            entry.defaultTemplate = templatename;

        if (entry.documentationURL.empty())
        {
            const std::string target = sofa_tostring(SOFA_TARGET);
            const auto modulePaths = sofa::helper::split(target, '.');
            if (modulePaths.size() > 2 && modulePaths[0] == "Sofa" && modulePaths[1] == "Component")
            {
                entry.documentationURL = std::string(sofa::SOFA_DOCUMENTATION_URL) + std::string("components/");
                entry.documentationURL += sofa::helper::join(modulePaths.begin() + 2, modulePaths.end(),
                    [](const std::string& m){ return sofa::helper::downcaseString(m);}, "/");
                entry.documentationURL += std::string("/") + sofa::helper::downcaseString(classname);
            }
        }

        auto objectCreator = std::make_shared<ObjectCreator<RealObject> >();
        if (strcmp(objectCreator->getTarget(), "") == 0)
        {
            dmsg_warning("ObjectFactory") << "Module name cannot be found when registering "
                << RealObject::GetClass()->className << "<" << RealObject::GetClass()->templateName << "> into the object factory";
        }
        return addCreator(classname, templatename, objectCreator);
    }

    /// This is the final operation that will actually commit the additions to the ObjectFactory.
    bool commitTo(sofa::core::ObjectFactory* objectFactory) const;

    friend class RegisterObject;
};


// Legacy structure, to keep compatibility with olden code
// using the singleton to get the instance of ObjectFactory
class SOFA_ATTRIBUTE_DEPRECATED__REGISTEROBJECT() SOFA_CORE_API RegisterObject
{
private:
    ObjectRegistrationData m_objectRegistrationdata;

public:
    explicit RegisterObject(const std::string& description);

    RegisterObject& addAlias(std::string val);
    RegisterObject& addDescription(std::string val);
    RegisterObject& addAuthor(std::string val);
    RegisterObject& addLicense(std::string val);
    RegisterObject& addDocumentationURL(std::string url);
    RegisterObject& addCreator(std::string classname, std::string templatename,
        ObjectFactory::BaseObjectCreator::SPtr creator);

    template<class RealObject>
    RegisterObject& add(bool defaultTemplate = false)
    {
        m_objectRegistrationdata.add<RealObject>(defaultTemplate);
        return *this;
    }

    operator int() const;

    int commitTo(ObjectFactory* factory) const;
};

} // namespace sofa::core
