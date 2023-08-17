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

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseClassNameHelper.h>
#include <numeric>

namespace sofa::core
{

/**
 *  \brief Main class used to register and dynamically create objects
 *
 *  It uses the Factory design pattern, where each class is registered in a map,
 *  and dynamically retrieved given the type name.
 *
 *  It also stores metainformation on each classes, such as description,
 *  authors, license, and available template types.
 *
 *  \see RegisterObject for how new classes should be registered.
 *
 */
typedef std::function<void(sofa::core::objectmodel::Base*, sofa::core::objectmodel::BaseObjectDescription*)> OnCreateCallback ;
class SOFA_CORE_API ObjectFactory
{
public:

    /// Abstract interface of objects used to create instances of a given type
    class Creator
    {
    public:
        typedef std::shared_ptr<Creator> SPtr;

        virtual ~Creator() { }
        /// Pre-construction check.
        ///
        /// \return true if the object can be created successfully.
        virtual bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) = 0;

        /// Construction method called by the factory.
        ///
        /// \pre canCreate(context, arg) == true.
        virtual objectmodel::BaseObject::SPtr createInstance(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) = 0;

        /// type_info structure associated with the type of intanciated objects.
        virtual const std::type_info& type() = 0;

        /// BaseClass structure associated with the type of intanciated objects.
        virtual const objectmodel::BaseClass* getClass() = 0;

        /// The name of the library or executable containing the binary code for this component
        virtual const char* getTarget() = 0;

        virtual const char* getHeaderFileLocation() = 0;
    };
    typedef std::map<std::string, Creator::SPtr> CreatorMap;

    /// Record storing information about a class
    class ClassEntry
    {
    public:
        typedef std::shared_ptr<ClassEntry> SPtr;

        std::string className;
        std::set<std::string> aliases;
        std::string description;
        std::string authors;
        std::string license;
        std::string defaultTemplate;
        CreatorMap creatorMap;
        std::map<std::string, std::vector<std::string>> m_dataAlias ;
    };
    typedef std::map<std::string, ClassEntry::SPtr> ClassEntryMap;

protected:
    /// Main class registry
    ClassEntryMap registry;
    OnCreateCallback m_callbackOnCreate ;

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
    void getAllEntries(std::vector<ClassEntry::SPtr>& result);

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
};

template<class BaseClass>
void ObjectFactory::getEntriesDerivedFrom(std::vector<ClassEntry::SPtr>& result) const
{
    result.clear();
    for (const auto& r : registry)
    {
        ClassEntry::SPtr entry = r.second;
        // Push the entry only if it is not an alias
        if (entry->className == r.first)
        {
            const auto creatorEntry = entry->creatorMap.begin();
            if (creatorEntry != entry->creatorMap.end())
            {
                const auto* baseClass = creatorEntry->second->getClass();
                if (baseClass && baseClass->hasParent(BaseClass::GetClass()))
                {
                    result.push_back(entry);
                }
            }
        }
    }
}

template<class BaseClass>
std::string ObjectFactory::listClassesDerivedFrom(const std::string& separator) const
{
    std::vector<ClassEntry::SPtr> entries;
    getEntriesDerivedFrom<BaseClass>(entries);
    if (entries.empty()) return std::string();

    const auto join = [&separator](std::string a, ClassEntry::SPtr b)
    {
        return std::move(a) + separator + b->className;
    };
    return std::accumulate(std::next(entries.begin()), entries.end(),
                           entries.front()->className, join);
}

/**
 *  \brief Typed Creator class used to create instances of object type RealObject
 */
template<class RealObject>
class ObjectCreator : public ObjectFactory::Creator
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
 *  for each supported template instanciation, to register a new entry in
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
class SOFA_CORE_API RegisterObject
{
protected:
    /// Class entry being constructed
    ObjectFactory::ClassEntry entry;
public:

    /// Start the registration by giving the description of this class.
    RegisterObject(const std::string& description);

    /// Add an alias name for this class
    RegisterObject& addAlias(std::string val);

    /// Add more descriptive text about this class
    RegisterObject& addDescription(std::string val);

    /// Specify a list of authors (separated with spaces)
    RegisterObject& addAuthor(std::string val);

    /// Specify a license (LGPL, GPL, ...)
    RegisterObject& addLicense(std::string val);

    /// Add a creator able to instance this class with the given templatename.
    ///
    /// See the add<RealObject>() method for an easy way to add a Creator.
    RegisterObject& addCreator(std::string classname, std::string templatename,
                               ObjectFactory::Creator::SPtr creator);

    /// Add a template instanciation of this class.
    ///
    /// \param defaultTemplate    set to true if this should be the default instance when no template name is given.
    template<class RealObject>
    RegisterObject& add(bool defaultTemplate=false)
    {
        const std::string classname = sofa::core::objectmodel::BaseClassNameHelper::getClassName<RealObject>();
        const std::string templatename = sofa::core::objectmodel::BaseClassNameHelper::getTemplateName<RealObject>();

        if (defaultTemplate)
            entry.defaultTemplate = templatename;

        return addCreator(classname, templatename, ObjectFactory::Creator::SPtr(new ObjectCreator<RealObject>));
    }

    /// This is the final operation that will actually commit the additions to the ObjectFactory.
    operator int();
};
} // namespace sofa::core
