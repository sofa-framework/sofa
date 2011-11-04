/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_OBJECTFACTORY_H
#define SOFA_CORE_OBJECTFACTORY_H

#include <sofa/helper/system/config.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObject.h>
/*
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/InteractionForceField.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/core/behavior/BaseLMConstraint.h>
#include <sofa/core/behavior/BaseMechanicalMapping.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseTopologyObject.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/core/loader/BaseLoader.h>
*/

#include <map>
#include <memory>
#include <iostream>
#include <typeinfo>

namespace sofa
{

namespace core
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
class SOFA_CORE_API ObjectFactory
{
public:
    class Creator;
    class ClassEntry;

    typedef std::map<std::string, Creator*>                 CreatorMap;
    typedef std::list< std::pair<std::string, Creator*> > CreatorList;
    typedef std::map<std::string, ClassEntry*>              ClassEntryMap;
    typedef std::vector<ClassEntry*>                        ClassEntryList;

    /// Abstract interface of objects used to create instances of a given type
    class Creator
    {
    public:
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

        virtual std::string shortName(objectmodel::BaseObjectDescription* arg) = 0;

        /// The name of the library or executable containing the binary code for this component
        virtual const char* getTarget() = 0;
    };

    /// Record storing information about a class
    class ClassEntry
    {
    public:
        std::string className;
        std::set<std::string> baseClasses;
        std::set<std::string> aliases;
        std::string description;
        std::string authors;
        std::string license;
        std::string defaultTemplate;
        CreatorList creatorList;
        CreatorMap creatorMap;
        //void print();

        ~ClassEntry()
        {
            for(CreatorMap::iterator it = creatorMap.begin(), itEnd = creatorMap.end();
                it != itEnd; ++it)
            {
                delete it->second;
                it->second = 0;
            }
        }
    };

protected:

    /// Main class registry
    ClassEntryMap registry;
    ClassEntryList classEntries;

public:

    ~ObjectFactory();

    /// Get an entry given a class name (or alias)
    ClassEntry* getEntry(std::string classname);

    /// Test if a creator exists for a given classname
    bool hasCreator(std::string classname);

    /// Return the shortname for this classname. Empty string if
    /// no creator exists for this classname.
    std::string shortName(std::string classname);

    /// Fill the given vector with all the registered classes
    void getAllEntries(std::vector<ClassEntry*>& result);

    /// Add an alias name for an already registered class
    ///
    /// \param name     name of the new alias
    /// \param result   class pointed to by the new alias
    /// \param force    set to true if this method should override any entry already registered for this name
    /// \param previous (output) previous ClassEntry registered for this name
    bool addAlias(std::string name, std::string result, bool force=false, ClassEntry** previous = NULL);

    /// Reset an alias to a previous state
    ///
    /// \param name     name of the new alias
    /// \param previous previous ClassEntry that need to be registered back for this name
    void resetAlias(std::string name, ClassEntry* previous);

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
    static bool AddAlias(std::string name, std::string result, bool force=false, ClassEntry** previous = NULL)
    {
        return getInstance()->addAlias(name, result, force, previous);
    }

    /// \copydoc resetAlias
    static void ResetAlias(std::string name, ClassEntry* previous)
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
};

/**
 *  \brief Typed Creator class used to create instances of object type RealObject
 */
template<class RealObject>
class ObjectCreator : public ObjectFactory::Creator
{
public:
    bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        RealObject* instance = NULL;
        return RealObject::canCreate(instance, context, arg);
    }
    objectmodel::BaseObject::SPtr createInstance(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        RealObject* instance = NULL;
        return RealObject::create(instance, context, arg);
    }
    const std::type_info& type()
    {
        return typeid(RealObject);
    }
    virtual const objectmodel::BaseClass* getClass()
    {
        return RealObject::GetClass();
    }
    /// The name of the library or executable containing the binary code for this component
    virtual const char* getTarget()
    {
#ifdef SOFA_TARGET
        return sofa_tostring(SOFA_TARGET);
#else
        return "";
#endif
    }

    virtual std::string shortName(objectmodel::BaseObjectDescription* arg)
    {
        RealObject* instance = NULL;
        return RealObject::shortName(instance,arg);
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

    /// Fill the base classes array using the BaseClass reflection system
    RegisterObject& addBaseClasses(const core::objectmodel::BaseClass* mclass);

    /// Add a creator able to instance this class with the given templatename.
    ///
    /// See the add<RealObject>() method for an easy way to add a Creator.
    RegisterObject& addCreator(std::string classname, std::string templatename, std::auto_ptr<ObjectFactory::Creator> creator);
    /*
        /// Test whether T* converts to U*,
        /// that is, if T is derived from U
        /// taken from Modern C++ Design
        template <class T, class U>
        class SOFA_CORE_API Conversion
        {
            typedef char Small;
            class Big {char dummy[2];};
            static Small Test(U*);
            static Big Test(...);
            static T* MakeT();
        public:
            enum { exists = sizeof(Test(MakeT())) == sizeof(Small) };
            static int Exists() { return exists; }
        };

        /// Test whether T* converts to U*,
        /// that is, if T is derived from U
        template<class RealClass, class BaseClass>
        bool implements()
        {
            bool res = Conversion<RealClass, BaseClass>::exists;
            //RealClass* p1=NULL;
            //BaseClass* p2=NULL;
            //if (res)
            //    sout << "class "<<RealClass::typeName(p1)<<" implements "<<BaseClass::typeName(p2)<<sendl;
            //else
            //    sout << "class "<<RealClass::typeName(p1)<<" does not implement "<<BaseClass::typeName(p2)<<sendl;
            return res;
        }
    */
    /// Add a template instanciation of this class.
    ///
    /// \param defaultTemplate    set to true if this should be the default instance when no template name is given.
    template<class RealObject>
    RegisterObject& add(bool defaultTemplate=false)
    {
        RealObject* p = NULL;
        std::string classname = RealObject::className(p);
        std::string templatename = RealObject::templateName(p);

        if (defaultTemplate)
            entry.defaultTemplate = templatename;

        // This is the only place where we can test which base classes are implemented by this particular object, without having to create any instance
        // Unfortunately, we have to enumerate all classes we are interested in...
        /*
                if (implements<RealObject,objectmodel::ContextObject>())
                    entry.baseClasses.insert("ContextObject");
                if (implements<RealObject,VisualModel>())
                    entry.baseClasses.insert("VisualModel");
                if (implements<RealObject,BehaviorModel>())
                    entry.baseClasses.insert("BehaviorModel");
                if (implements<RealObject,CollisionModel>())
                    entry.baseClasses.insert("CollisionModel");
                if (implements<RealObject,core::behavior::BaseMechanicalState>())
                    entry.baseClasses.insert("MechanicalState");
                if (implements<RealObject,core::behavior::BaseForceField>())
                    entry.baseClasses.insert("ForceField");
                if (implements<RealObject,core::behavior::InteractionForceField>())
                    entry.baseClasses.insert("InteractionForceField");
                if (implements<RealObject,core::behavior::BaseLMConstraint>())
                    entry.baseClasses.insert("Constraint");
                if (implements<RealObject,core::behavior::BaseConstraint>())
                    entry.baseClasses.insert("Constraint");
                if (implements<RealObject,core::BaseMapping>())
                    entry.baseClasses.insert("Mapping");
                if (implements<RealObject,core::behavior::BaseMechanicalMapping>())
        	  entry.baseClasses.insert("MechanicalMapping");
        	if (implements<RealObject,core::topology::TopologicalMapping>())
        	  entry.baseClasses.insert("TopologicalMapping");
                if (implements<RealObject,core::behavior::BaseMass>())
                    entry.baseClasses.insert("Mass");
                if (implements<RealObject,core::behavior::OdeSolver>())
        	  entry.baseClasses.insert("OdeSolver");
        	if (implements<RealObject,core::behavior::LinearSolver>())
        	  entry.baseClasses.insert("LinearSolver");
                if (implements<RealObject,core::behavior::BaseAnimationLoop>())
                    entry.baseClasses.insert("BaseAnimationLoop");
                if (implements<RealObject,core::topology::Topology>())
        	  entry.baseClasses.insert("Topology");
                if (implements<RealObject,core::topology::BaseTopologyObject>())
        	  entry.baseClasses.insert("TopologyObject");
        	if (implements<RealObject,core::behavior::BaseController>())
        	  entry.baseClasses.insert("Controller");
        	if (implements<RealObject,core::loader::BaseLoader>())
        	  entry.baseClasses.insert("Loader");
        */
        addBaseClasses(RealObject::GetClass());

        return addCreator(classname, templatename, std::auto_ptr<ObjectFactory::Creator>(new ObjectCreator<RealObject>));
    }

    /// This is the final operation that will actually commit the additions to the ObjectFactory.
    operator int();
};

} // namespace core

} // namespace sofa

#endif
