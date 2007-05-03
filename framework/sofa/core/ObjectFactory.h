/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_OBJECTFACTORY_H
#define SOFA_CORE_OBJECTFACTORY_H

#include <sofa/helper/system/config.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/core/componentmodel/behavior/BaseForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/BaseConstraint.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/BaseMass.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/MasterSolver.h>
#include <sofa/core/componentmodel/topology/Topology.h>

#include <map>
#include <iostream>
#include <typeinfo>

namespace sofa
{

namespace core
{

class ObjectFactory
{
public:

    class Creator
    {
    public:
        virtual ~Creator() { }
        virtual bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) = 0;
        virtual objectmodel::BaseObject* createInstance(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) = 0;
        virtual const std::type_info& type() = 0;
    };

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
        std::list< std::pair<std::string, Creator*> > creatorList;
        std::map<std::string, Creator*> creatorMap;
        //void print();
    };

protected:
    std::map<std::string,ClassEntry*> registry;

public:

    ClassEntry* getEntry(std::string classname);
    void getAllEntries(std::vector<ClassEntry*>& result);

    bool addAlias(std::string name, std::string result, bool force=false);

    objectmodel::BaseObject* createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    static ObjectFactory* getInstance();

    static objectmodel::BaseObject* CreateObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        return getInstance()->createObject(context, arg);
    }

    static bool AddAlias(std::string name, std::string result, bool force=false)
    {
        return getInstance()->addAlias(name, result, force);
    }

    void dump(std::ostream& out = std::cout);
    void dumpXML(std::ostream& out = std::cout);
    void dumpHTML(std::ostream& out = std::cout);
};

template<class RealObject>
class ObjectCreator : public ObjectFactory::Creator
{
public:
    bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        RealObject* instance = NULL;
        return RealObject::canCreate(instance, context, arg);
    }
    objectmodel::BaseObject *createInstance(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        RealObject* instance = NULL;
        RealObject::create(instance, context, arg);
        return instance;
    }
    const std::type_info& type()
    {
        return typeid(RealObject);
    }
};

class RegisterObject
{
protected:
    ObjectFactory::ClassEntry entry;
public:

    RegisterObject(const std::string& description);

    RegisterObject& addAlias(std::string val);

    RegisterObject& addDescription(std::string val);

    RegisterObject& addAuthor(std::string val);

    RegisterObject& addLicense(std::string val);

    RegisterObject& addCreator(std::string classname, std::string templatename, ObjectFactory::Creator* creator);

    // test whether T* converts to U*,
    // that is, if T is derived from U
    // taken from Modern C++ Design
    template <class T, class U>
    class Conversion
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

    template<class RealClass, class BaseClass>
    bool implements()
    {
        bool res = Conversion<RealClass, BaseClass>::exists;
        //RealClass* p1=NULL;
        //BaseClass* p2=NULL;
        //if (res)
        //    std::cout << "class "<<RealClass::typeName(p1)<<" implements "<<BaseClass::typeName(p2)<<std::endl;
        //else
        //    std::cout << "class "<<RealClass::typeName(p1)<<" does not implement "<<BaseClass::typeName(p2)<<std::endl;
        return res;
    }

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

        if (implements<RealObject,objectmodel::ContextObject>())
            entry.baseClasses.insert("ContextObject");
        if (implements<RealObject,VisualModel>())
            entry.baseClasses.insert("VisualModel");
        if (implements<RealObject,BehaviorModel>())
            entry.baseClasses.insert("BehaviorModel");
        if (implements<RealObject,CollisionModel>())
            entry.baseClasses.insert("CollisionModel");
        if (implements<RealObject,core::componentmodel::behavior::BaseMechanicalState>())
            entry.baseClasses.insert("MechanicalState");
        if (implements<RealObject,core::componentmodel::behavior::BaseForceField>())
            entry.baseClasses.insert("ForceField");
        if (implements<RealObject,core::componentmodel::behavior::InteractionForceField>())
            entry.baseClasses.insert("InteractionForceField");
        if (implements<RealObject,core::componentmodel::behavior::BaseConstraint>())
            entry.baseClasses.insert("Constraint");
        if (implements<RealObject,core::BaseMapping>())
            entry.baseClasses.insert("Mapping");
        if (implements<RealObject,core::componentmodel::behavior::BaseMechanicalMapping>())
            entry.baseClasses.insert("MechanicalMapping");
        if (implements<RealObject,core::componentmodel::behavior::BaseMass>())
            entry.baseClasses.insert("Mass");
        if (implements<RealObject,core::componentmodel::behavior::OdeSolver>())
            entry.baseClasses.insert("OdeSolver");
        if (implements<RealObject,core::componentmodel::behavior::MasterSolver>())
            entry.baseClasses.insert("MasterSolver");
        if (implements<RealObject,core::componentmodel::topology::Topology>())
            entry.baseClasses.insert("Topology");

        return addCreator(classname, templatename, new ObjectCreator<RealObject>);
    }

    /// Convert to an int
    /// This is the final operation that will actually commit the additions to the factory
    operator int();
};

} // namespace core

} // namespace sofa

#endif
