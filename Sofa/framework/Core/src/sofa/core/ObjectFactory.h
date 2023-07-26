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

#pragma once
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectfactory/ObjectFactory.h>
#include <sofa/core/objectfactory/ObjectFactoryInstance.h>

namespace sofa::core
{

// inject in the current namespace the needed objects so they are accessible in the current version.
using OnCreateCallback = sofa::core::objectfactory::OnCreateCallback;
using ObjectFactoryInstance = sofa::core::objectfactory::ObjectFactoryInstance;
using RegisterObject = sofa::core::objectfactory::RegisterObject;

using sofa::core::objectfactory::ClassEntrySPtr;
using sofa::core::objectfactory::BaseObjectCreatorSPtr;

class SOFA_CORE_API ObjectFactory
{
public:
    using ClassEntry = sofa::core::objectfactory::ClassEntry;
    using ClassEntryMap = sofa::core::objectfactory::ClassEntryMap;

    using Creator = sofa::core::objectfactory::BaseObjectCreator;
    using CreatorMap = sofa::core::objectfactory::BaseObjectCreatorMap;

    ObjectFactory(sofa::core::objectfactory::ObjectFactory* newfactory_){ newfactory=newfactory_; }

    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("short name has been removed. Use sofa::helper::NameDecoder::sortName instead.")
    std::string shortName(const std::string& classname);

    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace ObjectFactory::getInstance()->function() by the equivalent ObjectFactoryInstance::function()")
    static sofa::core::ObjectFactory* getInstance();

    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace 'ObjectFactory::CreateObject' by 'ObjectFactoryInstance::createObject'")
    static objectmodel::BaseObject::SPtr CreateObject(objectmodel::BaseContext* context,
                                                      objectmodel::BaseObjectDescription* arg);

    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace 'ObjectFactory::AddAlias' by 'ObjectFactoryInstance::addAlias'")
    static bool AddAlias(const std::string& name, const std::string& result, bool force=false,
                         ClassEntrySPtr* previous = nullptr);

    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace 'ObjectFactory::ResetAlias' by 'ObjectFactoryInstance::resetAlias'")
    static void ResetAlias(const std::string& name, ClassEntrySPtr previous);

    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace 'ObjectFactory::HasCreator' by 'ObjectFactoryInstance::hasCreator'")
    static bool HasCreator(const std::string& classname);

    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace 'ObjectFactory::ShortName' by 'sofa::helper::NameDecoder::shortName(classname)'")
    static std::string ShortName(const std::string& classname);

    /// Get an entry given a class name (or alias)
    ClassEntry& getEntry(const std::string& classname);

    /// Test if a creator exists for a given classname
    bool hasCreator(const std::string& classname);

    /// Fill the given vector with all the registered classes
    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace 'ObjectFactory::getAllEntries(results)' by 'ObjectFactory::getEntriesFromTarget(results, target=\"*\")")
    void getAllEntries(std::vector<ClassEntrySPtr>& result);

    /// Fill the given vector with the registered classes from a given target
    void getEntriesFromTarget(std::vector<ClassEntrySPtr>& result, const std::string& target="*");

    /// Return the list of classes from a given target
    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("'ObjectFactory::listClassesFromTarget(result)' has been deleted as equivalent behavior can be implemented using sofa::helper::join(ObjectFactory::getEntriesFromTarget());'")
    std::string listClassesFromTarget(const std::string& target, const std::string& separator = ", ");

    /// Fill the given vector with all the registered classes derived from BaseClass
    template<class BaseClass>
    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("Replace 'ObjectFactory::getEntriesDerivedFrom<Class>(result)' by 'ObjectFactory::getEntriesDerivedFrom(Class::GetClass(), result);'")
    void getEntriesDerivedFrom(std::vector<ClassEntrySPtr>& result) const;                                     //< old API (before 23.12)
    void getEntriesDerivedFrom(sofa::core::BaseClass* parentclass, std::vector<ClassEntrySPtr>& result) const; //< new API (post 23.12)

    /// Return the list of classes derived from BaseClass as a string
    template<class BaseClass>
    SOFA_ATTRIBUTE_DEPRECATED__OBJECTFACTORY("'ObjectFactory::listClassesDerivedFrom<Class>(result)' has been deleted as equivalent behavior can be implemented using sofa::helper::join(ObjectFactory::getEntriesDerivedFrom());'")
    std::string listClassesDerivedFrom(const std::string& separator = ", ") const; //< old API (pre 23.12)

    /// Add an alias name for an already registered class
    ///
    /// \param name     name of the new alias
    /// \param target   class pointed to by the new alias
    /// \param force    set to true if this method should override any entry already registered for this name
    /// \param previous (output) previous ClassEntry registered for this name
    bool addAlias(const std::string& name, const std::string& target, bool force=false,
                  ClassEntrySPtr* previous = nullptr);

    /// Reset an alias to a previous state
    ///
    /// \param name     name of the new alias
    /// \param previous previous ClassEntry that need to be registered back for this name
    void resetAlias(const std::string& name, ClassEntrySPtr previous);

    /// Create an object given a context and a description.
    objectmodel::BaseObject::SPtr createObject(objectmodel::BaseContext* context,
                                               objectmodel::BaseObjectDescription* arg);

    /// Dump the content of the factory to a text stream.
    void dump(std::ostream& out = std::cout);

    /// Dump the content of the factory to a XML stream.
    void dumpXML(std::ostream& out = std::cout);

    /// Dump the content of the factory to a HTML stream.
    void dumpHTML(std::ostream& out = std::cout);

    void setCallback(OnCreateCallback cb);

private:
    sofa::core::objectfactory::ObjectFactory* newfactory;

    std::string listClassesDerivedFrom(const sofa::core::BaseClass* parentclass,
                                       const std::string& separator) const;
};

template<class BaseClass>
void ObjectFactory::getEntriesDerivedFrom(std::vector<ClassEntrySPtr>& result) const
{
    return getEntriesDerivedFrom(BaseClass::GetClass(), result);
}

template<class BaseClass>
std::string ObjectFactory::listClassesDerivedFrom(const std::string& separator) const
{
    return listClassesDerivedFrom(BaseClass::GetClass(), separator);
}

} // namespace sofa::core
