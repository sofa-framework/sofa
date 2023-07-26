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
#include <sofa/core/objectfactory/BaseObjectCreator.h>
#include <numeric>

namespace sofa::core::objectfactory
{

/// Record storing information about a class
class ClassEntry
{
public:
    std::string className;
    std::set<std::string> aliases;
    std::string description;
    std::string authors;
    std::string license;
    std::string defaultTemplate;
    BaseObjectCreatorMap creatorMap;
    std::map<std::string, std::vector<std::string>> m_dataAlias ;

    typedef std::shared_ptr<ClassEntry> SPtr;
};
typedef std::shared_ptr<ClassEntry> ClassEntrySPtr;
typedef std::map<std::string, ClassEntrySPtr> ClassEntryMap;

typedef std::function<void(sofa::core::objectmodel::Base*,
                           sofa::core::objectmodel::BaseObjectDescription*)> OnCreateCallback ;

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

    /// Fill the given vector with the registered classes from a given target
    std::vector<ClassEntrySPtr>& getEntries(std::vector<ClassEntrySPtr>& result, const std::string& target="*");

    std::vector<ClassEntrySPtr>& getEntriesDerivedFrom(std::vector<ClassEntrySPtr>& result,
                                                       const sofa::core::BaseClass* parentclass) const;

    /// Add an alias name for an already registered class
    ///
    /// \param name     name of the new alias
    /// \param target   class pointed to by the new alias
    /// \param force    set to true if this method should override any entry already registered for this name
    /// \param previous (output) previous ClassEntry registered for this name
    bool addAlias(std::string name, std::string target, bool force=false,
                  ClassEntrySPtr* previous = nullptr);

    /// Reset an alias to a previous state
    ///
    /// \param name     name of the new alias
    /// \param previous previous ClassEntry that need to be registered back for this name
    void resetAlias(std::string name, ClassEntrySPtr previous);

    /// Create an object given a context and a description.
    objectmodel::BaseObject::SPtr createObject(objectmodel::BaseContext* context,
                                               objectmodel::BaseObjectDescription* arg);

    /// Dump the content of the factory to a text stream.
    void dump(std::ostream& out = std::cout);

    /// Dump the content of the factory to a XML stream.
    void dumpXML(std::ostream& out = std::cout);

    /// Dump the content of the factory to a HTML stream.
    void dumpHTML(std::ostream& out = std::cout);

    void setCallback(OnCreateCallback cb) { m_callbackOnCreate = cb ; }
};

} // namespace sofa::core
