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
class ObjectFactory;
class ClassEntry;
typedef std::shared_ptr<ClassEntry> ClassEntrySPtr;

class BaseObjectCreator;
typedef std::shared_ptr<BaseObjectCreator> BaseObjectCreatorSPtr;


typedef std::function<void(sofa::core::objectmodel::Base*,
                           sofa::core::objectmodel::BaseObjectDescription*)> OnCreateCallback ;

class SOFA_CORE_API ObjectFactoryInstance
{
public:
    /// Get the ObjectFactory singleton instance
    static ObjectFactory* getInstance();

    /// \copydoc createObject
    static objectmodel::BaseObject::SPtr createObject(objectmodel::BaseContext* context,
                                                      objectmodel::BaseObjectDescription* arg);

    /// \copydoc addAlias
    static bool addAlias(const std::string& name, const std::string& result, const bool force=false,
                         ClassEntrySPtr* previous = nullptr);

    /// \copydoc resetAlias
    static void resetAlias(const std::string& name,
                           ClassEntrySPtr previous);

    /// \copydoc hasCreator
    static bool hasCreator(const std::string& classname);

    static std::string shortName(const std::string& classname);

    static void dump(std::ostream& out);

    static void dumpXML(std::ostream& out);

    static void dumpHTML(std::ostream& out);

    static void setCallback(OnCreateCallback cb);
};

/**
 *  \brief Typed Creator class used to create instances of object type RealObject
 */
template<class RealObject>
class SOFA_CORE_API ObjectCreator : public objectfactory::BaseObjectCreator
{
public:
    bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) override
    {
        RealObject* instance = nullptr;
        return RealObject::canCreate(instance, context, arg);
    }
    objectmodel::BaseObject::SPtr createInstance(objectmodel::BaseContext* context,
                                                 objectmodel::BaseObjectDescription* arg) override
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

class SOFA_CORE_API RegisterObject
{
protected:
    /// Class entry being constructed
    ClassEntrySPtr entry;

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

    RegisterObject& setDefaultTemplateName(const std::string& name);

    /// Add a creator able to instance this class with the given templatename.
    ///
    /// See the add<RealObject>() method for an easy way to add a Creator.
    RegisterObject& addCreator(std::string classname,
                               std::string templatename,
                               BaseObjectCreatorSPtr creator);

    /// Add a template instanciation of this class.
    ///
    /// \param defaultTemplate    set to true if this should be the default instance when no template name is given.
    template<class RealObject>
    RegisterObject& add(bool defaultTemplate=false)
    {
        std::string classname = sofa::core::objectmodel::BaseClassNameHelper::getClassName<RealObject>();
        std::string templatename = sofa::core::objectmodel::BaseClassNameHelper::getTemplateName<RealObject>();

        auto& entry = addCreator(classname, templatename, BaseObjectCreatorSPtr(new ObjectCreator<RealObject>));
        if(defaultTemplate)
            entry.setDefaultTemplateName(templatename);
        return entry;
    }

    /// This is the final operation that will actually commit the additions to the ObjectFactory.
    operator int();
};

} // namespace sofa::core::objectfactory
