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
#include <sofa/core/objectfactory/ObjectFactoryInstance.h>
#include <sofa/core/objectfactory/ObjectFactory.h>

#include <sofa/defaulttype/TemplatesAliases.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/ComponentChange.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/DiffLib.h>

namespace sofa::core::objectfactory
{

auto ObjectFactoryInstance::getInstance() -> sofa::core::objectfactory::ObjectFactory*
{
    static ObjectFactory instance;
    return &instance;
}

sofa::core::objectmodel::BaseObject::SPtr ObjectFactoryInstance::createObject(objectmodel::BaseContext* context,
                                                                              objectmodel::BaseObjectDescription* arg)
{
    return getInstance()->createObject(context, arg);
}

bool ObjectFactoryInstance::addAlias(const std::string& name, const std::string& result, bool force,
                                     ClassEntrySPtr* previous)
{
    return getInstance()->addAlias(name, result, force, previous);
}

void ObjectFactoryInstance::resetAlias(const std::string& name,
                                       ClassEntrySPtr previous)
{
    return getInstance()->resetAlias(name, previous);
}

bool ObjectFactoryInstance::hasCreator(const std::string& classname)
{
    return getInstance()->hasCreator(classname);
}

void ObjectFactoryInstance::dump(std::ostream& out)
{
    return getInstance()->dump(out);
}

void ObjectFactoryInstance::dumpXML(std::ostream& out)
{
    return getInstance()->dumpXML(out);
}

void ObjectFactoryInstance::dumpHTML(std::ostream& out)
{
    return getInstance()->dumpHTML(out);
}

void ObjectFactoryInstance::setCallback(OnCreateCallback cb)
{
    return getInstance()->setCallback(cb);
}

RegisterObject::RegisterObject(const std::string& description)
{
    entry = std::make_shared<ClassEntry>();
    if (!description.empty())
    {
        addDescription(description);
    }
}

auto RegisterObject::addAlias(std::string val) -> RegisterObject&
{
    entry->aliases.insert(val);
    return *this;
}

auto RegisterObject::addDescription(std::string val) -> RegisterObject&
{
    val += '\n';
    entry->description += val;
    return *this;
}

auto RegisterObject::addAuthor(std::string val) -> RegisterObject&
{
    val += ' ';
    entry->authors += val;
    return *this;
}

auto RegisterObject::addLicense(std::string val) -> RegisterObject&
{
    entry->license += val;
    return *this;
}

auto RegisterObject::addCreator(std::string classname,
                                           std::string templatename,
                                           BaseObjectCreatorSPtr creator) -> RegisterObject&
{

    if (!entry->className.empty() && entry->className != classname)
    {
        msg_error("ObjectFactory") << "Template already instanciated with a different classname: " << entry->className << " != " << classname;
    }
    else if (entry->creatorMap.find(templatename) != entry->creatorMap.end())
    {
        msg_error("ObjectFactory") << "Component already registered: " << classname << "<" << templatename << ">";
    }
    else
    {
        entry->className = classname;
        entry->creatorMap[templatename] =  creator;
    }
    return *this;
}

auto RegisterObject::setDefaultTemplateName(const std::string& templateName) -> RegisterObject&
{
    entry->defaultTemplate = templateName;
    return *this;
}

RegisterObject::operator int()
{
    if (entry->className.empty())
    {
        return 0;
    }
    else
    {
        ClassEntry& reg = ObjectFactoryInstance::getInstance()->getEntry(entry->className);
        reg.description += entry->description;
        reg.authors += entry->authors;
        reg.license += entry->license;
        if (!entry->defaultTemplate.empty())
        {
            if (!reg.defaultTemplate.empty())
            {
                msg_warning("ObjectFactory") << "Default template for class " << entry->className << " already registered (" << reg.defaultTemplate << "), do not register " << entry->defaultTemplate << " as the default";
            }
            else
            {
                reg.defaultTemplate = entry->defaultTemplate;
            }
        }
        for (auto & creator_entry : entry->creatorMap)
        {
            const std::string & template_name = creator_entry.first;
            if (reg.creatorMap.find(template_name) != reg.creatorMap.end()) {
                if (template_name.empty()) {
                    msg_warning("ObjectFactory") << "Class already registered: " << entry->className;
                } else {
                    msg_warning("ObjectFactory") << "Class already registered: " << entry->className << "<" << template_name << ">";
                }
            } else {
                reg.creatorMap.insert(creator_entry);
            }
        }

        for (const auto & alias : entry->aliases)
        {
            if (reg.aliases.find(alias) == reg.aliases.end())
            {
                ObjectFactoryInstance::getInstance()->addAlias(alias,entry->className);
            }
        }
        return 1;
    }
}

} // namespace sofa::core
