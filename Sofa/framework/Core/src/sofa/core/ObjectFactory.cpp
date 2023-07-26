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
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/StringUtils.h>

namespace sofa::core
{

objectmodel::BaseObject::SPtr ObjectFactory::CreateObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    return ObjectFactoryInstance::createObject(context, arg);
}

bool ObjectFactory::AddAlias(const std::string& name, const std::string& result, bool force, ClassEntry::SPtr* previous)
{
    return ObjectFactoryInstance::addAlias(name, result, force, previous);
}

void ObjectFactory::ResetAlias(const std::string& name, ClassEntry::SPtr previous)
{
    ObjectFactoryInstance::resetAlias(name, previous);
}

bool ObjectFactory::HasCreator(const std::string& classname)
{
    return ObjectFactoryInstance::hasCreator(classname);
}

std::string ObjectFactory::ShortName(const std::string& classname)
{
    return sofa::helper::NameDecoder::shortName(classname);
}

std::string ObjectFactory::shortName(const std::string& classname)
{
    return sofa::helper::NameDecoder::shortName(classname);
}

sofa::core::ObjectFactory* ObjectFactory::getInstance()
{
    static ObjectFactory factory{ObjectFactoryInstance::getInstance()};
    return &factory;
}

void ObjectFactory::getAllEntries(std::vector<ClassEntry::SPtr>& result)
{
    newfactory->getEntries(result,"*");
}

/// Get an entry given a class name (or alias)
auto ObjectFactory::getEntry(const std::string& classname) -> ClassEntry&
{
    return newfactory->getEntry(classname);
}

/// Test if a creator exists for a given classname
bool ObjectFactory::hasCreator(const std::string& classname)
{
    return newfactory->hasCreator(classname);
}

/// Fill the given vector with the registered classes from a given target
void ObjectFactory::getEntriesFromTarget(std::vector<ClassEntry::SPtr>& result,
                                         const std::string& target)
{
    newfactory->getEntries(result, target);
}

/// Return the list of classes from a given target
std::string ObjectFactory::listClassesFromTarget(const std::string& target, const std::string& separator)
{
    std::vector<ClassEntry::SPtr> entries;
    getEntriesFromTarget(entries, target);
    return sofa::helper::join(entries, separator);
}

std::string ObjectFactory::listClassesDerivedFrom(const sofa::core::BaseClass* parentclass, const std::string& separator) const
{
    std::vector<ClassEntry::SPtr> entries;
    newfactory->getEntriesDerivedFrom(entries, parentclass);
    return sofa::helper::join(entries.begin(), entries.end(), [](ClassEntry::SPtr b){ return std::move(b->className); }, separator);
}

bool ObjectFactory::addAlias(const std::string& name, const std::string& target, bool force, ClassEntry::SPtr* previous)
{
    return newfactory->addAlias(name, target, force, previous);
}

void ObjectFactory::resetAlias(const std::string& name, ClassEntry::SPtr previous)
{
    return newfactory->resetAlias(name, previous);
}

sofa::core::objectmodel::BaseObject::SPtr ObjectFactory::createObject(objectmodel::BaseContext* context,
                                                                      objectmodel::BaseObjectDescription* arg)
{
    return newfactory->createObject(context, arg);
}

void ObjectFactory::dump(std::ostream& out)
{
    return newfactory->dump(out);
}

void ObjectFactory::dumpXML(std::ostream& out)
{
    return newfactory->dumpXML(out);
}

void ObjectFactory::dumpHTML(std::ostream& out)
{
    return newfactory->dumpHTML(out);
}

void ObjectFactory::setCallback(OnCreateCallback cb)
{
    return newfactory->setCallback(cb);
}

} // namespace sofa::core
