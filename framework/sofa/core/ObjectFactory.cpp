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
#include "ObjectFactory.h"
#include <sofa/core/Plugin.h>


namespace sofa
{
namespace core
{

ObjectFactory::ObjectFactory()
{
    std::string pluginDir("");
    pluginDir += "./bin/plugins/";
    m_pluginManager.addPluginDirectory(pluginDir);
}

ObjectFactory::~ObjectFactory()
{
}

ObjectFactory::ClassEntry& ObjectFactory::getEntry(std::string classname)
{
    if (registry.find(classname) == registry.end()) {
        registry[classname] = ClassEntry::SPtr(new ClassEntry);
        registry[classname]->className = classname;
    }

    return *registry[classname];
}

bool ObjectFactory::hasEntry(std::string componentName)
{
    return registry.find(componentName) != registry.end();
}

void ObjectFactory::addEntry(const std::string& componentName, const std::string& description,
                             const std::string& authors, const std::string& license)
{
    if (hasEntry(componentName)) {
        std::cerr << "ObjectFactory::addEntry(): entry already exists: '"
                  << componentName << "'" << std::endl;
        return;
    }
    ClassEntry& entry = getEntry(componentName);
    entry.description = description;
    entry.authors = authors;
    entry.license = license;
}
    
void ObjectFactory::addCreator(const std::string& componentName, const std::string& templateParameters,
                               Creator::SPtr creator, bool defaultTemplate)
{
    if (!hasEntry(componentName)) {
        std::cerr << "ObjectFactory::addCreator(): entry does not exist: '"
                  << componentName << "'" << std::endl;
        return;
    }
    ClassEntry& entry = getEntry(componentName);
    
    if (entry.creatorMap.find(templateParameters) != entry.creatorMap.end()) {
        std::cerr << "ObjectFactory::addCreator(): creator already exists: '"
                  << componentName << "<" << templateParameters << ">" << "'" << std::endl;
        return;
    }
    entry.creatorMap[templateParameters] = creator;
    if (defaultTemplate)
        entry.defaultTemplateParameters = templateParameters;
}

void ObjectFactory::removeCreator(const std::string& componentName, const std::string& templateParameters)
{
    if (!hasEntry(componentName)) {
        std::cerr << "ObjectFactory::removeCreator(): entry does not exist: '"
                  << componentName << "'" << std::endl;
        return;
    }
    ClassEntry& entry = getEntry(componentName);
    if (entry.creatorMap.find(templateParameters) == entry.creatorMap.end()) {
        std::cerr << "ObjectFactory::removeCreator(): creator does not exist: '"
                  << componentName << "<" << templateParameters << ">" << "'" << std::endl;
        return;
    }
    entry.creatorMap.erase(templateParameters);
    if (entry.creatorMap.empty()) {
        registry.erase(componentName);
        std::cout << "ObjectFactory::removeCreator(): erased empty entry: '" << componentName << "'" << std::endl;
    }
}

bool ObjectFactory::hasCreator(std::string classname)
{
    ClassEntryMap::iterator it = registry.find(classname);
    if (it == registry.end())
        return false;
    ClassEntry::SPtr entry = it->second;
    return (!entry->creatorMap.empty());
}

std::string ObjectFactory::shortName(std::string classname)
{
    std::string shortname;

    ClassEntryMap::iterator it = registry.find(classname);
    if (it != registry.end())
    {
        ClassEntry::SPtr entry = it->second;
        if(!entry->creatorMap.empty())
        {
            CreatorMap::iterator it = entry->creatorMap.begin();
	    Creator::SPtr c = it->second;
            shortname = c->getClass()->shortName;
        }
    }
    return shortname;
}

bool ObjectFactory::addAlias(std::string name, std::string result, bool force,
                             ClassEntry::SPtr* previous)
{
    // Check that the pointed class does exist
    ClassEntryMap::iterator it = registry.find(result);
    if (it == registry.end())
    {
        std::cerr << "ERROR: ObjectFactory: cannot create alias " << name << " to unknown class " << result << ".\n";
        return false;
    }

    ClassEntry::SPtr& pointedEntry = it->second;
    ClassEntry::SPtr& aliasEntry = registry[name];

    // Check that the alias does not already exist, unless 'force' is true
    if (aliasEntry.get()!=NULL && !force)
    {
        std::cerr << "ERROR: ObjectFactory: cannot create alias " << name << " because a class with this name already exists.\n";
        return false;
    }

    if (previous) {
        ClassEntry::SPtr& entry = aliasEntry;
        *previous = entry;
    }

    registry[name] = pointedEntry;
    pointedEntry->aliases.insert(name);
    return true;
}

void ObjectFactory::resetAlias(std::string name, ClassEntry::SPtr previous)
{
    registry[name] = previous;
}

ObjectFactory::Creator::SPtr ObjectFactory::getCreator(objectmodel::BaseContext* context,
                                                       objectmodel::BaseObjectDescription* arg)
{
    std::string className = arg->getAttribute("type", "");
    std::string templateParameters = arg->getAttribute("template", "");

    // std::cout << "getCreator: " << className << ", " << templateParameters << std::endl;
    ClassEntryMap::iterator i = registry.find(className);
    // std::cout << "found component: " << (i != registry.end()) << std::endl;
    if (i != registry.end()) { // Component found
        ClassEntry::SPtr entry = i->second;

        if(templateParameters.empty())
            templateParameters = entry->defaultTemplateParameters;
        // std::cout << "defaultTemplateParameters: " << templateParameters << std::endl;

        CreatorMap::iterator j = entry->creatorMap.find(templateParameters);
        // std::cout << "found creator: " << (j != entry->creatorMap.end()) << std::endl;
        if (j != entry->creatorMap.end()) { // Template instance found
            Creator::SPtr creator = j->second;
            if (creator->canCreate(context, arg))
                return creator;
        }

        for (CreatorMap::iterator k = entry->creatorMap.begin();
             k != entry->creatorMap.end();
             k++) {
            Creator::SPtr creator = k->second;
            // std::cout << "trying creator: " << (creator->getClass()->templateName) << std::endl;
            if (creator->canCreate(context, arg)) {
                // std::cout << "Yep !" << std::endl;
                return creator;
            }
        }
    }
    return Creator::SPtr();
}

objectmodel::BaseObject::SPtr ObjectFactory::createObject(objectmodel::BaseContext* context,
                                                          objectmodel::BaseObjectDescription* arg)
{
    objectmodel::BaseObject::SPtr object = NULL;
    std::string className = arg->getAttribute("type", "");
    std::string templateParameters = arg->getAttribute("template", "");

    // std::cout << "createObject: " << className << ", " << templateParameters << std::endl;
    Creator::SPtr creator = getCreator(context, arg);
    if (creator.get() == NULL) {
        if (m_pluginManager.canFindComponent(className, templateParameters)) {
            try {
                Plugin& plugin = m_pluginManager.loadPluginContaining(className, templateParameters);
                m_pluginManager.addComponentsToFactory(*this, plugin);
                creator = getCreator(context, arg);
            } catch (std::exception& e) {
                arg->logWarning(e.what());
            }
        }
    }
    if (creator.get() == NULL)
        arg->logWarning("Object type " + className + std::string("<") + templateParameters
                        + std::string("> creation failed"));
    else
        object = creator->createInstance(context, arg);

    return object;
}

ObjectFactory* ObjectFactory::getInstance()
{
    static ObjectFactory instance;
    return &instance;
}

sofa::core::PluginManager& ObjectFactory::getPluginManager()
{
    return m_pluginManager;
}

void ObjectFactory::getAllEntries(std::vector<ClassEntry::SPtr>& result)
{
    result.clear();
    for(ClassEntryMap::iterator it = registry.begin(), itEnd = registry.end();
        it != itEnd; ++it)
    {
        ClassEntry::SPtr entry = it->second;
        // Push the entry only if it is not an alias
        if (entry->className == it->first)
            result.push_back(entry);
    }
}

void ObjectFactory::getEntriesFromTarget(std::vector<ClassEntry::SPtr>& result, std::string target)
{
    result.clear();
    for(ClassEntryMap::iterator it = registry.begin(), itEnd = registry.end();
        it != itEnd; ++it)
    {
        ClassEntry::SPtr entry = it->second;
        bool inTarget = false;
        for (CreatorMap::iterator itc = entry->creatorMap.begin(), itcend = entry->creatorMap.end(); itc != itcend; ++itc)
        {
	    Creator::SPtr c = itc->second;
            if (target == c->getTarget())
                inTarget = true;
        }
        if (inTarget)
            result.push_back(entry);
    }
}

std::string ObjectFactory::listClassesFromTarget(std::string target, std::string separator)
{
    std::vector<ClassEntry::SPtr> entries;
    getEntriesFromTarget(entries, target);
    std::ostringstream oss;
    for (unsigned int i=0; i<entries.size(); ++i)
    {
        if (i) oss << separator;
        oss << entries[i]->className;
    }
    std::string result = oss.str();
    return result;
}

void ObjectFactory::dump(std::ostream& out)
{
    for (ClassEntryMap::iterator it = registry.begin(), itend = registry.end(); it != itend; ++it)
    {
        ClassEntry::SPtr entry = it->second;
        if (entry->className != it->first) continue;
        out << "class " << entry->className <<" :\n";
        if (!entry->aliases.empty())
        {
            out << "  aliases :";
            for (std::set<std::string>::iterator it = entry->aliases.begin(), itend = entry->aliases.end(); it != itend; ++it)
                out << " " << *it;
            out << "\n";
        }
        if (!entry->description.empty())
            out << entry->description;
        if (!entry->authors.empty())
            out << "  authors : " << entry->authors << "\n";
        if (!entry->license.empty())
            out << "  license : " << entry->license << "\n";
        for (CreatorMap::iterator itc = entry->creatorMap.begin(), itcend = entry->creatorMap.end(); itc != itcend; ++itc)
        {
            out << "  template instance : " << itc->first << "\n";
        }
    }
}

static std::string xmlencode(const std::string& str)
{
    std::string res;
    for (unsigned int i=0; i<str.length(); ++i)
    {
        switch(str[i])
        {
        case '<': res += "&lt;"; break;
        case '>': res += "&gt;"; break;
        case '&': res += "&amp;"; break;
        case '"': res += "&quot;"; break;
        case '\'': res += "&apos;"; break;
        default:  res += str[i];
        }
    }
    return res;
}

void ObjectFactory::dumpXML(std::ostream& out)
{
    for (ClassEntryMap::iterator it = registry.begin(), itend = registry.end(); it != itend; ++it)
    {
        ClassEntry::SPtr entry = it->second;
        if (entry->className != it->first) continue;
        out << "<class name=\"" << xmlencode(entry->className) <<"\">\n";
        for (std::set<std::string>::iterator it = entry->aliases.begin(), itend = entry->aliases.end(); it != itend; ++it)
            out << "<alias>" << xmlencode(*it) << "</alias>\n";
        if (!entry->description.empty())
            out << "<description>"<<entry->description<<"</description>\n";
        if (!entry->authors.empty())
            out << "<authors>"<<entry->authors<<"</authors>\n";
        if (!entry->license.empty())
            out << "<license>"<<entry->license<<"</license>\n";
        for (CreatorMap::iterator itc = entry->creatorMap.begin(), itcend = entry->creatorMap.end(); itc != itcend; ++itc)
        {
            out << "<creator";
            if (!itc->first.empty()) out << " template=\"" << xmlencode(itc->first) << "\"";
            out << "/>\n";
        }
        out << "</class>\n";
    }
}

void ObjectFactory::dumpHTML(std::ostream& out)
{
    out << "<ul>\n";
    for (ClassEntryMap::iterator it = registry.begin(), itend = registry.end(); it != itend; ++it)
    {
        ClassEntry::SPtr entry = it->second;
        if (entry->className != it->first) continue;
        out << "<li><b>" << xmlencode(entry->className) <<"</b>\n";
        if (!entry->description.empty())
            out << "<br/>"<<entry->description<<"\n";
        out << "<ul>\n";
        if (!entry->aliases.empty())
        {
            out << "<li>Aliases:<i>";
            for (std::set<std::string>::iterator it = entry->aliases.begin(), itend = entry->aliases.end(); it != itend; ++it)
                out << " " << xmlencode(*it);
            out << "</i></li>\n";
        }
        if (!entry->authors.empty())
            out << "<li>Authors: <i>"<<entry->authors<<"</i></li>\n";
        if (!entry->license.empty())
            out << "<li>License: <i>"<<entry->license<<"</i></li>\n";
        if (entry->creatorMap.size()>2 || (entry->creatorMap.size()==1 && !entry->creatorMap.begin()->first.empty()))
        {
            out << "<li>Template instances:<i>";
            for (CreatorMap::iterator itc = entry->creatorMap.begin(), itcend = entry->creatorMap.end(); itc != itcend; ++itc)
            {
                if (itc->first == entry->defaultTemplateParameters)
                    out << " <b>" << xmlencode(itc->first) << "</b>";
                else
                    out << " " << xmlencode(itc->first);
            }
            out << "</i></li>\n";
        }
        out << "</ul>\n";
        out << "</li>\n";
    }
    out << "</ul>\n";
}

RegisterObject::RegisterObject(const std::string& description)
{
    if (!description.empty())
    {
        //std::cerr<<"description.size() = "<<description.size()<<", value = "<<description<<std::endl;
        addDescription(description);
    }
}

RegisterObject& RegisterObject::addAlias(std::string val)
{
    entry.aliases.insert(val);
    return *this;
}

RegisterObject& RegisterObject::addDescription(std::string val)
{
    //std::cout << "ObjectFactory: add description "<<val<<std::endl;
    val += '\n';
    entry.description += val;
    return *this;
}

RegisterObject& RegisterObject::addAuthor(std::string val)
{
    //std::cout << "ObjectFactory: add author "<<val<<std::endl;
    val += ' ';
    entry.authors += val;
    return *this;
}

RegisterObject& RegisterObject::addLicense(std::string val)
{
    //std::cout << "ObjectFactory: add license "<<val<<std::endl;
    entry.license += val;
    return *this;
}

RegisterObject& RegisterObject::addCreator(std::string classname,
					   std::string templatename,
					   ObjectFactory::Creator::SPtr creator)
{
  // std::cout << "ObjectFactory: warning: RegisterObject called for ("
  //           << classname << ", " << templatename << ")" << std::endl;
    if (!entry.className.empty() && entry.className != classname)
    {
        std::cerr << "ERROR: ObjectFactory: all templated class should have the same base classname ("<<entry.className<<"!="<<classname<<")\n";
    }
    else if (entry.creatorMap.find(templatename) != entry.creatorMap.end())
    {
        std::cerr << "ERROR: ObjectFactory: class "<<classname<<"<"<<templatename<<"> already registered\n";
    }
    else
    {
        entry.className = classname;
        entry.creatorMap[templatename] =  creator;
    }
    return *this;
}

RegisterObject::operator int()
{
    if (entry.className.empty())
    {
        return 0;
    }
    else
    {
        //std::cout << "ObjectFactory: commit"<<std::endl;
        ObjectFactory::ClassEntry& reg = ObjectFactory::getInstance()->getEntry(entry.className);
        reg.description += entry.description;
        reg.authors += entry.authors;
        reg.license += entry.license;
        if (!entry.defaultTemplateParameters.empty())
        {
            if (!reg.defaultTemplateParameters.empty())
            {
                std::cerr << "ERROR: ObjectFactory: default template for class "<<entry.className<<" already registered <"<<reg.defaultTemplateParameters<<">, do not register <"<<entry.defaultTemplateParameters<<"> as default.\n";
            }
            else
            {
                reg.defaultTemplateParameters = entry.defaultTemplateParameters;
            }
        }
        for (ObjectFactory::CreatorMap::iterator itc = entry.creatorMap.begin(), itcend = entry.creatorMap.end(); itc != itcend; ++itc)
        {
            if (reg.creatorMap.find(itc->first) != reg.creatorMap.end())
            {
                std::cerr << "ERROR: ObjectFactory: class "<<entry.className<<"<"<<itc->first<<"> already registered\n";
            }
            else
            {
                reg.creatorMap.insert(*itc);
            }
        }
        for (std::set<std::string>::iterator it = entry.aliases.begin(), itend = entry.aliases.end(); it != itend; ++it)
        {
            if (reg.aliases.find(*it) == reg.aliases.end())
            {
                ObjectFactory::getInstance()->addAlias(*it,entry.className);
            }
        }
        return 1;
    }
}

} // namespace core

} // namespace sofa
