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


namespace sofa
{
namespace core
{

ObjectFactory::~ObjectFactory()
{
}

ObjectFactory::ClassEntry& ObjectFactory::getEntry(std::string classname)
{
    if (registry.find(classname) == registry.end()) {
        registry[classname] = boost::shared_ptr<ClassEntry>(new ClassEntry);
        registry[classname]->className = classname;
    }

    return *registry[classname];
}

/// Test if a creator exists for a given classname
bool ObjectFactory::hasCreator(std::string classname)
{
    ClassEntryMap::iterator it = registry.find(classname);
    if (it == registry.end())
        return false;
    ClassEntry* entry = it->second.get();
    return (!entry->creatorMap.empty());
}

std::string ObjectFactory::shortName(std::string classname)
{
    std::string shortname;

    ClassEntryMap::iterator it = registry.find(classname);
    if (it != registry.end())
    {
        ClassEntry* entry = it->second.get();
        if(!entry->creatorMap.empty())
        {
            CreatorMap::iterator it = entry->creatorMap.begin();
            Creator* c = it->second.get();
            shortname = c->getClass()->shortName;
        }
    }
    return shortname;
}

bool ObjectFactory::addAlias(std::string name, std::string result, bool force,
                             boost::shared_ptr<ClassEntry>* previous)
{
    // Check that the pointed class does exist
    ClassEntryMap::iterator it = registry.find(result);
    if (it == registry.end())
    {
        std::cerr << "ERROR: ObjectFactory: cannot create alias " << name << " to unknown class " << result << ".\n";
        return false;
    }

    boost::shared_ptr<ClassEntry>& pointedEntry = it->second;
    boost::shared_ptr<ClassEntry>& aliasEntry = registry[name];

    // Check that the alias does not already exist, unless 'force' is true
    if (aliasEntry.get()!=NULL && !force)
    {
        std::cerr << "ERROR: ObjectFactory: cannot create alias " << name << " because a class with this name already exists.\n";
        return false;
    }

    if (previous) {
        boost::shared_ptr<ClassEntry>& entry = aliasEntry;
        *previous = entry;
    }

    registry[name] = pointedEntry;
    pointedEntry->aliases.insert(name);
    return true;
}

void ObjectFactory::resetAlias(std::string name, boost::shared_ptr<ClassEntry> previous)
{
    registry[name] = previous;
}

objectmodel::BaseObject::SPtr ObjectFactory::createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    objectmodel::BaseObject::SPtr object = NULL;
    std::vector< std::pair<std::string, boost::shared_ptr<Creator> > > creators;
    std::string classname = arg->getAttribute( "type", "");
    std::string templatename = arg->getAttribute( "template", "");
    ClassEntryMap::iterator it = registry.find(classname);
    if (it == registry.end())
    {
        //std::cout << "ObjectFactory: class "<<classname<<" NOT FOUND."<<std::endl;
    }
    else
    {
//        std::cout << "ObjectFactory: class "<<classname<<" FOUND."<<std::endl;
        ClassEntry* entry = it->second.get();
        if(templatename.empty()) templatename = entry->defaultTemplate;
        CreatorMap::iterator it2 = entry->creatorMap.find(templatename);
        if (it2 != entry->creatorMap.end())
        {
//            std::cout << "ObjectFactory: template "<<templatename<<" FOUND."<<std::endl;
            Creator* c = it2->second.get();
            if (c->canCreate(context, arg))
                creators.push_back(*it2);
        }
        else
        {
            // std::cout << "ObjectFactory: template "<<templatename<<" NOT FOUND for "<<classname<<std::endl;
            CreatorMap::iterator it3;
            for (it3 = entry->creatorMap.begin(); it3 != entry->creatorMap.end(); ++it3)
            {
                Creator* c = it3->second.get();
                if (c->canCreate(context, arg))
                    creators.push_back(*it3);
            }
        }
    }
    if (creators.empty())
    {
//        std::cerr<<"ERROR: ObjectFactory: Object type "<<classname<<"<"<<templatename<<"> creation failed."<<std::endl;
        arg->logWarning("Object type " + classname + std::string("<") + templatename + std::string("> creation failed"));
    }
    else
    {
//          std::cout << "Create Instance : " << arg->getFullName() << "\n";
        object = creators[0].second->createInstance(context, arg);
        if (creators.size()>1)
        {
//                 std::cerr<<"WARNING: ObjectFactory: Several possibilities found for type "<<classname<<"<"<<templatename<<">:\n"; //<<std::endl;
            std::string w= "Template Unknown: <"+templatename+std::string("> : default used: <")+object->getTemplateName()+std::string("> in the list: ");
            for(unsigned int i=0; i<creators.size(); ++i)
            {
                w += std::string("\n\t* ") + creators[i].first; //creatorsobjectmodel::Base::decodeTemplateName(creators[i]->type());
            }
            object->serr<<w<<object->sendl;
        }
    }
    return object;
}

ObjectFactory* ObjectFactory::getInstance()
{
    static ObjectFactory instance;
    return &instance;
}

void ObjectFactory::getAllEntries(std::vector<ClassEntry*>& result)
{
    result.clear();
    for(ClassEntryMap::iterator it = registry.begin(), itEnd = registry.end();
        it != itEnd; ++it)
    {
        ClassEntry* entry = it->second.get();
        result.push_back(entry);
    }
}

void ObjectFactory::getEntriesFromTarget(std::vector<ClassEntry*>& result, std::string target)
{
    result.clear();
    for(ClassEntryMap::iterator it = registry.begin(), itEnd = registry.end();
        it != itEnd; ++it)
    {
        ClassEntry* entry = it->second.get();
        bool inTarget = false;
        for (CreatorMap::iterator itc = entry->creatorMap.begin(), itcend = entry->creatorMap.end(); itc != itcend; ++itc)
        {
            Creator* c = itc->second.get();
            if (target == c->getTarget())
                inTarget = true;
        }
        if (inTarget)
            result.push_back(entry);
    }
}

std::string ObjectFactory::listClassesFromTarget(std::string target, std::string separator)
{
    std::vector<ClassEntry*> entries;
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
        ClassEntry* entry = it->second.get();
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
        ClassEntry* entry = it->second.get();
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
        ClassEntry* entry = it->second.get();
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
                if (itc->first == entry->defaultTemplate)
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

RegisterObject& RegisterObject::addCreator(std::string classname, std::string templatename, boost::shared_ptr<ObjectFactory::Creator> creator)
{

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
        if (!entry.defaultTemplate.empty())
        {
            if (!reg.defaultTemplate.empty())
            {
                std::cerr << "ERROR: ObjectFactory: default template for class "<<entry.className<<" already registered <"<<reg.defaultTemplate<<">, do not register <"<<entry.defaultTemplate<<"> as default.\n";
            }
            else
            {
                reg.defaultTemplate = entry.defaultTemplate;
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
