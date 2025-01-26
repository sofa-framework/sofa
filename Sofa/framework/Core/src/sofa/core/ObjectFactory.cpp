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

#include <sofa/defaulttype/TemplatesAliases.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/ComponentChange.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/DiffLib.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::core
{

ObjectFactory::~ObjectFactory()
{
}

ObjectFactory::ClassEntry& ObjectFactory::getEntry(std::string classname)
{
    if (!registry.contains(classname))
    {
        registry[classname] = std::make_shared<ClassEntry>();
        registry[classname]->className = classname;
    }

    return *registry[classname];
}

/// Test if a creator exists for a given classname
bool ObjectFactory::hasCreator(std::string classname)
{
    const ClassEntryMap::iterator it = registry.find(classname);
    if (it == registry.end())
        return false;
    const ClassEntry::SPtr entry = it->second;
    return (!entry->creatorMap.empty());
}

std::string ObjectFactory::shortName(std::string classname)
{
    std::string shortname;

    const ClassEntryMap::iterator it = registry.find(classname);
    if (it != registry.end())
    {
        const ClassEntry::SPtr entry = it->second;
        if(!entry->creatorMap.empty())
        {
            const CreatorMap::iterator myit = entry->creatorMap.begin();
            const Creator::SPtr c = myit->second;
            shortname = c->getClass()->shortName;
        }
    }
    return shortname;
}

bool ObjectFactory::addAlias(std::string name, std::string target, bool force,
                             ClassEntry::SPtr* previous)
{
    // Check that the pointed class does exist
    const ClassEntryMap::iterator it = registry.find(target);
    if (it == registry.end())
    {
        msg_error("ObjectFactory::addAlias()") << "Target class for alias '" << target << "' not found: " << name;
        return false;
    }

    const ClassEntry::SPtr& pointedEntry = it->second;
    ClassEntry::SPtr& aliasEntry = registry[name];

    // Check that the alias does not already exist, unless 'force' is true
    if (aliasEntry.get()!=nullptr && !force)
    {
        msg_error("ObjectFactory::addAlias()") << "Name already exists: " << name;
        return false;
    }

    if (previous) {
        const ClassEntry::SPtr& entry = aliasEntry;
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


void findTemplatedCreator(
    objectmodel::BaseContext* context,
    const ObjectFactory::Creator::SPtr& creator, const std::string& templateName,
    std::map<std::string, std::vector<std::string>>& creatorsErrors,
    std::vector< std::pair<std::string, ObjectFactory::Creator::SPtr> >& creators,
    objectmodel::BaseObjectDescription* arg)
{
    if (helper::system::PluginManager::getInstance().isPluginUnloaded(creator->getTarget()))
    {
        creatorsErrors[templateName].emplace_back(
            "The object was previously registered, but the module that "
            "registered the object has been unloaded, preventing the object creation.");
        arg->clearErrors();
    }
    else
    {
        if (creator->canCreate(context, arg))
        {
            creators.emplace_back(templateName, creator);
        }
        else
        {
            creatorsErrors[templateName] = arg->getErrors();
            arg->clearErrors();
        }
    }
}

objectmodel::BaseObject::SPtr ObjectFactory::createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    objectmodel::BaseObject::SPtr object = nullptr;
    std::vector< std::pair<std::string, Creator::SPtr> > creators;
    std::string classname = arg->getAttribute( "type", "");
    std::string usertemplatename = arg->getAttribute( "template", "");
    ClassEntry::SPtr entry ;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// Process the template aliases.
    ///  (1) split in a vector the user provided templates by ','
    ///  (2) for each entry search if there is an alias
    ///  (3) if there is none then keep value as is
    ///      otherwise replace the value with the alias.
    ///      if there is one and it is "undefined" generate a warning.
    ///      and "undefined" behavior means that the template is converting a specifically given
    ///      type precision into a different one.
    ///  (4) rebuild the template string by joining them all with ','.
    std::vector<std::string> usertemplatenames = sofa::helper::split(usertemplatename, ',');
    std::vector<std::string> deprecatedTemplates;
    for(auto& name : usertemplatenames)
    {
        const sofa::defaulttype::TemplateAlias* alias;
        if( (alias=sofa::defaulttype::TemplateAliases::getTemplateAlias(name)) != nullptr )
        {
            assert(alias != nullptr);
            /// This alias results in "undefined" behavior.
            if( alias->second )
            {
                deprecatedTemplates.push_back("The deprecated template '"+name+"' has been replaced by "+alias->first+".");
            }

            name = alias->first;
        }
    }
    std::string templatename = sofa::helper::join(usertemplatenames, ",");
    std::string userresolved = templatename; // Copy in case we change for the default one
    ////////////////////////////////////////////////////////////////////////////////////////////////


    //Check if object has been renamed

    using sofa::helper::lifecycle::renamedComponents;
    auto renamedComponent = renamedComponents.find(classname);
    if( renamedComponent != renamedComponents.end() )
    {
        classname = renamedComponent->second.getNewName();
    }


    // In order to get the errors from the creators only, we save the current errors at this point
    // and we clear them. Once we extracted the errors from the creators, we put push them back.
    std::map<std::string, std::vector<std::string>> creators_errors; // (template_name, errors)
    const auto previous_errors = arg->getErrors();
    arg->clearErrors();

    // For every classes in the registry
    ClassEntryMap::iterator it = registry.find(classname);
    if (it != registry.end()) // Found the classname
    {
        entry = it->second;
        // If no template has been given or if the template does not exist, first try with the default one
        if(templatename.empty() || !entry->creatorMap.contains(templatename))
            templatename = entry->defaultTemplate;


        if (auto it2 = entry->creatorMap.find(templatename);
            it2 != entry->creatorMap.end())
        {
            findTemplatedCreator(context, it2->second, it2->first, creators_errors, creators, arg);
        }

        // If object cannot be created with the given template (or the default one), try all possible ones
        if (creators.empty())
        {
            for (const auto& [creatorTemplateName, creator] : entry->creatorMap)
            {
                if (creatorTemplateName != templatename)
                {
                    findTemplatedCreator(context, creator, creatorTemplateName, creators_errors, creators, arg);
                }
            }
        }
    }

    // Restore previous errors without the errors from the creator
    arg->logErrors(previous_errors);

    if (creators.empty())
    {
        //// The object cannot be created
        arg->logError("Object type " + classname + std::string("<") + templatename + std::string("> was not created"));

        using sofa::helper::lifecycle::ComponentChange;
        using sofa::helper::lifecycle::uncreatableComponents;
        using sofa::helper::lifecycle::movedComponents;
        using sofa::helper::lifecycle::dealiasedComponents;
        if(it == registry.end())
        {
            arg->logError("The component '" + classname + "' cannot be found in the factory.");
            auto uncreatableComponent = uncreatableComponents.find(classname);
            auto movedComponent = movedComponents.find(classname);
            auto dealiasedComponent = dealiasedComponents.find(classname);

            const bool isUncreatable =  uncreatableComponent != uncreatableComponents.end();
            const bool isMoved = movedComponent != movedComponents.end();
            const bool isDealiased = dealiasedComponent != dealiasedComponents.end();

            const bool multipleReasons = static_cast<std::size_t>(isUncreatable) + static_cast<std::size_t>(isMoved) + static_cast<std::size_t>(isDealiased) > 1;
            std::size_t reasonNumber = 1;
            const auto number = [&reasonNumber, multipleReasons]() -> std::string
            {
                return multipleReasons ? std::to_string(reasonNumber++) + ") " : "";
            };

            if (multipleReasons)
            {
                arg->logError("Several reasons are possible:");
            }

            if(isUncreatable)
            {
                arg->logError(number() + uncreatableComponent->second.getMessage() );
            }
            if (isMoved)
            {
                arg->logError(number() + movedComponent->second.getMessage() );
            }
            if (isDealiased)
            {
                arg->logError(number() + dealiasedComponent->second.getMessage());
            }
            else
            {
                std::vector<std::string> possibleNames;
                possibleNames.reserve(registry.size());
                for(auto& k : registry)
                {
                    possibleNames.emplace_back(k.first);
                }

                arg->logError("But the following object(s) exist:");
                for(auto& [name, score] : sofa::helper::getClosestMatch(classname, possibleNames, 5, 0.6))
                {
                    arg->logError( "                      : " + name + " ("+ std::to_string((int)(100*score))+"% match)");
                }
            }
        }
        else
        {
            std::stringstream tmp;
            tmp << "The object is in the factory but cannot be created." << msgendl;
            tmp << "Requested template : " << (usertemplatename.empty() ? "None" : usertemplatename) << msgendl;
            if (templatename.empty()) {
                tmp << "Used template      : None" << msgendl;
            } else {
                tmp << "Used template      : " << templatename;
                if (templatename == entry->defaultTemplate) {
                    tmp << " (default)";
                }
                tmp << msgendl;
            }

            // Collect the errors from the creator with the specified (or default) template name
            auto main_creator_errors_iterator = creators_errors.find(templatename);
            if (main_creator_errors_iterator != creators_errors.end()) {
                tmp << "Reason(s)          : ";
                if (main_creator_errors_iterator->second.empty()) {
                    tmp << "No reasons given" << msgendl;
                } else if (main_creator_errors_iterator->second.size() == 1) {
                    tmp << main_creator_errors_iterator->second[0] << msgendl;
                } else {
                    tmp << msgendl;
                    for (std::size_t i = 0; i < main_creator_errors_iterator->second.size(); ++i) {
                        tmp << "    " << (i+1) << ". " << main_creator_errors_iterator->second[i] << msgendl;
                    }
                }
                creators_errors.erase(main_creator_errors_iterator);
            }

            // Collect the errors from the creator with all remaining template names
            if (! creators_errors.empty()) {
                for (const auto & creator_errors_it : creators_errors) {
                    const std::string & creator_template_name = creator_errors_it.first;
                    const std::vector<std::string> & creator_errors = creator_errors_it.second;
                    tmp << "Also tried to create the object with the template '"<<creator_template_name << "' but failed ";
                    if (creator_errors.empty()) {
                        tmp << "(no reason given)." << msgendl;
                    } else {
                        tmp << "for the following reason(s):" << msgendl;
                        for (std::size_t i = 0; i < creator_errors.size(); ++i) {
                            tmp << "    " << (i+1) << ". " << creator_errors[i] << msgendl;
                        }
                    }
                }
            }
            arg->logError(tmp.str());
        }
        return nullptr;
    }

    object = creators[0].second->createInstance(context, arg);
    assert(object!=nullptr);

    /// The object has been created, but not with the template given by the user
    if (!usertemplatename.empty() && object->getTemplateName() != userresolved)
    {
        std::vector<std::string> templateList;
        if (entry)
            for (const auto& cr : entry->creatorMap)
                templateList.push_back(cr.first);
        std::stringstream ss;
        bool isUserTemplateNameInTemplateList = false;
        for(unsigned int i = 0; i < templateList.size(); ++i)
        {
            ss << templateList[i];
            isUserTemplateNameInTemplateList |= (templateList[i] == usertemplatename || templateList[i] == userresolved);
            if (i != templateList.size() - 1)
                ss << ", ";
        }
        if (isUserTemplateNameInTemplateList)
        {
            msg_error(object.get()) << "Requested template '" << usertemplatename << "' "
                                      << "is not compatible with the current context. "
                                      << "Falling back to the first compatible template: '"
                                      << object->getTemplateName() << "'.";
        }
        else
        {
            msg_error(object.get()) << "Requested template '" << usertemplatename << "' "
                                      << "cannot be found in the list of available templates [" << ss.str() << "]. "
                                      << "Falling back to the first compatible template: '"
                                      << object->getTemplateName() << "'.";
        }
    }
    else if (creators.size() > 1)
    {    // There were multiple possibilities, we used the first one (not necessarily the default, as it can be incompatible)
        std::string w = "Template '" + templatename + std::string("' incorrect, used ") + object->getTemplateName() + std::string(" in the list:");
        for(unsigned int i = 0; i < creators.size(); ++i)
            w += std::string("\n\t* ") + creators[i].first;
        msg_warning(object.get()) << w;
    }

    ////////////////////////// This code is emitting a warning messages if the scene is loaded
    if( m_callbackOnCreate )
        m_callbackOnCreate(object.get(), arg);

    ///////////////////////// All this code is just there to implement the MakeDataAlias component.
    std::vector<std::string> todelete;
    for(auto& kv : entry->m_dataAlias)
    {
        if(object->findData(kv.first)==nullptr)
        {
            msg_warning(object.get()) << "The object '"<< (object->getClassName()) <<"' does not have an alias named '"<< kv.first <<"'.  "
                                      << "To remove this error message you need to use a valid data name for the 'dataname field'. ";

            todelete.push_back(kv.first);
        }
    }

    for(auto& todeletename : todelete)
    {
        entry->m_dataAlias.erase( entry->m_dataAlias.find(todeletename) ) ;
    }

    for(auto& kv : entry->m_dataAlias)
    {
        objectmodel::BaseObjectDescription newdesc;
        for(std::string& alias : kv.second){
            object->addAlias(object->findData(kv.first), alias.c_str()) ;

            /// The Alias is used in the argument
            const std::string val(arg->getAttribute(alias));
            if( !val.empty() ){
                newdesc.setAttribute( alias, val );
            }
        }
        object->parse(&newdesc);
    }

    /// We managed to create an object but there is error message in the log. Thus we emit them
    /// as warning to this object.
    if(!deprecatedTemplates.empty())
    {
        msg_deprecated(object.get()) << sofa::helper::join(deprecatedTemplates, msgendl) ;
    }

    return object;
}

ObjectFactory* ObjectFactory::getInstance()
{
    static ObjectFactory instance;
    return &instance;
}

void ObjectFactory::getAllEntries(std::vector<ClassEntry::SPtr>& result, const bool filterUnloadedPlugins)
{
    result.clear();
    for (const auto& [className, entry] : registry)
    {
        // Push the entry only if it is not an alias
        if (entry->className == className)
        {
            result.push_back(entry);
        }
    }

    if (filterUnloadedPlugins)
    {
        for (auto itEntry = result.begin(); itEntry != result.end();)
        {
            auto& creatorMap = (*itEntry)->creatorMap;
            for (auto itCreator = creatorMap.begin(); itCreator != creatorMap.end();)
            {
                if (helper::system::PluginManager::getInstance().isPluginUnloaded(itCreator->second->getTarget()))
                {
                    itCreator = creatorMap.erase(itCreator);
                }
                else
                {
                    ++itCreator;
                }
            }

            if (creatorMap.empty())
            {
                itEntry = result.erase(itEntry);
            }
            else
            {
                ++itEntry;
            }
        }
    }
}

void ObjectFactory::getEntriesFromTarget(std::vector<ClassEntry::SPtr>& result, std::string target)
{
    result.clear();
    for(ClassEntryMap::iterator it = registry.begin(), itEnd = registry.end();
        it != itEnd; ++it)
    {
        ClassEntry::SPtr entry = it->second;
        if(entry->className == it->first)
        {

            bool inTarget = false;
            for (CreatorMap::iterator itc = entry->creatorMap.begin(), itcend = entry->creatorMap.end(); itc != itcend; ++itc)
            {
                const Creator::SPtr c = itc->second;
                if (target == c->getTarget())
                {
                    inTarget = true;
                    break;
                }
            }
            if (inTarget)
                result.push_back(entry);
        }
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
        const ClassEntry::SPtr entry = it->second;
        if (entry->className != it->first) continue;
        out << "class " << entry->className <<" :\n";
        if (!entry->aliases.empty())
        {
            out << "  aliases :";
            for (std::set<std::string>::iterator myit = entry->aliases.begin(), itend = entry->aliases.end(); myit != itend; ++myit)
                out << " " << *myit;
            out << "\n";
        }
        if (!entry->description.empty())
            out << entry->description;
        if (!entry->authors.empty())
            out << "  authors : " << entry->authors << "\n";
        if (!entry->license.empty())
            out << "  license : " << entry->license << "\n";
        if (!entry->documentationURL.empty())
            out << "  documentation : " << entry->documentationURL << "\n";
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
        const ClassEntry::SPtr entry = it->second;
        if (entry->className != it->first) continue;
        out << "<class name=\"" << xmlencode(entry->className) <<"\">\n";
        for (std::set<std::string>::iterator myit = entry->aliases.begin(), itend = entry->aliases.end(); myit != itend; ++myit)
            out << "<alias>" << xmlencode(*myit) << "</alias>\n";
        if (!entry->description.empty())
            out << "<description>"<<entry->description<<"</description>\n";
        if (!entry->authors.empty())
            out << "<authors>"<<entry->authors<<"</authors>\n";
        if (!entry->license.empty())
            out << "<license>"<<entry->license<<"</license>\n";
        if (!entry->documentationURL.empty())
            out << "<documentation>"<<entry->documentationURL<<"</documentation>\n";
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
        const ClassEntry::SPtr entry = it->second;
        if (entry->className != it->first) continue;
        out << "<li><b>" << xmlencode(entry->className) <<"</b>\n";
        if (!entry->description.empty())
            out << "<br/>"<<entry->description<<"\n";
        out << "<ul>\n";
        if (!entry->aliases.empty())
        {
            out << "<li>Aliases:<i>";
            for (std::set<std::string>::iterator myit = entry->aliases.begin(), itend = entry->aliases.end(); myit != itend; ++myit)
                out << " " << xmlencode(*myit);
            out << "</i></li>\n";
        }
        if (!entry->authors.empty())
            out << "<li>Authors: <i>"<<entry->authors<<"</i></li>\n";
        if (!entry->license.empty())
            out << "<li>License: <i>"<<entry->license<<"</i></li>\n";
        if (!entry->documentationURL.empty())
            out << "<li>Documentation: <i>"<<entry->documentationURL<<"</i></li>\n";
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

bool ObjectFactory::registerObjects(ObjectRegistrationData& ro)
{
    return ro.commitTo(this);
}

ObjectRegistrationData::ObjectRegistrationData(const std::string& description)
{
    if (!description.empty())
    {
        addDescription(description);
    }
}

ObjectRegistrationData& ObjectRegistrationData::addAlias(std::string val)
{
    entry.aliases.insert(val);
    return *this;
}

ObjectRegistrationData& ObjectRegistrationData::addDescription(std::string val)
{
    val += '\n';
    entry.description += val;
    return *this;
}

ObjectRegistrationData& ObjectRegistrationData::addAuthor(std::string val)
{
    val += ' ';
    entry.authors += val;
    return *this;
}

ObjectRegistrationData& ObjectRegistrationData::addLicense(std::string val)
{
    entry.license += val;
    return *this;
}

ObjectRegistrationData& ObjectRegistrationData::addDocumentationURL(std::string url)
{
    entry.documentationURL += url;
    return *this;
}

ObjectRegistrationData& ObjectRegistrationData::addCreator(std::string classname,
                                           std::string templatename,
                                           ObjectFactory::Creator::SPtr creator)
{

    if (!entry.className.empty() && entry.className != classname)
    {
        msg_error("ObjectFactory") << "Template already instantiated with a different classname: " << entry.className << " != " << classname;
    }
    else if (entry.creatorMap.contains(templatename))
    {
        msg_error("ObjectFactory") << "Component already registered: " << classname << "<" << templatename << ">";
    }
    else
    {
        entry.className = classname;
        entry.creatorMap[templatename] =  creator;
    }
    return *this;
}

bool ObjectRegistrationData::commitTo(sofa::core::ObjectFactory* objectFactory) const
{
    if (entry.className.empty() || objectFactory == nullptr)
    {
        return false;
    }
    else
    {
        ObjectFactory::ClassEntry& reg = objectFactory->getEntry(entry.className);
        reg.description += entry.description;
        reg.authors += entry.authors;
        reg.license += entry.license;
        reg.documentationURL += entry.documentationURL;
        if (!entry.defaultTemplate.empty())
        {
            if (!reg.defaultTemplate.empty())
            {
                msg_warning("ObjectFactory") << "Default template for class " << entry.className << " already registered (" << reg.defaultTemplate << "), do not register " << entry.defaultTemplate << " as the default";
            }
            else
            {
                reg.defaultTemplate = entry.defaultTemplate;
            }
        }
        for (const auto& creator_entry : entry.creatorMap)
        {
            const std::string & template_name = creator_entry.first;
            const auto [it, success] = reg.creatorMap.insert(creator_entry);
            if (!success)
            {
                std::string classType = entry.className;
                if (!template_name.empty())
                {
                    classType += "<" + template_name + ">";
                }

                msg_warning("ObjectFactory") << "Class already registered in the ObjectFactory: " << classType;
            }
        }

        for (const auto & alias : entry.aliases)
        {
            if (!reg.aliases.contains(alias))
            {
                objectFactory->addAlias(alias,entry.className);
            }
        }
        return true;
    }

}

typedef struct ObjectRegistrationEntry
{
    inline static const char* symbol = "registerObjects";
    typedef void (*FuncPtr) (sofa::core::ObjectFactory*);
    FuncPtr func;
    void operator()(sofa::core::ObjectFactory* data)
    {
        if (func) return func(data);
    }
    ObjectRegistrationEntry() :func(nullptr) {}
} ObjectRegistrationEntry;

bool ObjectFactory::registerObjectsFromPlugin(const std::string& pluginName)
{
    sofa::helper::system::PluginManager& pluginManager = sofa::helper::system::PluginManager::getInstance();
    auto* plugin = pluginManager.getPlugin(pluginName);
    if (plugin == nullptr)
    {
        msg_error("ObjectFactory") << pluginName << " has not been loaded yet.";
        return false;
    }

    // do not register if it was already done before
    if(m_registeredPluginSet.contains(pluginName))
    {
        // This warning should be generalized (i.e not only in dev mode) when runSofa will not auto-load modules/plugins by default anymore
        dmsg_warning("ObjectFactory") << pluginName << " has already registered its components.";
        return false;
    }

    ObjectRegistrationEntry registerObjects;
    if (pluginManager.getEntryFromPlugin(plugin, registerObjects))
    {
        registerObjects(this);
        m_registeredPluginSet.insert(pluginName);
        return true;
    }
    else
    {
        return false;
    }
}

RegisterObject::RegisterObject(const std::string& description)
    : m_objectRegistrationdata(description)
{

}

RegisterObject& RegisterObject::addAlias(std::string val)
{
    m_objectRegistrationdata.addAlias(val);
    return *this;
}

RegisterObject& RegisterObject::addDescription(std::string val)
{
    m_objectRegistrationdata.addDescription(val);
    return *this;
}

RegisterObject& RegisterObject::addAuthor(std::string val)
{
    m_objectRegistrationdata.addAuthor(val);
    return *this;
}

RegisterObject& RegisterObject::addLicense(std::string val)
{
    m_objectRegistrationdata.addLicense(val);
    return *this;
}

RegisterObject& RegisterObject::addDocumentationURL(std::string url)
{
    m_objectRegistrationdata.addDocumentationURL(url);
    return *this;
}

RegisterObject& RegisterObject::addCreator(std::string classname, std::string templatename,
    ObjectFactory::Creator::SPtr creator)
{
    m_objectRegistrationdata.addCreator(classname, templatename, creator);
    return *this;
}

RegisterObject::operator int() const
{
    dmsg_warning("RegisterObject") << m_objectRegistrationdata.entry.className
                                  << ": Implicit object registration is deprecated since v24.12. Check #4429 for more information.";
    return commitTo(ObjectFactory::getInstance());
}

int RegisterObject::commitTo(ObjectFactory* factory) const
{
    return (m_objectRegistrationdata.commitTo(factory) ? 1 : 0);
}

} // namespace sofa::core
