/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/logging/Messaging.h>
#include <map>
#include <typeinfo>
#include <string.h>
#include <sstream>


namespace sofa
{

namespace core
{

namespace objectmodel
{

using std::string;
static const std::string unnamed_label=std::string("unnamed");

Base::Base()
    : ref_counter(0)
    , serr(_serr)
    , sout(_sout)
    , name(initData(&name,unnamed_label,"name","object name"))
    , f_printLog(initData(&f_printLog, false, "printLog", "if true, print logs at run-time"))
    , f_tags(initData( &f_tags, "tags", "list of the subsets the objet belongs to"))
    , f_bbox(initData( &f_bbox, "bbox", "this object bounding box"))
{
    name.setOwnerClass("Base");
    name.setAutoLink(false);
    name.setReadOnly(true);
    f_printLog.setOwnerClass("Base");
    f_printLog.setAutoLink(false);
    f_tags.setOwnerClass("Base");
    f_tags.setAutoLink(false);
    f_bbox.setOwnerClass("Base");
    f_bbox.setReadOnly(true);
    f_bbox.setPersistent(false);
    f_bbox.setDisplayed(false);
    f_bbox.setAutoLink(false);
    sendl.setParent(this);
}

Base::~Base()
{
}

void Base::addRef()
{
    ++ref_counter;
}

void Base::release()
{
    if (ref_counter.dec_and_test_null())
    {
        delete this;
    }
}

/// Helper method used by initData()
void Base::initData0( BaseData* field, BaseData::BaseInitData& res, const char* name, const char* help, bool isDisplayed, bool isReadOnly )
{
    BaseData::DataFlags flags = BaseData::FLAG_DEFAULT;
    if(isDisplayed) flags |= (BaseData::DataFlags)BaseData::FLAG_DISPLAYED; else flags &= ~(BaseData::DataFlags)BaseData::FLAG_DISPLAYED;
    if(isReadOnly)  flags |= (BaseData::DataFlags)BaseData::FLAG_READONLY; else flags &= ~(BaseData::DataFlags)BaseData::FLAG_READONLY;

    initData0(field, res, name, help, flags);
}

/// Helper method used by initData()
void Base::initData0( BaseData* field, BaseData::BaseInitData& res, const char* name, const char* help, BaseData::DataFlags dataFlags )
{
    // Questionnable optimization: test a single 'uint32_t' rather that four 'char'
    static const char *draw_str = "draw";
    static const char *show_str = "show";
    static uint32_t draw_prefix = *(uint32_t*)draw_str;
    static uint32_t show_prefix = *(uint32_t*)show_str;

    /*
        std::string ln(name);
        if( ln.size()>0 && findField(ln) )
        {
            serr << "field name " << ln << " already used in this class or in a parent class !...aborting" << sendl;
            exit( 1 );
        }
        m_fieldVec.push_back( std::make_pair(ln,field));
        m_aliasData.insert(std::make_pair(ln,field));
    */
    res.owner = this;
    res.data = field;
    res.name = name;
    res.helpMsg = help;
    res.dataFlags = dataFlags;

    uint32_t prefix = *(uint32_t*)name;

    if (prefix == draw_prefix || prefix == show_prefix)
        res.group = "Visualization";
}

/// Add a data field.
/// Note that this method should only be called if the field was not initialized with the initData method
void Base::addData(BaseData* f)
{
    addData(f, f->getName());
}

/// Add a data field.
/// Note that this method should only be called if the field was not initialized with the initData method
void Base::addData(BaseData* f, const std::string& name)
{
    if (name.size() > 0 && (findData(name) || findLink(name)))
    {
        serr << "Data field name " << name
                << " already used in this class or in a parent class !"
                << sendl;
        //exit(1);
    }
    m_vecData.push_back(f);
    m_aliasData.insert(std::make_pair(name, f));
    f->setOwner(this);
}

/// Add an alias to a Data
void Base::addAlias( BaseData* field, const char* alias)
{
    m_aliasData.insert(std::make_pair(std::string(alias),field));
}

/// Add a link.
/// Note that this method should only be called if the link was not initialized with the initLink method
void Base::addLink(BaseLink* l)
{
    const std::string& name = l->getName();
    if (name.size() > 0 && (findData(name) || findLink(name)))
    {
        serr << "Link name " << name
                << " already used in this class or in a parent class !"
                << sendl;
        //exit(1);
    }
    m_vecLink.push_back(l);
    m_aliasLink.insert(std::make_pair(name, l));
    //l->setOwner(this);
}

/// Add an alias to a Link
void Base::addAlias( BaseLink* link, const char* alias)
{
    m_aliasLink.insert(std::make_pair(std::string(alias),link));
}

/// Copy the source aspect to the destination aspect for each Data in the component.
void Base::copyAspect(int destAspect, int srcAspect)
{
    for(VecData::const_iterator iData = m_vecData.begin(); iData != m_vecData.end(); ++iData)
    {
        //std::cout << "  " << iData->first;
        (*iData)->copyAspect(destAspect, srcAspect);
    }
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        //std::cout << "  " << iLink->first;
        (*iLink)->copyAspect(destAspect, srcAspect);
    }
    //std::cout << std::endl;
}

/// Release memory allocated for the specified aspect.
void Base::releaseAspect(int aspect)
{
    for(VecData::const_iterator iData = m_vecData.begin(); iData != m_vecData.end(); ++iData)
    {
        (*iData)->releaseAspect(aspect);
    }
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        (*iLink)->releaseAspect(aspect);
    }
}

/// Get the type name of this object (i.e. class and template types)
std::string Base::getTypeName() const
{
    //return BaseClass::decodeTypeName(typeid(*this));
    std::string c = getClassName();
    std::string t = getTemplateName();
    if (t.empty())
        return c;
    else
        return c + std::string("<") + t + std::string(">");
}

/// Get the class name of this object
std::string Base::getClassName() const
{
    return BaseClass::decodeClassName(typeid(*this));
}

/// Get the template type names (if any) used to instantiate this object
std::string Base::getTemplateName() const
{
    return BaseClass::decodeTemplateName(typeid(*this));
}

void Base::setName(const std::string& na)
{
    name.setValue(na);
}

/// Set the name of this object, adding an integer counter
void Base::setName(const std::string& n, int counter)
{
    std::ostringstream o;
    o << n << counter;
    setName(o.str());
}

#define MAXLOGSIZE 10000000

void Base::processStream(std::ostream& out)
{
    // const std::string name = getClassName() + " \"" + getName() + "\"";

    if (serr==out)
    {
        std::string str = serr.str();
        serr << "\n";

        helper::logging::MessageDispatcher::log(serr.messageClass(), serr.messageType(), this, serr.fileInfo()) << str;

        if (warnings.size()+str.size() >= MAXLOGSIZE)
        {
            const std::string msg = "Log overflow! Resetting serr buffer.";
            msg_warning(this) << msg;
            warnings.clear();
            warnings = msg;
        }
        warnings += str;
        serr.clear();
    }
    else if (sout==out)
    {
        std::string str = sout.str();
        sout << "\n";
        if (f_printLog.getValue())
        {
            helper::logging::MessageDispatcher::log(serr.messageClass(), sout.messageType(), this, sout.fileInfo()) << str;
        }
        if (outputs.size()+str.size() >= MAXLOGSIZE)
        {
            const std::string msg = "Log overflow! Resetting sout buffer.";
            msg_warning(this) << msg;
            outputs.clear();
            outputs = msg;
        }
        outputs += str;
        sout.clear();
    }
}

const std::string& Base::getWarnings() const
{
    return warnings;
}

const std::string& Base::getOutputs() const
{
    return outputs;
}

void Base::clearWarnings()
{
    warnings.clear();
}

void Base::clearOutputs()
{
    outputs.clear();
}


bool Base::hasTag(Tag t) const
{
    return (f_tags.getValue().count( t ) > 0 );
}


void Base::addTag(Tag t)
{
    f_tags.beginEdit()->insert(t);
    f_tags.endEdit();
}

void Base::removeTag(Tag t)
{
    f_tags.beginEdit()->erase(t);
    f_tags.endEdit();
}


/// Find a data field given its name.
/// Return NULL if not found. If more than one field is found (due to aliases), only the first is returned.
BaseData* Base::findData( const std::string &name ) const
{
    //Search in the aliases
    if(m_aliasData.size())
    {
        typedef MapData::const_iterator mapIterator;
        std::pair< mapIterator, mapIterator> range = m_aliasData.equal_range(name);
        if (range.first != range.second)
            return range.first->second;
        else
            return NULL;
    }
    else return NULL;
}

/// Find fields given a name: several can be found as we look into the alias map
std::vector< BaseData* > Base::findGlobalField( const std::string &name ) const
{
    std::vector<BaseData*> result;
    //Search in the aliases
    typedef MapData::const_iterator mapIterator;
    std::pair< mapIterator, mapIterator> range = m_aliasData.equal_range(name);
    for (mapIterator itAlias=range.first; itAlias!=range.second; ++itAlias)
        result.push_back(itAlias->second);
    return result;
}


/// Find a link given its name.
/// Return NULL if not found. If more than one link is found (due to aliases), only the first is returned.
BaseLink* Base::findLink( const std::string &name ) const
{
    //Search in the aliases
    typedef MapLink::const_iterator mapIterator;
    std::pair< mapIterator, mapIterator> range = m_aliasLink.equal_range(name);
    if (range.first != range.second)
        return range.first->second;
    else
        return NULL;
}

/// Find links given a name: several can be found as we look into the alias map
std::vector< BaseLink* > Base::findLinks( const std::string &name ) const
{
    std::vector<BaseLink*> result;
    //Search in the aliases
    typedef MapLink::const_iterator mapIterator;
    std::pair< mapIterator, mapIterator> range = m_aliasLink.equal_range(name);
    for (mapIterator itAlias=range.first; itAlias!=range.second; ++itAlias)
        result.push_back(itAlias->second);
    return result;
}

bool Base::findDataLinkDest(BaseData*& ptr, const std::string& path, const BaseLink* link)
{
    std::string pathStr, dataStr;
    if (link)
    {
        if (!link->parseString(path, &pathStr, &dataStr))
            return false;
    }
    else
    {
        if (!BaseLink::ParseString(path, &pathStr, &dataStr, this))
            return false;
    }
    if (pathStr.empty() || pathStr == std::string("[]"))
    {
        ptr = this->findData(dataStr);
        return (ptr != NULL);
    }
    Base* obj = NULL;
    if (!findLinkDest(obj, BaseLink::CreateString(pathStr), link))
        return false;
    if (!obj)
        return false;
    ptr = obj->findData(dataStr);
    return (ptr != NULL);
}

void* Base::findLinkDestClass(const BaseClass* /*destType*/, const std::string& /*path*/, const BaseLink* /*link*/)
{
    serr << "Base: calling unimplemented findLinkDest method" << sendl;
    return NULL;
}

bool Base::hasField( const std::string& attribute) const
{
    return m_aliasData.find(attribute) != m_aliasData.end()
            || m_aliasLink.find(attribute) != m_aliasLink.end();
}

/// Assign one field value (Data or Link)
bool Base::parseField( const std::string& attribute, const std::string& value)
{
    std::vector< BaseData* > dataVec = findGlobalField(attribute);
    std::vector< BaseLink* > linkVec = findLinks(attribute);
    if (dataVec.empty() && linkVec.empty())
    {
        serr << "Unknown Data field or Link: " << attribute << sendl;
        return false; // no field found
    }
    bool ok = true;
    for (unsigned int d=0; d<dataVec.size(); ++d)
    {
        // test if data is a link and can be linked
        if (value[0] == '@' && dataVec[d]->canBeLinked())
        {
            if (!dataVec[d]->setParent(value))
            {
                serr<<"Could not setup Data link between "<< value << " and " << attribute << "." << sendl;
                ok = false;
                continue;
            }
            else
            {
                BaseData* parentData = dataVec[d]->getParent();
                sout<<"Link from parent Data " << value << " (" << parentData->getValueTypeInfo()->name() << ") to Data " << attribute << "(" << dataVec[d]->getValueTypeInfo()->name() << ") OK" << sendl;
            }
            /* children Data cannot be modified changing the parent Data value */
            dataVec[d]->setReadOnly(true);
            continue;
        }
        if( !(dataVec[d]->read( value )) && !value.empty())
        {
            serr<<"Could not read value for data field "<< attribute <<": " << value << sendl;
            ok = false;
        }
    }
    for (unsigned int l=0; l<linkVec.size(); ++l)
    {
        if( !(linkVec[l]->read( value )) && !value.empty())
        {
            serr<<"Could not read value for link "<< attribute <<": " << value << sendl;
            ok = false;
        }
        sout << "Link " << linkVec[l]->getName() << " = " << linkVec[l]->getValueString() << sendl;
        unsigned int s = linkVec[l]->getSize();
        for (unsigned int i=0; i<s; ++i)
        {
            sout  << "  " << linkVec[l]->getLinkedPath(i) << " = ";
            Base* b = linkVec[l]->getLinkedBase(i);
            BaseData* d = linkVec[l]->getLinkedData(i);
            if (b) sout << b->getTypeName() << " " << b->getName();
            if (d) sout << " . " << d->getValueTypeString() << " " << d->getName();
            sout << sendl;
        }
    }
    return ok;
}

void  Base::parseFields ( const std::list<std::string>& str )
{
    string name,value;
    std::list<std::string>::const_iterator it = str.begin(), itend = str.end();
    while(it != itend)
    {
        name = *it;
        ++it;
        if (it == itend) break;
        value = *it;
        ++it;
        parseField(name, value);
    }
}

void  Base::parseFields ( const std::map<std::string,std::string*>& args )
{
    std::string key,val;
    for( std::map<string,string*>::const_iterator i=args.begin(), iend=args.end(); i!=iend; ++i )
    {
        if( (*i).second!=NULL )
        {
            key=(*i).first;
            val=*(*i).second;
            parseField(key, val);
        }
    }
}

/// Parse the given description to assign values to this object's fields and potentially other parameters
void  Base::parse ( BaseObjectDescription* arg )
{
    std::vector< std::string > attributeList;
    arg->getAttributeList(attributeList);
    for (unsigned int i=0; i<attributeList.size(); ++i)
    {
        std::string attrName = attributeList[i];
        // FIX: "type" is already used to define the type of object to instanciate, any Data with
        // the same name cannot be extracted from BaseObjectDescription
        if (attrName == std::string("type")) continue;
        if (!hasField(attrName)) continue;
        const char* val = arg->getAttribute(attrName);
        if (!val) continue;
        std::string valueString(val);
        parseField(attrName, valueString);
    }
    updateLinks(false);
}

/// Update pointers in case the pointed-to objects have appeared
void Base::updateLinks(bool logErrors)
{
    // update links
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        bool ok = (*iLink)->updateLinks();
        if (!ok && (*iLink)->storePath() && logErrors)
        {
            serr << "Link update failed for " << (*iLink)->getName() << " = " << (*iLink)->getValueString() << sendl;
        }
    }
}

void  Base::writeDatas ( std::map<std::string,std::string*>& args )
{
    for(VecData::const_iterator iData = m_vecData.begin(); iData != m_vecData.end(); ++iData)
    {
        BaseData* field = *iData;
        std::string name = field->getName();
        if( args[name] != NULL )
            *args[name] = field->getValueString();
        else
            args[name] =  new string(field->getValueString());
    }
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        BaseLink* link = *iLink;
        std::string name = link->getName();
        if( args[name] != NULL )
            *args[name] = link->getValueString();
        else
            args[name] =  new string(link->getValueString());
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

void  Base::writeDatas (std::ostream& out, const std::string& separator)
{
    for(VecData::const_iterator iData = m_vecData.begin(); iData != m_vecData.end(); ++iData)
    {
        BaseData* field = *iData;
        if (!field->getLinkPath().empty() )
        {
            out << separator << field->getName() << "=\""<< xmlencode(field->getLinkPath()) << "\" ";
        }
        else
        {
            if(field->isPersistent() && field->isSet())
            {
                std::string val = field->getValueString();
                if (!val.empty())
                    out << separator << field->getName() << "=\""<< xmlencode(val) << "\" ";
            }
        }
    }
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        BaseLink* link = *iLink;
        if(link->storePath())
        {
            std::string val = link->getValueString();
            if (!val.empty())
                out << separator << link->getName() << "=\""<< xmlencode(val) << "\" ";
        }
    }
}


} // namespace objectmodel

} // namespace core

} // namespace sofa
