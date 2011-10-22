/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/Factory.h>
#include <map>
#include <typeinfo>
#ifdef __GNUC__
#include <cxxabi.h>
#endif

#include <string.h>
namespace sofa
{

namespace core
{

namespace objectmodel
{

using std::string;

Base::Base()
    : ref_counter(0)
    , name(initData(&name,std::string("unnamed"),"name","object name"))
    , f_printLog(initData(&f_printLog, false, "printLog", "if true, print logs at run-time"))
    , f_tags(initData( &f_tags, "tags", "list of the subsets the objet belongs to"))
    , f_bbox(initData( &f_bbox, "bbox", "this object bounding box"))
{
    name.setOwnerClass("Base");
    name.setAutoLink(false);
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
    //if ((--ref_counter) == 0)
    if (ref_counter.dec_and_test_null())
    {
        //serr << "DELETE" << sendl;
        // Deletion of objects can be temporarily disabled by commenting the next line, until smart-pointers usage is corrected
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

    std::string nameStr(name);
    if (nameStr.size() >= 4)
    {
        const std::string prefix=nameStr.substr(0,4);
        if (prefix=="show" || prefix=="draw") res.group = "Visualization";
    }
}

/// Add a data field.
/// Note that this method should only be called if the field was not initialized with the initData method
void Base::addData(BaseData* f)
{
    std::string name = f->getName();
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
    std::string name = l->getName();
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
    //std::cout << std::endl;
}

/// Release memory allocated for the specified aspect.
void Base::releaseAspect(int aspect)
{
    for(VecData::const_iterator iData = m_vecData.begin(); iData != m_vecData.end(); ++iData)
    {
        (*iData)->releaseAspect(aspect);
    }
}

/// Get the type name of this object (i.e. class and template types)
std::string Base::getTypeName() const
{
    return decodeTypeName(typeid(*this));
}

/// Get the class name of this object
std::string Base::getClassName() const
{
    return decodeClassName(typeid(*this));
}

/// Get the template type names (if any) used to instantiate this object
std::string Base::getTemplateName() const
{
    return decodeTemplateName(typeid(*this));
}

std::string Base::getName() const
{
    return name.getValue();
}

void Base::setName(const std::string& na)
{
    name.setValue(na);
}

void Base::processStream(std::ostream& out)
{
    if (&out == &serr)
    {
        serr << "\n";
        //if (f_printLog.getValue())
        std::cerr<< "WARNING[" << getName() << "(" << getClassName() << ")]: "<<serr.str();
        warnings += serr.str();
        serr.str("");
    }
    else if (&out == &sout)
    {
        sout << "\n";
        if (f_printLog.getValue())
            std::cout<< "[" << getName() << "(" << getClassName() << ")]: "<< sout.str() << std::flush;
        outputs += sout.str();
        sout.str("");
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


/// Helper method to decode the type name
std::string Base::decodeFullName(const std::type_info& t)
{
    std::string name;
#ifdef __GNUC__
    int status;
    /* size_t length; */ // although it should, length would not be filled in by the following call
    char* allocname = abi::__cxa_demangle(t.name(), 0, /*&length*/0, &status);
    if(allocname == 0)
    {
        std::cerr << "Unable to demangle symbol: " << t.name() << std::endl;
    }
    else
    {
        int length = 0;
        while(allocname[length] != '\0')
        {
            length++;
        }
        name.resize(length);
        for(int i=0; i<(int)length; i++)
            name[i] = allocname[i];
        free(allocname);
    }
#else
    name = t.name();
#endif
    return name;
}
/// Decode the type's name to a more readable form if possible
std::string Base::decodeTypeName(const std::type_info& t)
{
    std::string name;
    std::string realname = decodeFullName(t);
    int len = realname.length();
    name.resize(len+1);
    int start = 0;
    int dest = 0;
    char cprev = '\0';
    //sout << "name = "<<realname<<sendl;
    for (int i=0; i<len; i++)
    {
        char c = realname[i];
        if (c == ':') // && cprev == ':')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z'))
        {
            // write result
            while (start < i)
            {
                name[dest++] = realname[start++];
            }
        }
        cprev = c;
        //sout << "i = "<<i<<" start = "<<start<<" dest = "<<dest<<" name = "<<name<<sendl;
    }
    while (start < len)
    {
        name[dest++] = realname[start++];
    }
    name.resize(dest);
    return name;
    /*
        // Remove namespaces
        std::string cname;
        for(;;)
        {
            std::string::size_type pos = name.find("::");
            if (pos == std::string::npos) break;
            std::string::size_type first = name.find_last_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",pos-1);
            if (first == std::string::npos) first = 0;
            else first++;
            name.erase(first,pos-first+2);
            //sout << "name="<<name<<sendl;
        }
        // Remove "class "
        for(;;)
        {
            std::string::size_type pos = name.find("class ");
            if (pos == std::string::npos) break;
            name.erase(pos,6);
        }
        //sout << "TYPE NAME="<<name<<sendl;
        return name;
    */
}

/// Extract the class name (removing namespaces and templates)
std::string Base::decodeClassName(const std::type_info& t)
{
    std::string name;
    std::string realname = decodeFullName(t);
    int len = realname.length();
    name.resize(len+1);
    int start = 0;
    int dest = 0;
    int i;
    char cprev = '\0';
    //sout << "name = "<<realname<<sendl;
    for (i=0; i<len; i++)
    {
        char c = realname[i];
        if (c == '<') break;
        if (c == ':') // && cprev == ':')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 6 && realname[i-6] == 's' && realname[i-5] == 't' && realname[i-4] == 'r' && realname[i-3] == 'u' && realname[i-2] == 'c' && realname[i-1] == 't')
        {
            start = i+1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z'))
        {
            // write result
            while (start < i)
            {
                name[dest++] = realname[start++];
            }
        }
        cprev = c;
    }

    while (start < i)
    {
        name[dest++] = realname[start++];
    }
    name.resize(dest);
    return name;
    /*
        std::string name = decodeTypeName(t);
        // Find template
        std::string::size_type pos = name.find("<");
        if (pos != std::string::npos)
        {
            name.erase(pos,name.length()-pos);
        }
        //sout << "CLASS NAME="<<name<<sendl;
        return name;
    */
}

/// Extract the namespace (removing class name and templates)
std::string Base::decodeNamespaceName(const std::type_info& t)
{
    std::string name;
    std::string realname = decodeFullName(t);
    int len = realname.length();
    int start = 0;
    int last = len-1;
    int i;
    for (i=0; i<len; i++)
    {
        char c = realname[i];
        if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c == ':' && (i<1 || realname[i-1]!=':'))
        {
            last = i-1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z'))
        {
            // write result
            break;
        }
    }
    name = realname.substr(start, last-start+1);
    return name;
}

/// Decode the template name (removing namespaces and class name)
std::string Base::decodeTemplateName(const std::type_info& t)
{
    std::string name;
    std::string realname = decodeFullName(t);
    int len = realname.length();
    name.resize(len+1);
    int start = 0;
    int dest = 0;
    int i = 0;
    char cprev = '\0';
    while (i < len && realname[i]!='<')
        ++i;
    start = i+1; ++i;
    for (; i<len; i++)
    {
        char c = realname[i];
        //if (c == '<') break;
        if (c == ':') // && cprev == ':')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z'))
        {
            // write result
            while (start <= i)
            {
                name[dest++] = realname[start++];
            }
        }
        cprev = c;
    }
    while (start < i)
    {
        name[dest++] = realname[start++];
    }
    name.resize(dest);
    return name;
    /*
        std::string name = decodeTypeName(t);
        // Find template
        std::string::size_type pos = name.find("<");
        if (pos != std::string::npos)
        {
            name = name.substr(pos+1,name.length()-pos-2);
        }
        else
        {
            name = "";
        }
        //sout << "TEMPLATE NAME="<<name<<sendl;
        return name;
    */
}

/// Find a data field given its name.
/// Return NULL if not found. If more than one field is found (due to aliases), only the first is returned.
BaseData* Base::findData( const std::string &name ) const
{
    //Search in the aliases
    typedef MapData::const_iterator mapIterator;
    std::pair< mapIterator, mapIterator> range = m_aliasData.equal_range(name);
    if (range.first != range.second)
        return range.first->second;
    else
        return NULL;
}

/// Find fields given a name: several can be found as we look into the alias map
std::vector< BaseData* > Base::findGlobalField( const std::string &name ) const
{
    std::vector<BaseData*> result;
    //Search in the aliases
    typedef MapData::const_iterator mapIterator;
    std::pair< mapIterator, mapIterator> range = m_aliasData.equal_range(name);
    for (mapIterator itAlias=range.first; itAlias!=range.second; itAlias++)
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
    for (mapIterator itAlias=range.first; itAlias!=range.second; itAlias++)
        result.push_back(itAlias->second);
    return result;
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
        std::vector< BaseData* > fields=findGlobalField(name);
        if( fields.size() > 0 )
        {
            for (unsigned int i=0; i<fields.size(); ++i)
            {
                if( !(fields[i]->read( value ))) serr<<"Could not read value for data field " << name <<": "<< value << sendl;
            }
        }
        else
        {
            serr<<"Unknown data field: "<< name << sendl;
        }
    }
}

void  Base::parseFields ( const std::map<std::string,std::string*>& args )
{
    std::string key,val;
    for( std::map<string,string*>::const_iterator i=args.begin(), iend=args.end(); i!=iend; i++ )
    {
        if( (*i).second!=NULL )
        {
            key=(*i).first;
            val=*(*i).second;
            std::vector< BaseData* > fields=findGlobalField(key);
            if( fields.size() > 0 )
            {
                for (unsigned int i=0; i<fields.size(); ++i)
                {
                    if( !(fields[i]->read( val ))) serr<<"Could not read value for data field "<<key<<": "<<val << sendl;
                }
            }
            else
            {
                if ((key!="name") && (key!="type")) serr<<"Unknown data field: " << key << sendl;
            }
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
        std::vector< BaseData* > dataModif = findGlobalField(attributeList[i]);
        for (unsigned int d=0; d<dataModif.size(); ++d)
        {
            const char* val = arg->getAttribute(attributeList[i]);
            if (val)
            {
                std::string valueString(val);
                if( !(dataModif[d]->read( valueString ))) serr<<"Could not read value for data field "<< attributeList[i] <<": " << val << sendl;
            }
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
}

void  Base::xmlWriteDatas (std::ostream& out, int)
{
    for(VecData::const_iterator iData = m_vecData.begin(); iData != m_vecData.end(); ++iData)
    {
        BaseData* field = *iData;
        if (!field->getLinkPath().empty() )
        {
            out << " " << field->getName() << "=\""<< field->getLinkPath() << "\" ";
        }
        else
        {
            if(  field->isPersistent() && field->isSet())
            {
                std::string val = field->getValueString();
                if (!val.empty())
                    out << " " << field->getName() << "=\""<< val << "\" ";
            }
        }
    }
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
