//
// C++ Implementation: Base
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/Factory.h>
#include <map>
#include <typeinfo>
#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace sofa
{

namespace core
{

namespace objectmodel
{

using std::string;

Base::Base()
{
    name = dataField(&name,std::string("unnamed"),"name","object name");
}

Base::~Base()
{}

std::string Base::getName() const
{
    //if( name.getValue().empty() )
    //    return getTypeName();
    return name.getValue();
}

void Base::setName(const std::string& na)
{
    name.setValue(na);
}


/// Decode the type's name to a more readable form if possible
std::string Base::decodeTypeName(const std::type_info& t)
{
    std::string name = t.name();
#ifdef __GNUC__
    char* realname = NULL;
    int status;
    realname = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
    if (realname!=NULL)
    {
        name = realname;
        free(realname);
    }
#endif
    // Remove namespaces
    for(;;)
    {
        std::string::size_type pos = name.find("::");
        if (pos == std::string::npos) break;
        std::string::size_type first = name.find_last_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",pos-1);
        if (first == std::string::npos) first = 0;
        else first++;
        name.erase(first,pos-first+2);
        std::cout << "name="<<name<<std::endl;
    }
    // Remove "class "
    for(;;)
    {
        std::string::size_type pos = name.find("class ");
        if (pos == std::string::npos) break;
        name.erase(pos,6);
    }
    //std::cout << "TYPE NAME="<<name<<std::endl;
    return name;
}

/// Extract the class name (removing namespaces and templates)
std::string Base::decodeClassName(const std::type_info& t)
{
    std::string name = decodeTypeName(t);
    // Find template
    std::string::size_type pos = name.find("<");
    if (pos != std::string::npos)
    {
        name.erase(pos,name.length()-pos);
    }
    //std::cout << "CLASS NAME="<<name<<std::endl;
    return name;
}

/// Decode the template name (removing namespaces and class name)
std::string Base::decodeTemplateName(const std::type_info& t)
{
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
    //std::cout << "TEMPLATE NAME="<<name<<std::endl;
    return name;
}

void  Base::parseFields ( std::list<std::string> str )
{
    string name;
    while( !str.empty() )
    {
        name = str.front();
        str.pop_front();

        // field name
        if( m_fieldMap.find(name) != m_fieldMap.end() )
        {
            std::string s = str.front();
            str.pop_front();
            if( !(m_fieldMap[ name ]->read( s )))
                std::cerr<< "\ncould not read value for option " << name <<": "<< s << std::endl << std::endl;
        }
        else
            std::cerr << "\nUnknown option: " << name << std::endl << std::endl;
    }
}

void  Base::parseFields ( const std::map<std::string,std::string*>& args )
{
    // build  std::list<std::string> str
    using std::string;
    string key,val;
    for( std::map<string,string*>::const_iterator i=args.begin(), iend=args.end(); i!=iend; i++ )
    {
        if( (*i).second!=NULL )
        {
            key=(*i).first;
            val=*(*i).second;
            if( m_fieldMap.find(key) != m_fieldMap.end() )
            {
                if( !(m_fieldMap[ key ]->read( val )))
                    std::cerr<< "\ncould not read value for option " << key <<": "<< val << std::endl << std::endl;
            }
            else
            {
                if ((key!="name") && (key!="type"))
                    std::cerr <<"\nUnknown option: " << key << std::endl;
            }
        }
    }
}

void    Base::writeFields ( std::map<std::string,std::string*>& args )
{
    for( std::map<string,FieldBase*>::const_iterator a=m_fieldMap.begin(), aend=m_fieldMap.end(); a!=aend; ++a )
    {
        string valueString;
        FieldBase* field = (*a).second;

        if( args[(*a).first] != NULL )
            *args[(*a).first] = field->getValueString();
        else
            args[(*a).first] =  new string(field->getValueString());
    }
}

void  Base::writeFields ( std::ostream& out )
{
    for( std::map<string,FieldBase*>::const_iterator a=m_fieldMap.begin(), aend=m_fieldMap.end(); a!=aend; ++a )
    {
        FieldBase* field = (*a).second;
        if( field->isSet() )
            out << (*a).first << "=\""<< field->getValueString() << "\" ";
    }
}

void  Base::xmlWriteFields ( std::ostream& out, unsigned level )
{
    for( std::map<string,FieldBase*>::const_iterator a=m_fieldMap.begin(), aend=m_fieldMap.end(); a!=aend; ++a )
    {
        FieldBase* field = (*a).second;
        if( field->isSet() )
        {
            for (unsigned i=0; i<=level; i++)
                out << "\t";
            out << (*a).first << "=\""<< field->getValueString() << "\""<<std::endl;
        }
    }
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

