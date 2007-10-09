/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
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
    const char* realname = NULL;
    char* allocname = strdup(name.c_str());
#ifdef __GNUC__
    int status;
    realname = allocname = abi::__cxa_demangle(allocname, 0, 0, &status);
    if (realname==NULL)
#endif
        realname = allocname;
    int len = strlen(realname);
    char* newname = (char*)malloc(len+1);
    int start = 0;
    int dest = 0;
    char cprev = '\0';
    //std::cout << "name = "<<realname<<std::endl;
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
                newname[dest++] = realname[start++];
                newname[dest] = 0;
            }
        }
        cprev = c;
        //std::cout << "i = "<<i<<" start = "<<start<<" dest = "<<dest<<" newname = "<<newname<<std::endl;
    }
    while (start < len)
    {
        newname[dest++] = realname[start++];
        newname[dest] = 0;
    }
    newname[dest] = '\0';
    //std::cout << "newname = "<<newname<<std::endl;
    name = newname;
    free(newname);
    //if (allocname)
    //    free(allocname);
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
            //std::cout << "name="<<name<<std::endl;
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
    */
}

/// Extract the class name (removing namespaces and templates)
std::string Base::decodeClassName(const std::type_info& t)
{
    std::string name = t.name();
    const char* realname = NULL;
    char* allocname = strdup(name.c_str());
#ifdef __GNUC__
    int status;
    realname = allocname = abi::__cxa_demangle(allocname, 0, 0, &status);
    if (realname==NULL)
#endif
        realname = allocname;
    int len = strlen(realname);
    char* newname = (char*)malloc(len+1);
    int start = 0;
    int dest = 0;
    int i;
    char cprev = '\0';
    //std::cout << "name = "<<realname<<std::endl;
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
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z'))
        {
            // write result
            while (start < i)
            {
                newname[dest++] = realname[start++];
                newname[dest] = 0;
            }
        }
        cprev = c;
        //std::cout << "i = "<<i<<" start = "<<start<<" dest = "<<dest<<" newname = "<<newname<<std::endl;
    }
    while (start < i)
    {
        newname[dest++] = realname[start++];
        newname[dest] = 0;
    }
    newname[dest] = '\0';
    //std::cout << "newname = "<<newname<<std::endl;
    name = newname;
    free(newname);
    //if (allocname)
    //    free(allocname);
    return name;
    /*
        std::string name = decodeTypeName(t);
        // Find template
        std::string::size_type pos = name.find("<");
        if (pos != std::string::npos)
        {
            name.erase(pos,name.length()-pos);
        }
        //std::cout << "CLASS NAME="<<name<<std::endl;
        return name;
    */
}

/// Extract the namespace (removing class name and templates)
std::string Base::decodeNamespaceName(const std::type_info& t)
{
    std::string name = t.name();
    const char* realname = NULL;
    char* allocname = strdup(name.c_str());
#ifdef __GNUC__
    int status;
    realname = allocname = abi::__cxa_demangle(allocname, 0, 0, &status);
    if (realname==NULL)
#endif
        realname = allocname;
    int len = strlen(realname);
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
    name.assign(realname+start, realname+last+1);
    //if (allocname)
    //    free(allocname);
    return name;
}

/// Decode the template name (removing namespaces and class name)
std::string Base::decodeTemplateName(const std::type_info& t)
{
    std::string name = t.name();
    const char* realname = NULL;
    char* allocname = strdup(name.c_str());
#ifdef __GNUC__
    int status;
    realname = allocname = abi::__cxa_demangle(allocname, 0, 0, &status);
    if (realname==NULL)
#endif
        realname = allocname;
    int len = strlen(realname);
    char* newname = (char*)malloc(len+1);
    newname[0] = '\0';
    int start = 0;
    int dest = 0;
    int i = 0;
    char cprev = '\0';
    //std::cout << "name = "<<realname<<std::endl;
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
                newname[dest++] = realname[start++];
                newname[dest] = 0;
            }
        }
        cprev = c;
        //std::cout << "i = "<<i<<" start = "<<start<<" dest = "<<dest<<" newname = "<<newname<<std::endl;
    }
    while (start < i)
    {
        newname[dest++] = realname[start++];
        newname[dest] = 0;
    }
    newname[dest] = '\0';
    //std::cout << "newname = "<<newname<<std::endl;
    name = newname;
    free(newname);
    //if (allocname)
    //    free(allocname);
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
        //std::cout << "TEMPLATE NAME="<<name<<std::endl;
        return name;
    */
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
            if( !(m_fieldMap[ name ]->read( s )))
                std::cerr<< "ERROR: could not read value for option " << name <<": "<< s << std::endl;
        }
        else
        {
            str.pop_front();
            std::cerr << "\nUnknown option: " << name << std::endl << std::endl;
        }
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
                    std::cerr<< "ERROR: could not read value for option " << key <<": "<< val << std::endl;
            }
            else
            {
                if ((key!="name") && (key!="type"))
                    std::cerr <<"\nUnknown option: " << key << std::endl;
            }
        }
    }
}

/// Parse the given description to assign values to this object's fields and potentially other parameters
void  Base::parse ( BaseObjectDescription* arg )
{
    //this->parseFields ( arg->getAttributeMap() );
    for( std::map<string,FieldBase*>::const_iterator a=m_fieldMap.begin(), aend=m_fieldMap.end(); a!=aend; ++a )
    {
        const char* val = arg->getAttribute(a->first);
        if (val)
        {
            string valueString = val;
            if( !(a->second->read( valueString )))
                std::cerr<< "ERROR: could not read value for option " << a->first <<": "<< val << std::endl;
        }
    }
}

void  Base::writeFields ( std::map<std::string,std::string*>& args )
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
        if( field->isSet() && !field->getValueString().empty())
            out << (*a).first << "=\""<< field->getValueString() << "\" ";
    }
}

void  Base::xmlWriteFields ( std::ostream& out, unsigned level )
{
    for( std::map<string,FieldBase*>::const_iterator a=m_fieldMap.begin(), aend=m_fieldMap.end(); a!=aend; ++a )
    {
        FieldBase* field = (*a).second;
        if( field->isSet() && !field->getValueString().empty())
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

