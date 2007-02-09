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

namespace sofa
{

namespace core
{

namespace objectmodel
{

using std::string;

Base::Base()
    : name(dataField(&name,std::string(""),"name","object name"))
{

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

const char* Base::getTypeName() const
{
    //return "UNKNOWN";
    // TODO: change the return type to std::string
    // BUG: this is not threadsafe!!!
    static std::map<const Base*, std::string> typenames;
    std::string& str = typenames[this];
    if (str.empty())
    {
        str = sofa::helper::gettypename(typeid(*this));
    }
    return str.c_str();
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

