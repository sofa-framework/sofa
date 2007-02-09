#ifndef SOFA_CORE_OBJECTMODEL_BASE_H
#define SOFA_CORE_OBJECTMODEL_BASE_H

#include <string>
#include <sofa/core/objectmodel/Field.h>
#include <sofa/core/objectmodel/DataField.h>
//#include <sofa/core/objectmodel/FieldContainer.h>
using sofa::core::objectmodel::Field;
using sofa::core::objectmodel::DataField;

namespace sofa
{

namespace core
{

namespace objectmodel
{

/// Base class for everything
class Base//:  public FieldContainer
{
public:
    Base();
    virtual ~Base();

    DataField<std::string> name;

    std::string getName() const;
    void setName(const std::string& n);
    virtual const char* getTypeName() const;

    void parseFields ( std::list<std::string> str );
    virtual void parseFields ( const std::map<std::string,std::string*>& str );
    void writeFields (std::map<std::string,std::string*>& str);
    void writeFields (std::ostream& out);
    void xmlWriteFields (std::ostream& out, unsigned level);

    template<class T>
    DataField<T> dataField( DataField<T>* field, char* name, char* help )
    {
        std::string ln(name);
        if( ln.size()>0 && m_fieldMap.find(ln) != m_fieldMap.end() )
        {
            std::cerr << "field name " << ln << " already used in this class or in a parent class !...aborting" << std::endl;
            exit( 1 );
        }
        //field = tmp;
        m_fieldMap[name] = field;
        return DataField<T>(help);
    }

    template<class T>
    DataField<T> dataField( DataField<T>* field, const T& value, char* name, char* help )
    {
        std::string ln(name);
        if( ln.size()>0 && m_fieldMap.find(ln) != m_fieldMap.end() )
        {
            std::cerr << "field name " << ln << " already used in this class or in a parent class !...aborting" << std::endl;
            exit( 1 );
        }
        //field = tmp;
        m_fieldMap[name] = field;
        return DataField<T>(value,help);
    }

    template<class T>
    Field<T> field( Field<T>* field, T* ptr, char* name, char* help )
    {
        std::string ln(name);
        if( ln.size()>0 && m_fieldMap.find(ln) != m_fieldMap.end() )
        {
            std::cerr << "field name " << ln << " already used in this class or in a parent class !...aborting" << std::endl;
            exit( 1 );
        }
        //field = tmp;
        m_fieldMap[name] = field;
        return Field<T>(ptr,help);
    }

protected:
    /// name -> Field object
    std::map< std::string, FieldBase* > m_fieldMap;

    void addField( FieldBase* f, char* name )
    {
        std::string ln(name);
        if( ln.size()>0 && m_fieldMap.find(ln) != m_fieldMap.end() )
        {
            std::cerr << "field name " << ln << " already used in this class or in a parent class !...aborting" << std::endl;
            exit( 1 );
        }
        m_fieldMap[name] = f;
    }

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif

