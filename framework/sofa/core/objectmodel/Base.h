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
#ifndef SOFA_CORE_OBJECTMODEL_BASE_H
#define SOFA_CORE_OBJECTMODEL_BASE_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/system/SofaOStream.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/objectmodel/BaseClass.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/Tag.h>
#include <string>
#include <map>

using sofa::core::objectmodel::Data;

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief Base class for everything
 *
 *  This class contains all functionnality shared by every objects in SOFA.
 *  Most importantly it defines how to retrieve information about an object (name, type, data fields).
 *  All classes deriving from Base should use the SOFA_CLASS macro within their declaration (see BaseClass.h).
 *
 */
class SOFA_CORE_API Base
{
public:
    typedef TClass< Base, void > MyClass;
    static const MyClass* GetClass() { return MyClass::get(); }
    virtual const BaseClass* getClass() const { return GetClass(); }

    Base();
    virtual ~Base();

private:
    /// Copy constructor is not allowed
    Base(const Base& b);

public:



    /// Accessor to the object name
    std::string getName() const;

    /// Set the name of this object
    void setName(const std::string& n);

    /// Get the type name of this object (i.e. class and template types)
    virtual std::string getTypeName() const;

    /// Get the class name of this object
    virtual std::string getClassName() const;

    /// Get the template type names (if any) used to instantiate this object
    virtual std::string getTemplateName() const;

    /// @name fields
    ///   Data fields managment
    /// @{

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse ( BaseObjectDescription* arg );

    /// Assign the field values stored in the given list of name + value pairs of strings
    void parseFields ( std::list<std::string> str );

    /// Assign the field values stored in the given map of name -> value pairs
    virtual void parseFields ( const std::map<std::string,std::string*>& str );

    /// Write the current field values to the given map of name -> value pairs
    void writeDatas (std::map<std::string,std::string*>& str);

    /// Write the current Node values to the given XML output stream
    void xmlWriteNodeDatas (std::ostream& out, unsigned level);

    /// Write the current field values to the given XML output stream
    void xmlWriteDatas (std::ostream& out, unsigned level, bool compact);

    /// Find a field given its name. Return NULL if not found. If more than one field is found (due to aliases), only the first is returned.
    BaseData* findField( const std::string &name ) const;

    /// Find fields given a name: several can be found as we look into the alias map
    std::vector< BaseData* > findGlobalField( const std::string &name ) const;

    /// Helper method used to initialize a field containing a value of type T
    template<class T>
    BaseData::BaseInitData initData( Data<T>* field, const char* name, const char* help, bool isDisplayed=true, bool isReadOnly=false )
    {
        BaseData::BaseInitData res;
        this->initData0(field, res, name, help, isDisplayed, isReadOnly);
        return res;
    }

    /// Helper method used to initialize a field containing a value of type T
    template<class T>
    typename Data<T>::InitData initData( Data<T>* field, const T& value, const char* name, const char* help, bool isDisplayed=true, bool isReadOnly=false  )
    {
        typename Data<T>::InitData res;
        this->initData0(field, res, value, name, help, isDisplayed, isReadOnly);
        return res;
    }

    /// Add an alias to a Data
    void addAlias( BaseData* field, const char* alias);

    /// Accessor to the vector containing all the fields of this object
    const std::vector< std::pair<std::string, BaseData*> >& getFields() const { return m_fieldVec; }
    /// Accessor to the map containing all the aliases of this object
    const std::multimap< std::string, BaseData* >& getAliases() const { return m_aliasData; }

    void copyAspect(int destAspect, int srcAspect);

    void releaseAspect(int aspect);
    /// @}

    /// @name tags
    ///   Methods related to tagged subsets
    /// @{

    /// Represents the subsets the object belongs to
    const sofa::core::objectmodel::TagSet& getTags() const { return f_tags.getValue(); }

    /// Return true if the object belong to the given subset
    bool hasTag( Tag t ) const;

    /// Add a subset qualification to the object
    void addTag(Tag t);
    /// Remove a subset qualification to the object
    void removeTag(Tag t);

    /// @}

    /// @name logs
    ///   Messages and warnings logging
    /// @{

    mutable sofa::helper::system::SofaOStream<Base> sendl;
    mutable std::ostringstream                      serr;
    mutable std::ostringstream                      sout;

    const std::string& getWarnings() const;
    const std::string& getOutputs() const;

    void clearWarnings();
    void clearOutputs();

    void processStream(std::ostream& out);

    /// @}

protected:
    /// Helper method used by initData()
    void initData0( BaseData* field, BaseData::BaseInitData& res, const char* name, const char* help, bool isDisplayed=true, bool isReadOnly=false );
    void initData0( BaseData* field, BaseData::BaseInitData& res, const char* name, const char* help, BaseData::DataFlags dataFlags );

    /// Helper method used by initData()
    template<class T>
    void initData0( Data<T>* field, typename Data<T>::InitData& res, const T& value, const char* name, const char* help, bool isDisplayed=true, bool isReadOnly=false )
    {
        initData0( field, res, name, help, isDisplayed, isReadOnly );
        res.value = value;
    }

    /// Add a field. Note that this method should only be called if the field was not initialized with the initData<T> of field<T> methods
    void addField(BaseData* f, const char* name);

public:
    /// Helper method to decode the type name
    static std::string decodeFullName(const std::type_info& t);

    /// Helper method to decode the type name to a more readable form if possible
    static std::string decodeTypeName(const std::type_info& t);

    /// Helper method to extract the class name (removing namespaces and templates)
    static std::string decodeClassName(const std::type_info& t);

    /// Helper method to extract the namespace (removing class name and templates)
    static std::string decodeNamespaceName(const std::type_info& t);

    /// Helper method to extract the template name (removing namespaces and class name)
    static std::string decodeTemplateName(const std::type_info& t);

    /// Helper method to get the type name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::typeName(ptr); \endcode
    /// This way derived classes can redefine the typeName method
    template<class T>
    static std::string typeName(const T* = NULL)
    {
        return decodeTypeName(typeid(T));
    }

    /// Helper method to get the class name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::className(ptr); \endcode
    /// This way derived classes can redefine the className method
    template<class T>
    static std::string className(const T* = NULL)
    {
        return decodeClassName(typeid(T));
    }

    /// Helper method to get the namespace name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::namespaceName(ptr); \endcode
    /// This way derived classes can redefine the namespaceName method
    template<class T>
    static std::string namespaceName(const T* = NULL)
    {
        return decodeNamespaceName(typeid(T));
    }

    /// Helper method to get the template name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::templateName(ptr); \endcode
    /// This way derived classes can redefine the templateName method
    template<class T>
    static std::string templateName(const T* = NULL)
    {
        return decodeTemplateName(typeid(T));
    }


    /// Helper method to get the shortname of a type derived from this class.
    /// The default implementation return the class name.
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::shortName(ptr); \endcode
    /// This way derived classes can redefine the shortName method
    template< class T>
    static std::string shortName( const T* ptr = NULL, BaseObjectDescription* = NULL )
    {
        std::string shortname = T::className(ptr);
        if( !shortname.empty() )
        {
            *shortname.begin() = ::tolower(*shortname.begin());
        }
        return shortname;
    }

protected:
    std::string warnings;
    std::string outputs;

    /// List of fields (Data instances)
    std::vector< std::pair<std::string, BaseData*> > m_fieldVec;
    /// name -> Data multi-map (includes names and aliases)
    std::multimap< std::string, BaseData* > m_aliasData;

public:
    /// Name of the object.
    Data<std::string> name;

    Data<bool> f_printLog;

    Data< sofa::core::objectmodel::TagSet > f_tags;

    Data< sofa::defaulttype::BoundingBox > f_bbox;

};

} // namespace objectmodel

} // namespace core

} // namespace sofa



#endif

