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
#ifndef SOFA_CORE_OBJECTMODEL_BASE_H
#define SOFA_CORE_OBJECTMODEL_BASE_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/system/atomic.h>
#include <sofa/helper/system/SofaOStream.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/objectmodel/BaseClass.h>
#include <sofa/core/objectmodel/SPtr.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/Tag.h>

#include <boost/intrusive_ptr.hpp>

#include <string>
#include <map>

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
    typedef Base* Ptr;
    typedef boost::intrusive_ptr<Base> SPtr;

    typedef TClass< Base, void > MyClass;
    static const MyClass* GetClass() { return MyClass::get(); }
    virtual const BaseClass* getClass() const { return GetClass(); }

    template<class T>
    static void dynamicCast(T*& ptr, Base* b)
    {
        ptr = dynamic_cast<T*>(b);
    }

protected:
    /// Constructor cannot be called directly
    /// Use the New() method instead
    Base();

    /// Direct calls to destructor are forbidden.
    /// Smart pointers must be used to manage creation/destruction of objects
    virtual ~Base();

private:
    /// Copy constructor is not allowed
    Base(const Base& b);

    sofa::helper::system::atomic<int> ref_counter;
    void addRef();
    void release();

    friend inline void intrusive_ptr_add_ref(Base* p)
    {
        p->addRef();
    }

    friend inline void intrusive_ptr_release(Base* p)
    {
        p->release();
    }

public:



    /// Accessor to the object name
    const std::string& getName() const
    {
        return name.getValue();
    }

    /// Set the name of this object
    void setName(const std::string& n);

    /// Set the name of this object, adding an integer counter
    void setName(const std::string& n, int counter);

    /// Get the type name of this object (i.e. class and template types)
    virtual std::string getTypeName() const;

    /// Get the class name of this object
    virtual std::string getClassName() const;

    /// Get the template type names (if any) used to instantiate this object
    virtual std::string getTemplateName() const;

    /// @name fields
    ///   Data fields management
    /// @{

    /// Assign one field value (Data or Link)
    virtual bool parseField( const std::string& attribute, const std::string& value);

    /// Check if a given Data field or Link exists
    virtual bool hasField( const std::string& attribute) const;

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse ( BaseObjectDescription* arg );

    /// Assign the field values stored in the given list of name + value pairs of strings
    void parseFields ( const std::list<std::string>& str );

    /// Assign the field values stored in the given map of name -> value pairs
    virtual void parseFields ( const std::map<std::string,std::string*>& str );

    /// Write the current field values to the given map of name -> value pairs
    void writeDatas (std::map<std::string,std::string*>& str);

    /// Write the current field values to the given output stream
    /// separated with the given separator (" " used by default for XML)
    void writeDatas (std::ostream& out, const std::string& separator = " ");

    /// Find a data field given its name. Return NULL if not found.
    /// If more than one field is found (due to aliases), only the first is returned.
    BaseData* findData( const std::string &name ) const;



    /// Find data fields given a name: several can be found as we look into the alias map
    std::vector< BaseData* > findGlobalField( const std::string &name ) const;

    /// Find a link given its name. Return NULL if not found.
    /// If more than one link is found (due to aliases), only the first is returned.
    BaseLink* findLink( const std::string &name ) const;

    /// Find link fields given a name: several can be found as we look into the alias map
    std::vector< BaseLink* > findLinks( const std::string &name ) const;

    /// Update pointers in case the pointed-to objects have appeared
    virtual void updateLinks(bool logErrors = true);

    /// Helper method used to initialize a data field containing a value of type T
    template<class T>
    BaseData::BaseInitData initData( Data<T>* field, const char* name, const char* help, bool isDisplayed=true, bool isReadOnly=false )
    {
        BaseData::BaseInitData res;
        this->initData0(field, res, name, help, isDisplayed, isReadOnly);
        return res;
    }

    /// Helper method used to initialize a data field containing a value of type T
    template<class T>
    typename Data<T>::InitData initData( Data<T>* field, const T& value, const char* name, const char* help, bool isDisplayed=true, bool isReadOnly=false  )
    {
        typename Data<T>::InitData res;
        this->initData0(field, res, value, name, help, isDisplayed, isReadOnly);
        return res;
    }

    /// Add a data field.
    /// Note that this method should only be called if the Data was not initialized with the initData method
    void addData(BaseData* f, const std::string& name);

    /// Add a data field.
    /// Note that this method should only be called if the Data was not initialized with the initData method
    void addData(BaseData* f);

    /// Remove a data field.
    void removeData(BaseData* f);


    /// Add an alias to a Data
    void addAlias( BaseData* field, const char* alias);

    /// Add a link.
    void addLink(BaseLink* l);

    /// Remove a link.
    void removeLink(BaseLink* l);

    /// Add an alias to a Link
    void addAlias( BaseLink* link, const char* alias);

    typedef helper::vector<BaseData*> VecData;
    typedef std::multimap<std::string, BaseData*> MapData;
    typedef helper::vector<BaseLink*> VecLink;
    typedef std::multimap<std::string, BaseLink*> MapLink;

    /// Accessor to the vector containing all the fields of this object
    const VecData& getDataFields() const { return m_vecData; }
    /// Accessor to the map containing all the aliases of this object
    const MapData& getDataAliases() const { return m_aliasData; }

    /// Accessor to the vector containing all the fields of this object
    const VecLink& getLinks() const { return m_vecLink; }
    /// Accessor to the map containing all the aliases of this object
    const MapLink& getLinkAliases() const { return m_aliasLink; }

    virtual bool findDataLinkDest(BaseData*& ptr, const std::string& path, const BaseLink* link);
    virtual void* findLinkDestClass(const BaseClass* destType, const std::string& path, const BaseLink* link);
    template<class T>
    bool findLinkDest(T*& ptr, const std::string& path, const BaseLink* link)
    {
        void* result = findLinkDestClass(T::GetClass(), path, link);
        ptr = reinterpret_cast<T*>(result);
        return (result != NULL);
    }

    virtual void copyAspect(int destAspect, int srcAspect);

    virtual void releaseAspect(int aspect);
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

public:

    /// Helper method to get the type name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::typeName(ptr); \endcode
    /// This way derived classes can redefine the typeName method
    template<class T>
    static std::string typeName(const T* ptr= NULL)
    {
        return BaseClass::defaultTypeName(ptr);
    }

    /// Helper method to get the class name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::className(ptr); \endcode
    /// This way derived classes can redefine the className method
    template<class T>
    static std::string className(const T* ptr= NULL)
    {
        return BaseClass::defaultClassName(ptr);
    }

    /// Helper method to get the namespace name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::namespaceName(ptr); \endcode
    /// This way derived classes can redefine the namespaceName method
    template<class T>
    static std::string namespaceName(const T* ptr= NULL)
    {
        return BaseClass::defaultNamespaceName(ptr);
    }

    /// Helper method to get the template name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::templateName(ptr); \endcode
    /// This way derived classes can redefine the templateName method
    template<class T>
    static std::string templateName(const T* ptr= NULL)
    {
        return BaseClass::defaultTemplateName(ptr);
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
    VecData m_vecData;
    /// name -> Data multi-map (includes names and aliases)
    MapData m_aliasData;

    /// List of links
    VecLink m_vecLink;
    /// name -> Link multi-map (includes names and aliases)
    MapLink m_aliasLink;

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
