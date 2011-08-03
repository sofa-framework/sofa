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
#ifndef SOFA_CORE_OBJECTMODEL_DATAFIELD_H
#define SOFA_CORE_OBJECTMODEL_DATAFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/helper/accessor.h>
#include <boost/shared_ptr.hpp>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief Abstract templated data, readable and writable from/to a string.
 *
 */
template < class T = void* >
class TData : public sofa::core::objectmodel::BaseData
{
public:
    typedef T value_type;

    explicit TData(const BaseInitData& init)
        : BaseData(init), parentData(NULL)
    {
    }

    TData( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, Base* owner=NULL, const char* name="")
        : BaseData(helpMsg, isDisplayed, isReadOnly, owner, name), parentData(NULL)
    {
    }

    virtual ~TData()
    {}

    inline void printValue(std::ostream& out) const;
    inline std::string getValueString() const;
    inline std::string getValueTypeString() const; // { return std::string(typeid(m_value).name()); }

    /// Get info about the value type of the associated variable
    virtual const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const
    {
        return sofa::defaulttype::VirtualTypeInfo<T>::get();
    }

    virtual const T& virtualGetValue() const = 0;
    virtual void virtualSetValue(const T& v) = 0;
    virtual void virtualSetLink(const BaseData& bd) = 0;
    virtual T* virtualBeginEdit() = 0;
    virtual void virtualEndEdit() = 0;

    /// Get current value as a void pointer (use getValueTypeInfo to find how to access it)
    virtual const void* getValueVoidPtr() const
    {
        return &(virtualGetValue());
    }

    /// Begin edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    virtual void* beginEditVoidPtr()
    {
        return virtualBeginEdit();
    }

    /// End edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    virtual void endEditVoidPtr()
    {
        virtualEndEdit();
    }

    /** Try to read argument value from an input stream.
    Return false if failed
     */
    virtual bool read( std::string& s )
    {
        if (s.empty())
            return false;
        //serr<<"Field::read "<<s.c_str()<<sendl;
        std::istringstream istr( s.c_str() );
        istr >> *virtualBeginEdit();
        virtualEndEdit();
        if( istr.fail() )
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    virtual bool isCounterValid() const {return true;}

    bool copyValue(const TData<T>* parent)
    {
        virtualSetValue(parent->virtualGetValue());
        return true;
    }

    virtual bool copyValue(const BaseData* parent)
    {
        const TData<T>* p = dynamic_cast<const TData<T>*>(parent);
        if (p)
        {
            virtualSetValue(p->virtualGetValue());
            return true;
        }
        return BaseData::copyValue(parent);
    }


    virtual bool validParent(BaseData* parent)
    {
        if (dynamic_cast<TData<T>*>(parent))
            return true;
        return BaseData::validParent(parent);
    }

protected:

    void doSetParent(BaseData* parent)
    {
        parentData = dynamic_cast<TData<T>*>(parent);
        BaseData::doSetParent(parent);
    }

    bool updateFromParentValue(const BaseData* parent)
    {
        if (parent == parentData)
        {
            //virtualSetValue(parentData->virtualGetValue());
            virtualSetLink(*parentData);
            return true;
        }
        else
            return BaseData::updateFromParentValue(parent);
    }

    TData<T>* parentData;
};

template <class T, bool COW>
class DataValue;

template <class T>
class DataValue<T, false>
{
    T data;
public:

    DataValue()
        : data(T())// BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    explicit DataValue(const T &value)
        : data(value)
    {
    }

    DataValue(const DataValue& dc)
        : data(dc.getValue())
    {
    }

    DataValue& operator=(const DataValue& dc )
    {
        data = dc.getValue();
        return *this;
    }

    T* beginEdit() { return &data; }
    void endEdit() {}
    const T& getValue() const { return data; }
    void setValue(const T& value)
    {
        data = value;
    }

//    T& value() { return data; }
};


template <class T>
class DataValue<T, true>
{
    boost::shared_ptr<T> ptr;
public:

    DataValue()
        : ptr(new T(T())) // BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    explicit DataValue(const T& value)
        : ptr(new T(value))
    {
    }

    DataValue(const DataValue& dc)
        : ptr(dc.ptr)
    {
    }

    ~DataValue()
    {
    }

    DataValue& operator=(const DataValue& dc )
    {
        //avoid self reference
        if(&dc != this)
        {
            ptr = dc.ptr;
        }

        return *this;
    }

    T* beginEdit()
    {
        if(!ptr.unique())
        {
            ptr.reset(new T(*ptr));
        }
        return ptr.get();
    }

    void endEdit()
    {
    }

    const T& getValue() const
    {
        return *ptr;
    }

    void setValue(const T& value)
    {
        if(!ptr.unique())
        {
            ptr.reset(new T(value));
        }
        else
        {
            *ptr = value;
        }
    }

    void release()
    {
        ptr.reset();
    }

//    T& value()
//    {
//        return *beginEdit();
//    }
};



/**
 *  \brief Container of data, readable and writable from/to a string.
 *
 */
template < class T = void* >
class Data : public TData<T>
{
public:

    /// @name Construction / destruction
    /// @{

    /// This internal class is used by the initData() methods to store initialization parameters of a Data
    class InitData : public BaseData::BaseInitData
    {
    public:
        InitData() : value(T()) {}
        InitData(const T& v) : value(v) {}
        InitData(const BaseData::BaseInitData& i) : BaseData::BaseInitData(i), value(T()) {}

        T value;
    };

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit Data(const BaseData::BaseInitData& init)
        : TData<T>(init)
        , shared(NULL)
    {
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit Data(const InitData& init)
        : TData<T>(init)
        , m_value(init.value)
        , shared(NULL)
    {
    }

    /** Constructor
    \param helpMsg help on the field
     */
    Data( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, Base* owner=NULL, const char* name="")
        : TData<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
        , m_value(T())// BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
        , shared(NULL)
    {

    }

    /** Constructor
    \param value default value
    \param helpMsg help on the field
     */
    Data( const T& value, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, Base* owner=NULL, const char* name="")
        : TData<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
        , m_value(value)
        , shared(NULL)
    {
    }

    virtual ~Data()
    {}

    /// @}

    /// @name Simple edition and retrieval API
    /// @{

    inline T* beginEdit()
    {
        this->updateIfDirty();
        ++this->m_counter;
        this->m_isSet = true;
        BaseData::setDirtyOutputs();
        return m_value.beginEdit();
    }

    inline void endEdit()
    {
        m_value.endEdit();
    }

    inline void setValue(const T& value )
    {
        *beginEdit()=value;
        endEdit();
    }

    inline const T& getValue() const
    {
        this->updateIfDirty();
        return m_value.getValue();
    }

    /// @}

    /// @name Optimized edition and retrieval API (for multi-threading performances)
    /// @{

    inline T* beginEdit(const core::ExecParams* /*params*/)
    {
        return beginEdit();
    }

    inline void endEdit(const core::ExecParams* /*params*/)
    {
        endEdit();
    }

    inline void setValue(const core::ExecParams* /*params*/, const T& value)
    {
        setValue(value);
    }

    inline const T& getValue(const core::ExecParams* /*params*/) const
    {
        return getValue();
    }

    /// @}

    /// @name Virtual edition and retrieval API (for generic TData parent API, deprecated)
    /// @{

    virtual const T& virtualGetValue() const { return getValue(); }
    virtual void virtualSetValue(const T& v) { setValue(v); }

    virtual void virtualSetLink(const BaseData& bd)
    {
        const Data<T>* d = dynamic_cast< const Data<T>* >(&bd);
        if (d)
            this->m_value = d->m_value;
    }

    virtual T* virtualBeginEdit() { return beginEdit(); }
    virtual void virtualEndEdit() { endEdit(); }


    /// @}

    inline friend std::ostream & operator << (std::ostream &out, const Data& df)
    {
        out<<df.getValue();
        return out;
    }

    inline bool operator ==( const T& value ) const
    {
        return getValue()==value;
    }

    inline bool operator !=( const T& value ) const
    {
        return getValue()!=value;
    }

    inline void operator =( const T& value )
    {
        this->setValue(value);
    }

protected:

    /// Value
    //T m_value;
    DataValue<T, sofa::defaulttype::DataTypeInfo<T>::CopyOnWrite> m_value;
    //DataValue<T, false> m_value;
    //DataValue<T, true> m_value;
public:
    mutable void* shared;

private:
    Data(const Data& );
    Data& operator=(const Data& );
};

#if defined(WIN32) && !defined(SOFA_CORE_OBJECTMODEL_DATA_CPP)

extern template class SOFA_CORE_API TData< std::string >;
extern template class SOFA_CORE_API Data< std::string >;
extern template class SOFA_CORE_API TData< bool >;
extern template class SOFA_CORE_API Data< bool >;

#endif

/// Specialization for reading strings
template<>
bool TData<std::string>::read( std::string& str );


/// Specialization for reading booleans
template<>
bool TData<bool>::read( std::string& str );


/// General case for printing default value
template<class T>
inline
void TData<T>::printValue( std::ostream& out=std::cout ) const
{
    out << virtualGetValue() << " ";
}

/// General case for printing default value
template<class T>
inline
std::string TData<T>::getValueString() const
{
    std::ostringstream out;
    out << virtualGetValue();
    return out.str();
}

template<class T>
inline
std::string TData<T>::getValueTypeString() const
{
    return BaseData::typeName(&virtualGetValue());
}


} // namespace objectmodel

} // namespace core

// Overload helper::ReadAccessor and helper::WriteAccessor

namespace helper
{

template<class T>
class ReadAccessor< core::objectmodel::Data<T> > : public ReadAccessor<T>
{
public:
    typedef ReadAccessor<T> Inherit;
    typedef core::objectmodel::Data<T> data_container_type;
    typedef T container_type;

protected:
    const data_container_type& data;
public:
    ReadAccessor(const data_container_type& d) : Inherit(d.getValue()), data(d) {}
    ReadAccessor(const data_container_type* d) : Inherit(d->getValue()), data(*d) {}
    ReadAccessor(const core::ExecParams* params, const data_container_type& d) : Inherit(d.getValue(params)), data(d) {}
    ReadAccessor(const core::ExecParams* params, const data_container_type* d) : Inherit(d->getValue(params)), data(*d) {}
    ~ReadAccessor() {}
};

template<class T>
class WriteAccessor< core::objectmodel::Data<T> > : public WriteAccessor<T>
{
public:
    typedef WriteAccessor<T> Inherit;
    typedef core::objectmodel::Data<T> data_container_type;
    typedef T container_type;

protected:
    data_container_type& data;
    const core::ExecParams* dparams;

public:
    WriteAccessor(data_container_type& d) : Inherit(*d.beginEdit()), data(d), dparams(NULL) {}
    WriteAccessor(data_container_type* d) : Inherit(*d->beginEdit()), data(*d), dparams(NULL) {}
    WriteAccessor(const core::ExecParams* params, data_container_type& d) : Inherit(*d.beginEdit(params)), data(d), dparams(params) {}
    WriteAccessor(const core::ExecParams* params, data_container_type* d) : Inherit(*d->beginEdit(params)), data(*d), dparams(params) {}
    ~WriteAccessor() { if (dparams) data.endEdit(dparams); else data.endEdit(); }
};

} // namespace helper

} // namespace sofa

#endif
