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

    const T& virtualGetValue() const
    {
        this->updateIfDirty();
        return value();
    }

    void virtualSetValue(const T& v)
    {
        ++this->m_counter;
        value() = v;
        BaseData::setDirtyOutputs();
    }

    virtual T* virtualBeginEdit()
    {
        this->updateIfDirty();
        ++this->m_counter;
        BaseData::setDirtyOutputs();
        return &(value());
    }

    virtual void virtualEndEdit()
    {
    }

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
        istr >> value();
        if( istr.fail() )
        {
            return false;
        }
        else
        {
            ++this->m_counter;
            BaseData::setDirtyOutputs();
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

protected:

    bool validParent(BaseData* parent)
    {
        if (dynamic_cast<TData<T>*>(parent))
            return true;
        return BaseData::validParent(parent);
    }

    void doSetParent(BaseData* parent)
    {
        parentData = dynamic_cast<TData<T>*>(parent);
        BaseData::doSetParent(parent);
    }

    bool updateFromParentValue(const BaseData* parent)
    {
        if (parent == parentData)
        {
            value() = parentData->value();
            ++this->m_counter;
            return true;
        }
        else
            return BaseData::updateFromParentValue(parent);
    }

    virtual const T& value() const = 0;
    virtual T& value() = 0;

    TData<T>* parentData;
};

template <class T, bool COW>
class DataContainer
{
public:
    DataContainer() {}

    virtual ~DataContainer() {std::cout << "destry datacontainer" << std::endl;}

    T* beginEdit() {return NULL; }
    void endEdit() {};
    const T& value() const { return T(); }
    T& value()  { return T(); }

};

template <class T>
class DataContainer<T, false>
{
protected:
    T data;
public:

    DataContainer(const DataContainer<T, false>& dc)
    {
        data = dc.value();
    }

    DataContainer(const T &value)
        : data(value)
    {

    }

    DataContainer<T, false>& operator=(const DataContainer<T, false>& dc )
    {
        data = dc.value();
        return *this;
    }

    T* beginEdit() { return &data; }
    void endEdit() {};
    const T& value() const { return data; }
    T& value() { return data; }
};


template <class T>
class DataContainer<T, true>
{
    //TODO: change this to be atomic
    typedef unsigned int Counter;

protected:
    T* data;
    Counter* cpt;
public:
    DataContainer()
    {
        cpt = new Counter;
        *cpt = 1;
        data = new T;
    }

    DataContainer(const T& value)
    {
        cpt = new Counter;
        *cpt = 1;
        data = new T;
        *data = value;
    }

    DataContainer(const DataContainer& dc)
    {
        this->data = dc.data;
        this->cpt = dc.cpt;

        this->cpt++;
    }

    ~DataContainer()
    {
        if(*cpt < 1)
        {
            delete cpt;
            delete data;
        }
        else *cpt--;
    }

    DataContainer<T, true>& operator=(const DataContainer<T, true>& dc )
    {
        //avoid self reference
        if(&dc != this)
        {
            if(this->data && this->cpt && *(this->cpt) < 1)
            {
                delete this->data;
                delete this->cpt;
            }
            this->data = dc.data;
            this->cpt = dc.cpt;

            *cpt++;
        }

        return *this;
    }

    T* beginEdit()
    {
        if (*cpt > 1)
        {
            *cpt--;

            cpt = new Counter;
            *cpt = 1;
            T* newData;
            newData = new T;
            std::copy(data, data+sizeof(T),newData);
            data = newData;
        }
        return data;
    }

    void endEdit()
    {

    };

    const T& value() const
    {
        return *data;
    }

    T& value()
    {
        return *beginEdit();
    }
};



/**
 *  \brief Container of data, readable and writable from/to a string.
 *
 */
template < class T = void* >
class Data : public TData<T>
{
public:

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
        , m_value(T())// BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit Data(const InitData& init)
        : TData<T>(init)
        , m_value(init.value)
    {
    }

    /** Constructor
    \param helpMsg help on the field
     */
    Data( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, Base* owner=NULL, const char* name="")
        : TData<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
        , m_value(T())// BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {

    }

    /** Constructor
    \param value default value
    \param helpMsg help on the field
     */
    Data( const T& value, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, Base* owner=NULL, const char* name="")
        : TData<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
        , m_value(value)
    {
    }

    Data(const Data& d)
        : TData<T>()
        ,m_value(d.getValue())
    {
    }

    virtual ~Data()
    {}

    inline T* beginEdit()
    {
        this->updateIfDirty();
        ++this->m_counter;
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
        return m_value.value();
    }

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
    DataContainer<T, sofa::defaulttype::DataTypeInfo<T>::CopyOnWrite> m_value;
    //DataContainer<T, false> m_value;

    const T& value() const
    {
        this->updateIfDirty();

        return m_value.value();
    }

    T& value()
    {
        this->updateIfDirty();
        return m_value.value();
    }
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
    out << value() << " ";
}

/// General case for printing default value
template<class T>
inline
std::string TData<T>::getValueString() const
{
    std::ostringstream out;
    out << value();
    return out.str();
}

template<class T>
inline
std::string TData<T>::getValueTypeString() const
{
    return BaseData::typeName(&value());
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
    ~ReadAccessor() {}
};

template<class T>
class WriteAccessor< core::objectmodel::Data<T> > : public WriteAccessor<T>
{
public:
    typedef WriteAccessor<T> Inherit;
    typedef core::objectmodel::Data<T> data_container_type;
    typedef T container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    data_container_type& data;

public:
    WriteAccessor(data_container_type& d) : Inherit(*d.beginEdit()), data(d) {}
    ~WriteAccessor() { data.endEdit(); }
};

} // namespace helper

} // namespace sofa

#endif
