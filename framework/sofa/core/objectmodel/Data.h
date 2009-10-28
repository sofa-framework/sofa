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

    virtual ~Data()
    {}

    inline T* beginEdit()
    {
        this->updateIfDirty();
        ++this->m_counter;
        BaseData::setDirtyOutputs();
        return &m_value;
    }
    inline void endEdit()
    {
    }
    inline void setValue(const T& value )
    {
        *beginEdit()=value;
        endEdit();
    }
    inline const T& getValue() const
    {
        this->updateIfDirty();
        return m_value;
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
    T m_value;
    const T& value() const
    {
        this->updateIfDirty();
        return m_value;
    }

    T& value()
    {
        this->updateIfDirty();
        return m_value;
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
class ReadAccessor< core::objectmodel::Data<T> >
{
public:
    typedef core::objectmodel::Data<T> data_container_type;
    typedef T container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    const data_container_type& data;
    const container_type& ref;
public:
    ReadAccessor(const data_container_type& d) : data(d), ref(d.getValue()) {}
    ~ReadAccessor() {}

    size_type size() const { return ref.size(); }
    bool empty() const { return ref.empty(); }

    const_reference operator[](size_type i) const { return ref[i]; }

    const_iterator begin() const { return ref.begin(); }
    const_iterator end() const { return ref.end(); }

    inline friend std::ostream& operator<< ( std::ostream& os, const ReadAccessor<data_container_type>& vec )
    {
        return os << vec.ref;
    }
};

template<class T>
class WriteAccessor< core::objectmodel::Data<T> >
{
public:
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
    container_type& ref;

public:
    WriteAccessor(data_container_type& d) : data(d), ref(*d.beginEdit()) {}
    ~WriteAccessor() { data.endEdit(); }

    size_type size() const { return ref.size(); }
    bool empty() const { return ref.empty(); }

    const_reference operator[](size_type i) const { return ref[i]; }
    reference operator[](size_type i) { return ref[i]; }

    const_iterator begin() const { return ref.begin(); }
    iterator begin() { return ref.begin(); }
    const_iterator end() const { return ref.end(); }
    iterator end() { return ref.end(); }

    void clear() { ref.clear(); }
    void resize(size_type s, bool /*init*/ = true) { /*if (init)*/ ref.resize(s); /*else ref.fastResize(s);*/ }
    void reserve(size_type s) { ref.reserve(s); }
    void push_back(const_reference v) { ref.push_back(v); }

    inline friend std::ostream& operator<< ( std::ostream& os, const WriteAccessor<data_container_type>& vec )
    {
        return os << vec.ref;
    }

    inline friend std::istream& operator>> ( std::istream& in, WriteAccessor<data_container_type>& vec )
    {
        return in >> vec.ref;
    }

};

} // namespace helper

} // namespace sofa

#endif
