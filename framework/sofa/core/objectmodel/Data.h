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

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/core.h>
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

    TData( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false )
        : BaseData(helpMsg, isDisplayed, isReadOnly)
    {
    }

    virtual ~TData()
    {}

    inline void printValue(std::ostream& out) const;
    inline std::string getValueString() const;
    inline std::string getValueTypeString() const; // { return std::string(typeid(m_value).name()); }

    inline bool setParentValue(BaseData* parent)
    {
        updateFromParentValue(parent);
        BaseData::setDirtyValue();
        return true;
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

    virtual bool isCounterValid() const {return true;};
protected:


    inline bool updateFromParentValue(BaseData* parent)
    {
        TData<T>* tData = dynamic_cast< TData<T>* >(parent);
        if (tData)
        {
            value() = tData->value();
            ++this->m_counter;
            return true;
        }
        return false;
    }

    virtual const T& value() const = 0;
    virtual T& value() = 0;
};

/**
 *  \brief Container of data, readable and writable from/to a string.
 *
 */
template < class T = void* >
class Data : public TData<T>
{
public:

    /** Constructor
    \param helpMsg help on the field
     */
    Data( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false )
        : TData<T>(helpMsg, isDisplayed, isReadOnly)
        , m_value(T())// BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    /** Constructor
    \param value default value
    \param helpMsg help on the field
     */
    Data( const T& value, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false  )
        : TData<T>(helpMsg, isDisplayed, isReadOnly)
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

} // namespace sofa

#endif
