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
#ifndef SOFA_CORE_OBJECTMODEL_DATAFIELD_H
#define SOFA_CORE_OBJECTMODEL_DATAFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/objectmodel/BaseData.h>
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
 *  \brief Pointer to data, readable and writable from/to a string.
 *
 */
template < class T = void* >
class Data : public sofa::core::objectmodel::BaseData
{
public:

    /** Constructor
    \param helpMsg help on the field
     */
    Data( const char* helpMsg=0, bool isDisplayed=true )
        : BaseData(helpMsg)
        , m_value(T()) // BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
        m_isDisplayed = isDisplayed;
    }

    /** Constructor
    \param value default value
    \param helpMsg help on the field
     */
    Data( const T& value, const char* helpMsg=0, bool isDisplayed=true  )
        : BaseData(helpMsg)
        , m_value(value)
    {
        m_isDisplayed = isDisplayed;
    }

    virtual ~Data()
    {}

    inline void setHelpMsg( const char* msg ) { this->help = msg; }
    inline void printValue(std::ostream& out) const ;
    inline std::string getValueString() const ;
    inline std::string getValueTypeString() const; // { return std::string(typeid(m_value).name()); }
    inline T* beginEdit()
    {
        m_isSet = true;
        return &m_value;
    }
    inline void endEdit()
    {}
    inline void setValue(const T& value )
    {
        *beginEdit()=value;
        endEdit();
    }
    inline const T& getValue() const
    {
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

    /** Try to read argument value from an input stream.
    Return false if failed
     */
    virtual bool read( std::string& s )
    {
        if (s.empty())
            return false;
        //std::cerr<<"Field::read "<<s.c_str()<<std::endl;
        std::istringstream istr( s.c_str() );
        istr >> m_value;
        if( istr.fail() )
        {
            return false;
        }
        else
        {
            m_isSet = true;
            return true;
        }
    }
protected:
    /// Value
    T m_value;


};

/// Specialization for reading strings
template<>
inline
bool Data<std::string>::read( std::string& str )
{
    m_value = str;
    m_isSet = true;
    return true;
}

/// Specialization for reading booleans
template<>
inline
bool Data<bool>::read( std::string& str )
{
    if (str.empty())
        return false;
    if (str[0] == 'T' || str[0] == 't')
        m_value = true;
    else if (str[0] == 'F' || str[0] == 'f')
        m_value = false;
    else if ((str[0] >= '0' && str[0] <= '9') || str[0] == '-')
        m_value = (atoi(str.c_str()) != 0);
    else return false;
    m_isSet = true;
    return true;
}

/// General case for printing default value
template<class T>
inline
void Data<T>::printValue( std::ostream& out=std::cout ) const
{
    out << m_value << " ";
}

/// General case for printing default value
template<class T>
inline
std::string Data<T>::getValueString() const
{
    std::ostringstream out;
    out << m_value;
    return out.str();
}

template<class T>
inline
std::string Data<T>::getValueTypeString() const
{
    return BaseData::typeName(&m_value);
}


} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
