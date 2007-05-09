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

#include <sofa/core/objectmodel/FieldBase.h>
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
class DataField : public sofa::core::objectmodel::FieldBase
{
public:

    /** Constructor
    \param helpMsg help on the field
     */
    DataField( const char* helpMsg=0 )
        : FieldBase(helpMsg)
    {}

    /** Constructor
    \param value default value
    \param helpMsg help on the field
     */
    DataField( const T& value, const char* helpMsg=0 )
        : FieldBase(helpMsg)
        , m_value(value)
    {}

    virtual ~DataField()
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

protected:
    /// Value
    T m_value;


    /** Try to read argument value from an input stream.
        Return false if failed
    */
    inline bool read( std::string& s )
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
};

/// Specialization for reading strings
template<>
inline
bool DataField<std::string>::read( std::string& str )
{
    m_value = str;
    m_isSet = true;
    return true;
}

/// General case for printing default value
template<class T>
inline
void DataField<T>::printValue( std::ostream& out=std::cout ) const
{
    out << m_value << " ";
}

/// General case for printing default value
template<class T>
inline
std::string DataField<T>::getValueString() const
{
    std::ostringstream out;
    out << m_value;
    return out.str();
}

template<class T>
inline
std::string DataField<T>::getValueTypeString() const
{
    return FieldBase::typeName(&m_value);
}


} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
