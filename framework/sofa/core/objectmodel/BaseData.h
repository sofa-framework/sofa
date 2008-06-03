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
#ifndef SOFA_CORE_OBJECTMODEL_BASEDATA_H
#define SOFA_CORE_OBJECTMODEL_BASEDATA_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <list>
#include <iostream>
#include <typeinfo>
namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief Abstract base class for all fields, independently of their type.
 *
 */
class BaseData
{
public:
    /** Constructor
     *  \param l long name
     *  \param h help
     *  \param m true iff the argument is mandatory
     */
    BaseData( const char* h)
        : help(h)
        , m_isSet(false), m_isDisplayed(true)
    {}

    /// Base destructor: does nothing.
    virtual ~BaseData() {}

    /// Read the command line
    virtual bool read( std::string& str ) = 0;

    /// Print the value of the associated variable
    virtual void printValue( std::ostream& ) const =0;

    /// Print the value of the associated variable
    virtual std::string getValueString() const=0;

    /// Print the value type of the associated variable
    virtual std::string getValueTypeString() const=0;

    /// Help message
    const char* help;

    /// True if the value has been modified
    inline bool isSet() const { return m_isSet; }

    /// True if the Data has to be displayed in the GUI
    inline bool isDisplayed() const { return m_isDisplayed; }

    /// Can dynamically change the status of a Data, by making it appear or disappear
    void setDisplayed(bool b) {m_isDisplayed = b;}
protected:
    /// True if a value has been read on the command line
    bool m_isSet;
    /// True if the Data will be displayed in GUI
    bool m_isDisplayed;

    /// Helper method to decode the type name to a more readable form if possible
    static std::string decodeTypeName(const std::type_info& t);

    /// Helper method to get the type name of type T
    template<class T>
    static std::string typeName(const T* = NULL)
    {
        return decodeTypeName(typeid(T));
    }
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
