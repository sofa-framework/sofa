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
#ifndef SOFA_CORE_OBJECTMODEL_BASEDATA_H
#define SOFA_CORE_OBJECTMODEL_BASEDATA_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <list>
#include <iostream>
#include <typeinfo>
#include <sofa/core/core.h>
#include <sofa/core/objectmodel/DDGNode.h>

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
class SOFA_CORE_API BaseData : public DDGNode
{
public:
    /** Constructor
     *  \param l long name
     *  \param h help
     *  \param m true iff the argument is mandatory
     */
    BaseData( const char* h, bool isDisplayed=true, bool isReadOnly=false )
        : help(h), group(""), widget("")
        , m_counter(0), m_isDisplayed(isDisplayed), m_isReadOnly(isReadOnly), m_isPersistent(true)/*, parent(NULL), writer(NULL)*/
    {}

    /// Base destructor
    virtual ~BaseData()
    {
    }

    /// Read the command line
    virtual bool read( std::string& str ) = 0;

    /// Print the value of the associated variable
    virtual void printValue( std::ostream& ) const =0;

    /// Print the value of the associated variable
    virtual std::string getValueString() const=0;

    /// Print the value type of the associated variable
    virtual std::string getValueTypeString() const=0;

    /// Get help message
    const char* getHelp() const { return help; }

    /// Set help message
    void setHelp(const char* val) { help = val; }

    /// @deprecated Set help message
    void setHelpMsg(const char* val) { help = val; }

    /// Get group
    const char* getGroup() const { return group; }

    /// Set group
    void setGroup(const char* val) { group = val; }

    /// Get widget
    const char* getWidget() const { return widget; }

    /// Set widget
    void setWidget(const char* val) { widget = val; }

    /// True if the value has been modified
    bool isSet() const { return m_counter > 0; }

    /// True if the Data has to be displayed in the GUI
    bool isDisplayed() const { return m_isDisplayed; }

    /// True if the Data will be readable only in the GUI
    bool isReadOnly() const { return m_isReadOnly; }

    /// True if the Data contain persistent information
    bool isPersistent() const { return m_isPersistent; }

    /// True if the counter of modification gives valid information. It is the case for Data, but not for DataPtr
    virtual bool isCounterValid() const =0;

    /// Can dynamically change the status of a Data, by making it appear or disappear
    void setDisplayed(bool b) {m_isDisplayed = b;}
    /// Can dynamically change the status of a Data, by making it readOnly
    void setReadOnly(bool b) {m_isReadOnly = b;}
    /// Can dynamically change the status of a Data, by making it persistent
    void setPersistent(bool b) {m_isPersistent = b;}

    /// Return the number of changes since creation
    /// This can be used to efficiently detect changes
    int getCounter() const { return m_counter; }

    /// Set for this Data the value of its parent value
    virtual bool setParentValue(BaseData* parent) = 0;

    /// Update the value of this Data
    void update()
    {
        dirty = false;
        for(std::list<DDGNode*>::iterator it=inputs.begin(); it!=inputs.end(); ++it)
        {
            if ((*it)->isDirty())
            {
                (*it)->update();
            }
            if (updateFromParentValue(dynamic_cast<BaseData*>(*it)))
                break;
        }
    }

protected:

    /// Update this Data from the value of its parent
    virtual bool updateFromParentValue(BaseData* parent) = 0;

    /// Help message
    const char* help;
    /// group
    const char* group;
    /// widget
    const char* widget;
    /// Number of changes since creation
    int m_counter;
    /// True if the Data will be displayed in the GUI
    bool m_isDisplayed;
    /// True if the Data will be readable only in the GUI
    bool m_isReadOnly;
    /// True if the Data contain persistent information
    bool m_isPersistent;
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
