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
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Base;

/**
 *  \brief Abstract base class for all fields, independently of their type.
 *
 */
class SOFA_CORE_API BaseData : public DDGNode
{
public:

    /// This internal class is used by the initData() methods to store initialization parameters of a Data
    class BaseInitData
    {
    public:
        BaseInitData() : data(NULL), helpMsg(""), isDisplayed(true), isReadOnly(false), isPersistent(true), owner(NULL), name(""), parentClass(""), group(""), widget("") {}
        BaseData* data;
        const char* helpMsg;
        bool isDisplayed;
        bool isReadOnly;
        bool isPersistent;
        Base* owner;
        const char* name;
        const char* parentClass;
        const char* group;
        const char* widget;
    };

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit BaseData(const BaseInitData& init);

    /** Constructor
     *  \param h help
     */
    BaseData( const char* h, bool isDisplayed=true, bool isReadOnly=false, Base* owner=NULL, const char* name="");

    /// Base destructor
    virtual ~BaseData();

    /// Read the command line
    virtual bool read( std::string& str ) = 0;

    /// Print the value of the associated variable
    virtual void printValue( std::ostream& ) const =0;

    /// Print the value of the associated variable
    virtual std::string getValueString() const=0;

    /// Print the value type of the associated variable
    virtual std::string getValueTypeString() const=0;

    /// Get info about the value type of the associated variable
    virtual const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const=0;

    /// Get current value as a void pointer (use getValueTypeInfo to find how to access it)
    virtual const void* getValueVoidPtr() const=0;

    /// Begin edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    virtual void* beginEditVoidPtr()=0;

    /// End edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    virtual void endEditVoidPtr()=0;

    /// Copy the value of another Data.
    /// Note that this is a one-time copy and not a permanent link (otherwise see setParent)
    /// @return true if copy was successfull
    virtual bool copyValue(const BaseData* parent);

    /// Get help message
    const char* getHelp() const { return help; }

    /// Set help message
    void setHelp(const char* val) { help = val; }

    /// @deprecated Set help message
    void setHelpMsg(const char* val) { help = val; }

    /// Get parentClass
    const char* getParentClass() const { return parentClass; }

    /// Set group
    void setParentClass(const char* val) { parentClass = val; }

    /// Get group
    const char* getGroup() const { return group; }

    /// Set group
    void setGroup(const char* val) { group = val; }

    /// Get widget
    const char* getWidget() const { return widget; }

    /// Set widget
    void setWidget(const char* val) { widget = val; }

    /// True if the value has been modified
    /// If this data is linked, the value of this data will be considered as modified
    /// (even if the parent's value has not been modified)
    bool isSet() const { return m_isSet; }

    /// True if the Data has to be displayed in the GUI
    bool isDisplayed() const { return m_isDisplayed; }

    /// True if the Data will be readable only in the GUI
    bool isReadOnly() const { return m_isReadOnly; }

    /// True if the Data contain persistent information
    bool isPersistent() const { return m_isPersistent; }

    /// True if the counter of modification gives valid information.
    virtual bool isCounterValid() const =0;

    /// Reset the isSet flag to false, to indicate that the current value is the default for this Data.
    void unset() { m_isSet = false; }

    /// Reset the isSet flag to true, to indicate that the current value has been modified.
    void forceSet() { m_isSet = true; }

    /// Can dynamically change the status of a Data, by making it appear or disappear
    void setDisplayed(bool b) {m_isDisplayed = b;}
    /// Can dynamically change the status of a Data, by making it readOnly
    void setReadOnly(bool b) {m_isReadOnly = b;}
    /// Can dynamically change the status of a Data, by making it persistent
    void setPersistent(bool b) {m_isPersistent = b;}
    /// If we use the Data as a link and not as value directly
    void setLinkPath(const std::string &path) {m_linkPath = path;};
    std::string getLinkPath() const {return m_linkPath;};
    /// Can this data be used as a linkPath
    /// True by default.
    /// Useful if you want to customize the use of @ syntax (see ObjectRef and DataObjectRef)
    virtual bool canBeLinked() const { return true; }

    /// Return the Base component owning this Data
    Base* getOwner() const { return m_owner; }
    void setOwner(Base* o) { m_owner=o; }

    /// Return the name of this Data within the Base component
    const std::string& getName() const { return m_name; }
    /// Set the name of this Data. Not that this methods should not be called directly, but the Data registration methods in Base should be used instead
    void setName(const std::string& name) { m_name=name; }

    /// Return the number of changes since creation
    /// This can be used to efficiently detect changes
    int getCounter() const { return m_counter; }

    /// Link to a parent data. The value of this data will automatically duplicate the value of the parent data.
    virtual bool setParent(BaseData* parent);

    /// Check if a given Data can be linked as a parent of this data
    virtual bool validParent(BaseData* parent);

    BaseData* getParent() const { return parentBaseData; }

    /// Update the value of this Data
    void update();

protected:

    virtual void doSetParent(BaseData* parent);

    virtual void doDelInput(DDGNode* n);

    /// Update this Data from the value of its parent
    virtual bool updateFromParentValue(const BaseData* parent);

    /// Help message
    const char* help;
    /// parent class
    const char* parentClass;
    /// group
    const char* group;
    /// widget
    const char* widget;
    /// Number of changes since creation
    int m_counter;
    /// True if the Data is set, i.e. its value is different from the default value
    bool m_isSet;
    /// True if the Data will be displayed in the GUI
    bool m_isDisplayed;
    /// True if the Data will be readable only in the GUI
    bool m_isReadOnly;
    /// True if the Data contain persistent information
    bool m_isPersistent;
    /// Return the Base component owning this Data
    Base* m_owner;
    /// Data name within the Base component
    std::string m_name;
    /// Link to another Data, if used as an input from another Data (@ typo).
    std::string m_linkPath;
    /// Parent Data
    BaseData* parentBaseData;

    /// Helper method to decode the type name to a more readable form if possible
    static std::string decodeTypeName(const std::type_info& t);

    /// Helper method to get the type name of type T
    template<class T>
    static std::string typeName(const T* = NULL)
    {
        if (defaulttype::DataTypeInfo<T>::ValidInfo)
            return defaulttype::DataTypeName<T>::name();
        else
            return decodeTypeName(typeid(T));
    }
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
