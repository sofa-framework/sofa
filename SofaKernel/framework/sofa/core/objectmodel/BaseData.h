/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
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
#include <sofa/core/ExecParams.h>
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Base;
class BaseData;

/**
 *  \brief Abstract base class for Data.
 *
 */
class SOFA_CORE_API BaseData : public DDGNode
{
public:
    /// Flags that describe some properties of a Data, and that can be OR'd together.
    /// \todo Probably remove FLAG_PERSISTENT, FLAG_ANIMATION_INSTANCE, FLAG_VISUAL_INSTANCE and FLAG_HAPTICS_INSTANCE, it looks like they are not used anywhere.
    enum DataFlagsEnum
    {
        FLAG_NONE       = 0,      ///< Means "no flag" when a value is required.
        FLAG_READONLY   = 1 << 0, ///< The Data will be read-only in GUIs.
        FLAG_DISPLAYED  = 1 << 1, ///< The Data will be displayed in GUIs.
        FLAG_PERSISTENT = 1 << 2, ///< The Data contains persistent information.
        FLAG_AUTOLINK   = 1 << 3, ///< The Data should be autolinked when using the src="..." syntax.
        FLAG_REQUIRED = 1 << 4, ///< True if the Data has to be set for the owner component to be valid (a warning is displayed at init otherwise) 
        FLAG_ANIMATION_INSTANCE = 1 << 10,
        FLAG_VISUAL_INSTANCE = 1 << 11,
        FLAG_HAPTICS_INSTANCE = 1 << 12,
    };
    /// Bit field that holds flags value.
    typedef unsigned DataFlags;

    /// Default value used for flags.
    enum { FLAG_DEFAULT = FLAG_DISPLAYED | FLAG_PERSISTENT | FLAG_AUTOLINK };

    /// @name Class reflection system
    /// @{
    typedef TClass<BaseData,DDGNode> MyClass;
    static const MyClass* GetClass() { return MyClass::get(); }
    virtual const BaseClass* getClass() const
    { return GetClass(); }
    /// @}

    /// This internal class is used by the initData() methods to store initialization parameters of a Data
    class BaseInitData
    {
    public:
        BaseInitData() : data(NULL), helpMsg(""), dataFlags(FLAG_DEFAULT), owner(NULL), name(""), ownerClass(""), group(""), widget("") {}
        BaseData* data;
        const char* helpMsg;
        DataFlags dataFlags;
        Base* owner;
        const char* name;
        const char* ownerClass;
        const char* group;
        const char* widget;
    };

    /** Constructor used via the Base::initData() methods. */
    explicit BaseData(const BaseInitData& init);

    /** Constructor.
     *  \param helpMsg A help message that describes this %Data.
     *  \param flags The flags for this %Data (see \ref DataFlagsEnum).
     */
    BaseData(const char* helpMsg, DataFlags flags = FLAG_DEFAULT);

    /** Constructor.
     *  \param helpMsg A help message that describes this %Data.
     *  \param isDisplayed Whether this %Data should be displayed in GUIs.
     *  \param isReadOnly Whether this %Data should be modifiable in GUIs.
     */
    BaseData(const char* helpMsg, bool isDisplayed=true, bool isReadOnly=false);

    /// Destructor.
    virtual ~BaseData();

    /// Assign a value to this %Data from a string representation.
    /// \return true on success.
    virtual bool read(const std::string& value) = 0;

    /// Print the value of this %Data to a stream.
    virtual void printValue(std::ostream&) const = 0;

    /// Get a string representation of the value held in this %Data.
    virtual std::string getValueString() const = 0;

    /// Get the name of the type of the value held in this %Data.
    virtual std::string getValueTypeString() const = 0;

    /// Get the TypeInfo for the type of the value held in this %Data.
    ///
    /// This can be used to access the content of the %Data generically, without
    /// knowing its type.
    /// \see sofa::defaulttype::AbstractTypeInfo
    virtual const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const = 0;

    /// Get a constant void pointer to the value held in this %Data, to be used with AbstractTypeInfo.
    ///
    /// This pointer should be used via the instance of AbstractTypeInfo
    /// returned by getValueTypeInfo().
    virtual const void* getValueVoidPtr() const = 0;

    /// Get a void pointer to the value held in this %Data, to be used with AbstractTypeInfo.
    ///
    /// This pointer should be used via the instance of AbstractTypeInfo
    /// returned by getValueTypeInfo().
    /// \warning You must call endEditVoidPtr() once you're done modifying the value.
    virtual void* beginEditVoidPtr() = 0;

    /// Must be called after beginEditVoidPtr(), after you are finished modifying this %Data.
    virtual void endEditVoidPtr() = 0;

    /// Copy the value from another Data.
    ///
    /// Note that this is a one-time copy and not a permanent link (otherwise see setParent())
    /// @return true if the copy was successful.
    virtual bool copyValue(const BaseData* parent);

    /// Copy the value of an aspect into another one.
    virtual void copyAspect(int destAspect, int srcAspect) = 0;

    /// Release memory allocated for the specified aspect.
    virtual void releaseAspect(int aspect) = 0;

    /// Get a help message that describes this %Data.
    const char* getHelp() const { return help; }

    /// Set the help message.
    void setHelp(const char* val) { help = val; }

    /// Get owner class
    const char* getOwnerClass() const { return ownerClass; }

    /// Set owner class
    void setOwnerClass(const char* val) { ownerClass = val; }

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
    bool isSet() const { return m_isSets[currentAspect()]; }

    /// True if the counter of modification gives valid information.
    virtual bool isCounterValid() const = 0;

    /// Reset the isSet flag to false, to indicate that the current value is the default for this %Data.
    void unset() { m_isSets[currentAspect()] = false; }

    /// Reset the isSet flag to true, to indicate that the current value has been modified.
    void forceSet() { m_isSets[currentAspect()] = true; }

    /// @name Flags
    /// @{

    /// Set one of the flags.
    void setFlag(DataFlagsEnum flag, bool b)  { if(b) m_dataFlags |= (DataFlags)flag;  else m_dataFlags &= ~(DataFlags)flag; }

    /// Get one of the flags.
    bool getFlag(DataFlagsEnum flag) const { return (m_dataFlags&(DataFlags)flag)!=0; }

    /// Return whether this %Data has to be displayed in GUIs.
    bool isDisplayed() const  { return getFlag(FLAG_DISPLAYED); }
    /// Return whether this %Data will be read-only in GUIs.
    bool isReadOnly() const   { return getFlag(FLAG_READONLY); }
    /// Return whether this %Data contains persistent information.
    bool isPersistent() const { return getFlag(FLAG_PERSISTENT); }
    /// Return whether this %Data should be autolinked when using the src="" syntax.
    bool isAutoLink() const { return getFlag(FLAG_AUTOLINK); }
    /// Return whether the Data has to be set by the user for the owner component to be valid
    bool isRequired() const { return getFlag(FLAG_REQUIRED); }

    /// Set whether this %Data should be displayed in GUIs.
    void setDisplayed(bool b)  { setFlag(FLAG_DISPLAYED,b); }
    /// Set whether this %Data is read-only.
    void setReadOnly(bool b)   { setFlag(FLAG_READONLY,b); }
    /// Set whether this %Data contains persistent information.
    void setPersistent(bool b) { setFlag(FLAG_PERSISTENT,b); }
    /// Set whether this data should be autolinked when using the src="" syntax
    void setAutoLink(bool b) { setFlag(FLAG_AUTOLINK,b); }
    /// Set whether the Data has to be set by the user for the owner component to be valid.
    void setRequired(bool b) { setFlag(FLAG_REQUIRED,b); }
    /// @}

    /// If we use the Data as a link and not as value directly
    //void setLinkPath(const std::string &path) { m_linkPath = path; }
    std::string getLinkPath() const { return parentBaseData.getPath(); }
    /// Return whether this %Data can be used as a linkPath.
    ///
    /// True by default.
    /// Useful if you want to customize the use of @ syntax (see ObjectRef and DataObjectRef)
    virtual bool canBeLinked() const { return true; }

    /// Return the Base component owning this %Data.
    Base* getOwner() const { return m_owner; }
    /// Set the owner of this %Data.
    void setOwner(Base* o) { m_owner=o; }

    /// This method is needed by DDGNode
    BaseData* getData() const
    {
        return const_cast<BaseData*>(this);
    }

    /// Return the name of this %Data within the Base component
    const std::string& getName() const { return m_name; }
    /// Set the name of this %Data.
    ///
    /// This method should not be called directly, the %Data registration methods in Base should be used instead.
    void setName(const std::string& name) { m_name=name; }

    /// Return the number of changes since creation.
    /// This can be used to efficiently detect changes.
    int getCounter() const { return m_counters[currentAspect()]; }


    /// @name Optimized edition and retrieval API (for multi-threading performances)
    /// @{

    /// True if the value has been modified
    /// If this data is linked, the value of this data will be considered as modified
    /// (even if the parent's value has not been modified)
    bool isSet(const core::ExecParams* params) const { return m_isSets[currentAspect(params)]; }

    /// Reset the isSet flag to false, to indicate that the current value is the default for this %Data.
    void unset(const core::ExecParams* params) { m_isSets[currentAspect(params)] = false; }

    /// Reset the isSet flag to true, to indicate that the current value has been modified.
    void forceSet(const core::ExecParams* params) { m_isSets[currentAspect(params)] = true; }

    /// Return the number of changes since creation
    /// This can be used to efficiently detect changes
    int getCounter(const core::ExecParams* params) const { return m_counters[currentAspect(params)]; }

    /// @}

    /// Link to a parent data. The value of this data will automatically duplicate the value of the parent data.
    bool setParent(BaseData* parent, const std::string& path = std::string());
    bool setParent(const std::string& path);

    /// Check if a given Data can be linked as a parent of this data
    virtual bool validParent(BaseData* parent);

    BaseData* getParent() const { return parentBaseData.get(); }

    /// Update the value of this %Data
    void update();

    /// @name Links management
    /// @{

    typedef std::vector<BaseLink*> VecLink;
    /// Accessor to the vector containing all the fields of this object
    const VecLink& getLinks() const { return m_vecLink; }

    virtual bool findDataLinkDest(DDGNode*& ptr, const std::string& path, const BaseLink* link);

    virtual bool findDataLinkDest(BaseData*& ptr, const std::string& path, const BaseLink* link);

    template<class DataT>
    bool findDataLinkDest(DataT*& ptr, const std::string& path, const BaseLink* link)
    {
        BaseData* base = NULL;
        if (!findDataLinkDest(base, path, link)) return false;
        ptr = dynamic_cast<DataT*>(base);
        return (ptr != NULL);
    }

    /// Add a link.
    void addLink(BaseLink* l);

protected:

    BaseLink::InitLink<BaseData>
    initLink(const char* name, const char* help)
    {
        return BaseLink::InitLink<BaseData>(this, name, help);
    }

    /// List of links
    VecLink m_vecLink;

    /// @}

    virtual void doSetParent(BaseData* parent);

    virtual void doDelInput(DDGNode* n);

    /// Update this %Data from the value of its parent
    virtual bool updateFromParentValue(const BaseData* parent);

    /// Help message
    const char* help;
    /// Owner class
    const char* ownerClass;
    /// group
    const char* group;
    /// widget
    const char* widget;
    /// Number of changes since creation
    helper::fixed_array<int, SOFA_DATA_MAX_ASPECTS> m_counters;
    /// True if this %Data is set, i.e. its value is different from the default value
    helper::fixed_array<bool, SOFA_DATA_MAX_ASPECTS> m_isSets;
    /// Flags indicating the purpose and behaviour of this %Data
    DataFlags m_dataFlags;
    /// Return the Base component owning this %Data
    Base* m_owner;
    /// Data name within the Base component
    std::string m_name;
//    /// Link to another Data, if used as an input from another Data (@ typo).
//    std::string m_linkPath;
    /// Parent Data
    SingleLink<BaseData,BaseData,BaseLink::FLAG_STOREPATH|BaseLink::FLAG_DATALINK|BaseLink::FLAG_DUPLICATE> parentBaseData;

    /// Helper method to decode the type name to a more readable form if possible
    static std::string decodeTypeName(const std::type_info& t);

public:

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

template<class Type>
class LinkTraitsPtrCasts
{
public:
    static sofa::core::objectmodel::Base* getBase(sofa::core::objectmodel::Base* b) { return b; }
    static sofa::core::objectmodel::Base* getBase(sofa::core::objectmodel::BaseData* d) { return d->getOwner(); }
    static sofa::core::objectmodel::BaseData* getData(sofa::core::objectmodel::Base* /*b*/) { return NULL; }
    static sofa::core::objectmodel::BaseData* getData(sofa::core::objectmodel::BaseData* d) { return d; }
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
