/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/core/config.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/DataLink.h>

namespace sofa::core::objectmodel
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

    /// This internal class is used by the initData() methods to store initialization parameters of a Data
    class BaseInitData
    {
    public:
        BaseInitData() : data(nullptr), helpMsg(""), dataFlags(FLAG_DEFAULT), owner(nullptr), name(""), group(""), widget("") {}
        BaseData* data;
        std::string helpMsg;
        DataFlags dataFlags;
        Base* owner;
        std::string name;
        std::string group;
        std::string widget;
    };

    virtual BaseData* getNewInstance() { return nullptr; }

    /** Constructor used via the Base::initData() methods. */
    explicit BaseData(const BaseInitData& init);

    /** Constructor.
     *  \param helpMsg A help message that describes this %Data.
     *  \param flags The flags for this %Data (see \ref DataFlagsEnum).
     */
    BaseData(const std::string& helpMsg, DataFlags flags = FLAG_DEFAULT);

    //TODO(dmarchal:08/10/2019)Uncomment the deprecated when VS2015 support will be dropped.
    //[[deprecated("Replaced with one with std::string instead of char* version")]]
    BaseData(const char* helpMsg, DataFlags flags = FLAG_DEFAULT);

    /** Constructor.
     *  \param helpMsg A help message that describes this %Data.
     *  \param isDisplayed Whether this %Data should be displayed in GUIs.
     *  \param isReadOnly Whether this %Data should be modifiable in GUIs.
     */
    BaseData(const std::string& helpMsg, bool isDisplayed=true, bool isReadOnly=false);

    //TODO(dmarchal:08/10/2019)Uncomment the deprecated when VS2015 support will be dropped.
    //[[deprecated("Replaced with one with std::string instead of char* version")]]
    BaseData(const char* helpMsg, bool isDisplayed=true, bool isReadOnly=false);

    /// Destructor.
    ~BaseData() override;

    /// Assign a value to this %Data from a string representation.
    /// \return true on success.
    virtual bool read(const std::string& value) = 0;

    /// Print the value of this %Data to a stream.
    virtual void printValue(std::ostream&) const = 0;

    /// Get a string representation of the value held in this %Data.
    virtual std::string getValueString() const = 0;

    /// Get a string representation of the default value held in this %Data.
    virtual std::string getDefaultValueString() const = 0;

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
    const void* getValueVoidPtr() const;

    /// Get a void pointer to the value held in this %Data, to be used with AbstractTypeInfo.
    ///
    /// This pointer should be used via the instance of AbstractTypeInfo
    /// returned by getValueTypeInfo().
    /// \warning You must call endEditVoidPtr() once you're done modifying the value.
    void* beginEditVoidPtr();

    /// Must be called after beginEditVoidPtr(), after you are finished modifying this %Data.
    void endEditVoidPtr();

    /// Get a help message that describes this %Data.
    const std::string& getHelp() const { return help; }

    /// Set the help message.
    void setHelp(const std::string& val) { help = val; }

    /// Get group
    const std::string& getGroup() const { return group; }

    /// Set group
    void setGroup(const std::string& val) { group = val; }

    /// Get widget
    const std::string& getWidget() const { return widget; }

    /// Set widget
    void setWidget(const char* val) { widget = val; }


    /// @name Flags
    /// @{

    /// Set one of the flags.
    void setFlag(DataFlagsEnum flag, bool b)  { if(b) m_dataFlags |= static_cast<DataFlags>(flag);  else m_dataFlags &= ~static_cast<DataFlags>(flag); }

    /// Get one of the flags.
    bool getFlag(DataFlagsEnum flag) const { return (m_dataFlags&static_cast<DataFlags>(flag))!=0; }

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
    virtual std::string getLinkPath() const;
    std::string getPathName()const;

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

    /// Return whether the Data has a default value or not
    bool hasDefaultValue() const { return m_hasDefaultValue; }

    /// @name Optimized edition and retrieval API (for multi-threading performances)
    /// @{
    /// True if the value has been modified
    /// If this data is linked, the value of this data will be considered as modified
    /// (even if the parent's value has not been modified)s
    bool isSet() const { return m_isSet; }

    /// Reset the isSet flag to false, to indicate that the current value is the default for this %Data.
    void unset() { m_isSet = false; }

    /// Reset the isSet flag to true, to indicate that the current value has been modified.
    void forceSet() { m_isSet = true; }

    /// Return the number of changes since creation
    /// This can be used to efficiently detect changes
    int getCounter() const { return m_counter; }
    /// @}

    /// Link to a parent data. The value of this data will automatically duplicate the value of the parent data.
    bool setParent(BaseData* parent, const std::string& path = std::string());
    bool setParent(const std::string& path);

    /// Check if a given Data can be linked as a parent of this data
    virtual bool validParent(const BaseData *parent);

    BaseData* getParent() const { return parentData.getTarget(); }

    /// Update the value of this %Data
    void update() override;

    /// Copy the value from another Data.
    ///
    /// Note that this is a one-time copy and not a permanent link (otherwise see setParent())
    /// @return true if the copy was successful.
    bool copyValueFrom(const BaseData* data);
    bool updateValueFromLink(const BaseData* data);

    /// Help message
    std::string help {""};
    /// Owner class
    std::string ownerClass {""} ;
    /// group
    std::string group {""};
    /// widget
    std::string widget {""};
    /// Number of changes since creation
    int m_counter;
    /// True if this %Data is set, i.e. its value is different from the default value
    bool m_isSet;
    /// Flags indicating the purpose and behaviour of this %Data
    DataFlags m_dataFlags;
    /// Return the Base component owning this %Data
    Base* m_owner {nullptr};
    /// Data name within the Base component
    std::string m_name;
    /// True if this %Data has a default value
    bool m_hasDefaultValue = false;

    /// Parent Data
    DataLink<BaseData> parentData;

    /// Helper method to decode the type name to a more readable form if possible
    static std::string decodeTypeName(const std::type_info& t);

    /// Helper method to get the type name of type T
    template<class T>
    static std::string typeName()
    {
        if (defaulttype::DataTypeInfo<T>::ValidInfo)
        {
            return defaulttype::DataTypeName<T>::name();
        }
        return decodeTypeName(typeid(T));
    }

    template<class T>
    SOFA_ATTRIBUTE_DISABLED__UNNECESSARY_PARAMETER_IN_TYPENAME() static std::string typeName(const T*) = delete;

protected:
    /// Try to update this Data from the value of its parent in "fast mode";
    bool genericCopyValueFrom(const BaseData* parent);

private:
    /// Delegates from DDGNode.
    void doDelInput(DDGNode* n) override;

    virtual bool doCopyValueFrom(const BaseData* parent) = 0;
    virtual bool doSetValueFromLink(const BaseData* parent) = 0;

    virtual bool doIsExactSameDataType(const BaseData* parent) = 0;
    virtual const void* doGetValueVoidPtr() const = 0;
    virtual void* doBeginEditVoidPtr() = 0;
    virtual void doEndEditVoidPtr() = 0;
    virtual void doOnUpdate() {}
};

/** A WriteAccessWithRawPtr is a RAII class, holding a reference to a given container
 *  and providing access to its data through a non-const void* ptr taking care of the
 * beginEdit/endEdit pairs.
 *
 *  Advantadges of using a WriteAccessWithRawPtr are :
 *
 *  - It can be faster that the default methods and operators of the container,
 *  as verifications and changes notifications can be handled in the accessor's
 *  constructor and destructor instead of at each item access.
 */
class WriteAccessWithRawPtr
{
public:
    explicit WriteAccessWithRawPtr(BaseData* data)
    {
        m_data = data;
        ptr = data->beginEditVoidPtr();
    }

    ~WriteAccessWithRawPtr()
    {
        m_data->endEditVoidPtr();
    }

    void*     ptr { nullptr };
private:
    WriteAccessWithRawPtr() = default;
    BaseData* m_data { nullptr };
};

SOFA_CORE_API std::ostream& operator<<(std::ostream &out, const sofa::core::objectmodel::BaseData& df);

} // namespace sofa::core::objectmodel
