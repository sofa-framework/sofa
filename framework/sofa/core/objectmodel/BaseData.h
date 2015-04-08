/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
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
class DDGDataNode;

/**
 *  \brief Class for a Data node in the Data dependency graph.
 *
 *  It's meant to be created and managed by the BaseData class, but it's presence is now optional,
 *  with any Data being able to be created disconnected from the DDG or to be disconnected later 
 *  (but for now not reconnected, as we also lose the RTTI information that is only available at creation).
 *
 *  The ability to skip the DDGDataNode part for a given BaseData allows for much faster creation and much
 *  lighter memory footprint, at the cost of losing some additional features 
 *  (serialisation and exposition for the editor, data propagation in the dependency graph)
 *  for those nodes beyond the basic data storage and ability to use their values during the simulation.
 *
 *  This is intended mainly for modules that dynamically create and destroy lots of nodes, 
 *  (typically collision detection and response), but can be used anywhere, if for example you
 *  create your objects in code rather than from a scene file (and do not specify references by name).
 */
class SOFA_CORE_API DDGDataNode : public DDGNode
{
public:
    /// @name Class reflection system
    /// @{
    typedef TClass<DDGDataNode,DDGNode> MyClass;
    static const MyClass* GetClass() { return MyClass::get(); }
    virtual const BaseClass* getClass() const
    { return GetClass(); }
    /// @}

	/// Bit field that holds flags value.
    typedef unsigned DataFlags;
	/// This internal class is used by the initData() methods to store initialization parameters of a Data
    class BaseInitData
    {
    public:
        BaseInitData();

		BaseData* data;
        const char* helpMsg;
        DataFlags dataFlags;
        Base* owner;
        const char* name;
        const char* ownerClass;
        const char* group;
        const char* widget;
    };

	explicit DDGDataNode(const BaseInitData& init);

	explicit DDGDataNode(BaseData* data, const char* h);

    /// @name DDGNode interface
    /// @{

	/// Return the name of this %Data within the Base component
    const std::string& getName() const { return m_name; }
    /// Set the name of this %Data.
    ///
    /// This method should not be called directly, the %Data registration methods in Base should be used instead.
    void setName(const std::string& name) { m_name=name; }

	///  This method is needed by DDGNode
    Base* getOwner() const;

	/// This method is needed by DDGNode
    BaseData* getData() const;

    /// Update the value of this %Data
    void update();

	/// @}


    /// @name Links management
    /// @{

    typedef std::vector<BaseLink*> VecLink;
    /// Accessor to the vector containing all the fields of this object
    const VecLink& getLinks() const { return m_vecLink; }

    bool setParent(DDGDataNode* parent, const std::string& path = std::string());
    bool setParent(const std::string& path);
	void removeParent();

    /// Add a link.
    void addLink(BaseLink* l);

    BaseLink::InitLink<DDGDataNode>
    initLink(const char* name, const char* help)
    {
        return BaseLink::InitLink<DDGDataNode>(this, name, help);
    }

	virtual bool findDataLinkDest(DDGDataNode*& ptr, const std::string& path, const BaseLink* link);

    /// List of links
    VecLink m_vecLink;

    /// @}

    virtual void doDelInput(DDGNode* n);

    /// Copy the value of an aspect into another one.
    virtual void copyAspect(int destAspect, int srcAspect);

    /// Release memory allocated for the specified aspect.
    virtual void releaseAspect(int aspect);

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

    /// Data name within the Base component
    std::string m_name;
    /// Parent Node
    SingleLink<DDGDataNode,DDGDataNode,BaseLink::FLAG_STOREPATH|BaseLink::FLAG_DATALINK|BaseLink::FLAG_DUPLICATE> m_parent;
	/// BaseData for value storage
	BaseData* m_data;

	static const std::string NO_NAME;
	static const DDGLinkContainer NO_LINKS;

	static void enableCreation(bool enable);

	static bool isCreationEnabled();

private:
	class CreationArray : public helper::fixed_array<bool, SOFA_DATA_MAX_ASPECTS>
	{
	public:
		CreationArray(bool init) { assign(init); }
	};

	static CreationArray m_creationEnabled;
};


/**
 *  \brief Abstract base class for Data.
 *
 */
class SOFA_CORE_API BaseData
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
    typedef DDGDataNode::DataFlags DataFlags;

    /// Default value used for flags.
    enum { FLAG_DEFAULT = FLAG_DISPLAYED | FLAG_PERSISTENT | FLAG_AUTOLINK };

    /// @name Class reflection system
    /// @{

    typedef TClass<BaseData> MyClass;
    static const MyClass* GetClass() { return MyClass::get(); }
    virtual const BaseClass* getClass() const
    { return GetClass(); }

    template<class T>
    static void dynamicCast(T*& ptr, Base* /*b*/)
    {
        ptr = NULL; // BaseData does not derive from Base
    }

    /// Helper method to get the type name of type T
    template<class T>
    static std::string typeName(const T* = NULL)
    {
        if (defaulttype::DataTypeInfo<T>::ValidInfo)
            return defaulttype::DataTypeName<T>::name();
        else
            return decodeTypeName(typeid(T));
    }

    /// Helper method to get the class name of a type derived from this class
    template<class T>
    static std::string className(const T* ptr= NULL)
    {
        return BaseClass::defaultClassName(ptr);
    }

    /// Helper method to get the namespace name of a type derived from this class
    template<class T>
    static std::string namespaceName(const T* ptr= NULL)
    {
        return BaseClass::defaultNamespaceName(ptr);
    }

    /// Helper method to get the template name of a type derived from this class
    template<class T>
    static std::string templateName(const T* ptr= NULL)
    {
        return BaseClass::defaultTemplateName(ptr);
    }

    /// Helper method to get the shortname of a type derived from this class.
    template< class T>
    static std::string shortName( const T* ptr = NULL, BaseObjectDescription* = NULL )
    {
        std::string shortname = T::className(ptr);
        if( !shortname.empty() )
        {
            *shortname.begin() = ::tolower(*shortname.begin());
        }
        return shortname;
    }

    /// @}

    /// This internal class is used by the initData() methods to store initialization parameters of a Data
    typedef DDGDataNode::BaseInitData BaseInitData;

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

    /// Update this %Data from the value of its parent
    virtual bool updateFromParentValue(const BaseData* parent);

    /// Returns true if the DDGNode needs to be updated
    bool isDirty(const core::ExecParams* params = 0) const { return m_ddg ? m_ddg->isDirty(params) : false; }

	/// Indicate the value needs to be updated
	void setDirtyValue(const core::ExecParams* params = 0) { if (m_ddg) m_ddg->setDirtyValue(params); }

    /// Set dirty flag to false
	void cleanDirty(const core::ExecParams* params = 0) { if (m_ddg) m_ddg->cleanDirty(params); }

	/// Utility method to call update if necessary.
	void updateIfDirty(const core::ExecParams* params = 0) const { if (m_ddg) m_ddg->updateIfDirty(params); }

	/// Update this value (forced).
	void update() { if (m_ddg) m_ddg->update(); }

    /// Copy the value of an aspect into another one.
    virtual void copyAspect(int destAspect, int srcAspect);

    /// Release memory allocated for the specified aspect.
    virtual void releaseAspect(int aspect);

	/// Return the name of this %Data within the Base component
	const std::string& getName() const { return m_ddg ? m_ddg->m_name : DDGDataNode::NO_NAME; }

	/// @deprecated Set the name of this %Data.
	///
	/// You should be using the initData constructor instead.
	/// @note If you do call this method, don't change the name of a %Data after adding it to the Base component.
	void setName(const std::string& name) { if (m_ddg) m_ddg->m_name = name; }

    /// Get a help message that describes this %Data.
    const char* getHelp() const { return m_ddg ? m_ddg->help : ""; }

    /// Set the help message.
    void setHelp(const char* val) { if (m_ddg) m_ddg->help = val; }

    /// Get owner class
    const char* getOwnerClass() const { return m_ddg ? m_ddg->ownerClass : ""; }

    /// Set owner class
    void setOwnerClass(const char* val) { if (m_ddg) m_ddg->ownerClass = val; }

    /// Get group
    const char* getGroup() const { return m_ddg ? m_ddg->group : ""; }

    /// Set group
    void setGroup(const char* val) { if (m_ddg) m_ddg->group = val; }

    /// Get widget
    const char* getWidget() const { return m_ddg ? m_ddg->widget : ""; }

    /// Set widget
    void setWidget(const char* val) { if (m_ddg) m_ddg->widget = val; }

    /// True if the value has been modified
    /// If this data is linked, the value of this data will be considered as modified
    /// (even if the parent's value has not been modified)
    bool isSet() const { return m_ddg ? m_ddg->m_isSets[DDGNode::currentAspect()] : true; }

    /// True if the counter of modification gives valid information.
    virtual bool isCounterValid() const = 0;

    /// Reset the isSet flag to false, to indicate that the current value is the default for this %Data.
    void unset() { if (m_ddg) m_ddg->m_isSets[DDGNode::currentAspect()] = false; }

    /// Reset the isSet flag to true, to indicate that the current value has been modified.
    void forceSet() { if (m_ddg) m_ddg->m_isSets[DDGNode::currentAspect()] = true; }

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
    std::string getLinkPath() const { return m_ddg ? m_ddg->m_parent.getPath() : std::string(); }
    /// Return whether this %Data can be used as a linkPath.
    ///
    /// True by default.
    /// Useful if you want to customize the use of @ syntax (see ObjectRef and DataObjectRef)
    virtual bool canBeLinked() const { return true; }

    /// Return the Base component owning this %Data.
    Base* getOwner() const { return m_owner; }
    /// Set the owner of this %Data.
    void setOwner(Base* o) { m_owner=o; }

    /// Return the number of changes since creation.
    /// This can be used to efficiently detect changes.
    int getCounter() const { return m_ddg ? m_ddg->m_counters[DDGNode::currentAspect()] : 0; }


    /// @name Optimized edition and retrieval API (for multi-threading performances)
    /// @{

    /// True if the value has been modified
    /// If this data is linked, the value of this data will be considered as modified
    /// (even if the parent's value has not been modified)
    bool isSet(const core::ExecParams* params) const { return m_ddg ? m_ddg->m_isSets[DDGNode::currentAspect(params)] : true; }

    /// Reset the isSet flag to false, to indicate that the current value is the default for this %Data.
    void unset(const core::ExecParams* params) { if (m_ddg) m_ddg->m_isSets[DDGNode::currentAspect(params)] = false; }

    /// Reset the isSet flag to true, to indicate that the current value has been modified.
    void forceSet(const core::ExecParams* params) { if (m_ddg) m_ddg->m_isSets[DDGNode::currentAspect(params)] = true; }

    /// Return the number of changes since creation
    /// This can be used to efficiently detect changes
    int getCounter(const core::ExecParams* params) const { return m_ddg ? m_ddg->m_counters[DDGNode::currentAspect(params)] : 0; }

    /// @}

    /// Link to a parent data. The value of this data will automatically duplicate the value of the parent data.
    bool setParent(BaseData* parent, const std::string& path = std::string());
    bool setParent(const std::string& path);
	void removeParent();

    /// Check if a given Data can be linked as a parent of this data
    virtual bool validParent(BaseData* parent);

	BaseData* getParent() const 
	{
		DDGDataNode* ddg = m_ddg ? m_ddg->m_parent.get() : NULL;
		return ddg ? ddg->getData() : NULL;
	}

    /// @deprecated Remove an input from this %Data (use clearParent, as it is the only valid input node for a %Data)
    void delInput(BaseData* data) { if (m_ddg) m_ddg->delInput(ddg(data)); }

    /// Add a new output to this %Data
    void addOutput(DDGNode* n) { if (m_ddg) m_ddg->addOutput(n); }

    /// Remove an output from this %Data
	void delOutput(DDGNode* n) { if (m_ddg) m_ddg->delOutput(n); }

    /// Get the list of outputs for this %Data
	const DDGNode::DDGLinkContainer& getOutputs() { return m_ddg ? m_ddg->getOutputs() : DDGDataNode::NO_LINKS; }

    virtual bool findDataLinkDest(BaseData*& ptr, const std::string& path, const BaseLink* link);

    template<class DataT>
    bool findDataLinkDest(DataT*& ptr, const std::string& path, const BaseLink* link)
    {
        BaseData* base = NULL;
        if (!findDataLinkDest(base, path, link)) return false;
        ptr = dynamic_cast<DataT*>(base);
        return (ptr != NULL);
    }

	/// Cast a BaseData to a dependency graph node, for when you want to use it as input or output in a graph.
	static DDGDataNode* ddg(BaseData* data) { return data ? data->m_ddg : NULL; }

	/// Check if the ddg part of this %Data is present or if it is a light version.
	bool hasDdg() const { return m_ddg != NULL; }

	/// Removes the ddg part of an existing data, but only if there is no input or output link.
	void cleanDdg();

protected:

	friend class DDGDataNode;
	virtual void onParentChanged(BaseData* /*parent*/) {}

    DataFlags m_dataFlags;
	/// Optional data dependency graph node.
	DDGDataNode* m_ddg;
    /// Return the Base component owning this %Data
    Base* m_owner;

    /// Helper method to decode the type name to a more readable form if possible
    static std::string decodeTypeName(const std::type_info& t);
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

template<>
class LinkTraitsPtrCasts<DDGDataNode>
{
public:
    static sofa::core::objectmodel::Base* getBase(sofa::core::objectmodel::DDGDataNode* n);
    static sofa::core::objectmodel::BaseData* getData(sofa::core::objectmodel::DDGDataNode* n);
};

inline 
DDGDataNode::BaseInitData::BaseInitData()
	: data(NULL)
	, helpMsg("")
	, dataFlags(BaseData::FLAG_DEFAULT)
	, owner(NULL)
	, name("")
	, ownerClass("")
	, group("")
	, widget("")
{
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
