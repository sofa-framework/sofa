/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_OBJECTMODEL_DATA_H
#define SOFA_CORE_OBJECTMODEL_DATA_H

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/accessor.h>

/// This is for DeprecatedBaseClass. To generate an error if some deprecated function are used.
#include <sofa/core/objectmodel/BaseClass.h>


namespace sofa::core::objectmodel
{

/// To handle the Data link:
/// - CopyOnWrite==false: an independent copy (duplicated memory)
/// - CopyOnWrite==true: shared memory while the Data is not modified (in that case the memory is duplicated to get an independent copy)
template <class T, bool CopyOnWrite>
class DataValue;

template <class T>
class DataValue<T, false>
{
    T data;
public:

    DataValue()
        : data(T())// BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    explicit DataValue(const T &value)
        : data(value)
    {
    }

    DataValue(const DataValue& dc)
        : data(dc.getValue())
    {
    }

    DataValue& operator=(const DataValue& dc )
    {
        data = dc.getValue(); // copy
        return *this;
    }

    T* beginEdit() { return &data; }
    void endEdit() {}
    const T& getValue() const { return data; }
    void setValue(const T& value)
    {
        data = value;
    }
    void release()
    {
    }
};


template <class T>
class DataValue<T, true>
{
    std::shared_ptr<T> ptr;
public:

    DataValue()
        : ptr(new T(T())) // BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    explicit DataValue(const T& value)
        : ptr(new T(value))
    {
    }

    DataValue(const DataValue& dc)
        : ptr(dc.ptr) // start with shared memory
    {
    }

    ~DataValue()
    {
    }

    DataValue& operator=(const DataValue& dc )
    {
        //avoid self reference
        if(&dc != this)
        {
            ptr = dc.ptr;
        }

        return *this;
    }

    T* beginEdit()
    {
        if(!ptr.unique())
        {
            ptr.reset(new T(*ptr)); // a priori the Data will be modified -> copy
        }
        return ptr.get();
    }

    void endEdit()
    {
    }

    const T& getValue() const
    {
        return *ptr;
    }

    void setValue(const T& value)
    {
        if(!ptr.unique())
        {
            ptr.reset(new T(value)); // the Data is modified -> copy
        }
        else
        {
            *ptr = value;
        }
    }

    void release()
    {
        ptr.reset();
    }
};

/** \brief Container that holds a variable for a component.
 *
 * This is a fundamental class template in Sofa.  Data are used to encapsulated
 * member variables of Sofa components (i.e. classes that somehow inherit from
 * Base) in order to access them dynamically and generically: briefly, Data can
 * be retrieved at run-time by their name, and they can be assigned a value from
 * a string, or be printed as a string.
 *
 * More concretely, from the perspective of XML scene files, each Data declared
 * in a component corresponds to an attribute of this component.
 *
 * <h4> Example </h4>
 *
 * If a component \c Foo has a boolean parameter \c bar, it does not simply declares it
 * as <tt>bool m_bar</tt>, but rather like this:
 *
 * \code{.cpp}
 *  Data<bool> d_bar;
 * \endcode
 *
 * Then, this %Data must be initialized to provide its name and default value.
 * This is typically done in the initialization list of \b each constructor of
 * the component, using the helper function Base::initData():
 *
 * \code{.cpp}
 * Foo::Foo(): d_bar(initData(&d_bar, true, "bar", "Here is a little description of this Data.")) {
 *     // ...
 * }
 * \endcode
 *
 * And this %Data can be assigned a value in XML scene files like so:
 * \code{.xml}
 * <Foo bar="false"/>
 * \endcode
 */
template < class T = void* >
class Data : public BaseData
{
public:
    using BaseData::m_counter;
    using BaseData::m_isSet;
    using BaseData::setDirtyOutputs;
    using BaseData::updateIfDirty;
    using BaseData::notifyEndEdit;
    using BaseData::cleanDirty;

    /// @name Construction / destruction
    /// @{

    /// This internal class is used by the initData() methods to store initialization parameters of a Data
    class InitData : public BaseData::BaseInitData
    {
    public:
        InitData() : value(T()) {}
        InitData(const T& v) : value(v) {}
        InitData(const BaseData::BaseInitData& i) : BaseData::BaseInitData(i), value(T()) {}

        T value;
    };

    /// It's used for getting a new instance from an existing instance. This function is used by the communication plugin
    virtual BaseData* getNewInstance() override { return new Data();}

    /** \copydoc BaseData(const BaseData::BaseInitData& init) */
    explicit Data(const BaseData::BaseInitData& init)
        : BaseData(init)
    {
    }

    /** \copydoc Data(const BaseData::BaseInitData&) */
    explicit Data(const InitData& init)
        : BaseData(init)
    {
        m_value = ValueType(init.value);
    }

    /** \copydoc BaseData() */
    Data() : Data(std::string(""), false) {}

    /** \copydoc BaseData(const std::string& , bool, bool) */
    Data( const std::string& helpMsg, bool isDisplayed=true, bool isReadOnly=false)
        : BaseData(helpMsg, isDisplayed, isReadOnly)
    {
        m_value = ValueType();
    }

    /** \copydoc BaseData(const char*, bool, bool)
     *  \param value The default value.
     */
    [[deprecated("2020-03-25: char* version are removed with taking std::string instead.")]]
    Data( const T& value, const char* helpMsg=nullptr, bool isDisplayed=true, bool isReadOnly=false) :
        Data(value, sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly)
    {}

    /** \copydoc BaseData(const char*, bool, bool)
     *  \param value The default value.
     */
    Data( const T& value, const std::string& helpMsg, bool isDisplayed=true, bool isReadOnly=false)
        : BaseData(helpMsg, isDisplayed, isReadOnly)
    {
        m_value = ValueType(value);
    }

    /// Destructor.
    virtual ~Data() override {}

    void printValue(std::ostream& out) const override;
    std::string getValueString() const override;
    std::string getValueTypeString() const override;

    /// @}

    /// @name Simple edition and retrieval API
    /// @{

    /// Return a pointer to edit the value of the data.
    /// Before returning the editting pointer the following action are done:
    ///     - update the value for the data from the connected parent (if any)
    ///     - unlink the connected parent(if any)
    inline T* beginEdit()
    {
        std::cout << "BEGINEDIT " << getName() << std::endl;
        if(m_parentData.isSet())
        {
            std::cout << "BEGINEDIT. " << std::endl;

            updateIfDirty();
            std::cout << "UNLINK. =>X2" << std::endl;

            //m_parentData.unSet();
        }
        m_counter++;
        m_isSet = true;
        setDirtyOutputs();
        return m_value.beginEdit();
    }

    /// Return a pointer to edit the value of the data.
    /// Before returning the editting pointer the following action are done:
    ///     - *does not* update the value from the connected parent (if any)
    ///     - unlink the connected parent(if any)
    inline T* beginWriteOnly()
    {
        std::cout << "BEGINWRITEONLY: =>" << getName() << std::endl;

        /// If there is a link we disconnect it.
        /// without updating its value
        if(m_parentData.isSet())
        {
            std::cout << "BEGINWRITEONLY: Unlink " << m_parentData.get()->getName() << std::endl;
            m_parentData.unSet();
            cleanDirty();
        }
        m_counter++;
        m_isSet=true;
        setDirtyOutputs();
        return m_value.beginEdit();
    }

    inline T* beginWriteOnlyNoUnlink()
    {
        m_counter++;
        m_isSet=true;
        setDirtyOutputs();
        return m_value.beginEdit();
    }

    inline void endEdit()
    {
        m_value.endEdit();
        notifyEndEdit();
    }

    /// @warning writeOnly (the Data is not updated before being set)
    inline void setValue(const T& value)
    {
        *beginWriteOnly()=value;
        endEdit();
    }

    inline const T& getValue() const
    {
        std::cout << "Data::getValue: " << getName() << std::endl;
        updateIfDirty();
        return m_value.getValue();
    }

    /// Get current value as a void pointer (use getValueTypeInfo to find how to access it)
    const void* getValueVoidPtr() const override
    {
        std::cout << "GET VALUEVOID PTR " << getName() << std::endl;
        return &(getValue());
    }

    /// Begin edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    void* beginEditVoidPtr() override
    {
        return beginEdit();
    }

    void* beginWriteOnlyVoidPtr() override
    {
        return beginWriteOnly();
    }

    /// End edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    void endEditVoidPtr() override
    {
        endEdit();
    }

    /// @}

    /// @name Virtual edition and retrieval API (for generic TData parent API, deprecated)
    /// @{

    [[deprecated("2020-03-25: virtualGetValue is now deprecated. use getValue() instead ")]]
    const T& virtualGetValue() const { return getValue(); }

    [[deprecated("2020-03-25: virtualGetValue is now deprecated. use setValue instead() ")]]
    void virtualSetValue(const T& v) { setValue(v); }

    [[deprecated("2020-03-25: virtualGetValue is now deprecated. Use setValueFromData")]]
    void virtualSetLink(const BaseData& bd){ copyValue(&bd); }

    [[deprecated("2020-03-25: virtualGetValue is now deprecated. use beginEdit() ")]]
    T* virtualBeginEdit() { return beginEdit(); }

    [[deprecated("2020-03-25: virtualGetValue is now deprecated. use endEdit() ")]]
    void virtualEndEdit() { endEdit(); }

    /// @}

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

    typedef DataValue<T, sofa::defaulttype::DataTypeInfo<T>::CopyOnWrite> ValueType;

    /// Value
    ValueType m_value;

private:
    Data(const Data& );
    Data& operator=(const Data& );

    ////////////////////////// DEPRECATED SECTION ///////////////////////////////////
public:
    [[deprecated("2020-03-25: Replaced with one with std::string instead of char* version")]]
    Data( const char* helpMsg, bool isDisplayed=true, bool isReadOnly=false)
        : Data(sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly) {}

    [[deprecated("2020-03-25: Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    inline void endEdit(const core::ExecParams*)
    {
        endEdit();
    }

    [[deprecated("2020-03-25: Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    inline T* beginWriteOnly(const core::ExecParams*)
    {
        return beginWriteOnly();
    }

    [[deprecated("2020-03-25: Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    inline T* beginEdit(const core::ExecParams*)
    {
        return beginEdit();
    }

    [[deprecated("2020-03-25: Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    inline void setValue(const core::ExecParams*, const T& value)
    {
        setValue(value);
    }

    [[deprecated("2020-03-25: Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    inline const T& getValue(const core::ExecParams*) const
    {
        return getValue();
    }

    [[deprecated("2020-03-25: Data are not part of the class system anymore. ")]]
    static const BaseClass* GetClass() { return DeprecatedBaseClass::GetSingleton(); }

    [[deprecated("2020-03-25: Data are not part of the class system anymore. ")]]
    virtual const BaseClass* getClass() const { return DeprecatedBaseClass::GetSingleton(); }

    [[deprecated("2020-03-25: Data are not part of the class system anymore. ")]]
    static std::string templateName(const Data<T>* = nullptr)
    {
        T* ptr = nullptr;
        return BaseData::typeName(ptr);
    }

    static sofa::defaulttype::AbstractTypeInfo* GetValueTypeInfo()
    {
        return sofa::defaulttype::VirtualTypeInfo<T>::get();
    }

    /// Get info about the value type of the associated variable
    const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const override
    {
        return GetValueTypeInfo();
    }
    virtual bool read( const std::string& s ) override;
    bool copyValue(const Data<T>* parent);
    bool copyValue(const BaseData* parent) override;
    bool canBeParent(BaseData* parent) override;
};

class EmptyData : public Data<void*> {};

#if !defined(SOFA_CORE_OBJECTMODEL_DATA_CPP)
extern template class SOFA_CORE_API Data< std::string >;
extern template class SOFA_CORE_API Data< sofa::helper::vector<std::string> >;
extern template class SOFA_CORE_API Data< bool >;
extern template class SOFA_CORE_API Data< double >;
extern template class SOFA_CORE_API Data< int >;
extern template class SOFA_CORE_API Data< unsigned int >;
#endif

} // namespace sofa

namespace sofa
{
    using sofa::core::objectmodel::Data;
}

#include <sofa/core/objectmodel/Data.inl>
namespace sofa::helper
{

/// @warning the Data is updated (if needed) only by the Accessor constructor
template<class T>
class ReadAccessor< core::objectmodel::Data<T> > : public ReadAccessor<T>
{
public:
    typedef ReadAccessor<T> Inherit;
    typedef core::objectmodel::Data<T> data_container_type;
    typedef T container_type;

public:
    ReadAccessor(const data_container_type& d) : Inherit(d.getValue()) {}
    ReadAccessor(const data_container_type* d) : Inherit(d->getValue()) {}

    [[deprecated("Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    ReadAccessor(const core::ExecParams*, const data_container_type& d) : Inherit(d.getValue()) {}

    [[deprecated("Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    ReadAccessor(const core::ExecParams*, const data_container_type* d) : Inherit(d->getValue()) {}
};

/// Read/Write Accessor.
/// The Data is updated before being accessible.
/// This means an expensive chain of Data link and Engine updates can be called
/// For a pure write only Accessor, prefer WriteOnlyAccessor< core::objectmodel::Data<T> >
/// @warning the Data is updated (if needed) only by the Accessor constructor
template<class T>
class WriteAccessor< core::objectmodel::Data<T> > : public WriteAccessor<T>
{
public:
    typedef WriteAccessor<T> Inherit;
    typedef core::objectmodel::Data<T> data_container_type;
    typedef typename Inherit::container_type container_type;

    // these are forbidden (until c++11 move semantics) as they break
    // RAII encapsulation. the reference member 'data' prevents them
    // anyways, but the intent is more obvious like this.
    WriteAccessor(const WriteAccessor& );
    WriteAccessor& operator=(const WriteAccessor& );

protected:
    data_container_type& data;

    /// @internal used by WriteOnlyAccessor
    WriteAccessor( container_type* c, data_container_type& d) : Inherit(*c), data(d) {}

public:
    WriteAccessor(data_container_type& d) : Inherit(*d.beginEdit()), data(d) {}
    WriteAccessor(data_container_type* d) : Inherit(*d->beginEdit()), data(*d) {}

    [[deprecated("Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    WriteAccessor(const core::ExecParams*, data_container_type& d) : WriteAccessor(d) {}

    [[deprecated("Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    WriteAccessor(const core::ExecParams*, data_container_type* d) : WriteAccessor(d) {}
    ~WriteAccessor() { data.endEdit(); }
};



/** @brief The WriteOnlyAccessor provides an access to the Data without triggering an engine update.
 * This should be the prefered writeAccessor for most of the cases as it avoids uncessary Data updates.
 * @warning read access to the Data is NOT up-to-date
 */
template<class T>
class WriteOnlyAccessor< core::objectmodel::Data<T> > : public WriteAccessor< core::objectmodel::Data<T> >
{
public:
    typedef WriteAccessor< core::objectmodel::Data<T> > Inherit;
    typedef typename Inherit::data_container_type data_container_type;
    typedef typename Inherit::container_type container_type;

    // these are forbidden (until c++11 move semantics) as they break
    // RAII encapsulation. the reference member 'data' prevents them
    // anyways, but the intent is more obvious like this.
    WriteOnlyAccessor(const WriteOnlyAccessor& );
    WriteOnlyAccessor& operator=(const WriteOnlyAccessor& );

    WriteOnlyAccessor(data_container_type& d) : Inherit( d.beginWriteOnly(), d ) {}
    WriteOnlyAccessor(data_container_type* d) : Inherit( d->beginWriteOnly(), *d ) {}

    [[deprecated("Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    WriteOnlyAccessor(const core::ExecParams*, data_container_type& d) : Inherit( d.beginWriteOnly(), d ) {}

    [[deprecated("Aspects have been removed. If the feature was of interest for you, please contact sofa-framework")]]
    WriteOnlyAccessor(const core::ExecParams*, data_container_type* d) : Inherit( d->beginWriteOnly(), *d ) {}
};

/// Easy syntax for getting read/write access to a Data using operator ->. Example: write(someFlagData)->setFlagValue(true);
template<class T>
inline WriteAccessor<core::objectmodel::Data<T> > write(core::objectmodel::Data<T>& data, const core::ExecParams*)
{
    return WriteAccessor<core::objectmodel::Data<T> >(data);
}


template<class T>
inline WriteAccessor<core::objectmodel::Data<T> > write(core::objectmodel::Data<T>& data) 
{ 
    return write(data);
}


template<class T>
inline ReadAccessor<core::objectmodel::Data<T> > read(const core::objectmodel::Data<T>& data, const core::ExecParams*)
{
    return ReadAccessor<core::objectmodel::Data<T> >(data);
}


template<class T>
inline ReadAccessor<core::objectmodel::Data<T> > read(const core::objectmodel::Data<T>& data)
{
    return read(data);
}

/// Easy syntax for getting write only access to a Data using operator ->. Example: writeOnly(someFlagData)->setFlagValue(true);
template<class T>
inline WriteOnlyAccessor<core::objectmodel::Data<T> > writeOnly(core::objectmodel::Data<T>& data) { return WriteOnlyAccessor<core::objectmodel::Data<T> >(data); }

} // namespace sofa::helper

#endif  // SOFA_CORE_OBJECTMODEL_DATA_H

