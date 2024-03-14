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
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/accessor.h>
#include <istream>
#include <sofa/core/objectmodel/DataContentValue.h>
namespace sofa
{
namespace core::objectmodel
{

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
    using value_type = T;

    using BaseData::m_counter;
    using BaseData::m_isSet;
    using BaseData::setDirtyOutputs;
    using BaseData::updateIfDirty;
    using BaseData::notifyEndEdit;

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

    static std::string templateName()
    {
        return sofa::core::objectmodel::BaseData::typeName<Data<T>>();
    }

    // It's used for getting a new instance from an existing instance. This function is used by the communication plugin
    BaseData* getNewInstance() override { return new Data();}

    /** \copydoc BaseData(const BaseData::BaseInitData& init) */
    explicit Data(const BaseData::BaseInitData& init) : BaseData(init)
    {
    }

    /** \copydoc Data(const BaseData::BaseInitData&) */
    explicit Data(const InitData& init) : BaseData(init)
    {
        m_value = ValueType(init.value);
        m_hasDefaultValue = true;
        m_defaultValue = init.value;
    }

    /** \copydoc BaseData(const char*, bool, bool) */
    //[[deprecated("Replaced with one with std::string instead of char* version")]]
    Data( const char* helpMsg=nullptr, bool isDisplayed=true, bool isReadOnly=false)
        : Data(sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly) {}

    /** \copydoc BaseData(const std::string& , bool, bool) */
    Data( const std::string& helpMsg, bool isDisplayed=true, bool isReadOnly=false)
        : BaseData(helpMsg, isDisplayed, isReadOnly)
    {
        m_value = ValueType();
    }

    /** \copydoc BaseData(const char*, bool, bool)
     *  \param value The default value.
     */
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
    ~Data() override {}

    /// @}

    /// @name Simple edition and retrieval API
    /// @{

    /// BeginEdit method if it is only to write the value
    /// checking that current value is up to date
    virtual T* beginEdit()
    {
        updateIfDirty();
        return beginWriteOnly();
    }

    /// beginWriteOnly method if it is only to write the value
    /// regardless of the current status of this value: no dirtiness check
    virtual T* beginWriteOnly()
    {
        m_counter++;
        m_isSet=true;
        BaseData::setDirtyOutputs();
        return m_value.beginEdit();
    }

    virtual void endEdit()
    {
        m_value.endEdit();
        BaseData::notifyEndEdit();
    }

    /// @warning writeOnly (the Data is not updated before being set)
    void setValue(const T& value)
    {
        *beginWriteOnly()=value;
        endEdit();
    }

    const T& getValue() const
    {
        updateIfDirty();
        return m_value.getValue();
    }

    /// @}

    /// Get info about the value type of the associated variable
    const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const override
    {
        return sofa::defaulttype::VirtualTypeInfo<T>::get();
    }

    /** Try to read argument value from an input stream.
    Return false if failed
     */
    bool read( const std::string& s ) override;
    void printValue(std::ostream& out) const override;
    std::string getValueString() const override;
    std::string getDefaultValueString() const override;
    std::string getValueTypeString() const override;

    void operator =( const T& value )
    {
        this->setValue(value);
    }

    bool copyValueFrom(const Data<T>* data);

    static constexpr bool isCopyOnWrite(){ return !std::is_scalar_v<T>; }

    Data(const Data& ) = delete;
    Data& operator=(const Data& ) = delete;

protected:
    typedef DataContentValue<T,  !std::is_scalar_v<T>> ValueType;
    T m_defaultValue;

    /// Value
    ValueType m_value;

    std::istream& readValue(std::istream& in);

private:


    bool doIsExactSameDataType(const BaseData* parent) override;
    bool doCopyValueFrom(const BaseData* parent) override;
    bool doSetValueFromLink(const BaseData* parent) override;
    const void* doGetValueVoidPtr() const override { return &getValue(); }
    void* doBeginEditVoidPtr() override  { return beginEdit(); }
    void doEndEditVoidPtr() override  { endEdit(); }
};

class EmptyData : public Data<void*> {};

template <class T>
std::istream& Data<T>::readValue(std::istream& in)
{
    in >> *beginEdit();
    endEdit();
    return in;
}

/// Specialization for reading strings
template<>
bool Data<std::string>::read( const std::string& str );

/// Specialization for reading booleans
template<>
bool Data<bool>::read( const std::string& str );


/// General case for printing default value
template<class T>
void Data<T>::printValue( std::ostream& out) const
{
    out << getValue() << " ";
}

/// General case for printing value
template<class T>
std::string Data<T>::getValueString() const
{
    std::ostringstream out;
    out << getValue();
    return out.str();
}

/// General case for printing default value
template<class T>
std::string Data<T>::getDefaultValueString() const
{
    if (hasDefaultValue())
    {
        std::ostringstream out;
        out << this->m_defaultValue;
        return out.str();
    }
    return {};
}

template<class T>
std::string Data<T>::getValueTypeString() const
{
    return BaseData::typeName(&getValue());
}

template <class T>
bool Data<T>::read(const std::string& s)
{
    if (s.empty())
    {
        const bool resized = getValueTypeInfo()->setSize( BaseData::beginEditVoidPtr(), 0 );
        BaseData::endEditVoidPtr();
        return resized;
    }
    std::istringstream istr( s.c_str() );

    // capture std::cerr output (if any)
    const std::stringstream cerrbuffer;
    std::streambuf* old = std::cerr.rdbuf(cerrbuffer.rdbuf());

    readValue(istr);

    // restore the previous cerr
    std::cerr.rdbuf(old);

    if( istr.fail() )
    {
        // transcript the std::cerr buffer into the Messaging system
        msg_warning(this->getName()) << cerrbuffer.str();

        return false;
    }

    return true;
}

template <class T>
bool Data<T>::copyValueFrom(const Data<T>* data)
{
    setValue(data->getValue());
    return true;
}

template <class T>
bool Data<T>::doCopyValueFrom(const BaseData* data)
{
    const Data<T>* typedata = dynamic_cast<const Data<T>*>(data);
    if(!typedata)
        return false;

    return copyValueFrom(typedata);
}

template <class T>
bool Data<T>::doSetValueFromLink(const BaseData* data)
{
    const Data<T>* typedata = dynamic_cast<const Data<T>*>(data);
    if(!typedata)
        return false;

    m_value = typedata->m_value;
    m_counter++;
    m_isSet = true;
    BaseData::setDirtyOutputs();
    return true;
}


template <class T>
bool Data<T>::doIsExactSameDataType(const BaseData* parent)
{
    return dynamic_cast<const Data<T>*>(parent) != nullptr;
}

#if !defined(SOFA_CORE_OBJECTMODEL_DATA_CPP)
extern template class SOFA_CORE_API Data< std::string >;
extern template class SOFA_CORE_API Data< sofa::type::vector<std::string> >;
extern template class SOFA_CORE_API Data< sofa::type::vector<Index> >;
extern template class SOFA_CORE_API Data< bool >;
#endif
} // namespace core::objectmodel

// Overload helper::ReadAccessor and helper::WriteAccessor

namespace helper
{


/// @warning the Data is updated (if needed) only by the Accessor constructor
template<class T>
class ReadAccessor< core::objectmodel::Data<T> > : public ReadAccessor<T>
{
public:
    typedef ReadAccessor<T> Inherit;
    typedef core::objectmodel::Data<T> data_container_type;
    typedef T container_type;

    ReadAccessor(const data_container_type& d) : Inherit(d.getValue()) {}
    ReadAccessor(const data_container_type* d) : Inherit(d->getValue()) {}
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
    WriteAccessor(const WriteAccessor& ) = delete;
    WriteAccessor& operator=(const WriteAccessor& ) = delete;

protected:
    data_container_type& data;

    /// @internal used by WriteOnlyAccessor
    WriteAccessor( container_type* c, data_container_type& d) : Inherit(*c), data(d) {}

public:
    WriteAccessor(data_container_type& d) : Inherit(*d.beginEdit()), data(d) {}
    WriteAccessor(data_container_type* d) : Inherit(*d->beginEdit()), data(*d) {}
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
    WriteOnlyAccessor(const WriteOnlyAccessor& ) = delete;
    WriteOnlyAccessor& operator=(const WriteOnlyAccessor& ) = delete;

    WriteOnlyAccessor(data_container_type& d) : Inherit( d.beginWriteOnly(), d ) {}
    WriteOnlyAccessor(data_container_type* d) : Inherit( d->beginWriteOnly(), *d ) {}
};


/// Returns a write only accessor from the provided Data<>
/// Example of use:
///   auto points = getWriteOnlyAccessor(d_points)
template<class T>
WriteAccessor<core::objectmodel::Data<T> > getWriteAccessor(core::objectmodel::Data<T>& data)
{ 
    return WriteAccessor<core::objectmodel::Data<T> >(data);
}


/// Returns a read accessor from the provided Data<>
/// Example of use:
///   auto points = getReadAccessor(d_points)
template<class T>
ReadAccessor<core::objectmodel::Data<T> > getReadAccessor(const core::objectmodel::Data<T>& data)
{
    return ReadAccessor<core::objectmodel::Data<T> >(data);
}

/// Returns a write only accessor from the provided Data<>
/// WriteOnly accessors are faster than WriteAccessor because
/// as the data is only read this means there is no need to pull
/// the data from the parents
/// Example of use:
///   auto points = getWriteOnlyAccessor(d_points)
template<class T>
WriteOnlyAccessor<core::objectmodel::Data<T> > getWriteOnlyAccessor(core::objectmodel::Data<T>& data)
{
    return WriteOnlyAccessor<core::objectmodel::Data<T> >(data);
}

} // namespace helper

// the Data class is used everywhere
using core::objectmodel::Data;

} // namespace sofa
