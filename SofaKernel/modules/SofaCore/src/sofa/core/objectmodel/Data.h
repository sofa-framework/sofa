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
#ifndef SOFA_CORE_OBJECTMODEL_DATA_H
#define SOFA_CORE_OBJECTMODEL_DATA_H

#include <sofa/core/config.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/accessor.h>
#include <istream>
#include <sofa/core/objectmodel/DataContentValue.h>
namespace sofa
{

namespace core
{

namespace objectmodel
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
    virtual ~Data() {}

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

    SOFA_BEGIN_DEPRECATION_AS_ERROR

    [[deprecated("2021-01-21: virtualSetLink has been deprecated. You can update your code by using copyValueFrom() or setParent() depending on the expected behavior.")]]
    void virtualSetLink(const BaseData& bd) { BaseData::copyValueFrom(&bd); }

    [[deprecated("2021-01-21: virtualGetValue has been deprecated. You can update your code by using .")]]
    void virtualSetValue(const T& v){ setValue(v); }

    [[deprecated("2021-01-21: virtualGetValue has been deprecated. You can update your code by using .")]]
    const T& virtualGetValue() { return getValue(); }

    [[deprecated("2021-01-21: virtualBeginEdit has been deprecated. You can update your code by using .")]]
    T* virtualBeginEdit() { return beginEdit(); }

    [[deprecated("2021-01-21: virtualEndEdit has been deprecated. You can update your code by using .")]]
    void virtualEndEdit() { return endEdit(); }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    void endEdit(const core::ExecParams*)
    {
        endEdit();
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    T* beginWriteOnly(const core::ExecParams*)
    {
        return beginWriteOnly();
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    T* beginEdit(const core::ExecParams*)
    {
        return beginEdit();
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    void setValue(const core::ExecParams*, const T& value)
    {
        setValue(value);
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    const T& getValue(const core::ExecParams*) const
    {
        return getValue();
    }
    SOFA_END_DEPRECATION_AS_ERROR

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
    std::string getValueTypeString() const override;

    friend std::ostream & operator << (std::ostream &out, const Data& df)
    {
        out<<df.getValue();
        return out;
    }

    [[deprecated("Deprecated before definitive removal (see PR#1639). Please update your code by replacing 'myData == aValue' with 'myData.getValue() == aValue'")]]
    bool operator ==( const T& value ) const
    {
        return getValue()==value;
    }

    [[deprecated("Deprecated before definitive removal (see PR#1639). Please update your code by replacing 'myData != aValue' with 'myData.getValue() != aValue'")]]
    bool operator!=( const T& value ) const {return getValue()!=value;}
            void operator =( const T& value )
    {
            this->setValue(value);
    }

    bool copyValueFrom(const BaseData* data){ return doCopyValueFrom(data); }
    bool copyValueFrom(const Data<T>* data);

    bool isCopyOnWrite(){ return sofa::defaulttype::DataTypeInfo<T>::CopyOnWrite; }

protected:
    typedef DataContentValue<T, sofa::defaulttype::DataTypeInfo<T>::CopyOnWrite> ValueType;

    /// Value
    ValueType m_value;

private:
    Data(const Data& );
    Data& operator=(const Data& );

    bool doIsExactSameDataType(const BaseData* parent) override;
    bool doCopyValueFrom(const BaseData* parent) override;
    bool doSetValueFromLink(const BaseData* parent) override;
    const void* doGetValueVoidPtr() const override { return &getValue(); }
    void* doBeginEditVoidPtr() override  { return beginEdit(); }
    void doEndEditVoidPtr() override  { endEdit(); }
};

class EmptyData : public Data<void*> {};

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

/// General case for printing default value
template<class T>
std::string Data<T>::getValueString() const
{
    std::ostringstream out;
    out << getValue();
    return out.str();
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
        bool resized = getValueTypeInfo()->setSize( BaseData::beginEditVoidPtr(), 0 );
        BaseData::endEditVoidPtr();
        return resized;
    }
    std::istringstream istr( s.c_str() );
    istr >> *beginEdit();
    endEdit();
    if( istr.fail() )
    {
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

#if  !defined(SOFA_CORE_OBJECTMODEL_DATA_CPP)
extern template class SOFA_CORE_API Data< std::string >;
extern template class SOFA_CORE_API Data< sofa::helper::vector<std::string> >;
extern template class SOFA_CORE_API Data< bool >;
#endif

} // namespace objectmodel

} // namespace core

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

public:
    ReadAccessor(const data_container_type& d) : Inherit(d.getValue()) {}
    ReadAccessor(const data_container_type* d) : Inherit(d->getValue()) {}

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    ReadAccessor(const core::ExecParams*, const data_container_type& d) : Inherit(d.getValue()) {}

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
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

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    WriteAccessor(const core::ExecParams*, data_container_type& d) : WriteAccessor(d) {}

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
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

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    WriteOnlyAccessor(const core::ExecParams*, data_container_type& d) : Inherit( d.beginWriteOnly(), d ) {}

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    WriteOnlyAccessor(const core::ExecParams*, data_container_type* d) : Inherit( d->beginWriteOnly(), *d ) {}
};

/// Easy syntax for getting read/write access to a Data using operator ->. Example: write(someFlagData)->setFlagValue(true);
template<class T>
WriteAccessor<core::objectmodel::Data<T> > write(core::objectmodel::Data<T>& data)
{ 
    return WriteAccessor<core::objectmodel::Data<T> >(data);
}

template<class T>
WriteAccessor<core::objectmodel::Data<T> > write(core::objectmodel::Data<T>& data, const core::ExecParams*)
{
    return write(data);
}

template<class T>
ReadAccessor<core::objectmodel::Data<T> > read(const core::objectmodel::Data<T>& data)
{
    return ReadAccessor<core::objectmodel::Data<T> >(data);
}

template<class T>
ReadAccessor<core::objectmodel::Data<T> > read(const core::objectmodel::Data<T>& data, const core::ExecParams*)
{
    return read(data);
}


/// Easy syntax for getting write only access to a Data using operator ->. Example: writeOnly(someFlagData)->setFlagValue(true);
template<class T>
WriteOnlyAccessor<core::objectmodel::Data<T> > writeOnly(core::objectmodel::Data<T>& data)
{
    return WriteOnlyAccessor<core::objectmodel::Data<T> >(data);
}


} // namespace helper

// the Data class is used everywhere
using core::objectmodel::Data;

} // namespace sofa

#endif  // SOFA_CORE_OBJECTMODEL_DATA_H

