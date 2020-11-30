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

#include <istream>
#include <sofa/core/config.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/accessor.h>
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

template<class T>
class HasDataTypeInfo
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::ValidInfo) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};


template <class T>
class Data : public BaseData
{
public:
    /// @name Construction / destruction
    /// @{

    /// This internal class is used by the initData() methods to store initialization parameters of a Data
    class InitData : public BaseData::BaseInitData
    {
    public:
        InitData() : value(*new T()) {}
        InitData(const T& v) : value{*new T(v)} {}
        InitData(const BaseData::BaseInitData& i) : BaseData::BaseInitData(i), value(*new T()) {}

        T& value;
    };

    static std::string templateName()
    {
        return sofa::core::objectmodel::BaseData::typeName<Data<T>>();
    }

    Data() : BaseData(std::string(""), false,false) {}

    // It's used for getting a new instance from an existing instance. This function is used by the communication plugin
    BaseData* getNewInstance() override { return new Data();}

    void printValue(std::ostream& out) const override;
    std::string getValueString() const override;

    /// Get info about the value type of the associated variable
    const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const override;

    bool read(const std::string& newvalue) override { return _doSetValueFromString_(newvalue); }

    [[deprecated("Never Used")]]
    bool isCounterValid() const override {return true;}

    bool copyValue(const Data<T>* parent);                ///< Fast path when you already know the content
    bool _doCopyValue_(const BaseData* parent) override;  ///< Slow copy path from the abstracted level


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


    /** \copydoc BaseData(const std::string& , bool, bool) */
    Data( const std::string& helpMsg, bool isDisplayed=true, bool isReadOnly=false)
        : BaseData(helpMsg, isDisplayed, isReadOnly)
    {
        m_value = ValueType();
    }


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
    T* beginEdit()
    {
        updateIfDirty();
        return beginWriteOnly();
    }

    /// beginWriteOnly method if it is only to write the value
    /// regardless of the current status of this value: no dirtiness check
    T* beginWriteOnly()
    {
        m_counter++;
        m_isSet=true;
        setDirtyOutputs();
        return m_value.beginEdit();
    }

    void endEdit()
    {
        m_value.endEdit();
        notifyEndEdit();
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


    /// Get current value as a void pointer (use getValueTypeInfo to find how to access it)
    const void* getValueVoidPtr() const override
    {
        return &(getValue());
    }

    /// Begin edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    void* beginEditVoidPtr() override
    {
        return beginEdit();
    }

    /// End edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    void endEditVoidPtr() override
    {
        endEdit();
    }

    void clearValue() override ;

    bool _doUpdateFromParentValue_(const BaseData*) override;
    bool _isACompatibleParent_(const BaseData* parent) override;
    bool _doSetValueFromString_(const std::string&) override;

    friend std::ostream & operator << (std::ostream &out, const Data& df)
    {
        out<<df.getValue();
        return out;
    }

    bool operator ==( const T& value ) const
    {
        return getValue()==value;
    }

    bool operator !=( const T& value ) const
    {
        return getValue()!=value;
    }

    void operator =( const T& value )
    {
        this->setValue(value);
    }

protected:

    typedef DataContentValue<T, true> ValueType;

    /// Value
    ValueType m_value;

private:
    Data(const Data& );
    Data& operator=(const Data& );
};

class EmptyData : public Data<void*> {};

/// General case for printing default value
template<class T>
inline
void Data<T>::printValue( std::ostream& out) const
{
    out << getValue() << " ";
}

/// General case for printing default value
template<class T>
inline
std::string Data<T>::getValueString() const
{
    std::ostringstream out;
    out << getValue();
    return out.str();
}

template <class T>
bool Data<T>::_doSetValueFromString_(const std::string& s)
{
    std::istringstream istr {s};
    istr >> *beginEdit();
    endEdit();
    return istr.fail();
}


template <class T>
void Data<T>::clearValue()
{
    setValue(T{});
}

template <class T>
bool Data<T>::copyValue(const Data<T>* parent)
{
    setValue(parent->getValue());
    return true;
}

template <class T>
bool Data<T>::_doCopyValue_(const BaseData* parent)
{
    auto pdata = dynamic_cast<const Data<T>*>(parent);
    if(pdata)
        return copyValue(pdata);
    return false;
}

template <class T>
bool Data<T>::_isACompatibleParent_(const BaseData* parent)
{
    if (dynamic_cast<const Data<T>*>(parent))
        return true;
    return false;
}

template <class T>
bool Data<T>::_doUpdateFromParentValue_(const BaseData* parent)
{
    auto typedParent = dynamic_cast<const Data<T>*>(parent);
    if (typedParent)
    {
        setValue(typedParent->getValue());
        return true;
    }
    return false;
}

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
inline WriteAccessor<core::objectmodel::Data<T> > write(core::objectmodel::Data<T>& data) 
{ 
    return WriteAccessor<core::objectmodel::Data<T> >(data);
}

template<class T>
inline WriteAccessor<core::objectmodel::Data<T> > write(core::objectmodel::Data<T>& data, const core::ExecParams*)
{
    return write(data);
}

template<class T>
inline ReadAccessor<core::objectmodel::Data<T> > read(const core::objectmodel::Data<T>& data)
{
    return ReadAccessor<core::objectmodel::Data<T> >(data);
}

template<class T>
inline ReadAccessor<core::objectmodel::Data<T> > read(const core::objectmodel::Data<T>& data, const core::ExecParams*)
{
    return read(data);
}


/// Easy syntax for getting write only access to a Data using operator ->. Example: writeOnly(someFlagData)->setFlagValue(true);
template<class T>
inline WriteOnlyAccessor<core::objectmodel::Data<T> > writeOnly(core::objectmodel::Data<T>& data)
{
    return WriteOnlyAccessor<core::objectmodel::Data<T> >(data);
}


} // namespace helper

// the Data class is used everywhere
using core::objectmodel::Data;

} // namespace sofa

#endif  // SOFA_CORE_OBJECTMODEL_DATA_H

