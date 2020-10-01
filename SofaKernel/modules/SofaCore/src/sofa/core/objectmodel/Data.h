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
#include <sofa/core/objectmodel/DataContentValue.h>
namespace sofa
{

namespace core
{

namespace objectmodel
{

/** \brief Abstract base class template for Data. */
template < class T >
class TData : public BaseData
{
public:
    typedef T value_type;


    explicit TData(const BaseInitData& init) : BaseData(init)
    {
    }

    //TODO(dmarchal:08/10/2019)Uncomment the deprecated when VS2015 support will be dropped. 
    //[[deprecated("Replaced with one with std::string instead of char* version")]]
    TData( const char* helpMsg=nullptr, bool isDisplayed=true, bool isReadOnly=false) :
        TData( sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly) {}

    TData( const std::string& helpMsg, bool isDisplayed=true, bool isReadOnly=false)
        : BaseData(helpMsg, isDisplayed, isReadOnly)
    {
    }


    ~TData() override
    {}

    inline void printValue(std::ostream& out) const override;
    inline std::string getValueString() const override;
    inline std::string getValueTypeString() const override;

    /// Get info about the value type of the associated variable
    const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const override
    {
        return sofa::defaulttype::VirtualTypeInfo<T>::get();
    }

    virtual const T& virtualGetValue() const = 0;
    virtual void virtualSetValue(const T& v) = 0;
    virtual void virtualSetLink(const BaseData& bd) = 0;
    virtual T* virtualBeginEdit() = 0;
    virtual void virtualEndEdit() = 0;

    /// Get current value as a void pointer (use getValueTypeInfo to find how to access it)
    const void* getValueVoidPtr() const override
    {
        return &(virtualGetValue());
    }

    /// Begin edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    void* beginEditVoidPtr() override
    {
        return virtualBeginEdit();
    }

    /// End edit current value as a void pointer (use getValueTypeInfo to find how to access it)
    void endEditVoidPtr() override
    {
        virtualEndEdit();
    }

    /** Try to read argument value from an input stream.
    Return false if failed
     */
    virtual bool read( const std::string& s ) override;

    bool isCounterValid() const override {return true;}

    bool copyValue(const TData<T>* parent);

    bool copyValue(const BaseData* parent) override;

    bool validParent(BaseData* parent) override;

protected:

    bool updateFromParentValue(const BaseData* parent) override;
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
class Data : public TData<T>
{
public:
    using TData<T>::m_counter;
    using TData<T>::m_isSet;
    using TData<T>::setDirtyOutputs;
    using TData<T>::updateIfDirty;
    using TData<T>::notifyEndEdit;

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
    virtual BaseData* getNewInstance() { return new Data();}

    /** \copydoc BaseData(const BaseData::BaseInitData& init) */
    explicit Data(const BaseData::BaseInitData& init)
        : TData<T>(init)
    {
    }

    /** \copydoc Data(const BaseData::BaseInitData&) */
    explicit Data(const InitData& init)
        : TData<T>(init)
    {
        m_value = ValueType(init.value);
    }

    /** \copydoc BaseData(const char*, bool, bool) */
    //[[deprecated("Replaced with one with std::string instead of char* version")]]
    Data( const char* helpMsg=nullptr, bool isDisplayed=true, bool isReadOnly=false)
        : Data(sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly) {}

    /** \copydoc BaseData(const std::string& , bool, bool) */
    Data( const std::string& helpMsg, bool isDisplayed=true, bool isReadOnly=false)
        : TData<T>(helpMsg, isDisplayed, isReadOnly)
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
        : TData<T>(helpMsg, isDisplayed, isReadOnly)
    {
        m_value = ValueType(value);
    }

    /// Destructor.
    virtual ~Data() {}

    /// @}

    /// @name Simple edition and retrieval API
    /// @{

    /// BeginEdit method if it is only to write the value
    inline T* beginEdit()
    {
        updateIfDirty();
        m_counter++;
        m_isSet = true;
        BaseData::setDirtyOutputs();
        return m_value.beginEdit();
    }

    inline T* beginWriteOnly()
    {
        m_counter++;
        m_isSet=true;
        BaseData::setDirtyOutputs();
        return m_value.beginEdit();
    }

    inline void endEdit()
    {
        m_value.endEdit();
        BaseData::notifyEndEdit();
    }

    /// @warning writeOnly (the Data is not updated before being set)
    inline void setValue(const T& value)
    {
        *beginWriteOnly()=value;
        endEdit();
    }

    inline const T& getValue() const
    {
        updateIfDirty();
        return m_value.getValue();
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    inline void endEdit(const core::ExecParams*)
    {
        endEdit();
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    inline T* beginWriteOnly(const core::ExecParams*)
    {
        return beginWriteOnly();
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    inline T* beginEdit(const core::ExecParams*)
    {
        return beginEdit();
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    inline void setValue(const core::ExecParams*, const T& value)
    {
        setValue(value);
    }

    [[deprecated("2020-03-25: Aspect have been deprecated for complete removal in PR #1269. You can probably update your code by removing aspect related calls. If the feature was important to you contact sofa-dev. ")]]
    inline const T& getValue(const core::ExecParams*) const
    {
        return getValue();
    }


    /// @}

    /// @name Virtual edition and retrieval API (for generic TData parent API, deprecated)
    /// @{

    virtual const T& virtualGetValue() const { return getValue(); }
    virtual void virtualSetValue(const T& v) { setValue(v); }

    virtual void virtualSetLink(const BaseData& bd)
    {
        const Data<T>* d = dynamic_cast< const Data<T>* >(&bd);
        if (d)
        {
            m_value = d->m_value;
            m_counter++;
            m_isSet = true;
            BaseData::setDirtyOutputs();
        }
    }

    virtual T* virtualBeginEdit() { return beginEdit(); }
    virtual void virtualEndEdit() { endEdit(); }

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
};

class EmptyData : public Data<void*> {};

/// Specialization for reading strings
template<>
bool TData<std::string>::read( const std::string& str );


/// Specialization for reading booleans
template<>
bool TData<bool>::read( const std::string& str );


/// General case for printing default value
template<class T>
inline
void TData<T>::printValue( std::ostream& out) const
{
    out << virtualGetValue() << " ";
}

/// General case for printing default value
template<class T>
inline
std::string TData<T>::getValueString() const
{
    std::ostringstream out;
    out << virtualGetValue();
    return out.str();
}

template<class T>
inline
std::string TData<T>::getValueTypeString() const
{
    return BaseData::typeName(&virtualGetValue());
}

template <class T>
bool TData<T>::read(const std::string& s)
{
    if (s.empty())
    {
        bool resized = getValueTypeInfo()->setSize( virtualBeginEdit(), 0 );
        virtualEndEdit();
        return resized;
    }
    std::istringstream istr( s.c_str() );
    istr >> *virtualBeginEdit();
    virtualEndEdit();
    if( istr.fail() )
    {
        return false;
    }
    return true;
}

template <class T>
bool TData<T>::copyValue(const TData<T>* parent)
{
    virtualSetValue(parent->virtualGetValue());
    return true;
}

template <class T>
bool TData<T>::copyValue(const BaseData* parent)
{
    const TData<T>* p = dynamic_cast<const TData<T>*>(parent);
    if (p)
    {
        virtualSetValue(p->virtualGetValue());
        return true;
    }
    return BaseData::copyValue(parent);
}

template <class T>
bool TData<T>::validParent(BaseData* parent)
{
    if (dynamic_cast<TData<T>*>(parent))
        return true;
    return BaseData::validParent(parent);
}


template <class T>
bool TData<T>::updateFromParentValue(const BaseData* parent)
{
    auto typedParent = dynamic_cast<const TData<T>*>(parent);
    if (typedParent)
    {
        virtualSetLink(*parent);
        return true;
    }
    else
        return BaseData::updateFromParentValue(parent);
}

#if  !defined(SOFA_CORE_OBJECTMODEL_DATA_CPP)

extern template class SOFA_CORE_API TData< std::string >;
extern template class SOFA_CORE_API Data< std::string >;
extern template class SOFA_CORE_API TData< sofa::helper::vector<std::string> >;
extern template class SOFA_CORE_API Data< sofa::helper::vector<std::string> >;
extern template class SOFA_CORE_API TData< bool >;
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

