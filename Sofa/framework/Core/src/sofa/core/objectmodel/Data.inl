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

#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/defaulttype/TypeInfoRegistry.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/StringUtils.h>
#include <istream>

namespace sofa::core::objectmodel
{

template <class T>
std::string Data<T>::templateName()
{
    return GetValueTypeInfo()->getTypeName();
}

template <class T>
BaseData* Data<T>::getNewInstance() 
{ 
    return new Data(); 
}

template <class T>
Data<T>::Data(const BaseData::BaseInitData& init) : BaseData(init)
{
}

template <class T>
Data<T>::Data(const InitData& init) : BaseData(init)
{
    m_value = ValueType(init.value);
    m_hasDefaultValue = true;
}

template <class T>
Data<T>::Data( const char* helpMsg, bool isDisplayed, bool isReadOnly)
    : Data(sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly) {}

template<class T>
Data<T>::Data( const std::string& helpMsg, bool isDisplayed, bool isReadOnly)
    : BaseData(helpMsg, isDisplayed, isReadOnly)
{
    m_value = ValueType();
}

template <class T>
Data<T>::Data( const T& value, const char* helpMsg, bool isDisplayed, bool isReadOnly) :
    Data(value, sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly)
{}



template <class T>
T* Data<T>::beginEdit()
{
    updateIfDirty();
    return beginWriteOnly();
}

/// beginWriteOnly method if it is only to write the value
/// regardless of the current status of this value: no dirtiness check
template <class T>
T* Data<T>::beginWriteOnly()
{
    m_counter++;
    m_isSet=true;
    BaseData::setDirtyOutputs();
    return m_value.beginEdit();
}

template <class T>
void Data<T>::endEdit()
{
    m_value.endEdit();
    BaseData::notifyEndEdit();
}

/// @warning writeOnly (the Data is not updated before being set)
template <class T>
void Data<T>::setValue(const T& value)
{
    *beginWriteOnly()=value;
    endEdit();
}

template<class T>
const T& Data<T>::getValue() const
{
    updateIfDirty();
    return m_value.getValue();
}

template<class T>
const defaulttype::AbstractTypeInfo* Data<T>::GetValueTypeInfo()
{
    return GetValueTypeInfoWithCompatibilityLayer();
}

template<class T>
const sofa::defaulttype::AbstractTypeInfo* Data<T>::getValueTypeInfo() const 
{
    return GetValueTypeInfoWithCompatibilityLayer();
}

template<class T>
void Data<T>::operator=( const T& value )
{
    this->setValue(value);
}

template<class T>
const void* Data<T>::doGetValueVoidPtr() const { return &getValue(); }

template<class T>
void* Data<T>::doBeginEditVoidPtr() { return beginEdit(); }

template<class T>
void Data<T>::doEndEditVoidPtr() { endEdit(); }

template<class T>
bool Data<T>::AbstractTypeInfoRegistration()
{
    auto info = sofa::defaulttype::VirtualTypeInfo<T>::get();
    sofa::defaulttype::TypeInfoRegistry::Set(sofa::defaulttype::TypeInfoId::GetTypeId<T>(), info, sofa_tostring(SOFA_TARGET));
    dmsg_deprecated("Data") << "registration type for " << info->getTypeName();                 
    return false;        
}

template<class T>
const sofa::defaulttype::AbstractTypeInfo* Data<T>::GetValueTypeInfoWithCompatibilityLayer()
{ 
    static bool __inited__ = AbstractTypeInfoRegistration();        
    SOFA_UNUSED(__inited__);
    return sofa::defaulttype::TypeInfoRegistry::Get(sofa::defaulttype::TypeInfoId::GetTypeId<T>()); 
}

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
SOFA_ATTRIBUTE_DEPRECATED__DATA_TYPEINFOAPI("Use sofa::getValueTypeInfo()->getTypeName().")
std::string Data<T>::getValueTypeString() const
{
    return getValueTypeInfo()->getTypeName();
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
    std::stringstream cerrbuffer;
    std::streambuf* old = std::cerr.rdbuf(cerrbuffer.rdbuf());
    
    istr >> *beginEdit();
    endEdit();
    
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

} // namespace core::objectmodel
