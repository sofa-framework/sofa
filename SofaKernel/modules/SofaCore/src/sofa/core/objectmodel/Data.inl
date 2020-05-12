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
#pragma once

#include <sofa/core/objectmodel/Data.h>

namespace sofa::core::objectmodel
{

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

template<class T>
inline
std::string Data<T>::getValueTypeString() const
{
    return getValueTypeInfo()->name();
}

template <class T>
bool Data<T>::read(const std::string& s)
{
    if (s.empty())
    {
        bool resized = getValueTypeInfo()->setSize( beginEdit(), 0 );
        endEdit();
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
bool Data<T>::copyValue(const Data<T>* parent)
{
    setValue(parent->getValue());
    return true;
}

template <class T>
bool Data<T>::copyValue(const BaseData* parent)
{
    const Data<T>* p = dynamic_cast<const Data<T>*>(parent);
    if (p)
    {
        setValue(p->getValue());
        return true;
    }
    return BaseData::copyValue(parent);
}

template <class T>
bool Data<T>::canBeParent(BaseData* parent)
{
    if (dynamic_cast<Data<T>*>(parent))
        return true;
    return BaseData::canBeParent(parent);
}


//template <class T>
//bool Data<T>::updateFromParentValue(const BaseData* parent)
//{
//    if (parent == parentData.get())
//    {
//        copyValue(parentData.get());
//        return true;
//    }
//    else
//        return BaseData::updateFromParentValue(parent);
//}

} /// namespace sofa::core::objectmodel
