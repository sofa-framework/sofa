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

#include <sofa/type/vector.h>
#include <sofa/defaulttype/typeinfo/models/VectorTypeInfo.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::defaulttype
{

template<class T, class Alloc>
struct DataTypeInfo< sofa::type::vector<T,Alloc> > : public VectorTypeInfo<sofa::type::vector<T,Alloc> >
{
};

// vector<bool> is a bitset, cannot get a pointer to the values
template<class Alloc>
struct DataTypeInfo< sofa::type::vector<bool,Alloc> > : public VectorTypeInfo<sofa::type::vector<bool,Alloc> >
{
    enum { SimpleLayout = 0 };
    static const void* getValuePtr(const sofa::type::vector<bool,Alloc>& /*data*/) { return nullptr; }
    static void* getValuePtr(sofa::type::vector<bool,Alloc>& /*data*/) { return nullptr; }
};

// Cannot use default impl of VectorTypeInfo for non-fixed size BaseTypes
template<class Alloc>
struct DataTypeInfo< sofa::type::vector<std::string,Alloc> > : public VectorTypeInfo<sofa::type::vector<std::string,Alloc> >
{
    // BaseType size is not fixed. Returning 1
    static sofa::Size size() { return 1; }

    // Total number of elements in the vector
    static sofa::Size size(const sofa::type::vector<std::string,Alloc>& data) { return sofa::Size(data.size()); }

    // Resizes the vector
    static bool setSize(sofa::type::vector<std::string,Alloc>& data, sofa::Size size) { data.resize(size); return true; }

    // Sets the value for element at index `index`
    static void setValueString(sofa::type::vector<std::string,Alloc>& data, sofa::Index index, const std::string& value)
    {
        if (data.size() <= index)
            data.resize(index + 1);
        data[index] = value;
    }

    // Gets the value for element at index `index`
    static void getValueString(const sofa::type::vector<std::string,Alloc>& data, sofa::Index index, std::string& value)
    {
        if (data.size() <= index)
            msg_error("DataTypeInfo<type::vector<std::string>") << "Index out of bounds for getValueString";
        else
            value = data[index];
    }
};

} /// namespace sofa::defaulttype

