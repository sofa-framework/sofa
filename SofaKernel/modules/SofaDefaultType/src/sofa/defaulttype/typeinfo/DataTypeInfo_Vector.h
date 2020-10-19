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

#include <sofa/defaulttype/typeinfo/models/VectorTypeInfo.h>
#include <sofa/helper/vector.h>

namespace sofa::defaulttype
{


template<class T, class Alloc>
struct DataTypeInfo< std::vector<T,Alloc> > : public VectorTypeInfo<std::vector<T,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "std::vector<" << DataTypeInfo<T>::name() << ">"; return o.str(); }
};

template<class T, class Alloc>
struct DataTypeInfo< sofa::helper::vector<T,Alloc> > : public VectorTypeInfo<sofa::helper::vector<T,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "vector<" << DataTypeInfo<T>::name() << ">"; return o.str(); }
};

// vector<bool> is a bitset, cannot get a pointer to the values
template<class Alloc>
struct DataTypeInfo< sofa::helper::vector<bool,Alloc> > : public VectorTypeInfo<sofa::helper::vector<bool,Alloc> >
{
    enum { SimpleLayout = 0 };

    static std::string name() { std::ostringstream o; o << "vector<bool>"; return o.str(); }

    static const void* getValuePtr(const sofa::helper::vector<bool,Alloc>& /*data*/) { return nullptr; }
    static void* getValuePtr(sofa::helper::vector<bool,Alloc>& /*data*/) { return nullptr; }
};

// Cannot use default impl of VectorTypeInfo for non-fixed size BaseTypes
template<class Alloc>
struct DataTypeInfo< sofa::helper::vector<std::string,Alloc> > : public VectorTypeInfo<sofa::helper::vector<std::string,Alloc> >
{
    static std::string name() { return "vector<string>"; }

    // BaseType size is not fixed. Returning 1
    static size_t size() { return 1; }

    // Total number of elements in the vector
    static size_t size(const sofa::helper::vector<std::string,Alloc>& data) { return data.size(); }

    // Resizes the vector
    static bool setSize(sofa::helper::vector<std::string,Alloc>& data, size_t size) { data.resize(size); return true; }

    // Sets the value for element at index `index`
    static void setValueString(sofa::helper::vector<std::string,Alloc>& data, size_t index, const std::string& value)
    {
        if (data.size() <= index)
            data.resize(index + 1);
        data[index] = value;
    }

    // Gets the value for element at index `index`
    static void getValueString(const sofa::helper::vector<std::string,Alloc>& data, size_t index, std::string& value)
    {
        if (data.size() <= index)
            msg_error("DataTypeInfo<helper::vector<std::string>") << "Index out of bounds for getValueString";
        else
            value = data[index];
    }
};

} /// namespace sofa::defaulttype

