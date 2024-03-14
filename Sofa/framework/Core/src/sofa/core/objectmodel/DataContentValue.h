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

#include <sofa/config.h>
#include <memory>

namespace sofa::core::objectmodel
{

/// To handle the Data link:
/// - CopyOnWrite==false: an independent copy (duplicated memory)
/// - CopyOnWrite==true: shared memory while the Data is not modified (in that case the memory is duplicated to get an independent copy)
template <class T, bool CopyOnWrite>
class DataContentValue;

template <class T>
class DataContentValue<T, false>
{
    T data;
public:

    DataContentValue()
        : data(T())// BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    explicit DataContentValue(const T &value)
        : data(value)
    {
    }

    DataContentValue(const DataContentValue& dc)
        : data(dc.getValue())
    {
    }

    DataContentValue& operator=(const DataContentValue& dc )
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
class DataContentValue<T, true>
{
    std::shared_ptr<T> ptr;
public:

    DataContentValue()
        : ptr(new T(T())) // BUGFIX (Jeremie A.): Force initialization of basic types to 0 (bool, int, float, etc).
    {
    }

    explicit DataContentValue(const T& value)
        : ptr(new T(value))
    {
    }

    DataContentValue(const DataContentValue& dc)
        : ptr(dc.ptr) // start with shared memory
    {
    }

    ~DataContentValue()
    {
    }

    DataContentValue& operator=(const DataContentValue& dc )
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
        if(!(ptr.use_count() == 1))
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


}

