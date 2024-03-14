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
#include <sofa/defaulttype/config.h>
#include <sofa/helper/NameDecoder.h>
#include <string>

namespace sofa::defaulttype
{
/** Type traits class for objects stored in Data.

    %DataTypeInfo is part of the introspection/reflection capabilities of the
    Sofa scene graph API; it is used to manipulate Data values generically in \a
    template code, working transparently with different types of containers
    (vector, fixed_array, etc), and different types of values (integers, scalars
    (float, double), strings, etc). For example, it can be used to work with
    arrays without having to handle all the possible array classes used in Sofa:
    fixed or dynamic size, CPU or GPU, etc.

    <h4>Small example</h4>

    Iterate over the values of a DataType in templated code:

    \code{.cpp}
    template<DataType>
    MyComponent::someMethod(DataType& data) {
        const sofa::Size dim = defaulttype::DataTypeInfo<Coord>::size();
        for(sofa::Size i = 0; i < dim; ++i) {
            DataTypeInfo<DataType>::ValueType value;
            DataTypeInfo<Coord>::getValue(data, i, value);
            // [...] Do something with 'value'
        }
    }
    \endcode


    <h4>Note about size and indices</h4>

    The getValue() and setValue() methods take an index as a parameter, with the
    following conventions:

    - If a type is not a container, then the index \b must be 0.

    - Multi-dimensional containers are abstracted to a single dimension.  This
      allows iterating over any container using a single index, at the price of
      some limitations.

    \see AbstractTypeInfo provides similar mechanisms to manipulate Data objects
    generically in non-template code.
*/
template<class TDataType>
struct DataTypeInfo;

template<class TDataType>
struct DefaultDataTypeInfo
{
    /// Template parameter.
    typedef TDataType DataType;
    /// If the type is a container, this the type of the values inside this
    /// container, otherwise this is DataType.
    typedef DataType BaseType;
    /// Type of the final atomic values (i.e. the values indexed by getValue()).
    typedef DataType ValueType;
    /// TypeInfo for BaseType
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    /// TypeInfo for ValueType
    typedef DataTypeInfo<ValueType> ValueTypeInfo;
    /**
       \{
     */
    enum { ValidInfo       = 0 /**< 1 if this type has valid infos*/ };
    enum { FixedSize       = 0 /**< 1 if this type has a fixed size*/ };
    enum { ZeroConstructor = 0 /**< 1 if the constructor is equivalent to setting memory to 0*/ };
    enum { SimpleCopy      = 0 /**< 1 if copying the data can be done with a memcpy*/ };
    enum { SimpleLayout    = 0 /**< 1 if the layout in memory is simply N values of the same base type*/ };
    enum { Integer         = 0 /**< 1 if this type uses integer values*/ };
    enum { Scalar          = 0 /**< 1 if this type uses scalar values*/ };
    enum { Text            = 0 /**< 1 if this type uses text values*/ };
    enum { CopyOnWrite     = 0 /**< 1 if this type uses copy-on-write. The memory is shared with its source Data while only the source is changing (and the source modifications are then visible in the current Data). As soon as modifications are applied to the current Data, it will allocate its own value, and no longer shares memory with the source.*/ };
    enum { Container       = 0 /**< 1 if this type is a container*/ };
    enum { Size            = 1 /**< largest known fixed size for this type, as returned by size() */ };

    // \}

    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return 1; }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    template <typename T>
    static void getValue(const DataType& /*data*/, Index /*index*/, T& /*value*/)
    {
    }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template<typename T>
    static void setValue(DataType& /*data*/, Index /*index*/, const T& /*value*/)
    {
    }

    static void getValueString(const DataType& /*data*/, Index /*index*/, std::string& /*value*/)
    {
    }

    static void setValueString(DataType& /*data*/, Index /*index*/, const std::string& /*value*/)
    {
    }

    static const void* getValuePtr(const DataType& /*type*/)
    {
        return nullptr;
    }

    static void* getValuePtr(DataType& /*type*/)
    {
        return nullptr;
    }

    static const std::string name() { return GetTypeName(); }
    static const std::string GetTypeName() { return sofa::helper::NameDecoder::decodeTypeName(typeid(DataType)); }
};

template<class TDataType>
struct DataTypeInfo : public DefaultDataTypeInfo<TDataType>
{
};

template<class T>
class DataTypeName : public DataTypeInfo<T> {};


} /// namespace sofa::defaulttype
