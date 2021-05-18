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
#include <string>
#include <typeinfo>

namespace sofa::defaulttype
{
    class TypeInfoId;

/** Information about the type of a value stored in a Data.

    %AbstractTypeInfo is part of the introspection/reflection capabilities of
    the Sofa scene graph API. It provides information about the type of the
    content of Data objects (Is it a simple type?  A container? How much memory
    should be allocated to copy it?), and allows manipulating Data generically,
    without knowing their exact type.

    This class is primarily used to copy information accross BaseData objects,
    for example when there exists a link between two instances of BaseData.
    E.g. this mecanism allows you to copy the content of a Data<vector<int>>
    into a Data<vector<double>>, because there is an acceptable conversion
    between integer and double, and because both Data use a resizable container.

    <h4>Using TypeInfo</h4>

    Use BaseData::getValueTypeInfo() to get a pointer to an AbtractTypeInfo, and
    BaseData::getValueVoidPtr() to get a pointer to the content of a Data. You
    can then use the methods of AbtractTypeInfo to access the Data generically.

    Very basic example:
    \code{.cpp}
    BaseData *data = getADataFromSomewhere();
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* ptr = data->getValueVoidPtr();
    for (int i = 0 ; i < typeinfo->size(ptr) ; i++)
    std::string value = typeinfo->getTextValue(ptr, 0);
    \endcode

    <h4>Note about size and indices</h4>

    All the getValue() and setValue() methods take an index as a parameter,
    which means that every type is abstracted to a one-dimensional container.
    See the detailed description of DataTypeInfo for more explanations.

    \see DataTypeInfo provides similar mechanisms to manipulate Data objects
    generically in template code.
*/
class SOFA_DEFAULTTYPE_API AbstractTypeInfo
{
public:
    /// If the type is a container, returns the TypeInfo for the type of the
    /// values inside this container.
    /// For example, if the type is `fixed_array<fixed_array<int, 2> 3>`, it
    /// returns the TypeInfo for `fixed_array<int, 2>`.
    virtual const AbstractTypeInfo* BaseType() const = 0;

    /// Returns the TypeInfo for the type of the values accessible by the
    /// get*Value() functions.
    /// For example, if the type is `fixed_array<fixed_array<int, 2> 3>`, it
    /// returns the TypeInfo for `int`.    
    virtual const AbstractTypeInfo* ValueType() const = 0;

    /// \brief Returns the name of this type.
    virtual std::string name() const = 0;
    virtual std::string getTypeName() const = 0;

    /// True iff the TypeInfo for this type contains valid information.
    /// A Type is considered "Valid" if there's at least one specialization of the ValueType
    virtual bool ValidInfo() const = 0;

    /// True iff this type has a fixed size.
    ///  (It cannot be resized)
    virtual bool FixedSize() const = 0;
    /// True iff the default constructor of this type is equivalent to setting the memory to 0.
    virtual bool ZeroConstructor() const = 0;
    /// True iff copying the data can be done with a memcpy().
    virtual bool SimpleCopy() const = 0;
    /// True iff the layout in memory is simply N values of the same base type.
    /// It means that you can use the abstract index system to iterate over the elements of the type.
    /// (It doesn't mean that the BaseType is of a fixed size)
    virtual bool SimpleLayout() const = 0;
    /// True iff this type uses integer values.
    virtual bool Integer() const = 0;
    /// True iff this type uses scalar values.
    virtual bool Scalar() const = 0;
    /// True iff this type uses text values.
    virtual bool Text() const = 0;
    /// True iff this type uses copy-on-write.
    virtual bool CopyOnWrite() const = 0;
    /// True iff this type is a container of some sort.
    ///
    /// That is, if it can contain several values. In particular, strings are
    /// not considered containers.
    virtual bool Container() const = 0;

    /// The size of this type, in number of elements.
    /// For example, the size of a `fixed_array<fixed_array<int, 2>, 3>` is 6,
    /// and those six elements are conceptually numbered from 0 to 5.  This is
    /// relevant only if FixedSize() is true. I FixedSize() is false,
    /// the return value will be equivalent to the one of byteSize()
    virtual sofa::Size size() const = 0;
    /// The size in bytes of the ValueType
    /// For example, the size of a fixed_array<fixed_array<int, 2>, 3>` is 4 on most systems,
    /// as it is the byte size of the smallest dimension in the array (int -> 32bit)
    virtual sofa::Size byteSize() const = 0;

    /// The size of \a data, in number of iterable elements
    /// (For containers, that'll be the number of elements in the 1st dimension).
    /// For example, with type == `
    virtual sofa::Size size(const void* data) const = 0;
    /// Resize \a data to \a size elements, if relevant.

    /// But resizing is not always relevant, for example:
    /// - nothing happens if FixedSize() is true;
    /// - sets can't be resized; they are cleared instead;
    /// - nothing happens for vectors containing resizable values (i.e. when
    ///   BaseType()::FixedSize() is false), because of the "single index"
    ///   abstraction;
    ///
    /// Returns true iff the data was resizable
    virtual bool setSize(void* data, sofa::Size size) const = 0;

    /// Get the value at \a index of \a data as an integer.
    /// Relevant only if this type can be casted to `long long`.
    virtual long long   getIntegerValue(const void* data, Index index) const = 0;
    /// Get the value at \a index of \a data as a scalar.
    /// Relevant only if this type can be casted to `double`.
    virtual double      getScalarValue (const void* data, Index index) const = 0;
    /// Get the value at \a index of \a data as a string.
    virtual std::string getTextValue   (const void* data, Index index) const = 0;

    /// Set the value at \a index of \a data from an integer value.
    virtual void setIntegerValue(void* data, Index index, long long value) const = 0;
    /// Set the value at \a index of \a data from a scalar value.
    virtual void setScalarValue (void* data, Index index, double value) const = 0;
    /// Set the value at \a index of \a data from a string value.
    virtual void setTextValue(void* data, Index index, const std::string& value) const = 0;

    /// Get a read pointer to the underlying memory
    /// Relevant only if this type is SimpleLayout
    virtual const void* getValuePtr(const void* type) const = 0;

    /// Get a write pointer to the underlying memory
    /// Relevant only if this type is SimpleLayout
    virtual void* getValuePtr(void* type) const = 0;

    /// Get the type_info for this type.
    virtual const std::type_info* type_info() const = 0;

    const std::string& getCompilationTarget() const { return m_compilationTarget; }
    void setCompilationTarget(const std::string& target) { m_compilationTarget=target; }

protected: // only derived types can instantiate this class
    AbstractTypeInfo() {}
    virtual ~AbstractTypeInfo() {}

    virtual const TypeInfoId& getBaseTypeId() const = 0;
    virtual const TypeInfoId& getValueTypeId() const = 0;

private: // copy constructor or operator forbidden
    AbstractTypeInfo(const AbstractTypeInfo&) {}
    void operator=(const AbstractTypeInfo&) {}

    std::string m_compilationTarget;
};

} /// namespace sofa::defaulttype
