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
#include <sofa/defaulttype/AbstractTypeInfo.h>
#include <sofa/defaulttype/TypeInfoID.h>

namespace sofa::defaulttype
{

class SOFA_DEFAULTTYPE_API NoTypeInfo : public AbstractTypeInfo
{
public:
    NoTypeInfo(){ setCompilationTarget("SofaDefaultType"); }

    const AbstractTypeInfo* BaseType() const override { return Get(); }
    const AbstractTypeInfo* ValueType() const override { return Get(); }

    static AbstractTypeInfo* Get(){ static NoTypeInfo t; return &t; }

    /// \brief Returns the name of this type.
    std::string name() const override {return "NoTypeInfo"; }
    std::string getTypeName() const override {return "NoTypeInfo"; }

    /// True iff the TypeInfo for this type contains valid information.
    /// A Type is considered "Valid" if there's at least one specialization of the ValueType
    bool ValidInfo() const override { return false; }

    /// True iff this type has a fixed size.
    ///  (It cannot be resized)
    bool FixedSize() const override {return false;}
    /// True iff the default constructor of this type is equivalent to setting the memory to 0.
    bool ZeroConstructor() const override {return false;}
    /// True iff copying the data can be done with a memcpy().
    bool SimpleCopy() const override {return false;}
    /// True iff the layout in memory is simply N values of the same base type.
    /// It means that you can use the abstract index system to iterate over the elements of the type.
    /// (It doesn't mean that the BaseType is of a fixed size)
    bool SimpleLayout() const override {return false;}
    /// True iff this type uses integer values.
    bool Integer() const override {return false;}
    /// True iff this type uses scalar values.
    bool Scalar() const override {return false;}
    /// True iff this type uses text values.
    bool Text() const override {return false;}
    /// True iff this type uses copy-on-write.
    bool CopyOnWrite() const override {return false;}
    /// True iff this type is a container of some sort.
    ///
    /// That is, if it can contain several values. In particular, strings are
    /// not considered containers.
    bool Container() const override {return false;}

    /// The size of this type, in number of elements.
    /// For example, the size of a `fixed_array<fixed_array<int, 2>, 3>` is 6,
    /// and those six elements are conceptually numbered from 0 to 5.  This is
    /// relevant only if FixedSize() is true. I FixedSize() is false,
    /// the return value will be equivalent to the one of byteSize()
    sofa::Size size() const override {return -1;}
    /// The size in bytes of the ValueType
    /// For example, the size of a fixed_array<fixed_array<int, 2>, 3>` is 4 on most systems,
    /// as it is the byte size of the smallest dimension in the array (int -> 32bit)
    sofa::Size byteSize() const override {return -1;}

    /// The size of \a data, in number of iterable elements
    /// (For containers, that'll be the number of elements in the 1st dimension).
    /// For example, with type == `
    sofa::Size size(const void* /*data*/) const override {return -1;}
    /// Resize \a data to \a size elements, if relevant.

    /// But resizing is not always relevant, for example:
    /// - nothing happens if FixedSize() is true;
    /// - sets can't be resized; they are cleared instead;
    /// - nothing happens for vectors containing resizable values (i.e. when
    ///   BaseType()::FixedSize() is false), because of the "single index"
    ///   abstraction;
    ///
    /// Returns true iff the data was resizable
    bool setSize(void*, sofa::Size) const override { return false; };

    /// Get the value at \a index of \a data as an integer.
    /// Relevant only if this type can be casted to `long long`.
     long long   getIntegerValue(const void*, Index) const override {return 0;}
    /// Get the value at \a index of \a data as a scalar.
    /// Relevant only if this type can be casted to `double`.
    double      getScalarValue (const void*, Index) const override {return 0;}
    /// Get the value at \a index of \a data as a string.
    std::string getTextValue   (const void*, Index) const override {return 0;}

    /// Set the value at \a index of \a data from an integer value.
    void setIntegerValue(void*, Index, long long ) const override {return;}
    /// Set the value at \a index of \a data from a scalar value.
    void setScalarValue (void*, Index, double ) const override {return;}
    /// Set the value at \a index of \a data from a string value.
    void setTextValue(void*, Index, const std::string&) const override {return;}

    /// Get a read pointer to the underlying memory
    /// Relevant only if this type is SimpleLayout
    const void* getValuePtr(const void*) const override {return 0;}

    /// Get a write pointer to the underlying memory
    /// Relevant only if this type is SimpleLayout
    void* getValuePtr(void*) const override {return 0;}

    /// Get the type_info for this type.
    const std::type_info* type_info() const override {return &typeid(this); }

protected:
    const TypeInfoId& getBaseTypeId() const override { return TypeInfoId::GetTypeId<NoTypeInfo>(); }
    const TypeInfoId& getValueTypeId() const override { return TypeInfoId::GetTypeId<NoTypeInfo>(); }

};

} /// namespace sofa::defaulttype
