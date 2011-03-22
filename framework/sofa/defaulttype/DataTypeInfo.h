/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_DATATYPEINFO_H
#define SOFA_DEFAULTTYPE_DATATYPEINFO_H

#include <vector>
#include <sofa/helper/fixed_array.h>
//#include <sofa/helper/vector.h>
#include <sofa/helper/set.h>
#include <sstream>

namespace sofa
{

namespace helper
{
template <class T, class MemoryManager >
class vector;
}

namespace defaulttype
{

/// Type traits template
template<class TDataType>
struct DataTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType; ///< class of type contained in DataType
    typedef DataType ValueType; ///< type of the final atomic values
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = 0 }; ///< 1 if this type has valid infos
    enum { FixedSize       = 0 }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = 0 }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = 0 }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 0 }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = 0 }; ///< 1 if this type uses integer values
    enum { Scalar          = 0 }; ///< 1 if this type uses scalar values
    enum { Text            = 0 }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 0 }; ///< 1 if this type uses copy-on-write

    enum { Size = 1 }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size() { return 1; }

    static unsigned int size(const DataType& /*type*/) { return 1; }

    template <typename T>
    static void getValue(const DataType& /*type*/, unsigned int /*index*/, T& /*value*/)
    {
    }

    static void setSize(DataType& /*type*/, unsigned int /*size*/) {  }

    template<typename T>
    static void setValue(DataType& /*type*/, unsigned int /*index*/, const T& /*value*/)
    {
    }

    static void getValueString(const DataType& /*type*/, unsigned int /*index*/, std::string& /*value*/)
    {
    }

    static void setValueString(DataType& /*type*/, unsigned int /*index*/, const std::string& /*value*/)
    {
    }

    static const char* name() { return "unknown"; }

};

/// Type name template: default to using DataTypeInfo::name(), but can be overriden for types with shorter typedefs
template<class TDataType>
struct DataTypeName : public DataTypeInfo<TDataType>
{
};

/// Abstract type traits class
class AbstractTypeInfo
{
public:
    virtual const AbstractTypeInfo* BaseType() const = 0;
    virtual const AbstractTypeInfo* ValueType() const = 0;

    virtual std::string name() const = 0;

    virtual bool ValidInfo() const = 0;
    virtual bool FixedSize() const = 0;
    virtual bool ZeroConstructor() const = 0;
    virtual bool SimpleCopy() const = 0;
    virtual bool SimpleLayout() const = 0;
    virtual bool Integer() const = 0;
    virtual bool Scalar() const = 0;
    virtual bool Text() const = 0;
    virtual bool CopyOnWrite() const = 0;

    virtual unsigned int size() const = 0;
    virtual unsigned int size(const void* type) const = 0;
    virtual void setSize(void* type, unsigned int size) const = 0;

    virtual long long   getIntegerValue(const void* type, unsigned int index) const = 0;
    virtual double      getScalarValue (const void* type, unsigned int index) const = 0;
    virtual std::string getTextValue   (const void* type, unsigned int index) const = 0;

    virtual void setIntegerValue(void* type, unsigned int index, long long value) const = 0;
    virtual void setScalarValue (void* type, unsigned int index, double value) const = 0;
    virtual void setTextValue(void* type, unsigned int index, const std::string& value) const = 0;

protected: // only derived types can instantiate this class
    AbstractTypeInfo() {}
    virtual ~AbstractTypeInfo() {}

private: // copy constructor or operator forbidden
    AbstractTypeInfo(const AbstractTypeInfo&) {}
    void operator=(const AbstractTypeInfo&) {}
};

/// Abstract type traits class
template<class TDataType>
class VirtualTypeInfo : public AbstractTypeInfo
{
public:
    typedef TDataType DataType;
    typedef DataTypeInfo<DataType> Info;

    static VirtualTypeInfo* get() { static VirtualTypeInfo<DataType> t; return &t; }

    virtual const AbstractTypeInfo* BaseType() const  { return VirtualTypeInfo<typename Info::BaseType>::get(); }
    virtual const AbstractTypeInfo* ValueType() const { return VirtualTypeInfo<typename Info::ValueType>::get(); }

    virtual std::string name() const { return DataTypeName<DataType>::name(); }

    virtual bool ValidInfo() const       { return Info::ValidInfo; }
    virtual bool FixedSize() const       { return Info::FixedSize; }
    virtual bool ZeroConstructor() const { return Info::ZeroConstructor; }
    virtual bool SimpleCopy() const      { return Info::SimpleCopy; }
    virtual bool SimpleLayout() const    { return Info::SimpleLayout; }
    virtual bool Integer() const         { return Info::Integer; }
    virtual bool Scalar() const          { return Info::Scalar; }
    virtual bool Text() const            { return Info::Text; }
    virtual bool CopyOnWrite() const     { return Info::CopyOnWrite; }

    virtual unsigned int size() const
    {
        return Info::size();
    }
    virtual unsigned int size(const void* type) const
    {
        return Info::size(*(const DataType*)type);
    }
    virtual void setSize(void* type, unsigned int size) const
    {
        Info::setSize(*(DataType*)type, size);
    }

    virtual long long getIntegerValue(const void* type, unsigned int index) const
    {
        long long v = 0;
        Info::getValue(*(const DataType*)type, index, v);
        return v;
    }

    virtual double    getScalarValue (const void* type, unsigned int index) const
    {
        double v = 0;
        Info::getValue(*(const DataType*)type, index, v);
        return v;
    }

    virtual std::string getTextValue   (const void* type, unsigned int index) const
    {
        std::string v;
        Info::getValueString(*(const DataType*)type, index, v);
        return v;
    }

    virtual void setIntegerValue(void* type, unsigned int index, long long value) const
    {
        Info::setValue(*(DataType*)type, index, value);
    }

    virtual void setScalarValue (void* type, unsigned int index, double value) const
    {
        Info::setValue(*(DataType*)type, index, value);
    }

    virtual void setTextValue(void* type, unsigned int index, const std::string& value) const
    {
        Info::setValueString(*(DataType*)type, index, value);
    }

protected: // only derived types can instantiate this class
    VirtualTypeInfo() {}
};

template<class TDataType>
struct IntegerTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType; ///< preferred type for conversions (i.e. long long for integers, double for scalars)
    typedef IntegerTypeInfo<DataType> BaseTypeInfo;
    typedef IntegerTypeInfo<DataType> ValueTypeInfo;

    enum { ValidInfo       = 1 }; ///< 1 if this type has valid infos
    enum { FixedSize       = 1 }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = 1 }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = 1 }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 1 }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = 1 }; ///< 1 if this type uses integer values
    enum { Scalar          = 0 }; ///< 1 if this type uses scalar values
    enum { Text            = 0 }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 0 }; ///< 1 if this type uses copy-on-write

    enum { Size = 1 }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size() { return 1; }

    static unsigned int size(const DataType& /*type*/) { return 1; }

    static void setSize(DataType& /*type*/, unsigned int /*size*/) {  }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(type);
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int index, const T& value )
    {
        if (index != 0) return;
        type = static_cast<DataType>(value);
    }

    static void getValueString(const DataType &type, unsigned int index, std::string& value)
    {
        if (index != 0) return;
        std::ostringstream o; o << type; value = o.str();
    }

    static void setValueString(DataType &type, unsigned int index, const std::string& value )
    {
        if (index != 0) return;
        std::istringstream i(value); i >> type;
    }
};

struct BoolTypeInfo
{
    typedef bool DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType; ///< preferred type for conversions (i.e. long long for integers, double for scalars)
    typedef IntegerTypeInfo<DataType> BaseTypeInfo;
    typedef IntegerTypeInfo<DataType> ValueTypeInfo;

    enum { ValidInfo       = 1 }; ///< 1 if this type has valid infos
    enum { FixedSize       = 1 }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = 1 }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = 1 }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 1 }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = 1 }; ///< 1 if this type uses integer values
    enum { Scalar          = 0 }; ///< 1 if this type uses scalar values
    enum { Text            = 0 }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 0 }; ///< 1 if this type uses copy-on-write

    enum { Size = 1 }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size() { return 1; }

    static unsigned int size(const DataType& /*type*/) { return 1; }

    static void setSize(DataType& /*type*/, unsigned int /*size*/) {  }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(type);
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int index, const T& value )
    {
        if (index != 0) return;
        type = (value != 0);
    }

    template<typename T>
    static void setValue(std::vector<DataType>::reference type, unsigned int index, const T& value )
    {
        if (index != 0) return;
        type = (value != 0);
    }

    static void getValueString(const DataType &type, unsigned int index, std::string& value)
    {
        if (index != 0) return;
        std::ostringstream o; o << type; value = o.str();
    }

    static void setValueString(DataType &type, unsigned int index, const std::string& value )
    {
        if (index != 0) return;
        std::istringstream i(value); i >> type;
    }

    static void setValueString(std::vector<DataType>::reference type, unsigned int index, const std::string& value )
    {
        if (index != 0) return;
        bool b = type;
        std::istringstream i(value); i >> b;
        type = b;
    }
};

template<class TDataType>
struct ScalarTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType; ///< preferred type for conversions (i.e. long long for integers, double for scalars)
    typedef ScalarTypeInfo<TDataType> BaseTypeInfo;
    typedef ScalarTypeInfo<TDataType> ValueTypeInfo;

    enum { ValidInfo       = 1 }; ///< 1 if this type has valid infos
    enum { FixedSize       = 1 }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = 1 }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = 1 }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 1 }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = 0 }; ///< 1 if this type uses integer values
    enum { Scalar          = 1 }; ///< 1 if this type uses scalar values
    enum { Text            = 0 }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 0 }; ///< 1 if this type uses copy-on-write

    enum { Size = 1 }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size() { return 1; }

    static unsigned int size(const DataType& /*type*/) { return 1; }

    static void setSize(DataType& /*type*/, unsigned int /*size*/) {  }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(type);
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int index, const T& value )
    {
        if (index != 0) return;
        type = static_cast<DataType>(value);
    }

    static void getValueString(const DataType &type, unsigned int index, std::string& value)
    {
        if (index != 0) return;
        std::ostringstream o; o << type; value = o.str();
    }

    static void setValueString(DataType &type, unsigned int index, const std::string& value )
    {
        if (index != 0) return;
        std::istringstream i(value); i >> type;
    }
};

template<class TDataType>
struct TextTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType; ///< preferred type for conversions (i.e. long long for integers, double for scalars)
    typedef ScalarTypeInfo<TDataType> BaseTypeInfo;
    typedef ScalarTypeInfo<TDataType> ValueTypeInfo;

    enum { ValidInfo       = 1 }; ///< 1 if this type has valid infos
    enum { FixedSize       = 0 }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = 0 }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = 0 }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 0 }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = 0 }; ///< 1 if this type uses integer values
    enum { Scalar          = 0 }; ///< 1 if this type uses scalar values
    enum { Text            = 1 }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 1 }; ///< 1 if this type uses copy-on-write

    enum { Size = 1 }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size() { return 1; }

    static unsigned int size(const DataType& /*type*/) { return 1; }

    static void setSize(DataType& /*type*/, unsigned int /*size*/) {  }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        if (index != 0) return;
        std::istringstream i(type); i >> value;
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int index, const T& value )
    {
        if (index != 0) return;
        std::ostringstream o; o << value; type = o.str();
    }

    static void getValueString(const DataType &type, unsigned int index, std::string& value)
    {
        if (index != 0) return;
        value = type;
    }

    static void setValueString(DataType &type, unsigned int index, const std::string& value )
    {
        if (index != 0) return;
        type = value;
    }
};

template<class TDataType, int static_size = TDataType::static_size>
struct FixedArrayTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::value_type BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       }; ///< 1 if this type has valid infos
    enum { FixedSize       = BaseTypeInfo::FixedSize       }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = BaseTypeInfo::ZeroConstructor }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = BaseTypeInfo::SimpleCopy      }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = BaseTypeInfo::SimpleLayout    }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = BaseTypeInfo::Integer         }; ///< 1 if this type uses integer values
    enum { Scalar          = BaseTypeInfo::Scalar          }; ///< 1 if this type uses scalar values
    enum { Text            = BaseTypeInfo::Text            }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 1                             }; ///< 1 if this type uses copy-on-write

    enum { Size = static_size * BaseTypeInfo::Size }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size()
    {
        return DataType::size() * BaseTypeInfo::size();
    }

    static unsigned int size(const DataType& type)
    {
        if (FixedSize)
            return size();
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<DataType::size(); ++i)
                s+= BaseTypeInfo::size(type[i]);
            return s;
        }
    }

    static void setSize(DataType& type, unsigned int size)
    {
        if (!FixedSize)
        {
            size /= DataType::size();
            for (unsigned int i=0; i<DataType::size(); ++i)
                BaseTypeInfo::setSize(type[i], size);
        }
    }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValue(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValue(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<DataType::size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValue(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int index, const T& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValue(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValue(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<DataType::size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValue(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void getValueString(const DataType &type, unsigned int index, std::string& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValueString(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValueString(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<DataType::size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &type, unsigned int index, const std::string& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValueString(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValueString(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<DataType::size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValueString(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }
};

template<class TDataType>
struct VectorTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::value_type BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       }; ///< 1 if this type has valid infos
    enum { FixedSize       = 0                             }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = 0                             }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = 0                             }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = BaseTypeInfo::SimpleLayout    }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = BaseTypeInfo::Integer         }; ///< 1 if this type uses integer values
    enum { Scalar          = BaseTypeInfo::Scalar          }; ///< 1 if this type uses scalar values
    enum { Text            = BaseTypeInfo::Text            }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 1                             }; ///< 1 if this type uses copy-on-write

    enum { Size = BaseTypeInfo::Size }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size()
    {
        return BaseTypeInfo::size();
    }

    static unsigned int size(const DataType& type)
    {
        if (BaseTypeInfo::FixedSize)
            return type.size()*BaseTypeInfo::size();
        else
        {
            unsigned int n = type.size();
            unsigned int s = 0;
            for (unsigned int i=0; i<n; ++i)
                s+= BaseTypeInfo::size(type[i]);
            return s;
        }
    }

    static void setSize(DataType& type, unsigned int size)
    {
        type.resize(size/BaseTypeInfo::size());
    }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValue(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValue(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<type.size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValue(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int index, const T& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValue(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValue(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<type.size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValue(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void getValueString(const DataType &type, unsigned int index, std::string& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValueString(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValueString(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<type.size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &type, unsigned int index, const std::string& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValueString(type[index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValueString(type[index/BaseTypeInfo::size()], index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (unsigned int i=0; i<type.size(); ++i)
            {
                unsigned int n = BaseTypeInfo::size(type[i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValueString(type[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }
};


template<class TDataType>
struct SetTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::value_type BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       }; ///< 1 if this type has valid infos
    enum { FixedSize       = 0                             }; ///< 1 if this type has a fixed size
    enum { ZeroConstructor = 0                             }; ///< 1 if the constructor is equivalent to setting memory to 0
    enum { SimpleCopy      = 0                             }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 0                             }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = BaseTypeInfo::Integer         }; ///< 1 if this type uses integer values
    enum { Scalar          = BaseTypeInfo::Scalar          }; ///< 1 if this type uses scalar values
    enum { Text            = BaseTypeInfo::Text            }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 1                             }; ///< 1 if this type uses copy-on-write

    enum { Size = BaseTypeInfo::Size }; ///< largest known fixed size for this type, as returned by size()
    static unsigned int size()
    {
        return BaseTypeInfo::size();
    }

    static unsigned int size(const DataType& type)
    {
        if (BaseTypeInfo::FixedSize)
            return type.size()*BaseTypeInfo::size();
        else
        {
            unsigned int s = 0;
            for (typename DataType::const_iterator it = type.begin(), end=type.end(); it!=end; ++it)
                s+= BaseTypeInfo::size(*it);
            return s;
        }
    }

    static void setSize(DataType& type, unsigned int /*size*/)
    {
        type.clear(); // we can't "resize" a set, so the only meaningfull operation is to clear it, as values will be added dynamically in setValue
    }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        if (BaseTypeInfo::FixedSize)
        {
            typename DataType::const_iterator it = type.begin();
            for (unsigned int i=0; i<index/BaseTypeInfo::size(); ++i) ++it;
            BaseTypeInfo::getValue(*it, index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (typename DataType::const_iterator it = type.begin(), end=type.end(); it!=end; ++it)
            {
                unsigned int n = BaseTypeInfo::size(*it);
                if (index < s+n)
                {
                    BaseTypeInfo::getValue(*it, index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int /*index*/, const T& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseType t;
            BaseTypeInfo::setValue(t, 0, value);
            type.insert(t);
        }
        else
        {
            std::cerr << "ERROR: SetTypeInfo::setValue not implemented for set with composite values." << std::endl;
        }
    }

    static void getValueString(const DataType &type, unsigned int index, std::string& value)
    {
        if (BaseTypeInfo::FixedSize)
        {
            typename DataType::const_iterator it = type.begin();
            for (unsigned int i=0; i<index/BaseTypeInfo::size(); ++i) ++it;
            BaseTypeInfo::getValueString(*it, index%BaseTypeInfo::size(), value);
        }
        else
        {
            unsigned int s = 0;
            for (typename DataType::const_iterator it = type.begin(), end=type.end(); it!=end; ++it)
            {
                unsigned int n = BaseTypeInfo::size(*it);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(*it, index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &type, unsigned int /*index*/, const std::string& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseType t;
            BaseTypeInfo::setValueString(t, 0, value);
            type.insert(t);
        }
        else
        {
            std::cerr << "ERROR: SetTypeInfo::setValueString not implemented for set with composite values." << std::endl;
        }
    }
};

template<>
struct DataTypeInfo<bool> : public BoolTypeInfo
{
    static const char* name() { return "bool"; }
};

template<>
struct DataTypeInfo<char> : public IntegerTypeInfo<char>
{
    static const char* name() { return "char"; }
};

template<>
struct DataTypeInfo<unsigned char> : public IntegerTypeInfo<unsigned char>
{
    static const char* name() { return "unsigned char"; }
};

template<>
struct DataTypeInfo<short> : public IntegerTypeInfo<short>
{
    static const char* name() { return "short"; }
};

template<>
struct DataTypeInfo<unsigned short> : public IntegerTypeInfo<unsigned short>
{
    static const char* name() { return "unsigned short"; }
};

template<>
struct DataTypeInfo<int> : public IntegerTypeInfo<int>
{
    static const char* name() { return "int"; }
};

template<>
struct DataTypeInfo<unsigned int> : public IntegerTypeInfo<unsigned int>
{
    static const char* name() { return "unsigned int"; }
};

template<>
struct DataTypeInfo<long> : public IntegerTypeInfo<long>
{
    static const char* name() { return "long"; }
};

template<>
struct DataTypeInfo<unsigned long> : public IntegerTypeInfo<unsigned long>
{
    static const char* name() { return "unsigned long"; }
};

template<>
struct DataTypeInfo<long long> : public IntegerTypeInfo<long long>
{
    static const char* name() { return "long long"; }
};

template<>
struct DataTypeInfo<unsigned long long> : public IntegerTypeInfo<unsigned long long>
{
    static const char* name() { return "unsigned long long"; }
};

template<>
struct DataTypeInfo<float> : public ScalarTypeInfo<float>
{
    static const char* name() { return "float"; }
};

template<>
struct DataTypeInfo<double> : public ScalarTypeInfo<double>
{
    static const char* name() { return "double"; }
};

template<>
struct DataTypeInfo<std::string> : public TextTypeInfo<std::string>
{
    static const char* name() { return "string"; }
};

template<class T, std::size_t N>
struct DataTypeInfo< sofa::helper::fixed_array<T,N> > : public FixedArrayTypeInfo<sofa::helper::fixed_array<T,N> >
{
    static std::string name() { std::ostringstream o; o << "fixed_array<" << DataTypeName<T>::name() << "," << N << ">"; return o.str(); }
};

template<class T, class Alloc>
struct DataTypeInfo< std::vector<T,Alloc> > : public VectorTypeInfo<std::vector<T,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "std::vector<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

template<class T, class Alloc>
struct DataTypeInfo< sofa::helper::vector<T,Alloc> > : public VectorTypeInfo<sofa::helper::vector<T,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "vector<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

template<class T, class Compare, class Alloc>
struct DataTypeInfo< std::set<T,Compare,Alloc> > : public SetTypeInfo<std::set<T,Compare,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "std::set<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

template<class T, class Compare, class Alloc>
struct DataTypeInfo< helper::set<T,Compare,Alloc> > : public SetTypeInfo<helper::set<T,Compare,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "set<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

} // namespace defaulttype

} // namespace sofa

#endif  // SOFA_DEFAULTTYPE_DATATYPEINFO_H
