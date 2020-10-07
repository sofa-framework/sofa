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
#ifndef SOFA_DEFAULTTYPE_EIGENMATRIXXD_H
#define SOFA_DEFAULTTYPE_EIGENMATRIXXD_H
#include <iostream>
#include <sofa/defaulttype/config.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <iostream>
#include <map>

#include <Eigen/Core>

///Eigen::MatrixXd
std::ostream & operator<< (std::ostream &out, const Eigen::MatrixXd* df)
{
    out<<(*df);
    return out;
}


std::ostream & operator<< (std::ostream &out, const Eigen::MatrixXd& df)
{
    out<<(df);
    return out;
}

///Eigen::VectorXd
std::ostream & operator<< (std::ostream &out, const Eigen::VectorXd* df)
{
    out<<(*df);
    return out;
}

std::ostream & operator<< (std::ostream &out, const Eigen::VectorXd& df)
{
    out<<(df);
    return out;
}


namespace sofa
{

namespace core::objectmodel
{
/////Eigen::MatrixXd
///
std::istream & operator>> (std::istream &in, Eigen::MatrixXd* df)
{
    in.setstate(std::ios_base::failbit) ;
    return in;
}


std::istream & operator>> (std::istream &in, Eigen::MatrixXd& df)
{
    in.setstate(std::ios_base::failbit) ;
    return in;
}

///Eigen::VectorXd
std::istream & operator>> (std::istream &in, Eigen::VectorXd* df)
{
    in.setstate(std::ios_base::failbit) ;
    return in;
}

std::istream & operator>> (std::istream &in, Eigen::VectorXd& df)
{
    in.setstate(std::ios_base::failbit) ;
    return in;
}

}

namespace defaulttype
{

struct EigenMatrixXDTypeInfo
{
    typedef Eigen::MatrixXd DataType;
    typedef double BaseType;
    typedef DataTypeInfo<Vec3d> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;

    enum
    {
        ValidInfo = 1
    };  ///< 1 if this type has valid infos
    enum
    {
        FixedSize = 0
    };  ///< 1 if this type has a fixed size  -> always 1 single pair of vec3
    enum
    {
        ZeroConstructor = 0
    };  ///< 1 if the constructor is equivalent to setting memory to 0
    enum
    {
        SimpleCopy = 0
    };  ///< 1 if copying the data can be done with a memcpy
    enum
    {
        SimpleLayout = 0
    };  ///< 1 if the layout in memory is simply N values of the same base type
    enum
    {
        Integer = 0
    };  ///< 1 if this type uses integer values
    enum
    {
        Scalar = 1
    };  ///< 1 if this type uses scalar values
    enum
    {
        Text = 0
    };  ///< 1 if this type uses text values
    enum
    {
        CopyOnWrite = 0
    };  ///< 1 if this type uses copy-on-write -> it seems to be THE important
    enum
    {
        Container = 1
    };  ///< 1 if this type is a container

    enum
    {
        Size = 0
    };  ///< largest known fixed size for this type, as returned by size()

    static size_t size() { return 3; } // supposed to be the total number of elements. Ends up being the number of elements in the 2nd dimension
    static size_t byteSize() { return sizeof (SReal); }  // Size of the smalest single element in the container: BoundingBox uses Vec3d internally, so double
    static size_t size(const DataType & /*data*/) { return 2; } // supposed to be the nb of elements in the 1st dimension. Ends up being the total number of elems.
    static bool setSize(DataType & /*data*/, size_t /*size*/) { return false; } // FixedArray -> ignore

    template <typename T>
    static void getValue(const DataType & data, size_t index,T & value) /// since TypeInfos abstract all containers as 1D arrays, T here is of ValueType
    {
        ///@todo
        /// Need to be implemented
        value = static_cast<T>(((ValueType*)&data)[index]);
    }

    template <typename T>
    static void setValue(DataType & /*data*/, size_t /*index*/,
                         const T & /*value*/)
    {}

    static double getScalarValue (const void* data, size_t index)
    {
    }

    static void setScalarValue (const void* data, size_t index, double
                                value)
    {
    }

    static void getValueString(const DataType &data, size_t index,
                               std::string &value)
    {
    }

    static void setValueString(DataType &data, size_t index,
                               const std::string &value)
    {
    }

    static const void *getValuePtr(const DataType & bbox) { return
                (const void*)(&bbox); }

    static void *getValuePtr(DataType &bbox) { return (void*)(&bbox); }

};

struct EigenVectorXDTypeInfo
{

    typedef Eigen::VectorXd DataType;
    typedef double BaseType;
    typedef DataTypeInfo<Vec3d> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;

    enum
    {
        ValidInfo = 1
    };  ///< 1 if this type has valid infos
    enum
    {
        FixedSize = 0
    };  ///< 1 if this type has a fixed size  -> always 1 single pair of vec3
    enum
    {
        ZeroConstructor = 0
    };  ///< 1 if the constructor is equivalent to setting memory to 0
    enum
    {
        SimpleCopy = 0
    };  ///< 1 if copying the data can be done with a memcpy
    enum
    {
        SimpleLayout = 0
    };  ///< 1 if the layout in memory is simply N values of the same base type
    enum
    {
        Integer = 0
    };  ///< 1 if this type uses integer values
    enum
    {
        Scalar = 1
    };  ///< 1 if this type uses scalar values
    enum
    {
        Text = 0
    };  ///< 1 if this type uses text values
    enum
    {
        CopyOnWrite = 0
    };  ///< 1 if this type uses copy-on-write -> it seems to be THE important
    enum
    {
        Container = 1
    };  ///< 1 if this type is a container

    enum
    {
        Size = 0
    };  ///< largest known fixed size for this type, as returned by size()

    static size_t size() { return 3; } // supposed to be the total number of elements. Ends up being the number of elements in the 2nd dimension
    static size_t byteSize() { return sizeof (SReal); }  // Size of the smalest single element in the container: BoundingBox uses Vec3d internally, so double
    static size_t size(const DataType & /*data*/) { return 2; } // supposed to be the nb of elements in the 1st dimension. Ends up being the total number of elems.
    static bool setSize(DataType & /*data*/, size_t /*size*/) { return false; } // FixedArray -> ignore

    template <typename T>
    static void getValue(const DataType & data, size_t index,T & value) /// since TypeInfos abstract all containers as 1D arrays, T here is of ValueType
    {
        ///@todo
        /// Need to be implemented
        value = static_cast<T>(((ValueType*)&data)[index]);
    }

    template <typename T>
    static void setValue(DataType & /*data*/, size_t /*index*/,
                         const T & /*value*/)
    {}

    static double getScalarValue (const void* data, size_t index)
    {
    }

    static void setScalarValue (const void* data, size_t index, double
                                value)
    {
    }

    static void getValueString(const DataType &data, size_t index,
                               std::string &value)
    {
    }

    static void setValueString(DataType &data, size_t index,
                               const std::string &value)
    {
    }

    static const void *getValuePtr(const DataType & bbox) { return
                (const void*)(&bbox); }

    static void *getValuePtr(DataType &bbox) { return (void*)(&bbox); }

};



template <>
class DataTypeInfo<Eigen::MatrixXd> : public EigenMatrixXDTypeInfo
{
public:
    static std::string name() { return "Eigen::MatrixXd"; }
};

template <>
class DataTypeInfo<Eigen::VectorXd> : public EigenVectorXDTypeInfo
{
public:
    static std::string name() { return "Eigen::VectorXd"; }
};

} // namespace defaulttype
} // namespace sofa


#endif

