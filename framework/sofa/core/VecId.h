/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_VECID_H
#define SOFA_CORE_VECID_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <boost/static_assert.hpp>

#include <sstream>
#include <iostream>

namespace sofa
{

namespace core
{

/// Types of vectors that can be stored in State
enum VecType
{
    V_ALL = 0,
    V_COORD,
    V_DERIV,
    V_MATDERIV,
};

/// Types of vectors that can be stored in State
enum VecAccess
{
    V_READ=0,
    V_WRITE,
};

template <VecType vtype, VecAccess vaccess>
class TVecId;

template <VecType vtype, VecAccess vaccess>
class TStandardVec;


template <VecAccess vaccess>
class TStandardVec<V_COORD, vaccess>
{
public:
    typedef TVecId<V_COORD, vaccess> MyVecId;
    static MyVecId position()      { return MyVecId(1);}
    static MyVecId restPosition()  { return MyVecId(2);}
    static MyVecId freePosition()  { return MyVecId(3);}
    static MyVecId resetPosition() { return MyVecId(4);}
    enum { V_FIRST_DYNAMIC_INDEX = 5 }; ///< This is the first index used for dynamically allocated vectors

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case 0: result+= "null";
            break;
        case 1: result+= "position";
            break;
        case 2: result+= "restPosition";
            break;
        case 3: result+= "freePosition";
            break;
        case 4: result+= "resetPosition";
            break;
        default:
            std::ostringstream out;
            out << v.getIndex();
            result+= out.str();
            break;
        }
        result+= "(V_COORD)";
        return result;
    }
};

template <VecAccess vaccess>
class TStandardVec<V_DERIV, vaccess>
{
public:
    typedef TVecId<V_DERIV, vaccess> MyVecId;

    static MyVecId velocity()       { return MyVecId(1); }
    static MyVecId resetVelocity()  { return MyVecId(2); }
    static MyVecId freeVelocity()   { return MyVecId(3); }
    static MyVecId normal()         { return MyVecId(4); }
    static MyVecId force()          { return MyVecId(5); }
    static MyVecId externalForce()  { return MyVecId(6); }
    static MyVecId dx()             { return MyVecId(7); }
    static MyVecId dforce()         { return MyVecId(8); }
    enum { V_FIRST_DYNAMIC_INDEX = 9 }; ///< This is the first index used for dynamically allocated vectors

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case 0: result+= "null";
            break;
        case 1: result+= "velocity";
            break;
        case 2: result+= "resetVelocity";
            break;
        case 3: result+= "freeVelocity";
            break;
        case 4: result+= "normal";
            break;
        case 5: result+= "force";
            break;
        case 6: result+= "externalForce";
            break;
        case 7: result+= "dx";
            break;
        case 8: result+= "dforce";
            break;
        default:
            std::ostringstream out;
            out << v.getIndex();
            result+= out.str();
            break;
        }
        result+= "(V_DERIV)";
        return result;
    }
};

template <VecAccess vaccess>
class TStandardVec<V_MATDERIV, vaccess>
{
public:
    typedef TVecId<V_MATDERIV, vaccess> MyVecId;

    static MyVecId holonomicC()    { return MyVecId(1);}
    static MyVecId nonHolonomicC() { return MyVecId(2);}
    enum { V_FIRST_DYNAMIC_INDEX = 3 }; ///< This is the first index used for dynamically allocated vectors

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case 0: result+= "null";
            break;
        case 1: result+= "holonomic";
            break;
        case 2: result+= "nonHolonomic";
            break;
        default:
            std::ostringstream out;
            out << v.getIndex();
            result+= out.str();
            break;
        }
        result+= "(V_MATDERIV)";
        return result;
    }
};

template <VecAccess vaccess>
class TStandardVec<V_ALL, vaccess>
    : public TStandardVec<V_COORD,vaccess>
    , public TStandardVec<V_DERIV,vaccess>
    , public TStandardVec<V_MATDERIV,vaccess>
{
public:
    typedef TVecId<V_ALL, vaccess> MyVecId;

    static unsigned int getFirstDynamicIndex(VecType t)
    {
        switch(t)
        {
        case V_COORD:
            return TStandardVec<V_COORD,vaccess>::V_FIRST_DYNAMIC_INDEX;
        case V_DERIV:
            return TStandardVec<V_DERIV,vaccess>::V_FIRST_DYNAMIC_INDEX;
        case V_MATDERIV:
            return TStandardVec<V_MATDERIV,vaccess>::V_FIRST_DYNAMIC_INDEX;
        default:
            return 0;
        }
    }

    static std::string getName(const MyVecId& v)
    {
        switch(v.getType())
        {
        case V_COORD:
            return TStandardVec<V_COORD,vaccess>::getName((TVecId<V_COORD,vaccess>)v);
        case V_DERIV:
            return TStandardVec<V_DERIV,vaccess>::getName((TVecId<V_DERIV,vaccess>)v);
        case V_MATDERIV:
            return TStandardVec<V_MATDERIV,vaccess>::getName((TVecId<V_MATDERIV,vaccess>)v);
        default:
            std::string result;
            std::ostringstream out;
            out << v.getIndex() << "(" << v.getType() << ")";
            result = out.str();
            return result;
        }
    }
};

/// Identify a vector of a given type stored in State
/// This class is templated in order to create different variations (generic versus specific type, read-only vs write access)
template <VecType vtype, VecAccess vaccess>
class TVecId : public TStandardVec<vtype, vaccess>
{
public:
    unsigned int index;
    TVecId() : index(0) { }
    TVecId(unsigned int i) : index(i) { }
    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecAccess vaccess2>
    TVecId(const TVecId<vtype, vaccess2>& v) : index(v.getIndex())
    {
        BOOST_STATIC_ASSERT(vaccess2 >= vaccess);
    }

    TVecId(const TVecId<vtype, V_WRITE>& v) : index(v.getIndex()) { }

    explicit TVecId(const TVecId<V_ALL, vaccess>& v) : index(v.getIndex())
    {
#ifndef NDEBUG
        assert(v.getType() == vtype);
#endif
    }

    VecType getType() const { return vtype; }
    unsigned int getIndex() const { return index; }

    template<VecType vtype2, VecAccess vaccess2>
    bool operator==(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() == v.getType() && getIndex() == v.getIndex();
    }

    template<VecType vtype2, VecAccess vaccess2>
    bool operator!=(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() != v.getType() || getIndex() != v.getIndex();
    }

    static TVecId null() { return TVecId(0);}
    bool isNull() const { return this->index == 0; }

    std::string getName() const
    {
        return TStandardVec<vtype, vaccess>::getName(*this);
    }
    friend inline std::ostream& operator << ( std::ostream& out, const TVecId& v )
    {
        out << v.getName();
        return out;
    }
};

/// Identify any vector stored in State
template<VecAccess vaccess>
class TVecId<V_ALL, vaccess> : public TStandardVec<V_ALL, vaccess>
{
public:
    typedef VecType Type;
    VecType type;
    unsigned int index;
    TVecId() : type(V_ALL), index(0) { }
    TVecId(VecType t, unsigned int i) : type(t), index(i) { }
    template<VecType vtype2, VecAccess vaccess2>
    /// Create a generic VecId from a specific or generic one, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    TVecId(const TVecId<vtype2, vaccess2>& v) : type(v.getType()), index(v.getIndex())
    {
        BOOST_STATIC_ASSERT(vaccess2 >= vaccess);
    }

    //operator TVecId<V_ALL, V_READ>() const { return TVecId<V_ALL, V_READ>(getType(), getIndex()); }

    VecType getType() const { return type; }
    unsigned int getIndex() const { return index; }

    template<VecType vtype2, VecAccess vaccess2>
    bool operator==(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() == v.getType() && getIndex() == v.getIndex();
    }

    template<VecType vtype2, VecAccess vaccess2>
    bool operator!=(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() != v.getType() || getIndex() != v.getIndex();
    }

    static TVecId null() { return TVecId(V_ALL, 0);}
    bool isNull() const { return this->index == 0; }

    std::string getName() const
    {
        return TStandardVec<V_ALL, vaccess>::getName(*this);
    }
    friend inline std::ostream& operator << ( std::ostream& out, const TVecId& v )
    {
        out << v.getName();
        return out;
    }
};


/// Identify one vector stored in State
/// A ConstVecId only provides a read-only access to the underlying vector.
typedef TVecId<V_ALL, V_READ> ConstVecId;

/// Identify one vector stored in State
/// A VecId provides a read-write access to the underlying vector.
typedef TVecId<V_ALL, V_WRITE> VecId;

/// Typedefs for each type of state vectors
typedef TVecId<V_COORD, V_READ> ConstVecCoordId;
typedef TVecId<V_COORD, V_WRITE>     VecCoordId;
typedef TVecId<V_DERIV, V_READ> ConstVecDerivId;
typedef TVecId<V_DERIV, V_WRITE>     VecDerivId;
typedef TVecId<V_MATDERIV, V_READ> ConstMatrixDerivId;
typedef TVecId<V_MATDERIV, V_WRITE>     MatrixDerivId;

} // namespace core

} // namespace sofa

#endif
