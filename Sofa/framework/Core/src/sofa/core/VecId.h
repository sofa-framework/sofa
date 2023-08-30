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

#include <sofa/core/config.h>

#include <string>
#include <sstream>
#include <cassert>
#include <unordered_map>

namespace sofa::core
{

/// Types of vectors that can be stored in State
enum VecType
{
    V_ALL = 0,
    V_COORD,
    V_DERIV,
    V_MATDERIV,
};

SOFA_CORE_API extern const std::unordered_map<VecType, std::string> VecTypeLabels;

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
        result+= VecTypeLabels.at(V_COORD);
        return result;
    }

    static std::string getGroup(const MyVecId& v)
    {
        switch(v.getIndex())
        {
            case 0: return {}; //null
            case 1: return "States"; //position
            case 2: return "Rest States"; //restPosition
            case 3: return "Free Motion"; //freePosition
            case 4: return "States"; //
            default: return {};
        }
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
        result+= VecTypeLabels.at(V_DERIV);
        return result;
    }

    static std::string getGroup(const MyVecId& v)
    {
        switch(v.getIndex())
        {
            case 0: return {}; //null
            case 1: return "States"; //velocity
            case 2: return "States"; //resetVelocity
            case 3: return "Free Motion"; //freeVelocity
            case 4: return "States"; //normal
            case 5: return "Force"; //force
            case 6: return "Force"; //externalForce
            case 7: return "States"; //dx
            case 8: return "Force"; //dforce
            default: return {};
        }
    }
};

template <VecAccess vaccess>
class TStandardVec<V_MATDERIV, vaccess>
{
public:
    typedef TVecId<V_MATDERIV, vaccess> MyVecId;

    static MyVecId constraintJacobian()    { return MyVecId(1);} // jacobian matrix of constraints
    static MyVecId mappingJacobian() { return MyVecId(2);}         // accumulated matrix of the mappings

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

    static std::string getGroup(const MyVecId& v)
    {
        switch(v.getIndex())
        {
        case 0: return {}; //null
            case 1: return "Jacobian"; //constraintJacobian
            case 2: return "Jacobian"; //mappingJacobian
            default: return {};
        }
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
            return TStandardVec<V_COORD,vaccess>::getName(static_cast<TVecId<V_COORD,vaccess>>(v));
        case V_DERIV:
            return TStandardVec<V_DERIV,vaccess>::getName(static_cast<TVecId<V_DERIV,vaccess>>(v));
        case V_MATDERIV:
            return TStandardVec<V_MATDERIV,vaccess>::getName(static_cast<TVecId<V_MATDERIV,vaccess>>(v));
        default:
            std::string result;
            std::ostringstream out;
            out << v.getIndex() << "(" << v.getType() << ")";
            result = out.str();
            return result;
        }
    }
};

/// This is a base class for TVecId that contains all the data stored.
///
/// @note TVecId itself stores no data, in order to be able to convert between templates inplace with reinterpret_cast
/// for performance reasons (typically when working with TMultiVecId instances, which would otherwise copy maps of TVecId).
/// This is (a little) less efficient for non V_ALL versions, but is without comparison with the loss of performance
/// with the typical operation of passing a stored "TMultiVecId<!V_ALL,V_WRITE>" to a method taking a "const TMultiVecId<V_ALL,V_READ>&".
class BaseVecId
{
public:
    VecType getType() const { return type; }
    unsigned int getIndex() const { return index; }

    VecType type;
    unsigned int index;

protected:
    BaseVecId(VecType t, unsigned int i) : type(t), index(i) {}
};

/// This class is only here as fix for a VC2010 compiler otherwise padding TVecId<V_ALL,?> with 4 more bytes than TVecId<?,?>, 
/// probably due to some weird rule requiring to have distinct base pointers with multiple inheritance that's imo
/// wrongly applied for base classes without data members, and hopefully should not make anything worse for other compilers.
/// @note Just in case, we have a static size assertion at the end of the file, so you will know if there is a problem.
class VecIdAlignFix {};

struct VecIdProperties
{
    std::string label;
    std::string group;
};

/// Identify a vector of a given type stored in State
/// This class is templated in order to create different variations (generic versus specific type, read-only vs write access)
template <VecType vtype, VecAccess vaccess>
class TVecId : public BaseVecId, public TStandardVec<vtype, vaccess>, public VecIdAlignFix
{
public:
    TVecId() : BaseVecId(vtype, 0) { }
    TVecId(unsigned int i) : BaseVecId(vtype, i) { }

    /// Copy constructor
    TVecId(const TVecId<vtype, vaccess> & v) : BaseVecId(vtype, v.getIndex()) {}

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecAccess vaccess2>
    TVecId(const TVecId<vtype, vaccess2>& v) : BaseVecId(vtype, v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
    }

    template<VecAccess vaccess2>
    explicit TVecId(const TVecId<V_ALL, vaccess2>& v) : BaseVecId(vtype, v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
#ifndef NDEBUG
        assert(v.getType() == vtype);
#endif
    }

    // Copy assignment

    TVecId<vtype, vaccess> & operator=(const TVecId<vtype, vaccess>& other) {
        this->index = other.index;
        this->type = other.type;
        return *this;
    }

    template<VecAccess vaccess2>
    TVecId<vtype, vaccess> & operator=(const TVecId<vtype, vaccess2>& other) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
        this->index = other.index;
        this->type = other.type;
        return *this;
    }

    template<VecAccess vaccess2>
    TVecId<vtype, vaccess> & operator=(const TVecId<V_ALL, vaccess2>& other) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
#ifndef NDEBUG
        assert(other.getType() == vtype);
#endif
        this->index = other.index;
        this->type = other.type;
        return *this;
    }


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
    std::string getGroup() const
    {
        return TStandardVec<vtype, vaccess>::getGroup(*this);
    }

    friend inline std::ostream& operator << ( std::ostream& out, const TVecId& v )
    {
        out << v.getName();
        return out;
    }
};

/// Identify any vector stored in State
template<VecAccess vaccess>
class TVecId<V_ALL, vaccess> : public BaseVecId, public TStandardVec<V_ALL, vaccess>
{
public:
    TVecId() : BaseVecId(V_ALL, 0) { }
    TVecId(VecType t, unsigned int i) : BaseVecId(t, i) { }
    /// Create a generic VecId from a specific or generic one, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecType vtype2, VecAccess vaccess2>
    TVecId(const TVecId<vtype2, vaccess2>& v) : BaseVecId(v.getType(), v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
    }

    // Copy assignment
    template<VecType vtype2, VecAccess vaccess2>
    TVecId<V_ALL, vaccess> & operator=(const TVecId<vtype2, vaccess2>& other) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
        this->index = other.index;
        this->type = other.type;
        return *this;
    }


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
    std::string getGroup() const
    {
        return TStandardVec<V_ALL, vaccess>::getGroup(*this);
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
typedef TVecId<V_COORD   , V_READ > ConstVecCoordId;
typedef TVecId<V_COORD   , V_WRITE>      VecCoordId;
typedef TVecId<V_DERIV   , V_READ > ConstVecDerivId;
typedef TVecId<V_DERIV   , V_WRITE>      VecDerivId;
typedef TVecId<V_MATDERIV, V_READ > ConstMatrixDerivId;
typedef TVecId<V_MATDERIV, V_WRITE>      MatrixDerivId;

static_assert(sizeof(VecId) == sizeof(VecCoordId), "");


/// Maps a VecType to a DataTypes member typedef representing the state variables
/// Example: StateType_t<DataTypes, core::V_COORD> returns the type DataTypes::Coord
template<class DataTypes, core::VecType vtype> struct StateType {};
template<class DataTypes, core::VecType vtype> using StateType_t = typename StateType<DataTypes, vtype>::type;

template<class DataTypes> struct StateType<DataTypes, core::V_COORD>
{
    using type = typename DataTypes::Coord;
};
template<class DataTypes> struct StateType<DataTypes, core::V_DERIV>
{
    using type = typename DataTypes::Deriv;
};

/// Maps a VecType to a DataTypes member static variable representing the size of the state variables
/// Example: StateTypeSize_v<DataTypes, core::V_COORD> is the value of DataTypes::coord_total_size
template<class DataTypes, core::VecType vtype> struct StateTypeSize {};
template<class DataTypes, core::VecType vtype> inline constexpr sofa::Size StateTypeSize_v = StateTypeSize<DataTypes, vtype>::total_size;

template<class DataTypes> struct StateTypeSize<DataTypes, core::V_COORD>
{
    static constexpr sofa::Size total_size = DataTypes::coord_total_size;
};
template<class DataTypes> struct StateTypeSize<DataTypes, core::V_DERIV>
{
    static constexpr sofa::Size total_size = DataTypes::deriv_total_size;
};

/// Maps a VecType to a DataTypes member typedef representing a vector of state variables
/// Example: StateVecType_t<DataTypes, core::V_COORD> returns the type DataTypes::VecCoord
template<class DataTypes, core::VecType vtype> struct StateVecType {};
template<class DataTypes, core::VecType vtype> using StateVecType_t = typename StateVecType<DataTypes, vtype>::type;

template<class DataTypes> struct StateVecType<DataTypes, core::V_COORD>
{
    using type = typename DataTypes::VecCoord;
};
template<class DataTypes> struct StateVecType<DataTypes, core::V_DERIV>
{
    using type = typename DataTypes::VecDeriv;
};

} // namespace sofa::core
