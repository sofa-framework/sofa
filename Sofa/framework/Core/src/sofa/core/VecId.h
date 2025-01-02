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
enum class VecType : uint8_t
{
    V_ALL = 0,
    V_COORD,
    V_DERIV,
    V_MATDERIV,
};

static constexpr inline VecType V_ALL = (VecType::V_ALL);
static constexpr inline VecType V_COORD = (VecType::V_COORD);
static constexpr inline VecType V_DERIV = (VecType::V_DERIV);
static constexpr inline VecType V_MATDERIV = (VecType::V_MATDERIV);

SOFA_CORE_API extern const std::unordered_map<VecType, std::string> VecTypeLabels;

inline std::ostream& operator<<( std::ostream& out, const VecType& v )
{
    out << VecTypeLabels.at(v);
    return out;
}

/// Types of vectors that can be stored in State
enum class VecAccess : uint8_t
{
    V_READ=0,
    V_WRITE,
};

static constexpr inline VecAccess V_READ = VecAccess::V_READ;
static constexpr inline VecAccess V_WRITE = VecAccess::V_WRITE;


template <VecType vtype, VecAccess vaccess>
class TVecId;

template <VecType vtype, VecAccess vaccess>
class TStandardVec;

enum class CoordState : uint8_t
{
    NULL_STATE,
    POSITION,
    REST_POSITION,
    FREE_POSITION,
    RESET_POSITION,
    DYNAMIC_INDEX
};

template <VecAccess vaccess>
class TStandardVec<V_COORD, vaccess>
{
public:

    using State = CoordState;

    typedef TVecId<V_COORD, vaccess> MyVecId;

    template<State v_state>
    static constexpr MyVecId state()
    {
        return MyVecId(static_cast<std::underlying_type_t<State>>(v_state));
    }

    SOFA_ATTRIBUTE_DEPRECATED__POSITION()
    static constexpr MyVecId position()      { return state<State::POSITION>();}
    SOFA_ATTRIBUTE_DEPRECATED__REST_POSITION()
    static constexpr MyVecId restPosition()  { return state<State::REST_POSITION>();}
    SOFA_ATTRIBUTE_DEPRECATED__FREE_POSITION()
    static constexpr MyVecId freePosition()  { return state<State::FREE_POSITION>();}
    SOFA_ATTRIBUTE_DEPRECATED__RESET_POSITION()
    static constexpr MyVecId resetPosition() { return state<State::RESET_POSITION>();}

    ///< This is the first index used for dynamically allocated vectors
    static constexpr uint8_t V_FIRST_DYNAMIC_INDEX = static_cast<uint8_t>(State::DYNAMIC_INDEX);

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case static_cast<uint8_t>(State::NULL_STATE): result+= "null";
            break;
        case static_cast<uint8_t>(State::POSITION): result+= "position";
            break;
        case static_cast<uint8_t>(State::REST_POSITION): result+= "restPosition";
            break;
        case static_cast<uint8_t>(State::FREE_POSITION): result+= "freePosition";
            break;
        case static_cast<uint8_t>(State::RESET_POSITION): result+= "resetPosition";
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
            case static_cast<uint8_t>(State::NULL_STATE): return {};
            case static_cast<uint8_t>(State::POSITION): return "States";
            case static_cast<uint8_t>(State::REST_POSITION): return "Rest States";
            case static_cast<uint8_t>(State::FREE_POSITION): return "Free Motion";
            case static_cast<uint8_t>(State::RESET_POSITION): return "States";
            default: return {};
        }
    }
};

enum class DerivState : uint8_t
{
    NULL_STATE,
    VELOCITY,
    RESET_VELOCITY,
    FREE_VELOCITY,
    NORMAL,
    FORCE,
    EXTERNAL_FORCE,
    DX,
    DFORCE,
    DYNAMIC_INDEX
};

template <VecAccess vaccess>
class TStandardVec<V_DERIV, vaccess>
{
public:
    typedef TVecId<V_DERIV, vaccess> MyVecId;

    using State = DerivState;

    template<State v_state>
    static constexpr MyVecId state()
    {
        return MyVecId(static_cast<std::underlying_type_t<State>>(v_state));
    }

    SOFA_ATTRIBUTE_DEPRECATED__VELOCITY()
    static constexpr MyVecId velocity()       { return state<State::VELOCITY>(); }
    SOFA_ATTRIBUTE_DEPRECATED__RESET_VELOCITY()
    static constexpr MyVecId resetVelocity()  { return state<State::RESET_VELOCITY>(); }
    SOFA_ATTRIBUTE_DEPRECATED__FREE_VELOCITY()
    static constexpr MyVecId freeVelocity()   { return state<State::FREE_VELOCITY>(); }
    SOFA_ATTRIBUTE_DEPRECATED__NORMAL()
    static constexpr MyVecId normal()         { return state<State::NORMAL>(); }
    SOFA_ATTRIBUTE_DEPRECATED__FORCE()
    static constexpr MyVecId force()          { return state<State::FORCE>(); }
    SOFA_ATTRIBUTE_DEPRECATED__EXTERNAL_FORCE()
    static constexpr MyVecId externalForce()  { return state<State::EXTERNAL_FORCE>(); }
    SOFA_ATTRIBUTE_DEPRECATED__DX()
    static constexpr MyVecId dx()             { return state<State::DX>(); }
    SOFA_ATTRIBUTE_DEPRECATED__DFORCE()
    static constexpr MyVecId dforce()         { return state<State::DFORCE>(); }

    ///< This is the first index used for dynamically allocated vectors
    static constexpr uint8_t V_FIRST_DYNAMIC_INDEX = static_cast<uint8_t>(State::DYNAMIC_INDEX);

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case static_cast<uint8_t>(State::NULL_STATE): result+= "null";
            break;
        case static_cast<uint8_t>(State::VELOCITY): result+= "velocity";
            break;
        case static_cast<uint8_t>(State::RESET_VELOCITY): result+= "resetVelocity";
            break;
        case static_cast<uint8_t>(State::FREE_VELOCITY): result+= "freeVelocity";
            break;
        case static_cast<uint8_t>(State::NORMAL): result+= "normal";
            break;
        case static_cast<uint8_t>(State::FORCE): result+= "force";
            break;
        case static_cast<uint8_t>(State::EXTERNAL_FORCE): result+= "externalForce";
            break;
        case static_cast<uint8_t>(State::DX): result+= "dx";
            break;
        case static_cast<uint8_t>(State::DFORCE): result+= "dforce";
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
            case static_cast<uint8_t>(State::NULL_STATE): return {};
            case static_cast<uint8_t>(State::VELOCITY):
            case static_cast<uint8_t>(State::DX):
            case static_cast<uint8_t>(State::NORMAL):
            case static_cast<uint8_t>(State::RESET_VELOCITY): return "States";

            case static_cast<uint8_t>(State::FREE_VELOCITY): return "Free Motion";

            case static_cast<uint8_t>(State::FORCE):
            case static_cast<uint8_t>(State::DFORCE):
            case static_cast<uint8_t>(State::EXTERNAL_FORCE): return "Force";
            default: return {};
        }
    }
};

enum class MatrixDerivState : uint8_t
{
    NULL_STATE,
    CONSTRAINT_JACOBIAN,
    MAPPING_JACOBIAN,
    DYNAMIC_INDEX
};

template <VecAccess vaccess>
class TStandardVec<V_MATDERIV, vaccess>
{
public:

    using State = MatrixDerivState;

    typedef TVecId<V_MATDERIV, vaccess> MyVecId;

    template<State v_state>
    static constexpr MyVecId state()
    {
        return MyVecId(static_cast<std::underlying_type_t<State>>(v_state));
    }

    SOFA_ATTRIBUTE_DEPRECATED__CONSTRAINT_JACOBIAN()
    static constexpr MyVecId constraintJacobian() { return state<State::CONSTRAINT_JACOBIAN>();} // jacobian matrix of constraints
    SOFA_ATTRIBUTE_DEPRECATED__MAPPING_JACOBIAN()
    static constexpr MyVecId mappingJacobian() { return state<State::MAPPING_JACOBIAN>();}         // accumulated matrix of the mappings

    ///< This is the first index used for dynamically allocated vectors
    static constexpr uint8_t V_FIRST_DYNAMIC_INDEX = static_cast<uint8_t>(State::DYNAMIC_INDEX);

    static std::string getName(const MyVecId& v)
    {
        std::string result;
        switch(v.getIndex())
        {
        case static_cast<uint8_t>(State::NULL_STATE): result+= "null";
            break;
        case static_cast<uint8_t>(State::CONSTRAINT_JACOBIAN): result+= "holonomic";
            break;
        case static_cast<uint8_t>(State::MAPPING_JACOBIAN): result+= "nonHolonomic";
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
            case static_cast<uint8_t>(State::NULL_STATE): return {}; //null
            case static_cast<uint8_t>(State::CONSTRAINT_JACOBIAN):
            case static_cast<uint8_t>(State::MAPPING_JACOBIAN): return "Jacobian";
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
class SOFA_CORE_API BaseVecId
{
public:
    [[nodiscard]] constexpr VecType getType() const
    {
        return type;
    }

    [[nodiscard]] constexpr unsigned int getIndex() const
    {
        return index;
    }

    VecType type;
    unsigned int index;

protected:
    constexpr BaseVecId(VecType t, unsigned int i)
        : type(t), index(i)
    {}
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
    constexpr TVecId() : BaseVecId(vtype, 0) { }
    explicit constexpr TVecId(unsigned int i) : BaseVecId(vtype, i) { }

    /// Copy constructor
    constexpr TVecId(const TVecId<vtype, vaccess> & v) : BaseVecId(vtype, v.getIndex()) {}

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecAccess vaccess2>
    constexpr TVecId(const TVecId<vtype, vaccess2>& v) : BaseVecId(vtype, v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
    }

    template<VecAccess vaccess2>
    constexpr explicit TVecId(const TVecId<V_ALL, vaccess2>& v) : BaseVecId(vtype, v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
#ifndef NDEBUG
        assert(v.getType() == vtype);
#endif
    }

    // Copy assignment

    constexpr TVecId<vtype, vaccess> & operator=(const TVecId<vtype, vaccess>& other) {
        this->index = other.index;
        this->type = other.type;
        return *this;
    }

    template<VecAccess vaccess2>
    constexpr TVecId<vtype, vaccess> & operator=(const TVecId<vtype, vaccess2>& other) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
        this->index = other.index;
        this->type = other.type;
        return *this;
    }

    template<VecAccess vaccess2>
    constexpr TVecId<vtype, vaccess> & operator=(const TVecId<V_ALL, vaccess2>& other) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
#ifndef NDEBUG
        assert(other.getType() == vtype);
#endif
        this->index = other.index;
        this->type = other.type;
        return *this;
    }


    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator==(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() == v.getType() && getIndex() == v.getIndex();
    }

    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator!=(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() != v.getType() || getIndex() != v.getIndex();
    }

    using TStandardVec<vtype, vaccess>::state;
    using State = typename TStandardVec<vtype, vaccess>::State;

    template<State v_state>
    static constexpr TVecId state()
    {
        return TVecId(static_cast<std::underlying_type_t<State>>(v_state));
    }

    static constexpr TVecId null()
    {
        return TStandardVec<vtype, vaccess>::template state<State::NULL_STATE>();
    }

    [[nodiscard]] constexpr bool isNull() const
    {
        return this->index == static_cast<unsigned int>(State::NULL_STATE);
    }

    [[nodiscard]] std::string getName() const
    {
        return TStandardVec<vtype, vaccess>::getName(*this);
    }
    [[nodiscard]] std::string getGroup() const
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
    constexpr TVecId() : BaseVecId(V_ALL, 0) { }
    constexpr TVecId(VecType t, unsigned int i) : BaseVecId(t, i) { }
    /// Create a generic VecId from a specific or generic one, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecType vtype2, VecAccess vaccess2>
    constexpr TVecId(const TVecId<vtype2, vaccess2>& v) : BaseVecId(v.getType(), v.getIndex())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
    }

    // Copy assignment
    template<VecType vtype2, VecAccess vaccess2>
    constexpr TVecId<V_ALL, vaccess> & operator=(const TVecId<vtype2, vaccess2>& other) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only vector id into a read/write vector id is forbidden.");
        this->index = other.index;
        this->type = other.type;
        return *this;
    }


    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator==(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() == v.getType() && getIndex() == v.getIndex();
    }

    template<VecType vtype2, VecAccess vaccess2>
    constexpr bool operator!=(const TVecId<vtype2, vaccess2>& v) const
    {
        return getType() != v.getType() || getIndex() != v.getIndex();
    }

    static constexpr TVecId null() { return TVecId(V_ALL, 0);}
    [[nodiscard]] constexpr bool isNull() const { return this->index == 0; }

    [[nodiscard]] std::string getName() const
    {
        return TStandardVec<V_ALL, vaccess>::getName(*this);
    }
    [[nodiscard]] std::string getGroup() const
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

static_assert(sizeof(VecId) == sizeof(VecCoordId));

namespace vec_id
{
namespace read_access
{
static constexpr inline auto position = ConstVecCoordId::state<CoordState::POSITION>();
static constexpr inline auto restPosition = ConstVecCoordId::state<CoordState::REST_POSITION>();
static constexpr inline auto freePosition = ConstVecCoordId::state<CoordState::FREE_POSITION>();
static constexpr inline auto resetPosition = ConstVecCoordId::state<CoordState::RESET_POSITION>();

static constexpr inline auto velocity = ConstVecDerivId::state<DerivState::VELOCITY>();
static constexpr inline auto resetVelocity = ConstVecDerivId::state<DerivState::RESET_VELOCITY>();
static constexpr inline auto freeVelocity = ConstVecDerivId::state<DerivState::FREE_VELOCITY>();
static constexpr inline auto normal = ConstVecDerivId::state<DerivState::NORMAL>();
static constexpr inline auto force = ConstVecDerivId::state<DerivState::FORCE>();
static constexpr inline auto externalForce = ConstVecDerivId::state<DerivState::EXTERNAL_FORCE>();
static constexpr inline auto dx = ConstVecDerivId::state<DerivState::DX>();
static constexpr inline auto dforce = ConstVecDerivId::state<DerivState::DFORCE>();

static constexpr inline auto constraintJacobian = ConstMatrixDerivId::state<MatrixDerivState::CONSTRAINT_JACOBIAN>();
static constexpr inline auto mappingJacobian = ConstMatrixDerivId::state<MatrixDerivState::MAPPING_JACOBIAN>();
}

namespace write_access
{
static constexpr inline auto position = VecCoordId::state<CoordState::POSITION>();
static constexpr inline auto restPosition = VecCoordId::state<CoordState::REST_POSITION>();
static constexpr inline auto freePosition = VecCoordId::state<CoordState::FREE_POSITION>();
static constexpr inline auto resetPosition = VecCoordId::state<CoordState::RESET_POSITION>();

static constexpr inline auto velocity = VecDerivId::state<DerivState::VELOCITY>();
static constexpr inline auto resetVelocity = VecDerivId::state<DerivState::RESET_VELOCITY>();
static constexpr inline auto freeVelocity = VecDerivId::state<DerivState::FREE_VELOCITY>();
static constexpr inline auto normal = VecDerivId::state<DerivState::NORMAL>();
static constexpr inline auto force = VecDerivId::state<DerivState::FORCE>();
static constexpr inline auto externalForce = VecDerivId::state<DerivState::EXTERNAL_FORCE>();
static constexpr inline auto dx = VecDerivId::state<DerivState::DX>();
static constexpr inline auto dforce = VecDerivId::state<DerivState::DFORCE>();

static constexpr inline auto constraintJacobian = MatrixDerivId::state<MatrixDerivState::CONSTRAINT_JACOBIAN>();
static constexpr inline auto mappingJacobian = MatrixDerivId::state<MatrixDerivState::MAPPING_JACOBIAN>();
}
}

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
template<class DataTypes> struct StateType<DataTypes, core::V_MATDERIV>
{
    using type = typename DataTypes::MatrixDeriv;
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
