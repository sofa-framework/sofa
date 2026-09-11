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

#include <sofa/core/VecId.h>
#include <sofa/core/objectmodel/Data.h>
#include <map>

namespace sofa::core
{

class SOFA_CORE_API BaseState;
template<class DataTypes> class State;

/// Identify a vector of a given type stored in multiple State instances
/// This class is templated in order to create different variations (generic versus specific type, read-only vs write access)
template<class DataTypes, VecType vtype, VecAccess vaccess>
struct StateVecAccessor
{
    using MyVecId = TVecId<vtype, vaccess>;

    // const State* for read-only access, State* for write access
    using MyStatePtr = std::conditional_t<vaccess == V_WRITE,
                                           State<DataTypes>*,
                                           const State<DataTypes>*>;

    StateVecAccessor(MyStatePtr st, const MyVecId& vecid) : state(st), id(vecid)
    {}

    operator MyVecId() const { return id; }

    auto* read() const requires (vtype != V_ALL)
    {
        assert(state);
        return state->read(id);
    }

    auto* write() const requires (vtype != V_ALL && vaccess == V_WRITE)
    {
        assert(state);
        return state->write(id);
    }

protected:
    MyStatePtr state { nullptr };
    MyVecId id;
};

template <VecType vtype, VecAccess vaccess>
class TMultiVecId
{
public:
    typedef TVecId<vtype, vaccess> MyVecId;

    typedef std::map<const BaseState*, MyVecId> IdMap;
    typedef typename IdMap::iterator IdMap_iterator;
    typedef typename IdMap::const_iterator IdMap_const_iterator;

protected:
    MyVecId defaultId;

private:
    std::shared_ptr< IdMap > idMap_ptr;

    /// Share the id map from a binary-compatible TMultiVecId instantiation
    /// without copying it. Safe because TVecId<vtype, V_READ> and
    /// TVecId<vtype, V_WRITE> have the same memory layout (same underlying
    /// integral index; access direction is a type-level tag only).
    /// This avoids an O(n) map copy when e.g. a writable id is passed to a
    /// function expecting a read-only one.
    template<VecType vtype2, VecAccess vaccess2>
    void sharedIdMapCast(const TMultiVecId<vtype2, vaccess2>& mv)
    {
        union {
            const std::shared_ptr<IdMap>* this_map_type;
            const std::shared_ptr<typename TMultiVecId<vtype2, vaccess2>::IdMap>* other_map_type;
        } ptr;
        ptr.other_map_type = &mv.idMap_ptr;
        idMap_ptr = *(ptr.this_map_type);
    }

    template <VecType vtype2, VecAccess vaccess2> friend class TMultiVecId;

protected:
    IdMap& writeIdMap()
    {
        if (!idMap_ptr)
            idMap_ptr.reset(new IdMap());
        else if(!(idMap_ptr.use_count() == 1))
            idMap_ptr.reset(new IdMap(*idMap_ptr));
        return *idMap_ptr;
    }
public:
    bool hasIdMap() const { return idMap_ptr != nullptr; }
    const IdMap& getIdMap() const
    {
        if (!idMap_ptr)
        {
            static const IdMap empty;
            return empty;
        }
        return *idMap_ptr;
    }

    TMultiVecId() = default;

    /// Copy from a TVecId.
    /// When vtype != V_ALL: only the same vtype is accepted (vtype2 must equal vtype).
    /// When vtype == V_ALL: any vtype2 is accepted (widening to V_ALL).
    /// In both cases, write->read is allowed but read->write is forbidden.
    template<VecType vtype2, VecAccess vaccess2>
        requires (vtype == V_ALL || vtype2 == vtype)
    TMultiVecId(const TVecId<vtype2, vaccess2>& v) : defaultId(v)
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
    }

    /// Copy assignment from a TVecId (same constraints as the constructor above).
    template<VecType vtype2, VecAccess vaccess2>
        requires (vtype == V_ALL || vtype2 == vtype)
    TMultiVecId<vtype, vaccess>& operator=(const TVecId<vtype2, vaccess2>& v)
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
        defaultId = v;
        return *this;
    }

    //// Copy constructor (exact same type)
    TMultiVecId(const TMultiVecId<vtype, vaccess>& mv)
        : defaultId(mv.getDefaultId())
        , idMap_ptr(mv.idMap_ptr)
    {
    }

    /// Copy assignment (exact same type)
    TMultiVecId<vtype, vaccess>& operator=(const TMultiVecId<vtype, vaccess>& mv)
    {
        defaultId = mv.getDefaultId();
        idMap_ptr = mv.idMap_ptr;
        return *this;
    }

    //// Copy constructor from a TMultiVecId of the same vtype but different access.
    //// Only available when vtype != V_ALL.
    //// For the vtype == V_ALL case, any-vtype2 widening is handled by the next
    //// constructor instead.
    //// When the access is compatible (write -> read), the id map is shared
    //// instead of copied, because these types are binary-compatible.
    template<VecAccess vaccess2>
        requires (vtype != V_ALL && vaccess2 != vaccess)
    TMultiVecId(const TMultiVecId<vtype, vaccess2>& mv) : defaultId(mv.getDefaultId())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
        if (mv.hasIdMap())
        {
            sharedIdMapCast(mv);
        }
    }

    template<VecAccess vaccess2>
        requires (vtype != V_ALL && vaccess2 != vaccess)
    TMultiVecId<vtype, vaccess>& operator=(const TMultiVecId<vtype, vaccess2>& mv)
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");

        defaultId = mv.defaultId;
        if (mv.hasIdMap())
        {
            sharedIdMapCast(mv);
        }

        return *this;
    }

    //// Copy constructor from any TMultiVecId<vtype2, vaccess2>.
    //// Only available when vtype == V_ALL (widening from a specific type to V_ALL).
    //// The id map is shared instead of copied when the access direction is
    //// compatible, for the same performance reason as above.
    template<VecType vtype2, VecAccess vaccess2>
        requires (vtype == V_ALL && (vtype2 != V_ALL || vaccess2 != vaccess))
    TMultiVecId(const TMultiVecId<vtype2, vaccess2>& mv) : defaultId(mv.getDefaultId())
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");

        if (mv.hasIdMap())
        {
            sharedIdMapCast(mv);
        }
    }

    template<VecType vtype2, VecAccess vaccess2>
        requires (vtype == V_ALL && (vtype2 != V_ALL || vaccess2 != vaccess))
    TMultiVecId<vtype, vaccess>& operator=(const TMultiVecId<vtype2, vaccess2>& mv)
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");

        defaultId = mv.defaultId;
        if (mv.hasIdMap())
        {
            sharedIdMapCast(mv);
        }

        return *this;
    }

    //// Provides explicit conversions from TMultiVecId<V_ALL> to a specific-vtype
    //// TMultiVecId (e.g. MultiVecId -> MultiVecCoordId).
    //// Only available when vtype != V_ALL.
    //// The explicit keyword forbids the compiler to use it automatically: the
    //// caller must have checked the type of the source vector before narrowing.
    template<VecAccess vaccess2>
        requires (vtype != V_ALL)
    explicit TMultiVecId(const TMultiVecId<V_ALL, vaccess2>& mv)
        : defaultId(static_cast<MyVecId>(mv.getDefaultId()))
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");

        if (mv.hasIdMap())
        {
            IdMap& map = writeIdMap();

            for (typename TMultiVecId<V_ALL,vaccess2>::IdMap_const_iterator it = mv.getIdMap().begin(), itend = mv.getIdMap().end();
                    it != itend; ++it)
                map[it->first] = MyVecId(it->second);
        }
    }

    template<VecAccess vaccess2>
        requires (vtype != V_ALL)
    TMultiVecId<vtype, vaccess>& operator=(const TMultiVecId<V_ALL, vaccess2>& mv)
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");

        defaultId = static_cast<MyVecId>(mv.defaultId);
        if (mv.hasIdMap())
        {
            IdMap& map = writeIdMap();

            for (typename TMultiVecId<V_ALL,vaccess2>::IdMap_const_iterator it = mv.getIdMap().begin(), itend = mv.getIdMap().end();
                    it != itend; ++it)
                map[it->first] = MyVecId(it->second);
        }

        return *this;
    }

    void setDefaultId(const MyVecId& id)
    {
        defaultId = id;
    }

    template<class State>
    void setId(const std::set<State>& states, const MyVecId& id)
    {
        if (!states.empty())
        {
            IdMap& map = writeIdMap();
            for (auto* state : states)
                map[state] = id;
        }
    }

    void setId(const BaseState* s, const MyVecId& id)
    {
        IdMap& map = writeIdMap();
        map[s] = id;
    }

    void assign(const MyVecId& id)
    {
        defaultId = id;
        idMap_ptr.reset();
    }

    const MyVecId& getId(const BaseState* s) const
    {
        if (!hasIdMap()) return defaultId;
        const IdMap& map = getIdMap();

        IdMap_const_iterator it = map.find(s);
        if (it != map.end()) return it->second;
        else                 return defaultId;
    }

    const MyVecId& getDefaultId() const
    {
        return defaultId;
    }

    std::string getName() const
    {
        if (!hasIdMap())
            return defaultId.getName();
        else
        {
            std::ostringstream out;
            out << '{';
            out << defaultId.getName() << "[*";
            const IdMap& map = getIdMap();
            MyVecId prev = defaultId;
            for (IdMap_const_iterator it = map.begin(), itend = map.end(); it != itend; ++it)
            {
                if (it->second != prev) // new id
                {
                    out << "],";
                    if (it->second.getType() == defaultId.getType())
                        out << it->second.getIndex();
                    else
                        out << it->second.getName();
                    out << '[';
                    prev = it->second;
                }
                else out << ',';
                if (it->first == nullptr) out << "nullptr";
                else
                    out << it->first->getName();
            }
            out << "]}";
            return out.str();
        }
    }

    friend inline std::ostream& operator<<(std::ostream& out, const TMultiVecId<vtype, vaccess>& v)
    {
        out << v.getName();
        return out;
    }

    static TMultiVecId<vtype, vaccess> null() { return TMultiVecId(MyVecId::null()); }
    bool isNull() const
    {
        if (!this->defaultId.isNull()) return false;
        if (hasIdMap())
            for (IdMap_const_iterator it = getIdMap().begin(), itend = getIdMap().end(); it != itend; ++it)
                if (!it->second.isNull()) return false;
        return true;
    }

    template <class DataTypes>
    StateVecAccessor<DataTypes,vtype,vaccess> operator[](State<DataTypes>* s) const
    {
        return StateVecAccessor<DataTypes,vtype,vaccess>(s,getId(s));
    }

    template <class DataTypes>
    StateVecAccessor<DataTypes,vtype,V_READ> operator[](const State<DataTypes>* s) const
    {
        return StateVecAccessor<DataTypes,vtype,V_READ>(s,getId(s));
    }
};


typedef TMultiVecId<V_COORD, V_READ> ConstMultiVecCoordId;
typedef TMultiVecId<V_COORD, V_WRITE>     MultiVecCoordId;
typedef TMultiVecId<V_DERIV, V_READ> ConstMultiVecDerivId;
typedef TMultiVecId<V_DERIV, V_WRITE>     MultiVecDerivId;
typedef TMultiVecId<V_MATDERIV, V_READ> ConstMultiMatrixDerivId;
typedef TMultiVecId<V_MATDERIV, V_WRITE>     MultiMatrixDerivId;
typedef TMultiVecId<V_ALL, V_READ>      ConstMultiVecId;
typedef TMultiVecId<V_ALL, V_WRITE>          MultiVecId;
} // namespace sofa::core
