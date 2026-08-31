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
template <VecType vtype, VecAccess vaccess>
class TMultiVecId;

/// Helper class to access vectors of a given type in a given State
template<class DataTypes, VecType vtype, VecAccess vaccess>
struct StateVecAccessor;

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_COORD, V_READ>
{
    typedef TVecId<V_COORD, V_READ> MyVecId;
    typedef Data<typename DataTypes::VecCoord> MyDataVec;

    StateVecAccessor(const State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_COORD, V_WRITE>
{
    typedef TVecId<V_COORD, V_WRITE> MyVecId;
    typedef Data<typename DataTypes::VecCoord> MyDataVec;

    StateVecAccessor(State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }
    MyDataVec* write() const {  return state->write(id);  }

protected:
    State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_DERIV, V_READ>
{
    typedef TVecId<V_DERIV, V_READ> MyVecId;
    typedef Data<typename DataTypes::VecDeriv> MyDataVec;

    StateVecAccessor(const State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_DERIV, V_WRITE>
{
    typedef TVecId<V_DERIV, V_WRITE> MyVecId;
    typedef Data<typename DataTypes::VecDeriv> MyDataVec;

    StateVecAccessor(State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }
    MyDataVec* write() const {  return state->write(id);  }

protected:
    State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_MATDERIV, V_READ>
{
    typedef TVecId<V_MATDERIV, V_READ> MyVecId;
    typedef Data<typename DataTypes::MatrixDeriv> MyDataVec;

    StateVecAccessor(const State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_MATDERIV, V_WRITE>
{
    typedef TVecId<V_MATDERIV, V_WRITE> MyVecId;
    typedef Data<typename DataTypes::MatrixDeriv> MyDataVec;

    StateVecAccessor(State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }
    MyDataVec* write() const {  return state->write(id);  }

protected:
    State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_ALL, V_READ>
{
    typedef TVecId<V_ALL, V_READ> MyVecId;
    //typedef BaseData MyDataVec;

    StateVecAccessor(const State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }
    //const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_ALL, V_WRITE>
{
    typedef TVecId<V_ALL, V_WRITE> MyVecId;

    StateVecAccessor(State<DataTypes>* st, const MyVecId& vecid) : state(st), id(vecid) {}
    operator MyVecId() const {  return id;  }

protected:
    State<DataTypes>* state;
    MyVecId id;
};

template <VecType vtype, VecAccess vaccess>
class TMultiVecId
{
public:
    using MyVecId = TVecId<vtype, vaccess>;
    using IdMap = std::map<const BaseState*, MyVecId>;
    using IdMap_iterator = typename IdMap::iterator;
    using IdMap_const_iterator = typename IdMap::const_iterator;

protected:
    MyVecId defaultId;

private:
    std::shared_ptr<IdMap> idMap_ptr;

    template <VecType vtype2, VecAccess vaccess2>
    friend class TMultiVecId;

    IdMap& writeIdMap()
    {
        if (!idMap_ptr)
            idMap_ptr = std::make_shared<IdMap>();
        else if (idMap_ptr.use_count() > 1)
            idMap_ptr = std::make_shared<IdMap>(*idMap_ptr);
        return *idMap_ptr;
    }

public:
    bool hasIdMap() const { return idMap_ptr != nullptr; }
    const  IdMap& getIdMap() const
    {
        if (!idMap_ptr)
        {
            static const IdMap empty;
            return empty;
        }
        return *idMap_ptr;
    }

    TMultiVecId() = default;

    // =========================================================================
    // 1. Construction from TVecId
    // =========================================================================

    // Implicit construction when types match, OR when converting any vtype2 to V_ALL
    template<VecType vtype2, VecAccess vaccess2>
    requires (vtype == vtype2 || vtype == V_ALL)
    TMultiVecId(const TVecId<vtype2, vaccess2>& v)
        : defaultId(v)
    {
        static_assert(vaccess2 >= vaccess,
            "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
    }

    // =========================================================================
    // 2. Implicit Construction from another TMultiVecId
    //    - Same vtype (with compatible access)
    //    - OR from ANY vtype2 to V_ALL (with compatible access)
    // =========================================================================
    template<VecType vtype2, VecAccess vaccess2>
    requires ((vtype == vtype2 || vtype == V_ALL) && !(vtype == vtype2 && vaccess == vaccess2))
    TMultiVecId(const TMultiVecId<vtype2, vaccess2>& mv)
        : defaultId(mv.getDefaultId())
    {
        static_assert(vaccess2 >= vaccess,
            "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");

        if (mv.hasIdMap())
        {
            if constexpr (vtype == vtype2)
            {
                // Share the map directly via type-punning / reinterpretation
                union {
                    const std::shared_ptr<IdMap>* this_map_type;
                    const std::shared_ptr<typename TMultiVecId<vtype2, vaccess2>::IdMap>* other_map_type;
                } ptr;
                ptr.other_map_type = &mv.idMap_ptr;
                idMap_ptr = *(ptr.this_map_type);
            }
            else
            {
                // Converting specific vtype to V_ALL: populate map
                IdMap& map = writeIdMap();
                for (const auto& [st, vecId] : mv.getIdMap())
                    map[st] = MyVecId(vecId);
            }
        }
    }

    // =========================================================================
    // 3. Explicit Conversion from TMultiVecId<V_ALL, ...> to specific vtype
    // =========================================================================
    template<VecAccess vaccess2>
    requires (vtype != V_ALL)
    explicit TMultiVecId(const TMultiVecId<V_ALL, vaccess2>& mv)
        : defaultId(static_cast<MyVecId>(mv.getDefaultId()))
    {
        static_assert(vaccess2 >= vaccess,
            "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");

        if (mv.hasIdMap())
        {
            IdMap& map = writeIdMap();
            for (const auto& [st, vecId] : mv.getIdMap())
                map[st] = MyVecId(vecId);
        }
    }

    // Standard copy/move constructors & assignment operators...
    TMultiVecId(const TMultiVecId&) = default;
    TMultiVecId& operator=(const TMultiVecId&) = default;

    void setDefaultId(const MyVecId& id)
    {
        defaultId = id;
    }

    template<class State>
    void setId(const std::set<State>& states, const MyVecId& id)
    {
        IdMap& map = writeIdMap();
        for (typename std::set<State>::const_iterator it = states.begin(), itend = states.end(); it != itend; ++it)
            map[*it] = id;
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

    friend inline std::ostream& operator << ( std::ostream& out, const TMultiVecId<vtype, vaccess>& v )
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
