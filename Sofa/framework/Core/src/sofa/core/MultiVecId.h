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
    typedef TVecId<vtype, vaccess> MyVecId;

    typedef std::map<const BaseState*, MyVecId> IdMap;
    typedef typename IdMap::iterator IdMap_iterator;
    typedef typename IdMap::const_iterator IdMap_const_iterator;

protected:
    MyVecId defaultId;

private:
    std::shared_ptr< IdMap > idMap_ptr;

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

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecAccess vaccess2>
    TMultiVecId(const TVecId<vtype, vaccess2>& v)
        :
        defaultId(v)
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
    }

    /// Copy assignment from another VecId
    template<VecAccess vaccess2>
    TMultiVecId<vtype, vaccess> & operator= (const TVecId<vtype, vaccess2>& v) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
        defaultId = v;
        return *this;
    }

    //// Copy constructor
    TMultiVecId( const TMultiVecId<vtype,vaccess>& mv)
        : defaultId( mv.getDefaultId() )
        , idMap_ptr( mv.idMap_ptr )
    {
    }

    /// Copy assignment
    TMultiVecId<vtype, vaccess> & operator= (const TMultiVecId<vtype, vaccess>& mv) {
        defaultId = mv.getDefaultId();
        idMap_ptr = mv.idMap_ptr;
        return *this;
    }

    //// Only TMultiVecId< V_ALL , vaccess> can declare copy constructors with all
    //// other kinds of TMultiVecIds, namely MultiVecCoordId, MultiVecDerivId...
    //// In other cases, the copy constructor takes a TMultiVecId of the same type
    //// ie copy construct a MultiVecCoordId from a const MultiVecCoordId& or a
    //// ConstMultiVecCoordId&. Other conversions should be done with the
    //// next constructor that can only be used if requested explicitly.
    template< VecAccess vaccess2>
    TMultiVecId( const TMultiVecId<vtype,vaccess2>& mv) : defaultId( mv.getDefaultId() )
    {
        static_assert( vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden." );
        if (mv.hasIdMap())
        {
            // When we assign a V_WRITE version to a V_READ version of the same type, which are binary compatible,
            // share the maps like with a copy constructor, because otherwise a simple operation like passing a
            // MultiVecCoordId to a method taking a ConstMultiVecCoordId to indicate it won't modify it
            // will cause a temporary copy of the map, which this define was meant to avoid!

            // Type-punning
            union {
                const std::shared_ptr< IdMap > * this_map_type;
                const std::shared_ptr< typename TMultiVecId<vtype,vaccess2>::IdMap > * other_map_type;
            } ptr;
            ptr.other_map_type = &mv.idMap_ptr;
            idMap_ptr = *(ptr.this_map_type);
        }
    }

    template<VecAccess vaccess2>
    TMultiVecId<vtype, vaccess> & operator= (const TMultiVecId<vtype, vaccess2>& mv) {
        static_assert( vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden." );

        defaultId = mv.defaultId;
        if (mv.hasIdMap()) {
            // When we assign a V_WRITE version to a V_READ version of the same type, which are binary compatible,
            // share the maps like with a copy constructor, because otherwise a simple operation like passing a
            // MultiVecCoordId to a method taking a ConstMultiVecCoordId to indicate it won't modify it
            // will cause a temporary copy of the map, which this define was meant to avoid!

            // Type-punning
            union {
                const std::shared_ptr< IdMap > * this_map_type;
                const std::shared_ptr< typename TMultiVecId<vtype,vaccess2>::IdMap > * other_map_type;
            } ptr;
            ptr.other_map_type = &mv.idMap_ptr;
            idMap_ptr = *(ptr.this_map_type);
        }

        return *this;
    }

    //// Provides explicit conversions from MultiVecId to MultiVecCoordId/...
    //// The explicit keyword forbid the compiler to use it automatically, as
    //// the user should check the type of the source vector before using this
    //// conversion.
    template< VecAccess vaccess2>
    explicit TMultiVecId( const TMultiVecId<V_ALL,vaccess2>& mv) : defaultId( static_cast<MyVecId>(mv.getDefaultId()) )
    {
        static_assert( vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden." );

        if (mv.hasIdMap())
        {
            IdMap& map = writeIdMap();

            for (typename TMultiVecId<V_ALL,vaccess2>::IdMap_const_iterator it = mv.getIdMap().begin(), itend = mv.getIdMap().end();
                    it != itend; ++it)
                map[it->first] = MyVecId(it->second);
        }
    }

    template<VecAccess vaccess2>
    TMultiVecId<vtype, vaccess> & operator= (const TMultiVecId<V_ALL, vaccess2>& mv) {
        static_assert( vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden." );

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



template <VecAccess vaccess>
class TMultiVecId<V_ALL, vaccess>
{
public:
    typedef TVecId<V_ALL, vaccess> MyVecId;

    typedef std::map<const BaseState*, MyVecId> IdMap;
    typedef typename IdMap::iterator IdMap_iterator;
    typedef typename IdMap::const_iterator IdMap_const_iterator;

protected:
    MyVecId defaultId;

private:
    std::shared_ptr< IdMap > idMap_ptr;

	template <VecType vtype2, VecAccess vaccess2> friend class TMultiVecId;

protected:
    IdMap& writeIdMap()
    {
        if (!idMap_ptr)
            idMap_ptr.reset(new IdMap());
        else if(!idMap_ptr.unique())
            idMap_ptr.reset(new IdMap(*idMap_ptr));
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

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecType vtype2, VecAccess vaccess2>
    TMultiVecId(const TVecId<vtype2, vaccess2>& v) : defaultId(v)
    {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
    }

    /// Copy assignment from another VecId
    template<VecType vtype2, VecAccess vaccess2>
    TMultiVecId<V_ALL, vaccess> & operator= (const TVecId<vtype2, vaccess2>& v) {
        static_assert(vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden.");
        defaultId = v;
        return *this;
    }

    //// Copy constructor
    TMultiVecId( const TMultiVecId<V_ALL,vaccess>& mv)
        : defaultId( mv.getDefaultId() )
        , idMap_ptr( mv.idMap_ptr )
    {
    }

    /// Copy assignment
    TMultiVecId<V_ALL, vaccess> & operator= (const TMultiVecId<V_ALL, vaccess>& mv) {
        defaultId = mv.getDefaultId();
        idMap_ptr = mv.idMap_ptr;
        return *this;
    }

    //// Only TMultiVecId< V_ALL , vaccess> can declare copy constructors with all
    //// other kinds of TMultiVecIds, namely MultiVecCoordId, MultiVecDerivId...
    //// In other cases, the copy constructor takes a TMultiVecId of the same type
    //// ie copy construct a MultiVecCoordId from a const MultiVecCoordId& or a
    //// ConstMultiVecCoordId&.
    template< VecType vtype2, VecAccess vaccess2>
    TMultiVecId( const TMultiVecId<vtype2,vaccess2>& mv) : defaultId( mv.getDefaultId() )
    {
        static_assert( vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden." );

        if (mv.hasIdMap())
        {
			// When we assign a V_WRITE version to a V_READ version of the same type, which are binary compatible,
			// share the maps like with a copy constructor, because otherwise a simple operation like passing a
			// MultiVecCoordId to a method taking a ConstMultiVecCoordId to indicate it won't modify it
			// will cause a temporary copy of the map, which this define was meant to avoid!

            // Type-punning
            union {
                const std::shared_ptr< IdMap > * this_map_type;
                const std::shared_ptr< typename TMultiVecId<vtype2,vaccess2>::IdMap > * other_map_type;
            } ptr;
            ptr.other_map_type = &mv.idMap_ptr;
            idMap_ptr = *(ptr.this_map_type);
        }
    }

    template<VecType vtype2, VecAccess vaccess2>
    TMultiVecId<V_ALL, vaccess> & operator= (const TMultiVecId<vtype2, vaccess2>& mv) {
        static_assert( vaccess2 >= vaccess, "Copy from a read-only multi-vector id into a read/write multi-vector id is forbidden." );

        defaultId = mv.defaultId;
        if (mv.hasIdMap()) {
            // When we assign a V_WRITE version to a V_READ version of the same type, which are binary compatible,
            // share the maps like with a copy constructor, because otherwise a simple operation like passing a
            // MultiVecCoordId to a method taking a ConstMultiVecCoordId to indicate it won't modify it
            // will cause a temporary copy of the map, which this define was meant to avoid!

            // Type-punning
            union {
                const std::shared_ptr< IdMap > * this_map_type;
                const std::shared_ptr< typename TMultiVecId<vtype2,vaccess2>::IdMap > * other_map_type;
            } ptr;
            ptr.other_map_type = &mv.idMap_ptr;
            idMap_ptr = *(ptr.this_map_type);
        }

        return *this;
    }

    void setDefaultId(const MyVecId& id)
    {
        defaultId = id;
    }

    template<class StateSet>
    void setId(const StateSet& states, const MyVecId& id)
    {
        IdMap& map = writeIdMap();
        for (typename StateSet::const_iterator it = states.begin(), itend = states.end(); it != itend; ++it)
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

    friend inline std::ostream& operator << ( std::ostream& out, const TMultiVecId<V_ALL, vaccess>& v )
    {
        out << v.getName();
        return out;
    }

    static TMultiVecId<V_ALL, vaccess> null() { return TMultiVecId(MyVecId::null()); }
    bool isNull() const
    {
        if (!this->defaultId.isNull()) return false;
        if (hasIdMap())
            for (IdMap_const_iterator it = getIdMap().begin(), itend = getIdMap().end(); it != itend; ++it)
                if (!it->second.isNull()) return false;
        return true;
    }

    template <class DataTypes>
    StateVecAccessor<DataTypes,V_ALL,vaccess> operator[](State<DataTypes>* s) const
    {
        return StateVecAccessor<DataTypes,V_ALL,vaccess>(s,getId(s));
    }

    template <class DataTypes>
    StateVecAccessor<DataTypes,V_ALL,V_READ> operator[](const State<DataTypes>* s) const
    {
        return StateVecAccessor<DataTypes,V_ALL,V_READ>(s,getId(s));
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
