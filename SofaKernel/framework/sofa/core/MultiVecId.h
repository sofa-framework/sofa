/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_MULTIVECID_H
#define SOFA_CORE_MULTIVECID_H

#include <sofa/core/VecId.h>

#include <map>

namespace sofa
{

namespace core
{

class SOFA_CORE_API BaseState;
template<class DataTypes> class State;

/// Identify a vector of a given type stored in multiple State instances
/// This class is templated in order to create different variations (generic versus specific type, read-only vs write access)
template <VecType vtype, VecAccess vaccess>
class TMultiVecId;

/*
/// Helper class to infer the types of elements, vectors, and Data for vectors of the given VecType in states with the given DataTypes
template<class DataTypes, VecType vtype>
struct DataTypesVecInfo;

template<class DataTypes>
struct DataTypesVecInfo<V_COORD>
{
    typedef typename DataTypes::Coord T;
    typedef typename DataTypes::VecCoord VecT;
    typedef Data<VecT> DataVecT;
};

template<class DataTypes>
struct DataTypesVecInfo<V_DERIV>
{
    typedef typename DataTypes::Deriv T;
    typedef typename DataTypes::VecDeriv VecT;
    typedef Data<VecT> DataVecT;
};

template<class DataTypes>
struct DataTypesVecInfo<V_MATDERIV>
{
    typedef typename DataTypes::MatrixDeriv T;
    typedef typename DataTypes::MatrixDeriv VecT;
    typedef Data<VecT> DataVecT;
};

template<class DataTypes>
struct DataTypesVecInfo<V_ALL>
{
    typedef void T;
    typedef void VecT;
    typedef BaseData DataVecT;
};
*/

/// Helper class to access vectors of a given type in a given State
template<class DataTypes, VecType vtype, VecAccess vaccess>
struct StateVecAccessor;

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_COORD, V_READ>
{
public:
    typedef TVecId<V_COORD, V_READ> MyVecId;
    typedef Data<typename DataTypes::VecCoord> MyDataVec;

    StateVecAccessor(const State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_COORD, V_WRITE>
{
public:
    typedef TVecId<V_COORD, V_WRITE> MyVecId;
    typedef Data<typename DataTypes::VecCoord> MyDataVec;

    StateVecAccessor(State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
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
public:
    typedef TVecId<V_DERIV, V_READ> MyVecId;
    typedef Data<typename DataTypes::VecDeriv> MyDataVec;

    StateVecAccessor(const State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_DERIV, V_WRITE>
{
public:
    typedef TVecId<V_DERIV, V_WRITE> MyVecId;
    typedef Data<typename DataTypes::VecDeriv> MyDataVec;

    StateVecAccessor(State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
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
public:
    typedef TVecId<V_MATDERIV, V_READ> MyVecId;
    typedef Data<typename DataTypes::MatrixDeriv> MyDataVec;

    StateVecAccessor(const State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
    operator MyVecId() const {  return id;  }
    const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_MATDERIV, V_WRITE>
{
public:
    typedef TVecId<V_MATDERIV, V_WRITE> MyVecId;
    typedef Data<typename DataTypes::MatrixDeriv> MyDataVec;

    StateVecAccessor(State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
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
public:
    typedef TVecId<V_ALL, V_READ> MyVecId;
    //typedef BaseData MyDataVec;

    StateVecAccessor(const State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
    operator MyVecId() const {  return id;  }
    //const MyDataVec* read()  const {  return state-> read(id);  }

protected:
    const State<DataTypes>* state;
    MyVecId id;
};

template<class DataTypes>
struct StateVecAccessor<DataTypes, V_ALL, V_WRITE>
{
public:
    typedef TVecId<V_ALL, V_WRITE> MyVecId;
    //typedef BaseData MyDataVec;

    StateVecAccessor(State<DataTypes>* state, const MyVecId& id) : state(state), id(id) {}
    operator MyVecId() const {  return id;  }
    //const MyDataVec* read()  const {  return state-> read(id);  }
    //      MyDataVec* write() const {  return state->write(id);  }

protected:
    State<DataTypes>* state;
    MyVecId id;
};

#define MAP_PTR

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

#ifndef MAP_PTR
    IdMap idMap;
    IdMap& writeIdMap()
    {
        return idMap;
    }
public:
    bool hasIdMap() const { return !idMap.empty(); }
    const  IdMap& getIdMap() const
    {
        return idMap;
    }
#else

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

#endif

public:

    TMultiVecId()
    {
    }

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecAccess vaccess2>
    TMultiVecId(const TVecId<vtype, vaccess2>& v)
        :
        defaultId(v)
    {
        static_assert(vaccess2 >= vaccess, "");
    }

    //// Copy constructor
    TMultiVecId( const TMultiVecId<vtype,vaccess>& mv)
        : defaultId( mv.getDefaultId() )
#ifdef MAP_PTR
        , idMap_ptr( mv.idMap_ptr )
#else
        , idMap( mv.idMap )
#endif
    {
    }

    //// Only TMultiVecId< V_ALL , vaccess> can declare copy constructors with all
    //// other kinds of TMultiVecIds, namely MultiVecCoordId, MultiVecDerivId...
    //// In other cases, the copy constructor takes a TMultiVecId of the same type
    //// ie copy construct a MultiVecCoordId from a const MultiVecCoordId& or a
    //// ConstMultiVecCoordId&. Other conversions should be done with the
    //// next constructor that can only be used if requested explicitly.
    template< VecType vtype2, VecAccess vaccess2>
    TMultiVecId( const TMultiVecId<vtype2,vaccess2>& mv) : defaultId( mv.getDefaultId() )
    {
        static_assert( vaccess2 > vaccess, "" );
        static_assert( vtype != V_ALL, "" ); // we should be using the V_ALL specific specialization in this case.
        static_assert( vtype2 == vtype, "" );
        if (mv.hasIdMap())
        {
#ifdef MAP_PTR
			// When we assign a V_WRITE version to a V_READ version of the same type, which are binary compatible,
			// share the maps like with a copy constructor, because otherwise a simple operation like passing a
			// MultiVecCoordId to a method taking a ConstMultiVecCoordId to indicate it won't modify it
			// will cause a temporary copy of the map, which this define was meant to avoid!
#	ifndef _MSC_VER
#if __GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 4
#		pragma GCC diagnostic push
#		pragma GCC diagnostic ignored "-Wstrict-aliasing" // this should not create problems here
#endif
#	endif
            idMap_ptr = *reinterpret_cast<const std::shared_ptr< IdMap > * >(&mv.idMap_ptr);
#	ifndef _MSC_VER
#if __GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 4
#		pragma GCC diagnostic pop
#endif
#	endif
#else
			IdMap& map = writeIdMap();
			std::copy(mv.getIdMap().begin(), mv.getIdMap().end(), std::inserter(map, map.begin()) );
#endif
        }
    }
    //// Provides explicit conversions from MultiVecId to MultiVecCoordId/...
    //// The explicit keyword forbid the compiler to use it automatically, as
    //// the user should check the type of the source vector before using this
    //// conversion.
    template< VecAccess vaccess2>
    explicit TMultiVecId( const TMultiVecId<V_ALL,vaccess2>& mv) : defaultId( MyVecId(mv.getDefaultId()) )
    {
        static_assert( vaccess2 >= vaccess, "" );
        static_assert( !(vtype == V_ALL), "" ); // for V_ALL vectors, this constructor is redundant with the previous one

        if (mv.hasIdMap())
        {
            IdMap& map = writeIdMap();

            for (typename TMultiVecId<V_ALL,vaccess2>::IdMap_const_iterator it = mv.getIdMap().begin(), itend = mv.getIdMap().end();
                    it != itend; ++it)
                map[it->first] = MyVecId(it->second);
        }
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
#ifndef MAP_PTR
        idMap.clear();
#else
        idMap_ptr.reset();
#endif
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
                if (it->first == NULL) out << "NULL";
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

    // fId.write(mstate);
    // fId[mstate].write();   <- THE CURRENT API
    // mstate->write(fId.getId(mstate));

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

    /*
        template<class DataTypes>
        typename const typename DataTypesVecInfo<DataTypes,vtype>::DataVecT* read(const State<DataTypes>* s) const
        {
            return s->read(getId(s));
        }

        template<class DataTypes>
        typename DataTypesVecInfo<DataTypes,vtype>::DataVecT* write(State<DataTypes>* s) const
        {
            static_assert(vaccess >= V_WRITE, "");
            return s->write(getId(s));
        }
    */

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

#ifndef MAP_PTR
    IdMap idMap;
    IdMap& writeIdMap()
    {
        return idMap;
    }
public:
    bool hasIdMap() const { return !idMap.empty(); }
    const  IdMap& getIdMap() const
    {
        return idMap;
    }
#else

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

#endif

public:

    TMultiVecId()
    {
    }

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecType vtype2, VecAccess vaccess2>
    TMultiVecId(const TVecId<vtype2, vaccess2>& v) : defaultId(v)
    {
        static_assert(vaccess2 >= vaccess, "");
    }

    //// Copy constructor
    TMultiVecId( const TMultiVecId<V_ALL,vaccess>& mv)
        : defaultId( mv.getDefaultId() )
#ifdef MAP_PTR
        , idMap_ptr( mv.idMap_ptr )
#else
        , idMap( mv.idMap )
#endif
    {
    }

    //// Only TMultiVecId< V_ALL , vaccess> can declare copy constructors with all
    //// other kinds of TMultiVecIds, namely MultiVecCoordId, MultiVecDerivId...
    //// In other cases, the copy constructor takes a TMultiVecId of the same type
    //// ie copy construct a MultiVecCoordId from a const MultiVecCoordId& or a
    //// ConstMultiVecCoordId&.
    template< VecType vtype2, VecAccess vaccess2>
    TMultiVecId( const TMultiVecId<vtype2,vaccess2>& mv) : defaultId( mv.getDefaultId() )
    {
        static_assert( vaccess2 >= vaccess, "" );
        //static_assert( vtype == V_ALL || vtype2 == vtype, "" );

        if (mv.hasIdMap())
        {
#ifdef MAP_PTR
			// When we assign a V_WRITE version to a V_READ version of the same type, which are binary compatible,
			// share the maps like with a copy constructor, because otherwise a simple operation like passing a
			// MultiVecCoordId to a method taking a ConstMultiVecCoordId to indicate it won't modify it
			// will cause a temporary copy of the map, which this define was meant to avoid!
#	ifndef _MSC_VER
#if __GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 4
#		pragma GCC diagnostic push
#		pragma GCC diagnostic ignored "-Wstrict-aliasing" // this should not create problems here
#endif
#	endif
            idMap_ptr = *reinterpret_cast<const std::shared_ptr< IdMap > * >(&mv.idMap_ptr);
#	ifndef _MSC_VER
#if __GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 4
#		pragma GCC diagnostic pop
#endif
#	endif
#else
			IdMap& map = writeIdMap();
			std::copy(mv.getIdMap().begin(), mv.getIdMap().end(), std::inserter(map, map.begin()) );
#endif
        }
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
#ifndef MAP_PTR
        idMap.clear();
#else
        idMap_ptr.reset();
#endif
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
                if (it->first == NULL) out << "NULL";
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

    // fId.write(mstate);
    // fId[mstate].write();   <- THE CURRENT API
    // mstate->write(fId.getId(mstate));

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

    /*
        template<class DataTypes>
        typename const typename DataTypesVecInfo<DataTypes,vtype>::DataVecT* read(const State<DataTypes>* s) const
        {
            return s->read(getId(s));
        }

        template<class DataTypes>
        typename DataTypesVecInfo<DataTypes,vtype>::DataVecT* write(State<DataTypes>* s) const
        {
            static_assert(vaccess >= V_WRITE, "");
            return s->write(getId(s));
        }
    */

};


typedef TMultiVecId<V_COORD, V_READ> ConstMultiVecCoordId;
typedef TMultiVecId<V_COORD, V_WRITE>     MultiVecCoordId;
typedef TMultiVecId<V_DERIV, V_READ> ConstMultiVecDerivId;
typedef TMultiVecId<V_DERIV, V_WRITE>     MultiVecDerivId;
typedef TMultiVecId<V_MATDERIV, V_READ> ConstMultiMatrixDerivId;
typedef TMultiVecId<V_MATDERIV, V_WRITE>     MultiMatrixDerivId;
typedef TMultiVecId<V_ALL, V_READ>      ConstMultiVecId;
typedef TMultiVecId<V_ALL, V_WRITE>          MultiVecId;
/*
//typedef TMultiVecId<V_ALL, V_READ>      ConstMultiVecId;
class ConstMultiVecId : public TMultiVecId<V_ALL, V_READ>
{
    typedef TMultiVecId<V_ALL, V_READ> Inherit;
public:

    ConstMultiVecId()
    {
    }

    template<VecType vtype2>
    ConstMultiVecId(const TVecId<vtype2, V_READ>& v) : Inherit((ConstVecId)v)
    {
    }
};

//typedef TMultiVecId<V_ALL, V_WRITE>          MultiVecId;
class MultiVecId : public TMultiVecId<V_ALL, V_WRITE>
{
    typedef TMultiVecId<V_ALL, V_WRITE> Inherit;
public:

    MultiVecId()
    {
    }

    template<VecType vtype2, VecAccess vaccess2>
    MultiVecId(const TVecId<vtype2, vaccess2>& v) : Inherit((TVecId<V_ALL,vaccess2>)v)
    {
    }
};
*/
} // namespace core

} // namespace sofa

#endif
