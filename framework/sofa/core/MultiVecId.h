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
    MyVecId id;
    const State<DataTypes>* state;
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
    MyVecId id;
    State<DataTypes>* state;
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
    MyVecId id;
    const State<DataTypes>* state;
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
    MyVecId id;
    State<DataTypes>* state;
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
    MyVecId id;
    const State<DataTypes>* state;
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
    MyVecId id;
    State<DataTypes>* state;
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
    MyVecId id;
    const State<DataTypes>* state;
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
    MyVecId id;
    State<DataTypes>* state;
};



template <VecType vtype, VecAccess vaccess>
class TMultiVecId
{
public:
    typedef TVecId<vtype, vaccess> MyVecId;

protected:
    MyVecId defaultId;

    typedef std::map<BaseState*, MyVecId> IdMap;
    typedef typename IdMap::iterator IdMap_iterator;
    typedef typename IdMap::const_iterator IdMap_const_iterator;
    IdMap idMap;

public:

    /// Copy from another VecId, possibly with another type of access, with the
    /// constraint that the access must be compatible (i.e. cannot create
    /// a write-access VecId from a read-only VecId.
    template<VecAccess vaccess2>
    TMultiVecId(const TVecId<vtype, vaccess2>& v) : defaultId(v)
    {
        BOOST_STATIC_ASSERT(vaccess2 >= vaccess);
    }

    void setDefaultId(const MyVecId& id)
    {
        defaultId = id;
    }

    template<class StateSet>
    void setId(const StateSet& states, const MyVecId& id)
    {
        for (typename StateSet::const_iterator it = states.begin(), itend = states.end(); it != itend; ++it)
            idMap[*it] = id;
    }

    void assign(const MyVecId& id)
    {
        defaultId = id;
        idMap.clear();
    }

    const MyVecId& getId(const BaseState* s) const
    {
        IdMap_const_iterator it = idMap.find(s);
        if (it != idMap.end()) return it->second;
        else                   return defaultId;
    }

    const MyVecId& getDefaultId() const
    {
        return defaultId;
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
            BOOST_STATIC_ASSERT(vaccess >= V_WRITE);
            return s->write(getId(s));
        }
    */

};

typedef TMultiVecId<V_ALL, V_READ> ConstMultiVecId;
typedef TMultiVecId<V_ALL, V_WRITE>     MultiVecId;
typedef TMultiVecId<V_COORD, V_READ> ConstMultiVecCoordId;
typedef TMultiVecId<V_COORD, V_WRITE>     MultiVecCoordId;
typedef TMultiVecId<V_DERIV, V_READ> ConstMultiVecDerivId;
typedef TMultiVecId<V_DERIV, V_WRITE>     MultiVecDerivId;
typedef TMultiVecId<V_MATDERIV, V_READ> ConstMultiMatrixDerivId;
typedef TMultiVecId<V_MATDERIV, V_WRITE>     MultiMatrixDerivId;

} // namespace core

} // namespace sofa

#endif
