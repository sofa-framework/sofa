/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
//
// C++ Interface: SpatialGridContainer
//
// Description:
//
//
// Author: The SOFA team <http://www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_H
#define SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_H
#include "config.h"

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/rmath.h>
#include <list>


// I need C++0x !!! 
// a: we all do ;-)

// TODO: the following should probably be moved outside this file (in
// the build system) and only deal with TR1 includes/namespaces

// also, no need to define a preprocessor macro for namespace aliasing:
//    namespace foo = std::tr1

#ifndef HASH_NAMESPACE
#  ifdef _MSC_VER
#    if _MSC_VER >= 1900
#      include <unordered_map>
#	   define HASH_NAMESPACE std
#	 else
#     if _MSC_VER >= 1300
#       include <hash_map>
//#       if _MSC_VER >= 1400
#         define HASH_NAMESPACE stdext
//#       else
//#         define HASH_NAMESPACE std
//#       endif
#     else
#       include <map>
#       define HASH_NAMESPACE std
#     endif
#	 endif
#  else
#  // TODO this test should not be about the compiler, but which stdc++ library to use
#    if __GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 3 )) || __llvm__ || __clang__ // I am not sure about clang version, but sofa compiles only from clang3.4 anyway
#     ifdef _LIBCPP_VERSION
#      // using libc++
#      include <unordered_map>
#      define HASH_NAMESPACE std
#     else
#      include <tr1/unordered_map>
#      define HASH_NAMESPACE std::tr1
#     endif
#    else
#		ifndef PS3
#      include <ext/hash_map>
#      define HASH_NAMESPACE __gnu_cxx
#		else
#		include <hash_map>
#      define HASH_NAMESPACE std
#		endif
#    endif
#  endif
#endif

namespace sofa
{

namespace component
{

namespace container
{


class EmptyClass
{
public:
    void clear() {}
};

template<class TDataTypes>
class SpatialGridTypes
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    //typedef EmptyClass                 CellData;
    typedef EmptyClass                 GridData;
    typedef EmptyClass                 ParticleField;
    class CellData
    {
    public:
        void clear()
        {
        }
        void add(ParticleField* /*field*/, int /*i*/, Real /*r2*/, Real /*h2*/)
        {
        }
    };

//	class NeighborListener
//	{
//		public:
//			void addNeighbor(int /*i1*/, int /*i2*/, Real /*r2*/, Real /*h2*/)
//			{
//			}
//	};

    enum { GRIDDIM_LOG2 = 3 };
};

template<class DataTypes>
class SpatialGrid
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::CellData CellData;
    typedef typename DataTypes::GridData GridData;
    //typedef typename DataTypes::NeighborListener NeighborListener;
    typedef typename DataTypes::ParticleField ParticleField;

public:
    SpatialGrid(Real cellWidth);

    void begin();

    void add(int i, const Coord& pos, bool allNeighbors = false);

    void end();

    void update(const VecCoord& x)
    {
        begin();
        for (unsigned int i=0; i<x.size(); i++)
        {
            add(i, x[i]);
        }
        end();
    }

    void draw(const core::visual::VisualParams* vparams);

    template<class NeighborListener>
    void findNeighbors(NeighborListener* dest, Real dist);

    void computeField(ParticleField* field, Real dist);

    /// Change particles ordering inside a given cell have contiguous indices
    ///
    /// Fill the old2new and new2old arrays giving the permutation to apply
    void reorderIndices(helper::vector<unsigned int>* old2new, helper::vector<unsigned int>* new2old);

    enum { GRIDDIM_LOG2 = DataTypes::GRIDDIM_LOG2, GRIDDIM = 1<<GRIDDIM_LOG2 };
    enum { NCELL = GRIDDIM*GRIDDIM*GRIDDIM };
    enum { DX = 1, DY = GRIDDIM, DZ = GRIDDIM*GRIDDIM };

    class Entry
    {
    public:
        int index;
        Coord pos;
        Entry(int i, const Coord& p) : index(i), pos(p) {}
    };

    class Cell
    {
    public:
        std::list<Entry> plist;
        CellData data;
        void clear()
        {
            plist.clear();
            data.clear();
        }
    };

    class Grid
    {
    public:
        Cell cell[NCELL];
        const Grid* neighbors[6];
        bool empty;
        GridData data;
        Grid() : empty(true) { std::fill(neighbors,neighbors+6, this); }
        void clear()
        {
            empty = true;
            for (unsigned int i=0; i<NCELL; i++)
                cell[i].clear();
            data.clear();
        }
    };

    static Grid emptyGrid;

    class Key : public helper::fixed_array<int, 3>
    {
    public:
        Key() {}
        Key(int i1, int i2, int i3) { (*this)[0] = i1; (*this)[1] = i2; (*this)[2] = i3; };
        bool operator==(const Key& a) const
        {
            return (*this)[0] == a[0] && (*this)[1] == a[1] && (*this)[2] == a[2];
        }
    };

    static std::size_t hash(const Key& x)
    {
//		return x[0]^x[1]^x[2];
        const unsigned int p0 = 73856093; // large prime numbers
        const unsigned int p1 = 19349663;
        const unsigned int p2 = 83492791;
        return (p0*x[0])^(p1*x[1])^(p2*x[2]);
    }

    class key_hash_fun
#if defined(_MSC_VER) || defined(PS3)
        : public HASH_NAMESPACE::hash_compare<Key>
    {
    public:
        //enum
        //{ // parameters for hash table
        //	bucket_size = 4, // 0 < bucket_size
        //	min_buckets = 8
        //}; // min_buckets = 2 ^^ N, 0 < N
        inline bool operator()(const Key& s1, const Key& s2) const
        {
            for (unsigned int i=0; i<s1.size(); ++i)
                if (s1[i] < s2[i]) return true;
                else if (s1[i] > s2[i]) return false;
            return false; // s1 == s2
        }
#else
    {
    public:
#endif
        inline std::size_t operator()(const Key &s) const
        {
            return hash(s);
        }
    };


#ifndef _MSC_VER
#    if __GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 3 )) || __llvm__ || __clang__       //hash_map is deprecated since gcc-4.3
    typedef HASH_NAMESPACE::unordered_map<Key, Grid*, key_hash_fun> Map;
#    else
    typedef HASH_NAMESPACE::hash_map<Key, Grid*, key_hash_fun> Map;
#    endif
#else
#	if _MSC_VER >= 1900
		typedef HASH_NAMESPACE::unordered_map<Key, Grid*, key_hash_fun> Map;
#	else
		typedef HASH_NAMESPACE::hash_map<Key, Grid*, key_hash_fun> Map;
#   endif
#endif



    typedef typename Map::const_iterator const_iterator;
    typedef typename Map::iterator iterator;

    const_iterator gridBegin() const { return map.begin(); }
    const_iterator gridEnd() const { return map.end(); }

    iterator gridBegin() { return map.begin(); }
    iterator gridEnd() { return map.end(); }

    Real getCellWidth() const { return cellWidth; }
    Real getInvCellWidth() const { return invCellWidth; }

    Map& getMap() { return map; }
    const Map& getMap() const { return map; }

protected:
    Map map;
    const Real cellWidth;
    const Real invCellWidth;

    const Grid* findGrid(const Key& k) const;

    Grid* getGrid(const Key& k);

    Cell* getCell(const Coord& x);

    const Cell* getCell(const Grid* g, int x, int y, int z);

    template<class NeighborListener>
    void findNeighbors(NeighborListener* dest, const Real dist2, const Cell** cellsBegin, const Cell** cellsEnd);

};

template<class DataTypes>
class SpatialGridContainer : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SpatialGridContainer,DataTypes),core::objectmodel::BaseObject);
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef SpatialGridTypes<DataTypes> GridTypes;
    typedef SpatialGrid< GridTypes > Grid;
    Grid* grid;
    Data<Real> d_cellWidth;
    Data<bool> d_showGrid;
    Data<bool> d_autoUpdate;
    Data<bool> d_sortPoints;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }
protected:
    SpatialGridContainer();
    virtual ~SpatialGridContainer();
public:
    virtual void init() override;
    virtual void reinit() override;
    virtual void draw(const core::visual::VisualParams* vparams) override;
    virtual void handleEvent(sofa::core::objectmodel::Event* event) override;

    Grid* getGrid() { return grid; }
    void updateGrid(const VecCoord& x)
    {
        grid->update(x);
        //grid->begin();
        //for (unsigned int i=0;i<x.size();i++)
        //{
        //    grid->add(i, x[i]);
        //}
        //grid->end();
    }

    core::behavior::MechanicalState<DataTypes>* getMState() { return mstate; }

    template<class NeighborListener>
    void findNeighbors(NeighborListener* listener, Real r)
    {
        grid->findNeighbors(listener, r);
    }
    bool sortPoints();

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const SpatialGridContainer<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
protected:
    core::behavior::MechanicalState<DataTypes>* mstate;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_CPP)
#ifndef SOFA_FLOAT
extern template class SpatialGridContainer< defaulttype::Vec3dTypes >;
extern template class SOFA_SPH_FLUID_API SpatialGrid< SpatialGridTypes< sofa::defaulttype::Vec3dTypes > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SpatialGridContainer< defaulttype::Vec3fTypes >;
extern template class SOFA_SPH_FLUID_API SpatialGrid< SpatialGridTypes< sofa::defaulttype::Vec3fTypes > >;
#endif
#endif

} // namespace container

} // namespace component

} // namespace sofa

#endif
