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

#ifndef SOFA_COMPONENTS_SPATIALGRIDCONTAINER_H
#define SOFA_COMPONENTS_SPATIALGRIDCONTAINER_H

#include "Common/Vec3Types.h"
#include "Common/rmath.h"
#include "Common/config.h"

#include <list>

// I need C++0x !!!
#ifndef HASH_NAMESPACE
#  ifdef _MSC_VER
#    if _MSC_VER >= 1300
#      include <hash_map>
//#      if _MSC_VER >= 1400
#        define HASH_NAMESPACE stdext
//#      else
//#        define HASH_NAMESPACE std
//#      endif
#    else
#      include <map>
#      define HASH_NAMESPACE std
#    endif
#  else
#    include <ext/hash_map>
#    define HASH_NAMESPACE __gnu_cxx
#  endif
#endif

namespace Sofa
{

namespace Components
{

using namespace Common;

class EmptyClass
{
public:
    void clear() {}
};

template<class TCoord>
class SpatialGridContainerTypes
{
public:
    typedef TCoord                     Coord;
    typedef typename Coord::value_type Real;
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
    class NeighborListener
    {
    public:
        void addNeighbor(int /*i1*/, int /*i2*/, Real /*r2*/, Real /*h2*/)
        {
        }
    };

    enum { GRIDDIM_LOG2 = 2 };
};

template<class DataTypes>
class SpatialGridContainer
{
public:
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::CellData CellData;
    typedef typename DataTypes::GridData GridData;
    typedef typename DataTypes::NeighborListener NeighborListener;
    typedef typename DataTypes::ParticleField ParticleField;

public:
    SpatialGridContainer(Real cellWidth);

    void begin();

    void add(int i, const Coord& pos, bool allNeighbors = false);

    void end();

    void draw();

    void findNeighbors(NeighborListener* dest, Real dist);

    void computeField(ParticleField* field, Real dist);

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

    class Key : public Common::fixed_array<int, 3>
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
        return x[0]^x[1]^x[2];
    }

    class key_hash_fun
#ifdef _MSC_VER
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

    typedef HASH_NAMESPACE::hash_map<Key, Grid*, key_hash_fun> Map;

    typedef typename Map::const_iterator const_iterator;
    typedef typename Map::iterator iterator;

    const_iterator gridBegin() const { return map.begin(); }
    const_iterator gridEnd() const { return map.end(); }

    iterator gridBegin() { return map.begin(); }
    iterator gridEnd() { return map.end(); }

protected:
    Map map;
    const Real cellWidth;
    const Real invCellWidth;

    const Grid* findGrid(const Key& k) const;

    Grid* getGrid(const Key& k);

    Cell* getCell(const Coord& x);

    const Cell* getCell(const Grid* g, int x, int y, int z);

    void findNeighbors(NeighborListener* dest, const Real dist2, const Cell** cellsBegin, const Cell** cellsEnd);

};

} // namespace Components

} // namespace Sofa

#endif
