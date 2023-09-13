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
#ifndef SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_H
#define SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_H
#include <SofaSphFluid/config.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/rmath.h>
#include <list>

#include <unordered_map>

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
    using Index = sofa::Index;

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
    void reorderIndices(type::vector<Index>* old2new, type::vector<Index>* new2old);

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

    class Key : public type::fixed_array<int, 3>
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
        const unsigned int p0 = 73856093; // large prime numbers
        const unsigned int p1 = 19349663;
        const unsigned int p2 = 83492791;
        return (p0*x[0])^(p1*x[1])^(p2*x[2]);
    }

    class key_hash_fun
    {
    public:
        inline std::size_t operator()(const Key &s) const
        {
            return hash(s);
        }
    };


    typedef std::unordered_map<Key, Grid*, key_hash_fun> Map;


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

    using Index = sofa::Index;

    Grid* grid;
    Data<Real> d_cellWidth; ///< Width each cell in the grid. If it is used to compute neighboors, it should be greater that the max radius considered.
    Data<bool> d_showGrid; ///< activate rendering of the grid
    Data<bool> d_autoUpdate; ///< Automatically update the grid at each iteration.
    Data<bool> d_sortPoints; ///< Sort points depending on which cell they are in the grid. This is required for efficient collision detection.

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr)
            return false;
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }
protected:
    SpatialGridContainer();
    ~SpatialGridContainer() override;
public:
    void init() override;
    void reinit() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

    Grid* getGrid() { return grid; }
    void updateGrid(const VecCoord& x)
    {
        grid->update(x);
    }

    core::behavior::MechanicalState<DataTypes>* getMState() { return mstate; }

    template<class NeighborListener>
    void findNeighbors(NeighborListener* listener, Real r)
    {
        grid->findNeighbors(listener, r);
    }
    bool sortPoints();

protected:
    core::behavior::MechanicalState<DataTypes>* mstate;
};

#if  !defined(SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_CPP)
extern template class SOFA_SPH_FLUID_API SpatialGridContainer< sofa::defaulttype::Vec3Types >;
extern template class SOFA_SPH_FLUID_API SpatialGrid< SpatialGridTypes< sofa::defaulttype::Vec3Types > >;

#endif

} // namespace container

} // namespace component

} // namespace sofa

#endif
