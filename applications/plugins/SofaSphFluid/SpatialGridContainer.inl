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

#ifndef SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_INL
#define SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_INL

#include <SofaSphFluid/SpatialGridContainer.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/PointSetTopologyModifier.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/helper/system/gl.h>
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa
{

namespace component
{

namespace container
{



template<class DataTypes>
typename SpatialGrid<DataTypes>::Grid SpatialGrid<DataTypes>::emptyGrid;

template<class DataTypes>
SpatialGrid<DataTypes>::SpatialGrid(Real cellWidth)
    : cellWidth(cellWidth), invCellWidth(1/cellWidth)
{
}

template<class DataTypes>
const typename SpatialGrid<DataTypes>::Grid* SpatialGrid<DataTypes>::findGrid(const Key& k) const
{
    typename Map::const_iterator it = map.find(k);
    if (it == map.end()) return &emptyGrid;
    else return it->second;
}

template<class DataTypes>
typename SpatialGrid<DataTypes>::Grid* SpatialGrid<DataTypes>::getGrid(const Key& k)
{
    Grid* & g = map[k];
    if (g == NULL)
    {
        g = new Grid;
        Grid* g2;
        typename Map::const_iterator it;
        typename Map::const_iterator end = map.end();
        it = map.find(Key(k[0]-1, k[1]  , k[2]  ));
        if (it != end)
        {
            g2 = it->second;
            g->neighbors[0] = g2; g2->neighbors[1] = g;
        }
        else
            g->neighbors[0] = &emptyGrid;
        it = map.find(Key(k[0]+1, k[1]  , k[2]  ));
        if (it != end)
        {
            g2 = it->second;
            g->neighbors[1] = g2; g2->neighbors[0] = g;
        }
        else
            g->neighbors[1] = &emptyGrid;
        it = map.find(Key(k[0]  , k[1]-1, k[2]  ));
        if (it != end)
        {
            g2 = it->second;
            g->neighbors[2] = g2; g2->neighbors[3] = g;
        }
        else
            g->neighbors[2] = &emptyGrid;
        it = map.find(Key(k[0]  , k[1]+1, k[2]  ));
        if (it != end)
        {
            g2 = it->second;
            g->neighbors[3] = g2; g2->neighbors[2] = g;
        }
        else
            g->neighbors[3] = &emptyGrid;
        it = map.find(Key(k[0]  , k[1]  , k[2]-1));
        if (it != end)
        {
            g2 = it->second;
            g->neighbors[4] = g2; g2->neighbors[5] = g;
        }
        else
            g->neighbors[4] = &emptyGrid;
        it = map.find(Key(k[0]  , k[1]  , k[2]+1));
        if (it != end)
        {
            g2 = it->second;
            g->neighbors[5] = g2; g2->neighbors[4] = g;
        }
        else
            g->neighbors[5] = &emptyGrid;
    }
    return g;
}

template<class DataTypes>
typename SpatialGrid<DataTypes>::Cell* SpatialGrid<DataTypes>::getCell(const Coord& x)
{
    int ix = sofa::helper::rfloor(x[0]*invCellWidth);
    int iy = sofa::helper::rfloor(x[1]*invCellWidth);
    int iz = sofa::helper::rfloor(x[2]*invCellWidth);
    Key k(ix >> GRIDDIM_LOG2, iy >> GRIDDIM_LOG2, iz >> GRIDDIM_LOG2);
    ix &= GRIDDIM-1;
    iy &= GRIDDIM-1;
    iz &= GRIDDIM-1;
    Grid* g = getGrid(k);
    return g->cell+(ix+GRIDDIM*iy+GRIDDIM*GRIDDIM*iz);
}

template<class DataTypes>
const typename SpatialGrid<DataTypes>::Cell* SpatialGrid<DataTypes>::getCell(const Grid* g, int x, int y, int z)
{
    if (x<0)
    {
        g = g->neighbors[0];
        x += GRIDDIM;
    }
    else if (x >= GRIDDIM)
    {
        g = g->neighbors[1];
        x -= GRIDDIM;
    }
    if (y<0)
    {
        g = g->neighbors[2];
        y += GRIDDIM;
    }
    else if (y >= GRIDDIM)
    {
        g = g->neighbors[3];
        y -= GRIDDIM;
    }
    if (z<0)
    {
        g = g->neighbors[4];
        z += GRIDDIM;
    }
    else if (z >= GRIDDIM)
    {
        g = g->neighbors[5];
        z -= GRIDDIM;
    }
    return g->cell + (x*DX + y*DY + z*DZ);
}

template<class DataTypes> template<class NeighborListener>
void SpatialGrid<DataTypes>::findNeighbors(NeighborListener* dest, const Real dist2, const Cell** cellsBegin, const Cell** cellsEnd)
{
    const Cell* c0 = *cellsBegin;
    const typename std::list<Entry>::const_iterator end = c0->plist.end();
    for (typename std::list<Entry>::const_iterator it = c0->plist.begin(); it != end; it++)
    {
        const Entry& p1 = *it;
        typename std::list<Entry>::const_iterator it2 = it;
        ++it2;
        for (; it2 != end; it2++)
        {
            const Entry& p2 = *it2;
            const Real r2 = (p2.pos - p1.pos).norm2();
            if (r2 < dist2)
                dest->addNeighbor(p1.index, p2.index, r2, dist2);
        }
        for (const Cell** c = cellsBegin+1; c != cellsEnd; ++c)
        {
            const typename std::list<Entry>::const_iterator end2 = (*c)->plist.end();
            for (typename std::list<Entry>::const_iterator it2 = (*c)->plist.begin(); it2 != end2; it2++)
            {
                const Entry& p2 = *it2;
                const Real r2 = (p2.pos - p1.pos).norm2();
                if (r2 < dist2)
                {
                    dest->addNeighbor(p1.index, p2.index, r2, dist2);
                }
            }
        }
    }
}

template<class DataTypes> template<class NeighborListener>
void SpatialGrid<DataTypes>::findNeighbors(NeighborListener* dest, Real dist)
{
    const Real dist2 = dist*dist;
    for (typename Map::iterator itg = map.begin(); itg != map.end(); itg++)
    {
        Grid* g = itg->second;
        if (g->empty) continue;
        const Cell* cell[14];
        int ic = 0;
        int x,y,z;
        cell[0] = g->cell;
        for (z = 0; z<GRIDDIM; z++)
        {
            for (y = 0; y<GRIDDIM; y++)
            {
                cell[ 0] = g->cell+ic;
                cell[ 2] = getCell(g,  0, y+1, z  );
                cell[ 4] = getCell(g,  0, y  , z+1); //(z < GRIDDIM-1) ? g->cell+(ic + DZ) : g->neighbors[5]->cell+(ic - (GRIDDIM-1)*DZ);
                cell[ 6] = getCell(g,  0, y+1, z+1); //(y < GRIDDIM-1) ? cell[4] + DY : (z < GRIDDIM-1) ? cell[2] + DZ : g->neighbors[3]->neighbors[5]->cell+(ic - (GRIDDIM-1)*(DY+DZ));
                cell[ 8] = getCell(g, -1, y+1, z  ); //(y < GRIDDIM-1) ? g->neighbors[0]->cell+(ic - (GRIDDIM-1)*DX + DY) : g->neighbors[0]->neighbors[3]->cell+(ic - (GRIDDIM-1)*(DX+DY));
                cell[ 9] = getCell(g, -1, y  , z+1); //(z < GRIDDIM-1) ? g->neighbors[0]->cell+(ic - (GRIDDIM-1)*DX + DZ) : g->neighbors[0]->neighbors[5]->cell+(ic - (GRIDDIM-1)*(DX+DZ));
                cell[10] = getCell(g, -1, y+1, z+1); //(y < GRIDDIM-1) ? cell[9] + DY : (z < GRIDDIM-1) ? cell[8] + DZ : g->neighbors[0]->neighbors[3]->neighbors[5]->cell+(ic - (GRIDDIM-1)*(DX+DY+DZ));
                cell[11] = getCell(g, -1, y-1, z+1); //(y > 0) ? cell[9] - DY : (z < GRIDDIM-1) ? cell[8] + DZ : g->neighbors[0]->neighbors[3]->neighbors[5]->cell+(ic - (GRIDDIM-1)*(DX+DY+DZ));
                cell[12] = getCell(g,  0, y-1, z+1);
                for (x = 0; x<GRIDDIM-1; x++)
                {
                    cell[ 1] = cell[ 0] + DX;
                    cell[ 3] = cell[ 2] + DX;
                    cell[ 5] = cell[ 4] + DX;
                    cell[ 7] = cell[ 6] + DX;
                    cell[13] = cell[12] + DX;
                    if (!cell[0]->plist.empty())
                        this->findNeighbors(dest, dist2, cell, cell+14);
                    cell[ 8] = cell[ 2];
                    cell[ 9] = cell[ 4];
                    cell[10] = cell[ 6];
                    cell[11] = cell[12];
                    cell[12] = cell[13];
                    cell[ 0] = cell[ 1];
                    cell[ 2] = cell[ 3];
                    cell[ 4] = cell[ 5];
                    cell[ 6] = cell[ 7];
                    ++ic;
                }
                if (!cell[0]->plist.empty())
                {
                    cell[ 1] = g->neighbors[1]->cell+(ic - (GRIDDIM-1)*DX);
                    cell[ 3] = getCell(g, GRIDDIM, y+1, z  ); //(y < GRIDDIM-1) ? cell[1] + DY : g->neighbors[3]->cell+(ic - (GRIDDIM-1)*(DX+DY));
                    cell[ 5] = getCell(g, GRIDDIM, y  , z+1); //(z < GRIDDIM-1) ? cell[1] + DZ : g->neighbors[5]->cell+(ic - (GRIDDIM-1)*(DX+DZ));
                    cell[ 7] = getCell(g, GRIDDIM, y+1, z+1); //(y < GRIDDIM-1) ? cell[5] + DY : (z < GRIDDIM-1) ? cell[3] + DZ : g->neighbors[3]->neighbors[5]->cell+(ic - (GRIDDIM-1)*(DX+DY+DZ));
                    cell[13] = getCell(g, GRIDDIM, y-1, z+1); //cell[12] + DX;
                    this->findNeighbors(dest, dist2, cell, cell+14);
                }
                ++ic;
            }
        }
    }
}

template<class DataTypes>
void SpatialGrid<DataTypes>::computeField(ParticleField* field, Real dist)
{
    //dist /= cellWidth;
    const Real dist2 = dist*dist;
    const int r = sofa::helper::rceil(dist/cellWidth)+1;
    int x,y,z;
    int x2,y2,z2;
    if (r > GRIDDIM)
    {
        dmsg_info("SpatalGrid") << "Distance too large in computeField ("<<r<<" > "<<GRIDDIM<<")" ;
        return;
    }
    for (typename Map::iterator itg = map.begin(); itg != map.end(); itg++)
    {
        Grid* g = itg->second;
        Coord pos;
        pos[0] = (Real)(itg->first[0]*GRIDDIM);
        pos[1] = (Real)(itg->first[1]*GRIDDIM);
        pos[2] = (Real)(itg->first[2]*GRIDDIM);
        if (!g->empty)
        {
            Cell* c;
            Cell* c2;
            for (z = 0, c = g->cell; z<GRIDDIM; z++)
            {
                int z0 = (z<r)?0:z-r+1;
                int z1 = (z>GRIDDIM-1-r)?GRIDDIM-1:z+r;
                for (y = 0; y<GRIDDIM; y++)
                {
                    int y0 = (y<r)?0:y-r+1;
                    int y1 = (y>GRIDDIM-1-r)?GRIDDIM-1:y+r;
                    for (x = 0; x<GRIDDIM; x++, c++)
                    {
                        if (!c->plist.empty())
                        {
                            int x0 = (x<r)?0:x-r+1;
                            int x1 = (x>GRIDDIM-1-r)?GRIDDIM-1:x+r;
                            typename std::list<Entry>::const_iterator begin = c->plist.begin();
                            typename std::list<Entry>::const_iterator end = c->plist.end();
                            typename std::list<Entry>::const_iterator it;
                            c2 = g->cell+(x0*DX+y0*DY+z0*DZ);
                            const int dy = DY-(x1-x0+1)*DX;
                            const int dz = DZ-(y1-y0+1)*DY;
                            Coord p;
                            for (z2 = z0; z2<=z1; z2++, c2+=dz)
                            {
                                p[2] = (pos[2]+z2)*cellWidth;
                                for (y2 = y0; y2<=y1; y2++, c2+=dy)
                                {
                                    p[1] = (pos[1]+y2)*cellWidth;
                                    for (x2 = x0; x2<=x1; x2++, c2+=DX)
                                    {
                                        p[0] = (pos[0]+x2)*cellWidth;
                                        for (it=begin; it != end; ++it)
                                        {
                                            Real r2 = (it->pos-p).norm2();
                                            if (r2 < dist2)
                                                c2->data.add(field, it->index, r2, dist2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int gz=-1; gz<=1; gz++)
        {
            for (int gy=-1; gy<=1; gy++)
            {
                for (int gx=-1; gx<=1; gx++)
                {
                    const Grid* g2 = g;
                    if (gx==-1) g2 = g2->neighbors[0]; else if (gx==1) g2 = g2->neighbors[1];
                    if (gy==-1) g2 = g2->neighbors[2]; else if (gy==1) g2 = g2->neighbors[3];
                    if (gz==-1) g2 = g2->neighbors[4]; else if (gz==1) g2 = g2->neighbors[5];
                    if (g2 == g) continue;
                    if (!g2->empty)
                    {
                        int g2_x0 = 0, g2_x1 = GRIDDIM-1; //, g_x0 = 0, g_x1 = GRIDDIM-1, g_dx0 = 0, g_dx1 = 0;
                        int g2_y0 = 0, g2_y1 = GRIDDIM-1; //, g_y0 = 0, g_y1 = GRIDDIM-1, g_dy0 = 0, g_dy1 = 0;
                        int g2_z0 = 0, g2_z1 = GRIDDIM-1; //, g_z0 = 0, g_z1 = GRIDDIM-1, g_dz0 = 0, g_dz1 = 0;

                        if (gx<0)      { g2_x0 = GRIDDIM-r; } // g_x1 = 0;  g_dx1 = 1; }
                        else if (gx>0) { g2_x1 = r-1; } // g_x0 = GRIDDIM-r;  g_dx0 = 1; }

                        if (gy<0)      { g2_y0 = GRIDDIM-r; } // g_y1 = 0;  g_dy1 = 1; }
                        else if (gy>0) { g2_y1 = r-1; } // g_y0 = GRIDDIM-r;  g_dy0 = 1; }

                        if (gz<0)      { g2_z0 = GRIDDIM-r; } // g_z1 = 0;  g_dz1 = 1; }
                        else if (gz>0) { g2_z1 = r-1; } // g_z0 = GRIDDIM-r;  g_dz0 = 1; }

                        //int z0 = g_z0;
                        //int z1 = g_z1;
                        const Cell* cz = g2->cell+(g2_x0*DX+g2_y0*DY+g2_z0*DZ);
                        for (z = g2_z0; z<=g2_z1; z++, cz+=DZ) //, z0+=g_dx0, z1+=g_dx1)
                        {
                            //int y0 = g_y0;
                            //int y1 = g_y1;
                            const Cell* cy = cz;
                            for (y = g2_y0; y<=g2_y1; y++, cy+=DY) //, y0+=g_dy0, y1+=g_dy1)
                            {
                                //int x0 = g_x0;
                                //int x1 = g_x1;
                                const Cell* c = cy;
                                for (x = g2_x0; x<=g2_x1; x++, c++) //, x0+=g_dx0, x1+=g_dx1)
                                {
                                    if (!c->plist.empty())
                                    {
                                        typename std::list<Entry>::const_iterator begin = c->plist.begin();
                                        typename std::list<Entry>::const_iterator end = c->plist.end();
                                        typename std::list<Entry>::const_iterator it;
                                        int x0 = x + gx*GRIDDIM - r+1; if (x0<0) x0 = 0;
                                        int x1 = x + gx*GRIDDIM + r; if (x1>GRIDDIM-1) x1 = GRIDDIM-1;
                                        int y0 = y + gy*GRIDDIM - r+1; if (y0<0) y0 = 0;
                                        int y1 = y + gy*GRIDDIM + r; if (y1>GRIDDIM-1) y1 = GRIDDIM-1;
                                        int z0 = z + gz*GRIDDIM - r+1; if (z0<0) z0 = 0;
                                        int z1 = z + gz*GRIDDIM + r; if (z1>GRIDDIM-1) z1 = GRIDDIM-1;
                                        Cell* c2 = g->cell+(x0*DX+y0*DY+z0*DZ);
                                        const int dy = DY-(x1-x0+1)*DX;
                                        const int dz = DZ-(y1-y0+1)*DY;
                                        Coord p;
                                        for (z2 = z0; z2<=z1; z2++, c2+=dz)
                                        {
                                            p[2] = (pos[2]+z2)*cellWidth;
                                            for (y2 = y0; y2<=y1; y2++, c2+=dy)
                                            {
                                                p[1] = (pos[1]+y2)*cellWidth;
                                                for (x2 = x0; x2<=x1; x2++, c2+=DX)
                                                {
                                                    p[0] = (pos[0]+x2)*cellWidth;
                                                    for (it=begin; it != end; ++it)
                                                    {
                                                        Real r2 = (it->pos-p).norm2();
                                                        if (r2 < dist2)
                                                            c2->data.add(field, it->index, r2, dist2);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<class DataTypes>
void SpatialGrid<DataTypes>::begin()
{
    for (typename Map::iterator itg = map.begin(); itg != map.end(); itg++)
    {
        Grid* g = itg->second;
        //g->clear();
        delete g;
    }
    map.clear();
}

template<class DataTypes>
void SpatialGrid<DataTypes>::add(int i, const Coord& pos, bool allNeighbors)
{
    int ix = sofa::helper::rfloor(pos[0]*invCellWidth);
    int iy = sofa::helper::rfloor(pos[1]*invCellWidth);
    int iz = sofa::helper::rfloor(pos[2]*invCellWidth);
    Key k(ix >> GRIDDIM_LOG2, iy >> GRIDDIM_LOG2, iz >> GRIDDIM_LOG2);
    ix &= GRIDDIM-1;
    iy &= GRIDDIM-1;
    iz &= GRIDDIM-1;
    Grid* g = getGrid(k);
    if (g->empty)
    {
        g->empty = false;
        // create direct neighbor grids
        if (g->neighbors[0] == &emptyGrid) getGrid(Key(k[0]-1, k[1]  , k[2]  ));
        if (g->neighbors[1] == &emptyGrid) getGrid(Key(k[0]+1, k[1]  , k[2]  ));
        if (g->neighbors[2] == &emptyGrid) getGrid(Key(k[0]  , k[1]-1, k[2]  ));
        if (g->neighbors[3] == &emptyGrid) getGrid(Key(k[0]  , k[1]+1, k[2]  ));
        if (g->neighbors[4] == &emptyGrid) getGrid(Key(k[0]  , k[1]  , k[2]-1));
        if (g->neighbors[5] == &emptyGrid) getGrid(Key(k[0]  , k[1]  , k[2]+1));
        if (allNeighbors)
        {
            // create all 26 neighbors
            if (g->neighbors[0]->neighbors[2] == &emptyGrid) getGrid(Key(k[0]-1, k[1]-1, k[2]  ));
            if (g->neighbors[1]->neighbors[2] == &emptyGrid) getGrid(Key(k[0]+1, k[1]-1, k[2]  ));
            if (g->neighbors[0]->neighbors[3] == &emptyGrid) getGrid(Key(k[0]-1, k[1]+1, k[2]  ));
            if (g->neighbors[1]->neighbors[3] == &emptyGrid) getGrid(Key(k[0]+1, k[1]+1, k[2]  ));
            if (g->neighbors[0]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]-1, k[1]  , k[2]-1));
            if (g->neighbors[1]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]+1, k[1]  , k[2]-1));
            if (g->neighbors[0]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]-1, k[1]  , k[2]+1));
            if (g->neighbors[1]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]+1, k[1]  , k[2]+1));
            if (g->neighbors[2]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]  , k[1]-1, k[2]-1));
            if (g->neighbors[3]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]  , k[1]+1, k[2]-1));
            if (g->neighbors[2]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]  , k[1]-1, k[2]+1));
            if (g->neighbors[3]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]  , k[1]+1, k[2]+1));

            if (g->neighbors[0]->neighbors[2]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]-1, k[1]-1, k[2]-1));
            if (g->neighbors[1]->neighbors[2]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]+1, k[1]-1, k[2]-1));
            if (g->neighbors[0]->neighbors[3]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]-1, k[1]+1, k[2]-1));
            if (g->neighbors[1]->neighbors[3]->neighbors[4] == &emptyGrid) getGrid(Key(k[0]+1, k[1]+1, k[2]-1));
            if (g->neighbors[0]->neighbors[2]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]-1, k[1]-1, k[2]+1));
            if (g->neighbors[1]->neighbors[2]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]+1, k[1]-1, k[2]+1));
            if (g->neighbors[0]->neighbors[3]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]-1, k[1]+1, k[2]+1));
            if (g->neighbors[1]->neighbors[3]->neighbors[5] == &emptyGrid) getGrid(Key(k[0]+1, k[1]+1, k[2]+1));
        }
    }
    g->cell[ix+GRIDDIM*iy+GRIDDIM*GRIDDIM*iz].plist.push_back(Entry(i,pos));
}

template<class DataTypes>
void SpatialGrid<DataTypes>::end()
{
    //for (typename Map::iterator itg = map.begin();itg != map.end(); itg++)
    //{
    //	const Key& k = itg->first;
    //	Grid* g = itg->second;
    //	g->neighbors[0] = g;
    //	for (int i=1;i<8;i++)
    //	{
    //		g->neighbors[i] = findGrid(Key(k[0]+(i&1),k[1]+((i>>1)&1),k[2]+((i>>2)&1)));
    //	}
    //	g->neighbors[ 8] = findGrid(Key(k[0]-1,k[1]+1,k[2]  ));
    //	g->neighbors[ 9] = findGrid(Key(k[0]-1,k[1]  ,k[2]+1));
    //	g->neighbors[10] = findGrid(Key(k[0]-1,k[1]+1,k[2]+1));
    //	g->neighbors[11] = findGrid(Key(k[0]-1,k[1]-1,k[2]+1));
    //	g->neighbors[12] = findGrid(Key(k[0]  ,k[1]-1,k[2]+1));
    //	g->neighbors[13] = findGrid(Key(k[0]+1,k[1]-1,k[2]+1));
    //}
}

/// Change particles ordering inside a given cell have contiguous indices
///
/// Fill the old2new and new2old arrays giving the permutation to apply
template<class DataTypes>
void SpatialGrid<DataTypes>::reorderIndices(helper::vector<unsigned int>* old2new, helper::vector<unsigned int>* new2old)
{
    unsigned int next = 0;
    for (typename Map::iterator itg = map.begin(); itg != map.end(); itg++)
    {
        //Key k = itg->first;
        Grid* g = itg->second;
        if (g->empty) continue;
        for (int i=0; i<NCELL; ++i)
        {
            int j=0;
            for (int s=0; s<GRIDDIM_LOG2; ++s)
                for(int c=0; c<3; ++c)
                    j += ((i>>(3*s+c))&1)<<(GRIDDIM_LOG2*c+s);
            Cell* c = g->cell+j;
            for (typename std::list<Entry>::iterator it = c->plist.begin(), itend = c->plist.end(); it != itend; ++it)
            {
                unsigned int old = it->index;
                if (old2new != NULL)
                {
                    if (old >= old2new->size()) old2new->resize(old+1);
                    (*old2new)[old] = next;
                }
                if (new2old != NULL)
                {
                    if (next >= new2old->size()) new2old->resize(next+1);
                    (*new2old)[next] = old;
                }
                it->index = next;
                ++next;
            }
        }
    }
}

template<class DataTypes>
void SpatialGrid<DataTypes>::draw(const core::visual::VisualParams* )
{
#ifndef SOFA_NO_OPENGL
    const float cscale = (float)(cellWidth);
    const float gscale = (float)(cellWidth*GRIDDIM);
    glBegin(GL_LINES);
    for (typename Map::iterator itg = map.begin(); itg != map.end(); itg++)
    {
        Key k = itg->first;
        Grid* g = itg->second;
        glColor3f(1,1,1);
        glVertex3f((k[0]  )*gscale, (k[1]  )*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]  )*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]+1)*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]+1)*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]  )*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]  )*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]+1)*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]+1)*gscale, (k[2]+1)*gscale);

        glVertex3f((k[0]  )*gscale, (k[1]  )*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]+1)*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]  )*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]+1)*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]  )*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]+1)*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]  )*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]+1)*gscale, (k[2]+1)*gscale);

        glVertex3f((k[0]  )*gscale, (k[1]  )*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]  )*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]+1)*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]  )*gscale, (k[1]+1)*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]  )*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]  )*gscale, (k[2]+1)*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]+1)*gscale, (k[2]  )*gscale);
        glVertex3f((k[0]+1)*gscale, (k[1]+1)*gscale, (k[2]+1)*gscale);

        if (g->neighbors[0] == &emptyGrid)
        {
            glColor3f(1.0f,0.0f,0.0f);
            glVertex3f((k[0]     )*gscale, (k[1]+0.3f)*gscale, (k[2]+0.3f)*gscale);
            glVertex3f((k[0]     )*gscale, (k[1]+0.7f)*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]     )*gscale, (k[1]+0.3f)*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]     )*gscale, (k[1]+0.7f)*gscale, (k[2]+0.3f)*gscale);
        }
        if (g->neighbors[1] == &emptyGrid)
        {
            glColor3f(1.0f,0.5f,0.5f);
            glVertex3f((k[0]+1   )*gscale, (k[1]+0.3f)*gscale, (k[2]+0.3f)*gscale);
            glVertex3f((k[0]+1   )*gscale, (k[1]+0.7f)*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]+1   )*gscale, (k[1]+0.3f)*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]+1   )*gscale, (k[1]+0.7f)*gscale, (k[2]+0.3f)*gscale);
        }
        if (g->neighbors[2] == &emptyGrid)
        {
            glColor3f(0.0f,1.0f,0.0f);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]     )*gscale, (k[2]+0.3f)*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]     )*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]     )*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]     )*gscale, (k[2]+0.3f)*gscale);
        }
        if (g->neighbors[3] == &emptyGrid)
        {
            glColor3f(0.5f,1.0f,0.5f);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]+1   )*gscale, (k[2]+0.3f)*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]+1   )*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]+1   )*gscale, (k[2]+0.7f)*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]+1   )*gscale, (k[2]+0.3f)*gscale);
        }
        if (g->neighbors[4] == &emptyGrid)
        {
            glColor3f(0.0f,0.0f,1.0f);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]+0.3f)*gscale, (k[2]     )*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]+0.7f)*gscale, (k[2]     )*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]+0.3f)*gscale, (k[2]     )*gscale);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]+0.7f)*gscale, (k[2]     )*gscale);
        }
        if (g->neighbors[5] == &emptyGrid)
        {
            glColor3f(0.5f,0.5f,1.0f);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]+0.3f)*gscale, (k[2]+1   )*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]+0.7f)*gscale, (k[2]+1   )*gscale);
            glVertex3f((k[0]+0.7f)*gscale, (k[1]+0.3f)*gscale, (k[2]+1   )*gscale);
            glVertex3f((k[0]+0.3f)*gscale, (k[1]+0.7f)*gscale, (k[2]+1   )*gscale);
        }
        if (!g->empty)
        {
            glColor3f(0.5f,0.5f,0.5f);
            int x,y,z;
            for (z = 0; z<=GRIDDIM; z++)
                for (y = 0; y<=GRIDDIM; y++)
                {
                    if ((y==0 || y==GRIDDIM) && (z==0 || z==GRIDDIM)) continue;
                    glVertex3f((k[0]*GRIDDIM          )*cscale, (k[1]*GRIDDIM + y    )*cscale, (k[2]*GRIDDIM + z    )*cscale);
                    glVertex3f((k[0]*GRIDDIM + GRIDDIM)*cscale, (k[1]*GRIDDIM + y    )*cscale, (k[2]*GRIDDIM + z    )*cscale);
                }
            for (z = 0; z<=GRIDDIM; z++)
                for (x = 0; x<=GRIDDIM; x++)
                {
                    if ((x==0 || x==GRIDDIM) && (z==0 || z==GRIDDIM)) continue;
                    glVertex3f((k[0]*GRIDDIM + x    )*cscale, (k[1]*GRIDDIM          )*cscale, (k[2]*GRIDDIM + z    )*cscale);
                    glVertex3f((k[0]*GRIDDIM + x    )*cscale, (k[1]*GRIDDIM + GRIDDIM)*cscale, (k[2]*GRIDDIM + z    )*cscale);
                }
            for (y = 0; y<=GRIDDIM; y++)
                for (x = 0; x<=GRIDDIM; x++)
                {
                    if ((x==0 || x==GRIDDIM) && (y==0 || y==GRIDDIM)) continue;
                    glVertex3f((k[0]*GRIDDIM + x    )*cscale, (k[1]*GRIDDIM + y    )*cscale, (k[2]*GRIDDIM          )*cscale);
                    glVertex3f((k[0]*GRIDDIM + x    )*cscale, (k[1]*GRIDDIM + y    )*cscale, (k[2]*GRIDDIM + GRIDDIM)*cscale);
                }
        }
    }
    glEnd();
#endif /* SOFA_NO_OPENGL */
}

template<class DataTypes>
SpatialGridContainer<DataTypes>::SpatialGridContainer()
    : grid(NULL)
    , d_cellWidth(initData(&d_cellWidth, (Real)1.0, "cellWidth", "Width each cell in the grid. If it is used to compute neighboors, it should be greater that the max radius considered."))
    , d_showGrid(initData(&d_showGrid, false, "showGrid", "activate rendering of the grid"))
    , d_autoUpdate(initData(&d_autoUpdate, false, "autoUpdate", "Automatically update the grid at each iteration."))
    , d_sortPoints(initData(&d_sortPoints, false, "sortPoints", "Sort points depending on which cell they are in the grid. This is required for efficient collision detection."))
    , mstate(NULL)
{
    this->f_listening.setValue(true);
}

template<class DataTypes>
SpatialGridContainer<DataTypes>::~SpatialGridContainer()
{
    if (grid != NULL)
        delete grid;
}

template<class DataTypes>
void SpatialGridContainer<DataTypes>::init()
{
    mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    grid = new Grid(d_cellWidth.getValue());
}

template<class DataTypes>
void SpatialGridContainer<DataTypes>::reinit()
{
    if (grid == NULL || grid->getCellWidth() != d_cellWidth.getValue())
    {
        if (grid != NULL)
            delete grid;
        grid = new Grid(d_cellWidth.getValue());
    }
}

template<class DataTypes>
bool SpatialGridContainer<DataTypes>::sortPoints()
{
    if (mstate)
        updateGrid(mstate->read(core::ConstVecCoordId::position())->getValue());

    msg_info() << "sortPoints(): sorting...";

    helper::vector<unsigned int> old2new, new2old;
    grid->reorderIndices(&old2new, &new2old);
    // check if the mapping actually changed something
    bool identity = true;
    for (unsigned int i=0; i<old2new.size(); ++i)
        if (old2new[i] != i)
        {
            identity = false;
            break;
        }
    if (identity)
    {
        msg_info() << "sortPoints(): no changes." ;
        return false;
    }

    if(notMuted())
    {
        std::stringstream tmp;
        tmp << "map:";
        for (unsigned int i=0; i<new2old.size(); ++i)
            tmp << " "<<new2old[i]<<"->"<<i;
        tmp << msgendl;
        tmp << "invmap:";
        for (unsigned int i=0; i<old2new.size(); ++i)
            tmp << " "<<i<<"->"<<old2new[i];
        tmp << msgendl;

        msg_info() << tmp.str() ;
    }

    sofa::component::topology::PointSetTopologyModifier* pointMod;
    this->getContext()->get(pointMod);

    if (pointMod)
    {
        msg_info() << "sortPoints(): renumber using PointSetTopologyModifier." ;

        pointMod->renumberPoints(new2old,old2new);
    }
    else
    {
        MechanicalObject<DataTypes>* object = dynamic_cast<MechanicalObject<DataTypes>*>(this->mstate);
        if (object != NULL)
        {
            msg_info() << "sortPoints(): renumber using MechanicalObject." ;
            object->renumberValues(new2old);
        }
        else
        {
            msg_info() << "sortPoints(): no external object supporting renumbering!";
        }
    }
    return true;
}
template<class DataTypes>
void SpatialGridContainer<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */simulation::AnimateBeginEvent::checkEventType(event))
        //if (simulation::AnimateEndEvent* ev =simulation::AnimateEndEvent::checkEventType(event))
    {
        if (d_sortPoints.getValue())
        {
            sortPoints();
        }
        else if (d_autoUpdate.getValue())
        {
            if (mstate)
                updateGrid(mstate->read(core::ConstVecCoordId::position())->getValue());
        }
    }
}

template<class DataTypes>
void SpatialGridContainer<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!d_showGrid.getValue())
        return;
    if (grid != NULL)
        grid->draw(vparams);
}

} // namespace container

} // namespace component

} // namespace sofa

#endif
