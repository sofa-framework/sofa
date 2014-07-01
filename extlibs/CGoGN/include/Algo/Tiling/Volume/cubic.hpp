/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#include "Algo/Modelisation/polyhedron.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Tilings
{

namespace Cubic
{

/*! Grid
 *************************************************************************/

template <typename PFP>
void Grid<PFP>::grid3D(unsigned int x, unsigned int y, unsigned int z)
{
    this->m_tableVertDarts.clear();
    this->m_tableVertDarts.reserve((x+1)*(y+1)*(z+1));

    Dart d0 = grid2D(x, y);
    Dart d1 = this->m_map.template phi<12>(d0);

    for (unsigned int i = 1; i < z; ++i)
    {
        // creation grille suivante
        Dart d2 = grid2D(x, y);
        Dart d3 = this->m_map.phi2(this->m_map.phi_1(d2));

        // couture des deux grilles 2D de cubes
        for (unsigned int j = 0; j < y; ++j)
        {
            Dart da = d1;
            Dart db = d3;
            for (unsigned int k = 0; k < x; ++k)
            {
                this->m_map.sewVolumes(da, db, false);
                da = this->m_map.template phi<11232>(da);
                db = this->m_map.template phi<11232>(db);
            }
            d1 = this->m_map.phi_1(d1);
            d1 = this->m_map.template phi<232>(d1);
            d1 = this->m_map.phi_1(d1);
            d3 = this->m_map.template phi<12321>(d3);
        }
        // passage a lignes suivante
        d1 = this->m_map.template phi<12>(d2);
    }

    // add last slice of vertices to the table
    unsigned int nb = (x+1)*(y+1);	// nb of vertices in one slice XY
    unsigned int index = nb*(z-1);	// last slice
    for (unsigned int i = 0; i < nb; ++i)
    {
        Dart dd = this->m_tableVertDarts[index++];
        dd = this->m_map.phi2(dd);
        this->m_tableVertDarts.push_back(dd);
    }

    this->m_map.closeMap();
}

template <typename PFP>
Dart Grid<PFP>::grid2D(unsigned int x, unsigned int y)
{
    // creation premiere ligne
    Dart d0 = grid1D(x);
    Dart d1 = this->m_map.template phi<112>(d0);

    for (unsigned int i = 1; i < y; ++i)
    {
        // creation ligne suivante
        Dart d2 = grid1D(x);
        Dart d3 = this-> m_map.phi2(d2);

        // couture des deux lignes de cubes
        for (unsigned int i = 0; i < x; ++i)
        {
            this->m_map.sewVolumes(d1, d3, false);
            d1 = this->m_map.template phi<11232>(d1);
            d3 = this->m_map.template phi<11232>(d3);
        }
        // passage a lignes suivante
        d1 = this->m_map.template phi<112>(d2);
    }

    // add last row of vertices (y = ny)

    int index = this->m_tableVertDarts.size()-(x+1); // pos of last inserted row of dart
    for (unsigned int i = 0; i < x; ++i)
    {
        Dart dd = this->m_tableVertDarts[index++];
        dd = this->m_map.template phi<112>(dd);
        this->m_tableVertDarts.push_back(dd);
    }
    // warning last vertex of row has not same dart
    Dart dd = this->m_tableVertDarts[index++];
    dd = this->m_map.template phi<211>(dd);
    this->m_tableVertDarts.push_back(dd);

    return d0;
}

template <typename PFP>
Dart Grid<PFP>::grid1D(unsigned int x)
{
    // first cube

    Dart d0 = Surface::Modelisation::createHexahedron<PFP>(this->m_map,false);
    this->m_tableVertDarts.push_back(d0);

    Dart d1 = this->m_map.template phi<2112>(d0);

    for (unsigned int i = 1; i < x; ++i)
    {
        Dart d2 = Surface::Modelisation::createHexahedron<PFP>(this->m_map,false);

        this->m_tableVertDarts.push_back(d2);
        this->m_map.sewVolumes(d1, d2, false);
        d1 = this->m_map.template phi<2112>(d2);
    }

    // add last vertex (x=nx)
    d1 = this->m_map.phi2(d1); //TODO can take phi1 instead > same vertex ??
    this->m_tableVertDarts.push_back(d1);

    return d0;
}

template <typename PFP>
void Grid<PFP>::embedIntoGrid(VertexAttribute<VEC3, MAP>& position, float x, float y, float z)
{
    float dx = x/float(this->m_nx);
    float dy = y/float(this->m_ny);
    float dz = z/float(this->m_nz);

    unsigned int nbs = (this->m_nx+1)*(this->m_ny+1);

    for(unsigned int i = 0; i <= this->m_nz; ++i)
    {
        for(unsigned int j = 0; j <= this->m_ny; ++j)
        {
            for(unsigned int k = 0; k <= this->m_nx; ++k)
            {
                typename PFP::VEC3 pos(-x/2.0f + dx*float(k), -y/2.0f + dy*float(j), -z/2.0f + dz*float(i));
                Dart d = this->m_tableVertDarts[ i*nbs+j*(this->m_nx+1)+k ];

				Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(this->m_map, d);
                position[d] = pos;
            }
        }
    }
}

template <typename PFP>
void Grid<PFP>::embedIntoTwistedStrip(VertexAttribute<VEC3, MAP>& position, float radius_min, float radius_max, float turns)
{
    float alpha = float(2.0*M_PI/this->m_ny);
    float beta = turns/float(this->m_ny);

    float radius = (radius_max + radius_min)/2.0f;
    float rdiff = (radius_max - radius_min)/2.0f;

    for(unsigned int i = 0; i <= this->m_ny; ++i)
    {
        for(unsigned int j = 0; j <= this->m_nx; ++j)
        {
            float rw = -rdiff + float(j)*2.0f*rdiff/float(this->m_nx);
            float r = radius + rw*cos(beta*float(i));
            VEC3 pos(r*cos(alpha*float(i)), r*sin(alpha*float(i)), rw*sin(beta*float(i)));
            position[this->m_tableVertDarts[ i*(this->m_nx+1)+j ] ] = pos;
        }
    }
}

template <typename PFP>
void Grid<PFP>::embedIntoHelicoid(VertexAttribute<VEC3, MAP>& position, float radius_min,  float radius_max, float maxHeight, float nbTurn, int orient)
{
    float alpha = float(2.0*M_PI/this->m_nx)*nbTurn;
    float hS = maxHeight/this->m_nx;

    // 	float radius = (radius_max + radius_min)/2.0f;
    // 	float rdiff = (radius_max - radius_min)/2.0f;

    for(unsigned int i = 0; i <= this->m_ny; ++i)
    {
        for(unsigned int j = 0; j <= this->m_nx; ++j)
        {
            // 			float r = radius_max + radius_min*cos(beta*float(j));
            float r,x,y;
            // 			if(i==1) {
            // 				r = radius_max;
            // 			}
            // 			else {
            r = radius_min+(radius_max-radius_min)*float(i)/float(this->m_ny);
            // 			}
            x = orient*r*sin(alpha*float(j));
            y = orient*r*cos(alpha*float(j));

            VEC3 pos(x, y, j*hS);
            Dart d = this->m_tableVertDarts[i*(this->m_nx+1)+j];
            position[d] = pos;
        }
    }
}



} // namespace Cubic

} // namespace Tilings

} // namespace Volume

} // namespace Algo

} // namespace CGoGN
