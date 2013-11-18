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

namespace Modelisation
{

template <typename PFP>
Dart Primitive3D<PFP>::HexaGrid1Topo(unsigned int nx)
{
	// first cube

	Dart d0 = Surface::Modelisation::createHexahedron<PFP>(m_map,false);
	m_tableVertDarts.push_back(d0);

	Dart d1 = m_map.template phi<2112>(d0);

	for (unsigned int i = 1; i < nx; ++i)
	{
		Dart d2 = Surface::Modelisation::createHexahedron<PFP>(m_map,false);

		m_tableVertDarts.push_back(d2);
		m_map.sewVolumes(d1, d2, false);
		d1 = m_map.template phi<2112>(d2);
	}

	// add last vertex (x=nx)
	d1 = m_map.phi2(d1); //TODO can take phi1 instead > same vertex ??
	m_tableVertDarts.push_back(d1);

	return d0;
}

template <typename PFP>
Dart Primitive3D<PFP>::HexaGrid2Topo(unsigned int nx, unsigned int ny)
{
	// creation premiere ligne
	Dart d0 = HexaGrid1Topo(nx);
	Dart d1 = m_map.template phi<112>(d0);

	for (unsigned int i = 1; i < ny; ++i)
	{
		// creation ligne suivante
		Dart d2 = HexaGrid1Topo(nx);
		Dart d3 = m_map.phi2(d2);

		// couture des deux lignes de cubes
		for (unsigned int i = 0; i < nx; ++i)
		{
			m_map.sewVolumes(d1, d3, false);
			d1 = m_map.template phi<11232>(d1);
			d3 = m_map.template phi<11232>(d3);
		}
		// passage a lignes suivante
		d1 = m_map.template phi<112>(d2);
	}

	// add last row of vertices (y = ny)

	int index = m_tableVertDarts.size()-(nx+1); // pos of last inserted row of dart
	for (unsigned int i = 0; i < nx; ++i)
	{
		Dart dd = m_tableVertDarts[index++];
		dd = m_map.template phi<112>(dd);
		m_tableVertDarts.push_back(dd);	
	}
	// warning last vertex of row has not same dart
	Dart dd = m_tableVertDarts[index++];
	dd = m_map.template phi<211>(dd);
	m_tableVertDarts.push_back(dd);	

	return d0;
}

template <typename PFP>
Dart Primitive3D<PFP>::hexaGrid_topo(unsigned int nx, unsigned int ny, unsigned int nz)
{
	m_kind = HEXAGRID;
	m_nx = nx;
	m_ny = ny;
	m_nz = nz;
	m_tableVertDarts.clear();
	m_tableVertDarts.reserve((nx+1)*(ny+1)*(nz+1));

	Dart d0 = HexaGrid2Topo(nx, ny);
	Dart d1 = m_map.template phi<12>(d0);

	for (unsigned int i = 1; i < nz; ++i)
	{
		// creation grille suivante
		Dart d2 = HexaGrid2Topo(nx, ny);
		Dart d3 = m_map.phi2(m_map.phi_1(d2));
		
		// couture des deux grilles 2D de cubes
		for (unsigned int j = 0; j < ny; ++j)
		{
			Dart da = d1;
			Dart db = d3;
			for (unsigned int k = 0; k < nx; ++k)
			{
				m_map.sewVolumes(da, db, false);
				da = m_map.template phi<11232>(da);
				db = m_map.template phi<11232>(db);
			}
			d1 = m_map.phi_1(d1);
			d1 = m_map.template phi<232>(d1);
			d1 = m_map.phi_1(d1);
			d3 = m_map.template phi<12321>(d3);
		}
		// passage a lignes suivante
		d1 = m_map.template phi<12>(d2);
	}

	// add last slice of vertices to the table
	unsigned int nb = (nx+1)*(ny+1);	// nb of vertices in one slice XY
	unsigned int index = nb*(nz-1);	// last slice
	for (unsigned int i = 0; i < nb; ++i)
	{
		Dart dd = m_tableVertDarts[index++];
		dd = m_map.phi2(dd);
		m_tableVertDarts.push_back(dd);
	}

	std::cout << m_map.closeMap() << std::endl;

	return d0;
}

template <typename PFP>
void Primitive3D<PFP>::embedHexaGrid(float x, float y, float z)
{
	if (m_kind != HEXAGRID)
	{
		CGoGNerr << "Warning try to embedHexaGrid something that is not a grid of hexahedron"<<CGoGNendl;
		return;
	}

	float dx = x/float(m_nx);
	float dy = y/float(m_ny);
	float dz = z/float(m_nz);

	unsigned int nbs = (m_nx+1)*(m_ny+1);

	for(unsigned int i = 0; i <= m_nz; ++i)
	{
		for(unsigned int j = 0; j <= m_ny; ++j)
		{
			for(unsigned int k = 0; k <= m_nx; ++k)
			{
				typename PFP::VEC3 pos(-x/2.0f + dx*float(k), -y/2.0f + dy*float(j), -z/2.0f + dz*float(i));
				Dart d = m_tableVertDarts[ i*nbs+j*(m_nx+1)+k ];

				m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d);
				m_positions[d] = pos;
			}
		}
	}
}

template <typename PFP>
void Primitive3D<PFP>::embedHexaGrid(typename PFP::VEC3 origin, float x, float y, float z)
{
	if (m_kind != HEXAGRID)
	{
		CGoGNerr << "Warning try to embedHexaGrid something that is not a grid of hexahedron"<<CGoGNendl;
		return;
	}

	float dx = x/float(m_nx);
	float dy = y/float(m_ny);
	float dz = z/float(m_nz);

	unsigned int nbs = (m_nx+1)*(m_ny+1);

	for(unsigned int i = 0; i <= m_nz; ++i)
	{
		for(unsigned int j = 0; j <= m_ny; ++j)
		{
			for(unsigned int k = 0; k <= m_nx; ++k)
			{
				typename PFP::VEC3 pos(-x/2.0f + dx*float(k), -y/2.0f + dy*float(j), -z/2.0f + dz*float(i));
				Dart d = m_tableVertDarts[ i*nbs+j*(m_nx+1)+k ];

				m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d);
				m_positions[d] = origin + pos;
			}
		}
	}
}

template <typename PFP>
void Primitive3D<PFP>::transform(const Geom::Matrix44f& matrice)
{
	for(typename std::vector<Dart>::iterator di = m_tableVertDarts.begin(); di != m_tableVertDarts.end(); ++di)
	{
		typename PFP::VEC3& pos = m_positions[*di];
		pos = Geom::transform(pos, matrice);
	}

}

//template <typename PFP>
//void Primitive3D<PFP>::mark(Mark m)
//{
//	for(typename std::vector<Dart>::iterator di = m_tableVertDarts.begin(); di != m_tableVertDarts.end(); ++di)
//	{
//		m_map.markOrbit(0, *di, m);
//	}
//}

} // namespace Modelisation

}

} // namespace Algo

} // namespace CGoGN
