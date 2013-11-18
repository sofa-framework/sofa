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

#include "Topology/generic/traversorCell.h"
#include "Algo/Geometry/centroid.h"
#include "Algo/Modelisation/tetrahedralization.h"

namespace CGoGN
{

namespace Algo
{

namespace Geometry
{

template <typename PFP>
typename PFP::REAL tetrahedronSignedVolume(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typename PFP::VEC3 p1 = position[d] ;
	typename PFP::VEC3 p2 = position[map.phi1(d)] ;
	typename PFP::VEC3 p3 = position[map.phi_1(d)] ;
	typename PFP::VEC3 p4 = position[map.phi_1(map.phi2(d))] ;

	return Geom::tetraSignedVolume(p1, p2, p3, p4) ;
}

template <typename PFP>
typename PFP::REAL tetrahedronVolume(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typename PFP::VEC3 p1 = position[d] ;
	typename PFP::VEC3 p2 = position[map.phi1(d)] ;
	typename PFP::VEC3 p3 = position[map.phi_1(d)] ;
	typename PFP::VEC3 p4 = position[map.phi_1(map.phi2(d))] ;

	return Geom::tetraVolume(p1, p2, p3, p4) ;
}

template <typename PFP>
typename PFP::REAL convexPolyhedronVolume(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position, unsigned int thread)
{
	typedef typename PFP::VEC3 VEC3;

	if (Volume::Modelisation::Tetrahedralization::isTetrahedron<PFP>(map,d,thread))
		return tetrahedronVolume<PFP>(map,d,position) ;
	else
	{
		typename PFP::REAL vol = 0 ;
		VEC3 vCentroid = Algo::Surface::Geometry::volumeCentroid<PFP>(map, d, position, thread) ;

		DartMarkerStore mark(map,thread);		// Lock a marker

		std::vector<Dart> visitedFaces ;
		visitedFaces.reserve(100) ;

		visitedFaces.push_back(d) ;
		mark.markOrbit<FACE>(d) ;

		for(unsigned int  iface = 0; iface != visitedFaces.size(); ++iface)
		{
			Dart e = visitedFaces[iface] ;
			if(map.isCycleTriangle(e))
			{
				VEC3 p1 = position[e] ;
				VEC3 p2 = position[map.phi1(e)] ;
				VEC3 p3 = position[map.phi_1(e)] ;
				vol += Geom::tetraVolume(p1, p2, p3, vCentroid) ;
			}
			else
			{
				VEC3 fCentroid = Algo::Surface::Geometry::faceCentroid<PFP>(map, e, position) ;
				Dart f = e ;
				do
				{
					VEC3 p1 = position[f] ;
					VEC3 p2 = position[map.phi1(f)] ;
					vol += Geom::tetraVolume(p1, p2, fCentroid, vCentroid) ;
					f = map.phi1(f) ;
				} while(f != e) ;
			}
			Dart currentFace = e;
			do	// add all face neighbours to the table
			{
				Dart ee = map.phi2(e) ;
				if(!mark.isMarked(ee)) // not already marked
				{
					visitedFaces.push_back(ee) ;
					mark.markOrbit<FACE>(ee) ;
				}
				e = map.phi1(e) ;
			} while(e != currentFace) ;
		}

		return vol ;
	}
}


template <typename PFP>
typename PFP::REAL totalVolume(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, unsigned int thread)
{
	double vol = 0.0 ;

	TraversorW<typename PFP::MAP> t(map, thread) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vol += convexPolyhedronVolume<PFP>(map, d, position,thread) ;
	return typename PFP::REAL(vol) ;
}


namespace Parallel
{

template <typename PFP>
class FunctorTotalVolume: public FunctorMapThreaded<typename PFP::MAP >
{
	 const VertexAttribute<typename PFP::VEC3>& m_position;
	 double m_vol;
public:
	 FunctorTotalVolume<PFP>( typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_vol(0.0)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		m_vol += convexPolyhedronVolume<PFP>(this->m_map, d, m_position,threadID) ;
	}

	double getVol() const
	{
		return m_vol;
	}
};



template <typename PFP>
typename PFP::REAL totalVolume(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, unsigned int nbth)
{
	if (nbth==0)
		nbth = Algo::Parallel::optimalNbThreads();


	std::vector<FunctorMapThreaded<typename PFP::MAP>*> functs;
	for (unsigned int i=0; i < nbth; ++i)
	{
		functs.push_back(new FunctorTotalVolume<PFP>(map,position));
	}

	double total=0.0;

	Algo::Parallel::foreach_cell<typename PFP::MAP,VOLUME>(map, functs, true);

	for (unsigned int i=0; i < nbth; ++i)
	{
		total += reinterpret_cast<FunctorTotalVolume<PFP>*>(functs[i])->getVol();
		delete functs[i];
	}
	return typename PFP::REAL(total);
}

}

} // namespace Geometry

} // namespace Algo

} // namespace CGoGN
