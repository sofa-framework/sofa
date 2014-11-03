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

#include "Algo/Geometry/basic.h"
#include "Algo/Geometry/feature.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Remeshing
{

template <typename PFP>
void pliantRemeshing(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	// compute the mean edge length
	DartMarker<MAP> m1(map) ;
	REAL meanEdgeLength = 0 ;
	unsigned int nbEdges = 0 ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!m1.isMarked(d))
		{
			m1.template markOrbit<EDGE>(d) ;
			meanEdgeLength += Geometry::edgeLength<PFP>(map, d, position) ;
			++nbEdges ;
		}
	}
	meanEdgeLength /= REAL(nbEdges) ;

	// compute the min and max edge lengths
	REAL edgeLengthInf = REAL(3) / REAL(4) * meanEdgeLength ;
	REAL edgeLengthSup = REAL(4) / REAL(3) * meanEdgeLength ;

	// split long edges
	DartMarker<MAP> m2(map) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!m2.isMarked(d))
		{
			m2.template markOrbit<EDGE>(d) ;
			REAL length = Geometry::edgeLength<PFP>(map, d, position) ;
			if(length > edgeLengthSup)
			{
				Dart dd = map.phi2(d) ;
				VEC3 p = REAL(0.5) * (position[d] + position[dd]) ;
				map.cutEdge(d) ;
				position[map.phi1(d)] = p ;
				map.splitFace(map.phi1(d), map.phi_1(d)) ;
				if(dd != d)
					map.splitFace(map.phi1(dd), map.phi_1(dd)) ;
			}
		}
	}

	// compute feature edges
	CellMarker<MAP, EDGE> featureEdge(map) ;
	Geometry::featureEdgeDetection<PFP>(map, position, featureEdge) ;

	// compute feature vertices
	CellMarker<MAP, VERTEX> featureVertex(map) ;
	CellMarker<MAP, VERTEX> cornerVertex(map) ;
	DartMarker<MAP> m3(map) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!m3.isMarked(d))
		{
			m3.template markOrbit<VERTEX>(d) ;
			unsigned int nbFeatureEdges = 0 ;
			Dart vit = d ;
			do
			{
				if(featureEdge.isMarked(vit))
					++nbFeatureEdges ;
				vit = map.phi2_1(vit) ;
			} while(vit != d) ;
			if(nbFeatureEdges > 0)
			{
				if(nbFeatureEdges == 2)
					featureVertex.mark(d) ;
				else
					cornerVertex.mark(d) ;
			}
		}
	}

	// collapse short
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(m3.isMarked(d))
		{
			m3.template unmarkOrbit<EDGE>(d) ;
			Dart d1 = map.phi1(d) ;
			if(!cornerVertex.isMarked(d) && !cornerVertex.isMarked(d1) &&
				( (featureVertex.isMarked(d) && featureVertex.isMarked(d1)) || (!featureVertex.isMarked(d) && !featureVertex.isMarked(d1)) ))
			{
				REAL length = Geometry::edgeLength<PFP>(map, d, position) ;
				if(length < edgeLengthInf && map.edgeCanCollapse(d))
				{
					bool collapse = true ;
					VEC3 p = position[d1] ;
					Dart vit = map.phi2_1(d) ;
					do
					{
						VEC3 vec = position[d1] - position[map.phi1(vit)] ;
						if(vec.norm() > edgeLengthSup)
							collapse = false ;
						vit = map.phi2_1(vit) ;
					} while(vit != d && collapse) ;
					if(collapse)
					{
						Dart v = map.collapseEdge(d) ;
						position[v] = p ;
					}
				}
			}
		}
	}

	// equalize valences with edge flips
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!m3.isMarked(d))
		{
			m3.template markOrbit<EDGE>(d) ;
			Dart e = map.phi2(d) ;
			if(!featureEdge.isMarked(d) && e != d)
			{
				unsigned int w = map.vertexDegree(d) ;
				unsigned int x = map.vertexDegree(e) ;
				unsigned int y = map.vertexDegree(map.phi1(map.phi1(d))) ;
				unsigned int z = map.vertexDegree(map.phi1(map.phi1(e))) ;
				int flip = 0 ;
				flip += w > 6 ? 1 : (w < 6 ? -1 : 0) ;
				flip += x > 6 ? 1 : (x < 6 ? -1 : 0) ;
				flip += y < 6 ? 1 : (y > 6 ? -1 : 0) ;
				flip += z < 6 ? 1 : (z > 6 ? -1 : 0) ;
				if(flip > 1)
				{
					map.flipEdge(d) ;
					m3.template markOrbit<EDGE>(map.phi1(d)) ;
					m3.template markOrbit<EDGE>(map.phi_1(d)) ;
					m3.template markOrbit<EDGE>(map.phi1(e)) ;
					m3.template markOrbit<EDGE>(map.phi_1(e)) ;
				}
			}
		}
	}

	// update vertices normals
	Algo::Surface::Geometry::computeNormalVertices<PFP>(map, position, normal) ;

	// tangential relaxation
	VertexAttribute<VEC3, MAP> centroid = map.template addAttribute<VEC3, VERTEX>("centroid") ;
	Surface::Geometry::computeNeighborhoodCentroidVertices<PFP>(map, position, centroid) ;

	CellMarker<MAP, VERTEX> vm(map) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!vm.isMarked(d))
		{
			vm.mark(d) ;
			if(!cornerVertex.isMarked(d) && !featureVertex.isMarked(d) && !map.isBoundaryVertex(d))
			{
				VEC3 l = position[d] - centroid[d] ;
				REAL e = l * normal[d] ;
				VEC3 displ = e * normal[d] ;
				position[d] = centroid[d] + displ ;
			}
		}
	}

	map.removeAttribute(centroid) ;
}

} // namespace Remeshing

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
