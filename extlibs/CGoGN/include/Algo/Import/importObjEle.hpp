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
#include <vector>

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Import
{

template<typename PFP>
bool importOFFWithELERegions(typename PFP::MAP& map, const std::string& filenameOFF, const std::string& filenameELE, std::vector<std::string>& attrNames)
{
	typedef typename PFP::VEC3 VEC3;

	VertexAttribute<VEC3> position = map.template addAttribute<VEC3, VERTEX>("position") ;
	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;

	unsigned int m_nbVertices = 0, m_nbFaces = 0, m_nbEdges = 0, m_nbVolumes = 0;

	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	// open files
	std::ifstream foff(filenameOFF.c_str(), std::ios::in);
	if (!foff.good())
	{
		CGoGNerr << "Unable to open OFF file " << CGoGNendl;
		return false;
	}

	std::ifstream fele(filenameELE.c_str(), std::ios::in);
	if (!fele.good())
	{
		CGoGNerr << "Unable to open ELE file " << CGoGNendl;
		return false;
	}

	std::string line;

	//OFF reading
	std::getline(foff, line);
	if(line.rfind("OFF") == std::string::npos)
	{
		CGoGNerr << "Problem reading off file: not an off file"<<CGoGNendl;
		CGoGNerr << line << CGoGNendl;
		return false;
	}

	//Reading number of vertex/faces/edges in OFF file
	unsigned int nbe;
	{
		do
		{
			std::getline(foff,line);
		}while(line.size() == 0);

		std::stringstream oss(line);
		oss >> m_nbVertices;
		oss >> m_nbFaces;
		oss >> m_nbEdges;
		oss >> nbe;
	}

	//Reading number of tetrahedra in ELE file
	unsigned int nbv;
	{
		do
		{
			std::getline(fele,line);
		}while(line.size() == 0);

		std::stringstream oss(line);
		oss >> m_nbVolumes;
		oss >> nbv ; oss >> nbv;
	}

	CGoGNout << "nb points = " << m_nbVertices << " / nb faces = " << m_nbFaces << " / nb edges = " << m_nbEdges << " / nb tet = " << m_nbVolumes << CGoGNendl;

	//Reading vertices
	std::vector<unsigned int> verticesID;
	verticesID.reserve(m_nbVertices);

	for(unsigned int i = 0 ; i < m_nbVertices ; ++i)
	{
		do
		{
			std::getline(foff,line);
		}while(line.size() == 0);

		std::stringstream oss(line);

		float x,y,z;
		oss >> x;
		oss >> y;
		oss >> z;
		//we can read colors informations if exists
		VEC3 pos(x,y,z);

		unsigned int id = container.insertLine();
		position[id] = pos;
		verticesID.push_back(id);
	}

	std::vector<std::vector<Dart> > vecDartPtrEmb;
	vecDartPtrEmb.reserve(m_nbVertices);

	DartMarkerNoUnmark m(map) ;

	//Read and embed tetrahedra TODO
	for(unsigned i = 0; i < m_nbVolumes ; ++i)
	{
		do
		{
			std::getline(fele,line);
		} while(line.size() == 0);

		std::stringstream oss(line);
		oss >> nbe;

		Dart d = Surface::Modelisation::createTetrahedron<PFP>(map,false);

		Geom::Vec4ui pt;
		oss >> pt[0];
		oss >> pt[1];
		oss >> pt[2];
		oss >> pt[3];

//		oss >> pt[1];
//		oss >> pt[2];
//		oss >> pt[3];
//		oss >> pt[0];


		//regions ?
		//oss >> nbe;

		// Embed three vertices
		for(unsigned int j = 0 ; j < 3 ; ++j)
		{
			FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[pt[2-j]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

			//store darts per vertices to optimize reconstruction
			Dart dd = d;
			do
			{
				m.mark(dd) ;
				vecDartsPerVertex[pt[2-j]].push_back(dd);
				dd = map.phi1(map.phi2(dd));
			} while(dd != d);

			d = map.phi1(d);

		}

		//Embed the last vertex
		d = map.phi_1(map.phi2(d));

		FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[pt[3]]);
		map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

		//store darts per vertices to optimize reconstruction
		Dart dd = d;
		do
		{
			m.mark(dd) ;
			vecDartsPerVertex[pt[3]].push_back(dd);
			dd = map.phi1(map.phi2(dd));
		} while(dd != d);

	}

	foff.close();
	fele.close();

//	//Association des phi3
//	unsigned int nbBoundaryFaces = 0 ;
//	for (Dart d = map.begin(); d != map.end(); map.next(d))
//	{
//		if (m.isMarked(d))
//		{
//			std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];
//
//			Dart good_dart = NIL;
//			for(typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
//			{
//				if(map.template getEmbedding<VERTEX>(map.phi1(*it)) == map.template getEmbedding<VERTEX>(d) &&
//				   map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.phi_1(d)) /*&&
//				   map.template getEmbedding<VERTEX>(*it) == map.template getEmbedding<VERTEX>(map.phi1(d)) */)
//				{
//					good_dart = *it ;
//				}
//			}
//
//			if (good_dart != NIL)
//			{
//				map.sewVolumes(d, good_dart, false);
//				m.template unmarkOrbit<FACE>(d);
//			}
//			else
//			{
//				m.template unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(d);
//				++nbBoundaryFaces;
//			}
//		}
//	}
//
//	if (nbBoundaryFaces > 0)
//	{
//		std::cout << "closing" << std::endl ;
//		map.closeMap();
//		CGoGNout << "Map closed (" << nbBoundaryFaces << " boundary faces)" << CGoGNendl;
//	}

	return true;
}

} // namespace Import
}
} // namespace Algo
} // namespace CGoGN
