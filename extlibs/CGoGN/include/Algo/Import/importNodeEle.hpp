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
#include "Geometry/orientation.h"

#include <vector>

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Import 
{

template <typename PFP>
bool importNodeWithELERegions(typename PFP::MAP& map, const std::string& filenameNode, const std::string& filenameELE, std::vector<std::string>& attrNames)
{
	typedef typename PFP::VEC3 VEC3;

	VertexAttribute<VEC3> position = map.template addAttribute<VEC3, VERTEX>("position") ;
	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;

	unsigned int m_nbVertices = 0, m_nbFaces = 0, m_nbEdges = 0, m_nbVolumes = 0;
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	//open file
	std::ifstream fnode(filenameNode.c_str(), std::ios::in);
	if (!fnode.good())
	{
		CGoGNerr << "Unable to open file " << filenameNode << CGoGNendl;
		return false;
	}

	std::ifstream fele(filenameELE.c_str(), std::ios::in);
	if (!fele.good())
	{
		CGoGNerr << "Unable to open file " << filenameELE << CGoGNendl;
		return false;
	}

	std::string line;

//	do
//	{
//		std::getline(fnode,line);
//	}while(line.rfind("#") == 0);

	//Reading NODE file
	//First line: [# of points] [dimension (must be 3)] [# of attributes] [# of boundary markers (0 or 1)]
	unsigned int nbe;
	{
		do
		{
			std::getline(fnode,line);
		}while(line.size() == 0);

		std::stringstream oss(line);
		oss >> m_nbVertices;
		oss >> nbe;
		oss >> nbe;
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
	//Remaining lines: [point #] [x] [y] [z] [optional attributes] [optional boundary marker]
//	std::vector<unsigned int> verticesID;
//	verticesID.reserve(m_nbVertices);

	std::map<unsigned int,unsigned int> verticesMapID;

	for(unsigned int i = 0 ; i < m_nbVertices ; ++i)
	{
		do
		{
			std::getline(fnode,line);
		}while(line.size() == 0);

		std::stringstream oss(line);

		int idv;
		oss >> idv;

		float x,y,z;
		oss >> x;
		oss >> y;
		oss >> z;
		//we can read colors informations if exists
		VEC3 pos(x,y,z);

		unsigned int id = container.insertLine();
		position[id] = pos;

//		verticesID.push_back(id);
		verticesMapID.insert(std::pair<unsigned int, unsigned int>(idv,id));
	}


	DartMarkerNoUnmark m(map) ;
	bool invertVol=false;
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

		// test orientation of first tetra
		if (i==0)
		{
			oss >> pt[0];
			oss >> pt[1];
			oss >> pt[2];
			oss >> pt[3];

			typename PFP::VEC3 P = position[verticesMapID[pt[0]]];
			typename PFP::VEC3 A = position[verticesMapID[pt[1]]];
			typename PFP::VEC3 B = position[verticesMapID[pt[2]]];
			typename PFP::VEC3 C = position[verticesMapID[pt[3]]];

			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
			{
				invertVol=true;
				unsigned int ui=pt[0];
				pt[0] = pt[3];
				pt[3] = pt[2];
				pt[2] = pt[1];
				pt[1] = ui;
			}
		}
		else
		{
			if (invertVol)
			{
				oss >> pt[1];
				oss >> pt[2];
				oss >> pt[3];
				oss >> pt[0];
			}
			else
			{
				oss >> pt[0];
				oss >> pt[1];
				oss >> pt[2];
				oss >> pt[3];
			}
		}

		// Embed three vertices
		for(unsigned int j = 0 ; j < 3 ; ++j)
		{
			FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesMapID[pt[2-j]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

			//store darts per vertices to optimize reconstruction
			Dart dd = d;
			do
			{
				m.mark(dd) ;
				vecDartsPerVertex[verticesMapID[pt[2-j]]].push_back(dd);
				dd = map.phi1(map.phi2(dd));
			} while(dd != d);

			d = map.phi1(d);

		}

		//Embed the last vertex
		d = map.phi_1(map.phi2(d));

		FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesMapID[pt[3]]);
		map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

		//store darts per vertices to optimize reconstruction
		Dart dd = d;
		do
		{
			m.mark(dd) ;
			vecDartsPerVertex[verticesMapID[pt[3]]].push_back(dd);
			dd = map.phi1(map.phi2(dd));
		} while(dd != d);

	}

	fnode.close();
	fele.close();

	//Association des phi3
	unsigned int nbBoundaryFaces = 0 ;
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (m.isMarked(d))
		{
			std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];

			Dart good_dart = NIL;
			for(typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
			{
				if(map.template getEmbedding<VERTEX>(map.phi1(*it)) == map.template getEmbedding<VERTEX>(d) &&
				   map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.phi_1(d)) /*&&
				   map.template getEmbedding<VERTEX>(*it) == map.template getEmbedding<VERTEX>(map.phi1(d)) */)
				{
					good_dart = *it ;
				}
			}

			if (good_dart != NIL)
			{
				map.sewVolumes(d, good_dart, false);
				m.template unmarkOrbit<FACE>(d);
			}
			else
			{
				m.unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(d);
				++nbBoundaryFaces;
			}
		}
	}

	if (nbBoundaryFaces > 0)
	{
		std::cout << "closing" << std::endl ;
		map.closeMap();
		CGoGNout << "Map closed (" << nbBoundaryFaces << " boundary faces)" << CGoGNendl;
	}

	return true;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN
