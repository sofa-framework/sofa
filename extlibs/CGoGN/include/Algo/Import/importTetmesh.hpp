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
bool importTetmesh(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scaleFactor)
{
	typedef typename PFP::VEC3 VEC3;

	VertexAttribute<VEC3> position = map.template addAttribute<VEC3, VERTEX>("position") ;
	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;

	unsigned int nbVertices = 0, nbVolumes = 0;
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	//open file
	std::ifstream fp(filename.c_str(), std::ios::in);
	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}

	std::string ligne;

	fp >> ligne;

	std::cout << "READ: "<< ligne << std::endl;

	if (ligne!="Vertices")
		CGoGNerr << "Warning tetmesh file problem"<< CGoGNendl;

	fp >> nbVertices;

	std::cout << "READ: "<< nbVertices << std::endl;
	std::getline (fp, ligne);
	std::cout << "READ: "<< ligne << std::endl;

//	// reading number of vertices
//	std::getline (fp, ligne);
//	std::stringstream oss(ligne);
//	oss >> nbv;

//	// reading number of tetrahedra
//	std::getline (fp, ligne);
//	std::stringstream oss2(ligne);
//	oss2 >> nbt;

	//reading vertices
	std::vector<unsigned int> verticesID;
	verticesID.reserve(nbVertices+1);
	verticesID.push_back(0xffffffff);
	for(unsigned int i = 0; i < nbVertices;++i)
	{
		do
		{
			std::getline (fp, ligne);
		} while (ligne.size() == 0);

		std::stringstream oss(ligne);

		float x,y,z;
		oss >> x;
		oss >> y;
		oss >> z;
		// TODO : if required read other vertices attributes here
		VEC3 pos(x*scaleFactor,y*scaleFactor,z*scaleFactor);

		unsigned int id = container.insertLine();
		position[id] = pos;

		verticesID.push_back(id);
	}

	fp >> ligne;
	if (ligne!="Tetrahedra")
		CGoGNerr << "Warning tetmesh file problem"<< CGoGNendl;

	std::cout << "READ: "<< ligne << std::endl;


	fp >> nbVolumes;

	std::cout << "READ: "<< nbVolumes << std::endl;

	std::getline (fp, ligne);
	std::cout << "READ: "<< ligne << std::endl;


	CGoGNout << "nb points = " << nbVertices  << " / nb tet = " << nbVolumes << CGoGNendl;

	DartMarkerNoUnmark m(map) ;

	unsigned int invertTetra = 0;
	//Read and embed all tetrahedrons
	for(unsigned int i = 0; i < nbVolumes ; ++i)
	{
		//start one tetra

		do
		{
			std::getline(fp,ligne);
		} while(ligne.size() == 0);

		std::stringstream oss(ligne);

		Dart d = Surface::Modelisation::createTetrahedron<PFP>(map,false);

		Geom::Vec4ui pt;

		if (i==0)
		{
			oss >> pt[0];
			oss >> pt[1];
			oss >> pt[2];
			oss >> pt[3];

			typename PFP::VEC3 P = position[verticesID[pt[0]]];
			typename PFP::VEC3 A = position[verticesID[pt[1]]];
			typename PFP::VEC3 B = position[verticesID[pt[2]]];
			typename PFP::VEC3 C = position[verticesID[pt[3]]];

			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
			{
				invertTetra=1;
				unsigned int ui=pt[1];
				pt[1] = pt[2];
				pt[2] = ui;
			}
		}

		oss >> pt[0];
		oss >> pt[1+invertTetra];
		oss >> pt[2-invertTetra];
		oss >> pt[3];

		// Embed three "base" vertices
		for(unsigned int j = 0 ; j < 3 ; ++j)
		{
			FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[pt[2-j]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

			//store darts per vertices to optimize reconstruction
			Dart dd = d;
			do
			{
				m.mark(dd) ;
				vecDartsPerVertex[verticesID[pt[2-j]]].push_back(dd);
				dd = map.phi1(map.phi2(dd));
			} while(dd != d);

			d = map.phi1(d);
		}

		//Embed the last "top" vertex
		d = map.phi_1(map.phi2(d));

		FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[pt[3]]);
		map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

		//store darts per vertices to optimize reconstruction
		Dart dd = d;
		do
		{
			m.mark(dd) ;
			vecDartsPerVertex[verticesID[pt[3]]].push_back(dd);
			dd = map.phi1(map.phi2(dd));
		} while(dd != d);

		//end of tetra
	}

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

	fp.close();
	return true;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN
