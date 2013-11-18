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
bool importVBGZ(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scaleFactor)
{
	typedef typename PFP::VEC3 VEC3;

	VertexAttribute<VEC3> position = map.template addAttribute<VEC3, VERTEX>("position") ;
	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;

	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	//open file
	igzstream fs(filename.c_str(), std::ios::in|std::ios::binary);
	if (!fs.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}


	unsigned int numbers[3];

	// read nb of points
	fs.read(reinterpret_cast<char*>(numbers), 3*sizeof(unsigned int));

	VEC3* bufposi;
	bufposi = new VEC3[ numbers[0] ];
	fs.read(reinterpret_cast<char*>(bufposi), numbers[0]*sizeof(VEC3));

	std::vector<unsigned int> verticesID;
	verticesID.reserve(numbers[0]);

	for(unsigned int i = 0; i < numbers[0];++i)
	{
		unsigned int id = container.insertLine();
		position[id] = bufposi[i]*scaleFactor;
		verticesID.push_back(id);
	}
	delete bufposi;

	unsigned int* bufTetra=NULL;
	if (numbers[1] != 0)
	{
		bufTetra = new unsigned int[ 4*numbers[1] ];
		fs.read(reinterpret_cast<char*>(bufTetra), 4*numbers[1]*sizeof(unsigned int));
	}

	unsigned int* bufHexa=NULL;
	if (numbers[2] != 0)
	{
		bufHexa = new unsigned int[ 8*numbers[2] ];
		fs.read(reinterpret_cast<char*>(bufHexa), 8*numbers[2]*sizeof(unsigned int));
	}
	CGoGNout << "nb vertices = " << numbers[0];


	DartMarkerNoUnmark m(map) ;

	if (numbers[1] > 0)
	{
		//Read and embed all tetrahedrons
		for(unsigned int i = 0; i < numbers[1] ; ++i)
		{
			Geom::Vec4ui pt;

			pt[0] = bufTetra[4*i];
			pt[1] = bufTetra[4*i+1];
			pt[2] = bufTetra[4*i+2];
			pt[3] = bufTetra[4*i+3];

			Dart d = Surface::Modelisation::createTetrahedron<PFP>(map,false);

			// Embed three "base" vertices
			for(unsigned int j = 0 ; j < 3 ; ++j)
			{
				FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[pt[j]]);
				map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

				//store darts per vertices to optimize reconstruction
				Dart dd = d;
				do
				{
					m.mark(dd) ;
					vecDartsPerVertex[verticesID[pt[j]]].push_back(dd);
					dd = map.phi1(map.phi2(dd));
				} while(dd != d);

				d = map.phi1(d);
			}

			//Embed the last "top" vertex
			d = map.template phi<211>(d);
			//d = map.phi_1(map.phi2(d));

			FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[pt[3]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

			//store darts per vertex to optimize reconstruction
			Dart dd = d;
			do
			{
				m.mark(dd) ;
				vecDartsPerVertex[verticesID[pt[3]]].push_back(dd);
				dd = map.phi1(map.phi2(dd));
			} while(dd != d);

			//end of tetra
		}
		CGoGNout << " / nb tetra = " << numbers[1];
		delete[] bufTetra;
	}

	if (numbers[2] > 0)
	{

		//Read and embed all tetrahedrons
		for(unsigned int i = 0; i < numbers[2] ; ++i)
		{
			// one hexa
			unsigned int pt[8];
			pt[0] = bufHexa[8*i];
			pt[1] = bufHexa[8*i+1];
			pt[2] = bufHexa[8*i+2];
			pt[3] = bufHexa[8*i+3];
			pt[4] = bufHexa[8*i+4];
			pt[5] = bufHexa[8*i+5];
			pt[6] = bufHexa[8*i+6];
			pt[7] = bufHexa[8*i+7];

			Dart d = Surface::Modelisation::createHexahedron<PFP>(map,false);

			FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[pt[0]]);

			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			Dart dd = d;
			vecDartsPerVertex[verticesID[pt[0]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[0]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[0]]].push_back(dd); m.mark(dd);

			d = map.phi1(d);
			fsetemb.changeEmb(verticesID[pt[1]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[1]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[1]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[1]]].push_back(dd); m.mark(dd);


			d = map.phi1(d);
			fsetemb.changeEmb(verticesID[pt[2]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[2]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[2]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[2]]].push_back(dd); m.mark(dd);


			d = map.phi1(d);
			fsetemb.changeEmb(verticesID[pt[3]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[3]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[3]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[3]]].push_back(dd); m.mark(dd);

			d = map.template phi<2112>(d);
			fsetemb.changeEmb(verticesID[pt[4]]);

			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[4]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[4]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[4]]].push_back(dd); m.mark(dd);

			d = map.phi1(d);
			fsetemb.changeEmb(verticesID[pt[5]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[5]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[5]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[5]]].push_back(dd); m.mark(dd);

			d = map.phi1(d);
			fsetemb.changeEmb(verticesID[pt[6]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[6]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[6]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[6]]].push_back(dd); m.mark(dd);

			d = map.phi1(d);
			fsetemb.changeEmb(verticesID[pt[7]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[7]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[7]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[7]]].push_back(dd); m.mark(dd);
			//end of hexa
		}
		CGoGNout << " / nb hexa = " << numbers[2];
		delete[] bufHexa;
	}

	CGoGNout << CGoGNendl;

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
				   map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.template phi<11>(d)))
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
		map.closeMap();
		CGoGNout << "Map closed (" << nbBoundaryFaces << " boundary faces)" << CGoGNendl;
	}

	fs.close();
	return true;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN
