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
bool importMSH(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scaleFactor)
{
	typedef typename PFP::VEC3 VEC3;
    typedef typename PFP::MAP MAP;
    VertexAttribute<VEC3, MAP> position = map.template addAttribute<VEC3, VERTEX, MAP>("position") ;
	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;

	unsigned int m_nbVertices = 0, m_nbVolumes = 0;
    VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> >, MAP > vecDartsPerVertex(map, "incidents");

	//open file
	std::ifstream fp(filename.c_str(), std::ios::in);
	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}

	std::string ligne;
	unsigned int nbv=0;
	//read $NODE
	std::getline (fp, ligne);

	// reading number of vertices
	std::getline (fp, ligne);
	std::stringstream oss(ligne);
	oss >> nbv;


	//reading vertices
//	std::vector<unsigned int> verticesID;
	std::map<unsigned int, unsigned int> verticesMapID;


//	verticesID.reserve(nbv);
	for(unsigned int i = 0; i < nbv;++i)
	{
		do
		{
			std::getline (fp, ligne);
		} while (ligne.size() == 0);

		std::stringstream oss(ligne);
		unsigned int pipo;
		float x,y,z;
		oss >> pipo;
		oss >> x;
		oss >> y;
		oss >> z;
		// TODO : if required read other vertices attributes here
		VEC3 pos(x*scaleFactor,y*scaleFactor,z*scaleFactor);

		unsigned int id = container.insertLine();
		position[id] = pos;

		verticesMapID.insert(std::pair<unsigned int, unsigned int>(pipo,id));
//		verticesID.push_back(id);
	}

	// ENNODE
	std::getline (fp, ligne);

	m_nbVertices = nbv;


	// ELM
	std::getline (fp, ligne);

	// reading number of elements
	std::getline (fp, ligne);
	unsigned int nbe=0;
	std::stringstream oss2(ligne);
	oss2 >> nbe;

	std::vector<Geom::Vec4ui> tet;
	tet.reserve(1000);
	std::vector<Geom::Vec4ui> hexa;
	tet.reserve(1000);

	bool invertVol = false;

	for(unsigned int i=0; i<nbe; ++i)
	{
		unsigned int pipo,type_elm,nb;
		fp >> pipo;
		fp >> type_elm;
		fp >> pipo;
		fp >> pipo;
		fp >> nb;

		if ((type_elm==4) && (nb==4))
		{
			Geom::Vec4ui v;

			// test orientation of first tetra
			if (i==0)
			{
				fp >> v[0];
				fp >> v[1];
				fp >> v[2];
				fp >> v[3];

				typename PFP::VEC3 P = position[verticesMapID[v[0]]];
				typename PFP::VEC3 A = position[verticesMapID[v[1]]];
				typename PFP::VEC3 B = position[verticesMapID[v[2]]];
				typename PFP::VEC3 C = position[verticesMapID[v[3]]];

				if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
				{
					invertVol=true;
					unsigned int ui=v[0];
					v[0] = v[3];
					v[3] = v[2];
					v[2] = v[1];
					v[1] = ui;
				}
			}
			else
			{
				if (invertVol)
				{
					fp >> v[1];
					fp >> v[2];
					fp >> v[3];
					fp >> v[0];
				}
				else
				{
					fp >> v[0];
					fp >> v[1];
					fp >> v[2];
					fp >> v[3];
				}
			}
			tet.push_back(v);
		}
		else
		{
			if ((type_elm==5) && (nb==8))
			{
				Geom::Vec4ui v;

				if (i==0)
				{
					unsigned int last;
					fp >> v[0];
					fp >> v[1];
					fp >> v[2];
					fp >> v[3];
					fp >> last;

					typename PFP::VEC3 P = position[verticesMapID[last]];
					typename PFP::VEC3 A = position[verticesMapID[v[0]]];
					typename PFP::VEC3 B = position[verticesMapID[v[1]]];
					typename PFP::VEC3 C = position[verticesMapID[v[2]]];

					if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
					{

						invertVol=true;
						unsigned int ui = v[3];
						v[3] = v[0];
						v[0] = ui;
						ui = v[2];
						v[2] = v[1];
						v[1] = ui;
						hexa.push_back(v);
						v[3] = last;
						fp >> v[2];
						fp >> v[1];
						fp >> v[0];
						hexa.push_back(v);
					}
					else
					{
						hexa.push_back(v);
						v[0] = last;
						fp >> v[1];
						fp >> v[2];
						fp >> v[3];
						hexa.push_back(v);
					}
				}
				else
				{
					if (invertVol)
					{
						fp >> v[3];
						fp >> v[2];
						fp >> v[1];
						fp >> v[0];
						hexa.push_back(v);
						fp >> v[3];
						fp >> v[2];
						fp >> v[1];
						fp >> v[0];
						hexa.push_back(v);

					}
					else
					{
						fp >> v[0];
						fp >> v[1];
						fp >> v[2];
						fp >> v[3];
						hexa.push_back(v);
						fp >> v[0];
						fp >> v[1];
						fp >> v[2];
						fp >> v[3];
						hexa.push_back(v);
					}
				}
			}
			else
			{
				for (unsigned int j=0; j<nb; ++j)
				{
					unsigned int v;
					fp >> v;
				}
			}
		}
	}

	CGoGNout << "nb points = " << m_nbVertices ;


	m_nbVolumes = 0;

    DartMarkerNoUnmark<MAP> m(map) ;

	if (tet.size() > 0)
	{
		m_nbVolumes += tet.size();

		//Read and embed all tetrahedrons
		for(unsigned int i = 0; i < tet.size() ; ++i)
		{
			//start one tetra
			//		Geom::Vec4ui& pt = tet[i];
			Geom::Vec4ui pt;

			pt[0] = tet[i][0];
			pt[1] = tet[i][1];
			pt[2] = tet[i][2];
			pt[3] = tet[i][3];

			Dart d = Surface::Modelisation::createTetrahedron<PFP>(map,false);


			// Embed three "base" vertices
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

			//Embed the last "top" vertex
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

			//end of tetra
		}
		CGoGNout << " / nb tetra = " << tet.size() << CGoGNendl;

	}

	if (hexa.size() > 0)
	{

		m_nbVolumes += hexa.size()/2;

		//Read and embed all tetrahedrons
		for(unsigned int i = 0; i < hexa.size()/2 ; ++i)
		{
			// one hexa
			Geom::Vec4ui pt;
			pt[0] = hexa[2*i][0];
			pt[1] = hexa[2*i][1];
			pt[2] = hexa[2*i][2];
			pt[3] = hexa[2*i][3];
			Geom::Vec4ui ppt;
			ppt[0] = hexa[2*i+1][0];
			ppt[1] = hexa[2*i+1][1];
			ppt[2] = hexa[2*i+1][2];
			ppt[3] = hexa[2*i+1][3];

			Dart d = Surface::Modelisation::createHexahedron<PFP>(map,false);

			FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesMapID[pt[0]]);

			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			Dart dd = d;
			vecDartsPerVertex[verticesMapID[pt[0]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[0]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[0]]].push_back(dd); m.mark(dd);

			d = map.phi1(d);
			fsetemb.changeEmb(verticesMapID[pt[1]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesMapID[pt[1]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[1]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[1]]].push_back(dd); m.mark(dd);


			d = map.phi1(d);
			fsetemb.changeEmb(verticesMapID[pt[2]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesMapID[pt[2]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[2]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[2]]].push_back(dd); m.mark(dd);


			d = map.phi1(d);
			fsetemb.changeEmb(verticesMapID[pt[3]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesMapID[pt[3]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[3]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[pt[3]]].push_back(dd); m.mark(dd);

			d = map.template phi<2112>(d);
			fsetemb.changeEmb(verticesMapID[ppt[0]]);

			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesMapID[ppt[0]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[0]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[0]]].push_back(dd); m.mark(dd);

			d = map.phi_1(d);
			fsetemb.changeEmb(verticesMapID[ppt[1]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesMapID[ppt[1]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[1]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[1]]].push_back(dd); m.mark(dd);

			d = map.phi_1(d);
			fsetemb.changeEmb(verticesMapID[ppt[2]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesMapID[ppt[2]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[2]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[2]]].push_back(dd); m.mark(dd);

			d = map.phi_1(d);
			fsetemb.changeEmb(verticesMapID[ppt[3]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesMapID[ppt[3]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[3]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesMapID[ppt[3]]].push_back(dd); m.mark(dd);

			//end of hexa
		}
		CGoGNout << " / nb hexa = " << hexa.size()/2 << CGoGNendl;

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
