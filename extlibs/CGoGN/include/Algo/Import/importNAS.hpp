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

inline float floatFromNas(std::string& s_v)
{
	float x = 0.0f;

	std::size_t pos1 = s_v.find_last_of('-');
	if ((pos1!=std::string::npos) && (pos1!=0))
	{
		std::string res = s_v.substr(0,pos1) + "e" + s_v.substr(pos1,8-pos1);
		x = atof(res.c_str());
	}
	else
	{
		std::size_t pos2 = s_v.find_last_of('+');
		if ((pos2!=std::string::npos) && (pos2!=0))
		{
			std::string res = s_v.substr(0,pos2) + "e" + s_v.substr(pos2,8-pos2);
			x = atof(res.c_str());
		}
		else
		{
			x = atof(s_v.c_str());
		}
	}
	return x;
}


template <typename PFP>
bool importNAS(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scaleFactor)
{
	typedef typename PFP::VEC3 VEC3;

	VertexAttribute<VEC3> position = map.template addAttribute<VEC3, VERTEX>("position") ;
	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;


	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	//open file
	std::ifstream fp(filename.c_str(), std::ios::in);
	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}

	std::string ligne;
	std::string tag;

	std::getline (fp, ligne);
	do
	{
		std::getline (fp, ligne);
		tag = ligne.substr(0,4);
	} while (tag !="GRID");

	unsigned int m_nbVertices = 0;
	//reading vertices
	std::map<unsigned int, unsigned int> verticesMapID;
	do
	{
		std::string s_v = ligne.substr(8,8);
		unsigned int ind = atoi(s_v.c_str());

		s_v = ligne.substr(24,8);
		float x = floatFromNas(s_v);
		s_v = ligne.substr(32,8);
		float y = floatFromNas(s_v);
		s_v = ligne.substr(40,8);
		float z = floatFromNas(s_v);

		VEC3 pos(x*scaleFactor,y*scaleFactor,z*scaleFactor);
		unsigned int id = container.insertLine();
		position[id] = pos;
		verticesMapID.insert(std::pair<unsigned int, unsigned int>(ind,id));
//		std::cout << "P: "<< ind << "   "<<x<<", "<<y<<", "<<z << std::endl;
		std::getline (fp, ligne);
		tag = ligne.substr(0,4);
		m_nbVertices++;
	} while (tag =="GRID");

	std::vector<Geom::Vec4ui> tet;
	tet.reserve(1000);
	std::vector<Geom::Vec4ui> hexa;
	tet.reserve(1000);

	do
	{
		std::string s_v = ligne.substr(0,6);

		if (s_v == "CHEXA ")
		{
			s_v = ligne.substr(24,8);
			unsigned int ind1 = atoi(s_v.c_str());
			s_v = ligne.substr(32,8);
			unsigned int ind2 = atoi(s_v.c_str());
			s_v = ligne.substr(40,8);
			unsigned int ind3 = atoi(s_v.c_str());
			s_v = ligne.substr(48,8);
			unsigned int ind4 = atoi(s_v.c_str());
			s_v = ligne.substr(56,8);
			unsigned int ind5 = atoi(s_v.c_str());
			s_v = ligne.substr(64,8);
			unsigned int ind6 = atoi(s_v.c_str());

			std::getline (fp, ligne);
			s_v = ligne.substr(8,8);
			unsigned int ind7 = atoi(s_v.c_str());
			s_v = ligne.substr(16,8);
			unsigned int ind8 = atoi(s_v.c_str());

			typename PFP::VEC3 P = position[verticesMapID[ind5]];
			typename PFP::VEC3 A = position[verticesMapID[ind1]];
			typename PFP::VEC3 B = position[verticesMapID[ind2]];
			typename PFP::VEC3 C = position[verticesMapID[ind3]];

			Geom::Vec4ui v;

			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
			{
				v[0] = ind4;
				v[1] = ind3;
				v[2] = ind2;
				v[3] = ind1;
				hexa.push_back(v);
				v[0] = ind8;
				v[1] = ind7;
				v[2] = ind6;
				v[3] = ind5;
				hexa.push_back(v);
			}
			else
			{
				v[0] = ind1;
				v[1] = ind2;
				v[2] = ind3;
				v[3] = ind4;
				hexa.push_back(v);
				v[0] = ind5;
				v[1] = ind6;
				v[2] = ind7;
				v[3] = ind8;
				hexa.push_back(v);
			}
		}
		if (s_v == "CTETRA")
		{
			s_v = ligne.substr(24,8);
			unsigned int ind1 = atoi(s_v.c_str());
			s_v = ligne.substr(32,8);
			unsigned int ind2 = atoi(s_v.c_str());
			s_v = ligne.substr(40,8);
			unsigned int ind3 = atoi(s_v.c_str());
			s_v = ligne.substr(48,8);
			unsigned int ind4 = atoi(s_v.c_str());

			typename PFP::VEC3 P = position[verticesMapID[ind1]];
			typename PFP::VEC3 A = position[verticesMapID[ind2]];
			typename PFP::VEC3 B = position[verticesMapID[ind3]];
			typename PFP::VEC3 C = position[verticesMapID[ind4]];

			Geom::Vec4ui v;

			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
			{
				v[0] = ind4;
				v[1] = ind3;
				v[2] = ind2;
				v[3] = ind1;
			}
			else
			{
				v[0] = ind1;
				v[1] = ind2;
				v[2] = ind3;
				v[3] = ind4;
			}
			tet.push_back(v);
		}
		std::getline (fp, ligne);
		tag = ligne.substr(0,4);
	} while (!fp.eof());



	CGoGNout << "nb points = " << m_nbVertices ;


	unsigned int m_nbVolumes = 0;

	DartMarkerNoUnmark m(map) ;

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
