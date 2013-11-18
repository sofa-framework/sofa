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

#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#include <libxml/parser.h>

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
bool importVTU(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scaleFactor)
{
	typedef typename PFP::VEC3 VEC3;

	VertexAttribute<VEC3> position = map.template addAttribute<VEC3, VERTEX>("position") ;
	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;

	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	xmlDocPtr doc = xmlReadFile(filename.c_str(), NULL, 0);
	xmlNodePtr vtu_node = xmlDocGetRootElement(doc);


//	std::cout << " NAME "<<vtu_node->name << std::endl;

	xmlChar *prop = xmlGetProp(vtu_node, BAD_CAST "type");
//	std::cout << "type = "<< prop << std::endl;

	xmlNode* grid_node = vtu_node->children;
	while (strcmp((char*)(grid_node->name),(char*)"UnstructuredGrid")!=0)
		grid_node = grid_node->next;

	xmlNode* piece_node = grid_node->children;
	while (strcmp((char*)(piece_node->name),(char*)"Piece")!=0)
		piece_node = piece_node->next;

	prop = xmlGetProp(piece_node, BAD_CAST "NumberOfPoints");
	unsigned int nbVertices = atoi((char*)(prop));

	prop = xmlGetProp(piece_node, BAD_CAST "NumberOfCells");
	unsigned int nbVolumes = atoi((char*)(prop));

	std::cout << "Number of points = "<< nbVertices<< std::endl;
	std::cout << "Number of cells = "<< nbVolumes << std::endl;

	xmlNode* points_node = piece_node->children;
	while (strcmp((char*)(points_node->name),(char*)"Points")!=0)
		points_node = points_node->next;

	points_node = points_node->children;
	while (strcmp((char*)(points_node->name),(char*)"DataArray")!=0)
		points_node = points_node->next;

	std::vector<unsigned int> verticesID;
	verticesID.reserve(nbVertices);

	std::stringstream ss((char*)(xmlNodeGetContent(points_node->children)));
	for (unsigned int i=0; i< nbVertices; ++i)
	{
		typename PFP::VEC3 P;
		ss >> P[0]; ss >> P[1]; ss >> P[2];
		P *= scaleFactor;
		unsigned int id = container.insertLine();
		position[id] = P;
		verticesID.push_back(id);
	}


	xmlNode* cell_node = piece_node->children;
	while (strcmp((char*)(cell_node->name),(char*)"Cells")!=0)
		cell_node = cell_node->next;

	std::cout <<"CELL NODE = "<< cell_node->name << std::endl;


	std::vector<unsigned char> typeVols;
	typeVols.reserve(nbVolumes);
	std::vector<unsigned int> offsets;
	offsets.reserve(nbVolumes);
	std::vector<unsigned int> indices;
	indices.reserve(nbVolumes*4);


	for (xmlNode* x_node = cell_node->children; x_node!=NULL; x_node = x_node->next)
	{
		while ((x_node!=NULL) && (strcmp((char*)(x_node->name),(char*)"DataArray")!=0))
			x_node = x_node->next;

		if (x_node == NULL)
			break;
		else
		{
			xmlChar* type = xmlGetProp(x_node, BAD_CAST "Name");

			if (strcmp((char*)(type),(char*)"connectivity")==0)
			{
				std::stringstream ss((char*)(xmlNodeGetContent(x_node->children)));
				while (!ss.eof())
				{
					unsigned int ind;
					ss >> ind;
					indices.push_back(ind);
				}
			}
			if (strcmp((char*)(type),(char*)"offsets")==0)
			{
				std::stringstream ss((char*)(xmlNodeGetContent(x_node->children)));
				for (unsigned int i=0; i< nbVolumes; ++i)
				{
					unsigned int o;
					ss >> o;
					offsets.push_back(o);
				}
			}
			if (strcmp((char*)(type),(char*)"types")==0)
			{
				bool unsupported = false;
				std::stringstream ss((char*)(xmlNodeGetContent(x_node->children)));
				for (unsigned int i=0; i< nbVolumes; ++i)
				{
					unsigned int t;
					ss >> t;
					if ((t != 12) && (t!= 10))
					{
						unsupported = true;
						typeVols.push_back(0);
					}
					else
					{
						typeVols.push_back((unsigned char)t);
					}
				}
				if (unsupported)
					CGoGNerr << "warning, some unsupported volume cell types"<< CGoGNendl;
			}
		}
	}

	xmlFreeDoc(doc);

	DartMarkerNoUnmark m(map) ;

	unsigned int currentOffset = 0;
	for (unsigned int i=0; i< nbVolumes; ++i)
	{
		if (typeVols[i]==12)
		{
			Dart d = Surface::Modelisation::createHexahedron<PFP>(map,false);

			unsigned int pt[8];
			pt[0] = indices[currentOffset];
			pt[1] = indices[currentOffset+1];
			pt[2] = indices[currentOffset+2];
			pt[3] = indices[currentOffset+3];
			pt[4] = indices[currentOffset+4];
			typename PFP::VEC3 P = position[verticesID[indices[currentOffset+4]]];
			typename PFP::VEC3 A = position[verticesID[indices[currentOffset  ]]];
			typename PFP::VEC3 B = position[verticesID[indices[currentOffset+1]]];
			typename PFP::VEC3 C = position[verticesID[indices[currentOffset+2]]];

			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
			{

				pt[0] = indices[currentOffset+3];
				pt[1] = indices[currentOffset+2];
				pt[2] = indices[currentOffset+1];
				pt[3] = indices[currentOffset+0];
				pt[4] = indices[currentOffset+7];
				pt[5] = indices[currentOffset+6];
				pt[6] = indices[currentOffset+5];
				pt[7] = indices[currentOffset+4];
			}
			else
			{
				pt[0] = indices[currentOffset+0];
				pt[1] = indices[currentOffset+1];
				pt[2] = indices[currentOffset+2];
				pt[3] = indices[currentOffset+3];
				pt[4] = indices[currentOffset+4];
				pt[5] = indices[currentOffset+5];
				pt[6] = indices[currentOffset+6];
				pt[7] = indices[currentOffset+7];
			}


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

			d = map.phi_1(d);
			fsetemb.changeEmb(verticesID[pt[5]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[5]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[5]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[5]]].push_back(dd); m.mark(dd);

			d = map.phi_1(d);
			fsetemb.changeEmb(verticesID[pt[6]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[6]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[6]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[6]]].push_back(dd); m.mark(dd);

			d = map.phi_1(d);
			fsetemb.changeEmb(verticesID[pt[7]]);
			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
			dd = d;
			vecDartsPerVertex[verticesID[pt[7]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[7]]].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
			vecDartsPerVertex[verticesID[pt[7]]].push_back(dd); m.mark(dd);
		}
		else if (typeVols[i]==10)
		{
			Dart d = Surface::Modelisation::createTetrahedron<PFP>(map,false);

			Geom::Vec4ui pt;
			pt[0] = indices[currentOffset];
			pt[1] = indices[currentOffset+1];
			pt[2] = indices[currentOffset+2];
			pt[3] = indices[currentOffset+3];

			typename PFP::VEC3 P = position[verticesID[pt[0]]];
			typename PFP::VEC3 A = position[verticesID[pt[1]]];
			typename PFP::VEC3 B = position[verticesID[pt[2]]];
			typename PFP::VEC3 C = position[verticesID[pt[3]]];

			if (Geom::testOrientation3D<typename PFP::VEC3>(P,A,B,C) == Geom::OVER)
			{
				unsigned int ui=pt[1];
				pt[1] = pt[2];
				pt[2] = ui;
			}

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
					vecDartsPerVertex[pt[2-j]].push_back(dd);
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
				vecDartsPerVertex[pt[3]].push_back(dd);
				dd = map.phi1(map.phi2(dd));
			} while(dd != d);

		}
		currentOffset = offsets[i];
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

	return true;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN
