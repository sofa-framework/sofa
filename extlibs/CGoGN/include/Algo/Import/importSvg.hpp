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

#include <iostream>
#include "Geometry/bounding_box.h"
#include "Geometry/plane_3d.h"
#include "Algo/BooleanOperator/mergeVertices.h"
#include "Container/fakeAttribute.h"
#include <limits>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{

inline bool checkXmlNode(xmlNodePtr node, const std::string& name)
{
	return (strcmp((char*)(node->name),(char*)(name.c_str())) == 0);
}

template<typename T>
inline bool valueOf(const std::string &s, T &obj)
{
  std::istringstream is(s);
  return is >> obj;
}

template <typename VEC>
bool posSort(const std::pair<VEC, Dart>& a1, const std::pair<VEC, Dart>& a2)
{
	VEC v1 = a1.first;
	VEC v2 = a2.first;
	return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]);
}

template <typename VEC3>
void getPolygonFromSVG(std::string allcoords, std::vector<VEC3>& curPoly, bool& closedPoly)
{
	closedPoly=false;
	std::stringstream is(allcoords);
	bool relative=false;
	bool push_point;
	std::string coord;
	int mode = -1;

	while ( std::getline( is, coord, ' ' ) )
	{
		float x,y;
		push_point=false;

		if(coord[0]=='m' || coord[0]=='l' || coord[0]=='t') //start point, line or quadratic bezier curve
		{
			mode = 0;
			relative=true;
		}
		else if(coord[0]=='M' || coord[0] == 'L' || coord[0]=='T') //same in absolute coordinates
		{
			mode = 1;
			relative=false;
		}
		else if(coord[0]=='h' || coord[0] == 'H') //horizontal line
		{
			mode = 2;
			relative=(coord[0]=='h');
		}
		else if(coord[0]=='v' || coord[0] == 'V') //vertical line
		{
			mode = 3;
			relative=(coord[0]=='v');
		}
		else if(coord[0]=='c' || coord[0] == 'C') //bezier curve
		{
			mode = 4;
			relative=(coord[0]=='c');
		}
		else if(coord[0]=='s' || coord[0] == 'S' || coord[0]=='q' || coord[0] == 'Q') //bezier curve 2
		{
			mode = 5;
			relative= ((coord[0]=='s') || (coord[0]=='q'));
		}
		else if(coord[0]=='a' || coord[0] == 'A') //elliptic arc
		{
			mode =6;
			relative= (coord[0]=='a');
		}
		else if(coord[0]=='z') //end of path
		{
			closedPoly = true;

		}
		else //coordinates
		{
			switch(mode)
			{
				case 0 : //relative
				break;
				case 1 : //absolute
				break;
				case 2 : //horizontal
				{
					std::stringstream streamCoord(coord);
					std::string xS;
					std::getline(streamCoord, xS, ',' );

					valueOf(xS,x);

					VEC3 previous = (curPoly)[(curPoly).size()-1];
					y = previous[1];

					push_point=true;
				}
				break;
				case 3 : //vertical
				{
					std::stringstream streamCoord(coord);
					std::string yS;
					std::getline(streamCoord, yS, ',' );

					valueOf(yS,y);

					VEC3 previous = (curPoly)[(curPoly).size()-1];
					x = previous[0];

					push_point=true;
				}
				break;
				case 4 : //bezier
				{
					std::getline( is, coord, ' ' ); //ignore first control point
					std::getline( is, coord, ' ' ); //ignore second control point
				}
				break;
				case 5 : //bezier 2
				{
					std::getline( is, coord, ' ' ); //ignore control point

				}
				break;
				case 6 : //elliptic
					std::getline( is, coord, ' ' ); //ignore rx
					std::getline( is, coord, ' ' ); //ignore ry
					std::getline( is, coord, ' ' ); //ignore x-rotation
					std::getline( is, coord, ' ' ); //ignore large arc flag
					std::getline( is, coord, ' ' ); //ignore sweep flag
				break;
			}

			std::stringstream streamCoord(coord);
			std::string xS,yS;
			std::getline(streamCoord, xS, ',' );
			std::getline(streamCoord, yS, ',' );

			valueOf(xS,x);
			valueOf(yS,y);

			push_point = true;
		}

		//if there is a point to push
		if(push_point)
		{

			VEC3 previous;

			if(curPoly.size()>0)
				previous = (curPoly)[(curPoly).size()-1];

			if(relative)
			{
				x += previous[0];
				y += previous[1];
			}

			if(curPoly.size()==0 || (curPoly.size()>0 && (x!=previous[0] || y!= previous[1])))
				curPoly.push_back(VEC3(x,y,0));
		}
	}
}

template <typename PFP>
void readCoordAndStyle(xmlNode* cur_path,
		std::vector<std::vector<VEC3 > >& allPoly,
		std::vector<std::vector<VEC3 > >& allBrokenLines,
		std::vector<float>& allBrokenLinesWidth)
{
	typedef typename PFP::VEC3 VEC3;
	typedef std::vector<VEC3 > POLYGON;

	bool closedPoly;
	POLYGON curPoly;

//	CGoGNout << "--load a path--"<< CGoGNendl;
	xmlChar* prop = xmlGetProp(cur_path, BAD_CAST "d");
	std::string allcoords((reinterpret_cast<const char*>(prop)));
	getPolygonFromSVG(allcoords,curPoly,closedPoly);

	//check orientation : set in CCW
	if(curPoly.size()>2)
	{
		VEC3 v(0), v1, v2;
		typename std::vector<VEC3 >::iterator it0, it1, it2;
		it0 = curPoly.begin();
		it1 = it0+1;
		it2 = it1+1;
		for(unsigned int i = 0 ; i < curPoly.size() ; ++i)
		{
			VEC3 t = (*it1)^(*it0);
			v += t;

			it0=it1;
			it1=it2;
			it2++;
			if(it2 == curPoly.end())
				it2 = curPoly.begin();
		}

		if(v[2]>0)
		{
			std::reverse(curPoly.begin(), curPoly.end());
		}
	}

	//closed polygon ?
	if(closedPoly)
		allPoly.push_back(curPoly);
	else
	{
		//if not : read the linewidth for further dilatation
		allBrokenLines.push_back(curPoly);
		xmlChar* prop = xmlGetProp(cur_path, BAD_CAST "style");
		std::string allstyle((reinterpret_cast<const char*>(prop)));
		std::stringstream is(allstyle);
		std::string style;
		while ( std::getline( is, style, ';' ) )
		{
			if(style.find("stroke-width:")!=std::string::npos)
			{
				std::stringstream isSize(style);
				std::getline( isSize, style, ':' );
				float sizeOfLine;
				isSize >> sizeOfLine;
				allBrokenLinesWidth.push_back(sizeOfLine);
			}
		}
	}
}

template <typename PFP>
bool importSVG(typename PFP::MAP& map, const std::string& filename, VertexAttribute<typename PFP::VEC3>& position, CellMarker<EDGE>& obstacleMark, CellMarker<FACE>& buildingMark)
{
	//TODO : remove auto-intersecting faces
	//TODO : handling polygons with holes

	typedef typename PFP::VEC3 VEC3;
	typedef std::vector<VEC3> POLYGON;

	xmlDocPtr doc = xmlReadFile(filename.c_str(), NULL, 0);
	xmlNodePtr map_node = xmlDocGetRootElement(doc);

	if (!checkXmlNode(map_node,"svg"))
	{
		CGoGNerr << "Wrong xml format: Root node != svg"<< CGoGNendl;
		return false;
	}

	std::vector<POLYGON> allPoly;
	std::vector<POLYGON> allBrokenLines;
	std::vector<float> allBrokenLinesWidth;

	for (xmlNode* cur_node = map_node->children; cur_node; cur_node = cur_node->next)
	{
		// for each layer
		if (checkXmlNode(cur_node, "g"))
			for (xmlNode* cur_path = cur_node->children ; cur_path; cur_path = cur_path->next)
			{
				if (checkXmlNode(cur_path, "path"))
					readCoordAndStyle<PFP>(cur_path, allPoly, allBrokenLines, allBrokenLinesWidth);
			}
		else if (checkXmlNode(cur_node, "path"))
				readCoordAndStyle<PFP>(cur_node, allPoly, allBrokenLines, allBrokenLinesWidth);
	}

	xmlFreeDoc(doc);

	std::cout << "importSVG : XML read." << std::endl;

	CellMarker<EDGE> brokenMark(map);
	EdgeAttribute<float> edgeWidth = map.template addAttribute<float, EDGE>("width");
//	EdgeAttribute<NoMathAttribute<Geom::Plane3D<typename PFP::REAL> > > edgePlanes = map.template addAttribute<NoMathAttribute<Geom::Plane3D<typename PFP::REAL> >, EDGE>("planes");
	EdgeAttribute<NoTypeNameAttribute<Geom::Plane3D<typename PFP::REAL> > > edgePlanes = map.template addAttribute<NoTypeNameAttribute<Geom::Plane3D<typename PFP::REAL> >, EDGE>("planes");
	/////////////////////////////////////////////////////////////////////////////////////////////
	//create broken lines
	DartMarker brokenL(map);

	typename std::vector<POLYGON >::iterator it;
	std::vector<float >::iterator itW = allBrokenLinesWidth.begin();
	for(it = allBrokenLines.begin() ; it != allBrokenLines.end() ; ++it, ++itW)
	{
		if(it->size()<2)
		{
			it = allBrokenLines.erase(it);
			itW = allBrokenLinesWidth.erase(itW);
		}
		else
		{
			Dart d = map.newPolyLine(it->size()-1);

			for(typename POLYGON::iterator emb = it->begin(); emb != it->end() ; emb++)
			{
				brokenL.mark(d);
				brokenL.mark(map.phi2(d));

				edgeWidth[d] = *itW;
				if (*itW == 0)
					std::cout << "importSVG : null path width" << std::endl ;
				position[d] = *emb;
				d = map.phi1(d);
			}
		}
	}
	std::cout << "importSVG : broken lines created : " << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////////
	// Merge near vertices
	BooleanOperator::mergeVertices<PFP>(map,position,1);
	std::cout << "importSVG : Merging of vertices." << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "buildings " << allPoly.size() << std::endl;
	unsigned int c = 0;

	//create polygons
	for(it = allPoly.begin() ; it != allPoly.end() ; ++it)
	{
		if(it->size()<3)
		{
			it = allPoly.erase(it);
		}
		else
		{
			Dart d = map.newFace(it->size());
			c++;
			buildingMark.mark(d);
			buildingMark.mark(map.phi2(d));

			for(typename POLYGON::iterator emb = it->begin(); emb != it->end() ; emb++)
			{
				position[d] = *emb;
				obstacleMark.mark(d);
				d = map.phi1(d);
			}
		}
	}

	Geom::BoundingBox<typename PFP::VEC3> bb ;
	bb = Algo::Geometry::computeBoundingBox<PFP>(map, position) ;
	float tailleX = bb.size(0) ;
	float tailleY = bb.size(1) ;
	float tailleM = std::max<float>(tailleX, tailleY) / 30 ;
	std::cout << "bounding box = " << tailleX << " X " << tailleY << std::endl;

	for(Dart d = map.begin();d != map.end(); map.next(d))
	{
		if(position[d][0] == position[map.phi1(d)][0] && position[d][1] == position[map.phi1(d)][1])
			std::cout << "prob d " << d << std::endl;
	}

	std::cout << "importSVG : Polygons generated." << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////////
	unsigned int count = 0 ;

	//cut the edges to have a more regular sampling
	TraversorE<typename PFP::MAP> edges(map) ;
	for (Dart d = edges.begin() ; d != edges.end() ; d = edges.next())
	{
		if (!buildingMark.isMarked(d))
		{
			VEC3 p1 = position[d] ;
			VEC3 p2 = position[map.phi1(d)] ;
			VEC3 v  = p2 - p1 ;
			float length = v.norm() ;

			if (length > tailleM)
			{
				unsigned int nbSeg = (unsigned int)(length / tailleM) ;
				v /= nbSeg ;
				count += nbSeg ;

				for (unsigned int i = 0 ; i < nbSeg - 1 ; ++i)
					map.cutEdge(d) ;

				brokenL.mark(d);
				brokenL.mark(map.phi2(d));
				Dart dd = map.phi1(d) ;

				for (unsigned int i = 1 ; i < nbSeg ; ++i)
				{
					brokenL.mark(dd);
					brokenL.mark(map.phi2(dd));
					position[dd] = p1 + v * i ;
					dd = map.phi1(dd) ;
				}
			}
		}
	}
	std::cout << "importSVG : Subdivision of long edges : " << count << " morceaux."<< std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////////
	//simplify the edges to have a more regular sampling
	count = 0 ;
	for (Dart d = map.begin() ; d != map.end() ; map.next(d))
	{
		if(!buildingMark.isMarked(d))
		{
			bool canSimplify = true ;
			while ( canSimplify && (Geometry::edgeLength<PFP>(map,d,position) < edgeWidth[d]) )
			{
				if (map.vertexDegree(map.phi1(d)) == 2)
				{
					map.uncutEdge(d) ;
					count++;
				}
				else
					canSimplify = false ;
			}
		}
	}
	std::cout << "importSVG : Downsampling of vertices : " << count << " sommets supprimés." << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////////
	//process broken lines
	CellMarker<EDGE> eMTreated(map) ;
	for (Dart d = map.begin() ; d != map.end() ; map.next(d))
	{
		if (brokenL.isMarked(d) && !eMTreated.isMarked(d))
		{
			eMTreated.mark(d) ;
			//insert a quadrangular face in the broken line
			// -> we convert broken lines to faces to represent their width
			// -> the intersection are then closed

			Dart d1 = d ;
			Dart d2 = map.phi2(d1) ;

			map.unsewFaces(d1) ;
			Dart dN = map.newFace(4) ;

			VEC3 p1 = position[d1] ;
			VEC3 p2 = position[d2] ;
			VEC3 v = p2 - p1 ;
			VEC3 ortho = v ^ VEC3(0, 0, 1);
			float width = edgeWidth[d1] / 2.0f ;
			ortho.normalize() ;
			v.normalize() ;

			//if the valence of one of the vertex is equal to one
			//cut the edge to insert the quadrangular face
			if(map.vertexDegree(d1)==2)
			{
				map.cutEdge(d2) ;
				brokenL.mark(map.phi1(d2)) ;
				eMTreated.mark(map.phi1(d2)) ;
				map.sewFaces(map.phi_1(d1), map.phi1(dN)) ;
				obstacleMark.mark(map.phi_1(d1)) ;
				position[map.phi_1(d1)] = p1 ;
				edgePlanes[map.phi_1(d1)] = Geom::Plane3D<typename PFP::REAL>(v, p1) ;
			}

			if(map.vertexDegree(d2)==2)
			{
				map.cutEdge(d1) ;
				brokenL.mark(map.phi1(d1)) ;
				eMTreated.mark(map.phi1(d1)) ;
				map.sewFaces(map.phi_1(d2), map.phi_1(dN)) ;
				obstacleMark.mark(map.phi_1(d2)) ;
				position[map.phi_1(d2)] = p2 ;
				edgePlanes[map.phi_1(d2)] = Geom::Plane3D<typename PFP::REAL>(-1.0f * v, p2) ;
			}

			map.sewFaces(d1, dN) ;
			obstacleMark.mark(d1) ;
			edgePlanes[d1] = Geom::Plane3D<typename PFP::REAL>(ortho, p1 - (width * ortho)) ;

			map.sewFaces(d2, map.phi1(map.phi1(dN))) ;
			obstacleMark.mark(d2) ;
			edgePlanes[d2] = Geom::Plane3D<typename PFP::REAL>(-1.0f * ortho, p2 + (width * ortho)) ;
		}
	}

	if(allBrokenLines.size()>0)
	{
		for (Dart d = map.begin() ; d != map.end() ; map.next(d))
		{
			if(map.isBoundaryMarked2(d))
			{
				map.fillHole(d);
			}

			if(map.faceDegree(d)==2)
			{
				map.mergeFaces(d);
			}
		}

		//embed the path
		for (Dart d = map.begin() ; d != map.end() ; map.next(d))
		{
			if (brokenL.isMarked(d))
			{
				VEC3 pos;

				Geom::Plane3D<typename PFP::REAL> pl;
				pos = position[d] ;

				pl = edgePlanes[d] ;
				pl.project(pos) ;
				position[d] = pos ;

				pos = position[map.phi1(d)] ;
				pl.project(pos) ;
				position[map.phi1(d)] = pos ;
			}
		}

		map.template initAllOrbitsEmbedding<FACE>(true);


		for (Dart d = map.begin() ; d != map.end() ; map.next(d))
		{
			if (!map.isBoundaryMarked2(d) && brokenL.isMarked(d))
			{
				map.deleteFace(d,false);
			}
		}

		map.closeMap();

		for (Dart d = map.begin() ; d != map.end() ; map.next(d))
		{
			if (map.isBoundaryMarked2(d))
				buildingMark.mark(d);
		}

	}

	return true ;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN
