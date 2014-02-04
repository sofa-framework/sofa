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

#include "Algo/Modelisation/subdivision3.h"
#include "Topology/generic/traversor3.h"
#include "Algo/Modelisation/subdivision.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Modelisation
{

namespace Tetrahedralization
{

//template <typename PFP>
//void hexahedronToTetrahedron(typename PFP::MAP& map, Dart d)
//{
//	Dart d1 = d;
//	Dart d2 = map.phi1(map.phi1(d));
//	Dart d3 = map.phi_1(map.phi2(d));
//	Dart d4 = map.phi1(map.phi1(map.phi2(map.phi_1(d3))));
//
//	Algo::Modelisation::cut3Ear<PFP>(map,d1);
//	Algo::Modelisation::cut3Ear<PFP>(map,d2);
//	Algo::Modelisation::cut3Ear<PFP>(map,d3);
//	Algo::Modelisation::cut3Ear<PFP>(map,d4);
//}
//
//template <typename PFP>
//void hexahedronsToTetrahedrons(typename PFP::MAP& map)
//{
//    TraversorV<typename PFP::MAP> tv(map);
//
//    //for each vertex
//    for(Dart d = tv.begin() ; d != tv.end() ; d = tv.next())
//    {
//        bool vertToTet=true;
//        std::vector<Dart> dov;
//        dov.reserve(32);
//        FunctorStore fs(dov);
//        map.foreach_dart_of_vertex(d,fs);
//        CellMarkerStore<VOLUME> cmv(map);
//
//        //check if all vertices degree is equal to 3 (= no direct adjacent vertex has been split)
//        for(std::vector<Dart>::iterator it=dov.begin();vertToTet && it!=dov.end();++it)
//        {
//            if(!cmv.isMarked(*it) && !map.isBoundaryMarked3(*it))
//            {
//                cmv.mark(*it);
//                vertToTet = (map.phi1(map.phi2(map.phi1(map.phi2(map.phi1(map.phi2(*it))))))==*it); //degree = 3
//            }
//        }
//
//        //if ok : create tetrahedrons around the vertex
//        if(vertToTet)
//        {
//            for(std::vector<Dart>::iterator it=dov.begin();it!=dov.end();++it)
//            {
//                if(cmv.isMarked(*it) && !map.isBoundaryMarked3(*it))
//                {
//                    cmv.unmark(*it);
//                    cut3Ear<PFP>(map,*it);
//                }
//            }
//        }
//    }
//}
//
//template <typename PFP>
//void tetrahedrizeVolume(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position)
//{
//	//mark bad edges
//	DartMarkerStore mBadEdge(map);
//
//	std::vector<Dart> vEdge;
//	vEdge.reserve(1024);
//
////	unsignzed int i = 0;
//
//	unsigned int nbEdges = map.template getNbOrbits<EDGE>();
//	unsigned int i = 0;
//
//	for(Dart dit = map.begin() ; dit != map.end() ; map.next(dit))
//	{
//		//check if this edge is an "ear-edge"
//		if(!mBadEdge.isMarked(dit))
//		{
//			++i;
//			std::cout << i << " / " << nbEdges << std::endl;
//
//			//search three positions
//			typename PFP::VEC3 tris1[3];
//			tris1[0] = position[dit];
//			tris1[1] = position[map.phi_1(dit)];
//			tris1[2] = position[map.phi_1(map.phi2(dit))];
//
//			//search if the triangle formed by these three points intersect the rest of the mesh (intersection triangle/triangle)
//			TraversorF<typename PFP::MAP> travF(map);
//			for(Dart ditF = travF.begin() ; ditF != travF.end() ; ditF = travF.next())
//			{
//				//get vertices position
//				typename PFP::VEC3 tris2[3];
//				tris2[0] = position[ditF];
//				tris2[1] = position[map.phi1(ditF)];
//				tris2[2] = position[map.phi_1(ditF)];
//
//				bool intersection = false;
//
//				for (unsigned int i = 0; i < 3 && !intersection; ++i)
//				{
//					typename PFP::VEC3 inter;
//					intersection = Geom::intersectionSegmentTriangle(tris1[i], tris1[(i+1)%3], tris2[0], tris2[1], tris2[2], inter);
//				}
//
//				if(!intersection)
//				{
//					for (unsigned int i = 0; i < 3 && !intersection; ++i)
//					{
//						typename PFP::VEC3 inter;
//						intersection = Geom::intersectionSegmentTriangle(tris2[i], tris2[(i+1)%3], tris1[0], tris1[1], tris1[2], inter);
//					}
//				}
//
//				//std::cout << "intersection ? " << (intersection ? "true" : "false") << std::endl;
//
//				if(intersection)
//				{
//					mBadEdge.markOrbit<EDGE>(dit);
//				}
//				else //cut a tetrahedron
//				{
//					vEdge.push_back(dit);
//				}
//
//
////
////				if(i == 16)
////					return;
//			}
//		}
//	}
//
//	std::cout << "nb edges to split = " << vEdge.size() << std::endl;
//	i = 0;
//	for(std::vector<Dart>::iterator it = vEdge.begin() ; it != vEdge.end() ; ++it)
//	{
//		++i;
//		std::cout << i << " / " << vEdge.size() << std::endl;
//
//		Dart dit = *it;
//
//		//std::cout << "cut cut " << std::endl;
//		std::vector<Dart> vPath;
//
//		vPath.push_back(map.phi1(dit));
//		vPath.push_back(map.phi1(map.phi2(map.phi_1(dit))));
//		vPath.push_back(map.phi_1(map.phi2(dit)));
//
//		map.splitVolume(vPath);
//
//		map.splitFace(map.phi2(map.phi1(dit)), map.phi2(map.phi1(map.phi2(dit))));
//	}
//
//	std::cout << "finished " << std::endl;
//}


/************************************************************************************************
 * 									Collapse / Split Operators
 ************************************************************************************************/
template <typename PFP>
Dart splitVertex(typename PFP::MAP& map, std::vector<Dart>& vd)
{
    //split the vertex
    Dart dres = map.splitVertex(vd);

    //split the faces incident to the new vertex
    Dart dbegin = map.phi1(map.phi2(vd.front()));
    Dart dit = dbegin;
    do
    {
        map.splitFace(map.phi1(dit),map.phi_1(dit));
        dit = map.alpha2(dit);
    }
    while(dbegin != dit);

    //split the volumes incident to the new vertex
    for(unsigned int i = 0; i < vd.size(); ++i)
    {
        Dart dit = vd[i];

        std::vector<Dart> v;
        v.push_back(map.phi1(map.phi1(map.phi2(dit))));
        std::cout << "[" << v.back();
        v.push_back(map.phi1(dit));
        std::cout << " - " << v.back();
        v.push_back(map.phi1(map.phi2(map.phi_1(dit))));
        std::cout << " - " << v.back() << "]" << std::endl;
        map.splitVolume(v);
    }

    return dres;
}

/*************************************************************************************************
 *		 								Tetrahedron functions									 *
 *************************************************************************************************/

template <typename PFP>
bool isTetrahedron(typename PFP::MAP& the_map, Dart d, unsigned int thread)
{
    unsigned int nbFaces = 0;

    //Test the number of faces end its valency
    Traversor3WF<typename PFP::MAP> travWF(the_map, d, false, thread);
    for(Dart dit = travWF.begin() ; dit != travWF.end(); dit = travWF.next())
    {
        //increase the number of faces
        nbFaces++;
        if(nbFaces > 4)	//too much faces
            return false;

        //test the valency of this face
        if(the_map.faceDegree(dit) != 3)
            return false;
    }

    return true;
}

template <typename PFP>
bool isTetrahedralization(typename PFP::MAP& map)
{
    TraversorW<typename PFP::MAP> travW(map);
    for(Dart dit = travW.begin() ; dit != travW.end() ; dit = travW.next())
    {
        if(!isTetrahedron<PFP>(map, dit))
            return false;
    }

    return true;
}

/***********************************************************************************************
 * 										swap functions										   *
 ***********************************************************************************************/

template <typename PFP>
Dart swap2To2(typename PFP::MAP& map, Dart d)
{
    std::vector<Dart> edges;

    Dart d2_1 = map.phi_1(map.phi2(d));
    map.mergeVolumes(d);
    map.mergeFaces(map.phi1(d2_1));
    map.splitFace(d2_1, map.phi1(map.phi1(d2_1)));

        Dart stop = map.phi_1(d2_1);
        Dart dit = stop;
        do
        {
            edges.push_back(dit);
            dit = map.phi1(map.phi2(map.phi1(dit)));
        }
        while(dit != stop);

        map.splitVolume(edges);

    return map.phi2(stop);
}

template <typename PFP>
void swap4To4(typename PFP::MAP& map, Dart d)
{
    Dart e = map.phi2(map.phi3(d));
    Dart dd = map.phi2(d);

    //unsew middle crossing darts
    map.unsewVolumes(d);
    map.unsewVolumes(map.phi2(map.phi3(dd)));

    Dart d1 = Tetrahedralization::swap2To2<PFP>(map, dd);
    Dart d2 = Tetrahedralization::swap2To2<PFP>(map, e);

    //sew middle darts so that they do not cross
    map.sewVolumes(map.phi2(d1),map.phi2(map.phi3(d2)));
    map.sewVolumes(map.phi2(map.phi3(d1)),map.phi2(d2));
}

template <typename PFP>
Dart swap3To2(typename PFP::MAP& map, Dart d)
{
    std::vector<Dart> edges;

    Dart stop = map.phi_1(map.phi2(map.phi1(d)));
    Dart d2 = map.phi2(d);
    Dart d21 = map.phi1(d2);
    map.mergeVolumes(d);
    map.mergeFaces(d2);
    map.mergeVolumes(d21);

    Dart dit = stop;
    do
    {
        edges.push_back(dit);
        dit = map.phi1(map.phi2(map.phi1(dit)));
    }
    while(dit != stop);
    map.splitVolume(edges);

    return map.phi3(stop);
}

//[precond] le brin doit venir d'une face partagé par 2 tetraèdres
// renvoie un brin de l'ancienne couture entre les 2 tetras qui est devenu une arête
template <typename PFP>
Dart swap2To3(typename PFP::MAP& map, Dart d)
{
    std::vector<Dart> edges;

    Dart d2_1 = map.phi_1(map.phi2(d));
    map.mergeVolumes(d);

    //
    // Cut the 1st tetrahedron
    //
    Dart stop = d2_1;
    Dart dit = stop;
    do
    {
        edges.push_back(dit);
        dit = map.phi1(map.phi2(map.phi1(dit)));
    }
    while(dit != stop);

    map.splitVolume(edges);
    map.splitFace(map.alpha2(edges[0]), map.alpha2(edges[2]));

    //
    // Cut the 2nd tetrahedron
    //
    edges.clear();
    stop = map.phi1(map.phi2(d2_1));
    dit = stop;
    do
    {
        edges.push_back(dit);
        dit = map.phi1(map.phi2(map.phi1(dit)));
    }
    while(dit != stop);
    map.splitVolume(edges);

    return map.phi1(d2_1);
}

template <typename PFP>
Dart swap5To4(typename PFP::MAP& map, Dart d)
{
    Dart t1 = map.phi3(d);
    Dart t2 = map.phi3(map.phi2(d));

    Dart d323 = map.phi_1(map.phi2(map.phi1(d)));
    Dart dswap = map.phi2(map.phi3(d323));

    map.unsewVolumes(t1);
    map.unsewVolumes(t2);
    map.unsewVolumes(d323);
    map.unsewVolumes(map.phi2(d323));
    map.deleteVolume(d);

    Dart d1 = Tetrahedralization::swap2To2<PFP>(map, dswap);

    map.sewVolumes(map.phi2(d1), t1);
    map.sewVolumes(map.phi2(map.phi3(d1)),t2);

    return t1;
}

template <typename PFP>
void swapGen3To2(typename PFP::MAP& map, Dart d)
{
    unsigned int n = map.edgeDegree(d);


    if(n >= 4)
    {
        Dart dit = d;
        for(unsigned int i = 0 ; i < n - 4 ; ++i)
        {
            dit = map.phi2(Tetrahedralization::swap2To3<PFP>(map, dit));
        }
        Tetrahedralization::swap4To4<PFP>(map,  map.alpha2(dit));
    }




//	if(n >= 4)
//	{
//		Dart dit = d;
//		if(map.isBoundaryEdge(dit))
//		{
//			for(unsigned int i = 0 ; i < n - 2 ; ++i)
//			{
//				dit = map.phi2(Tetrahedralization::swap2To3<PFP>(map, dit));
//			}
//			Tetrahedralization::swap2To2<PFP>(map, dit);
//		}
//		else
//		{
//			for(unsigned int i = 0 ; i < n - 4 ; ++i)
//			{
//				dit = map.phi2(Tetrahedralization::swap2To3<PFP>(map, dit));
//			}
//			Tetrahedralization::swap4To4<PFP>(map,  map.alpha2(dit));
//		}
//	}
//	else if (n == 3)
//	{
//		Dart dres = Tetrahedralization::swap2To3<PFP>(map, d);
//		Tetrahedralization::swap2To2<PFP>(map, map.phi2(dres));
//	}
//	else // si (n == 2)
//	{
//		Tetrahedralization::swap2To2<PFP>(map, d);
//	}

}

template <typename PFP>
void swapGen2To3(typename PFP::MAP& map, Dart d)
{
//	unsigned int n = map.edgeDegree(d);

//- a single 2-3 swap, followed by n − 3 3-2 swaps, or
//- a single 4-4 swap, followed by n − 4 3-2 swaps.
}




/************************************************************************************************
 *										Flip Functions 											*
 ************************************************************************************************/

template <typename PFP>
Dart flip1To4(typename PFP::MAP& map, Dart d)
{
    std::vector<Dart> edges;

    //
    // Cut the 1st tetrahedron
    //
    edges.push_back(map.phi2(d));
    edges.push_back(map.phi2(map.phi1(d)));
    edges.push_back(map.phi2(map.phi_1(d)));
    map.splitVolume(edges);

    Dart x = Surface::Modelisation::trianguleFace<PFP>(map,map.phi2(d));

    //
    // Cut the 2nd tetrahedron
    //
    Dart dit = map.phi2(map.phi3(x));
    edges.clear();
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);
    dit = map.phi1(dit);
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);

    map.splitVolume(edges);
    map.splitFace(map.phi1(map.phi2(edges[0])),map.phi1(map.phi2(edges[2])));

    //
    // Cut the 3rd tetrahedron
    //
    dit = map.phi3(map.phi1(map.phi2(edges[0])));
    edges.clear();
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);

    map.splitVolume(edges);

    return x;
}

template <typename PFP>
Dart flip1To3(typename PFP::MAP& map, Dart d)
{
    std::vector<Dart> edges;

    //
    // Triangule one face
    //
    Dart x = Surface::Modelisation::trianguleFace<PFP>(map,d);

    //
    // Cut the 1st Tetrahedron
    //
    Dart dit = x;
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);

    map.splitVolume(edges);

    // Cut the 2nd Tetrahedron
    map.splitFace(map.phi1(map.phi2(edges[0])),map.phi1(map.phi2(edges[2])));

    // Cut the 3rd Tetrahedron
    dit = map.phi1(map.phi2(edges[0]));
    edges.clear();
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);
    dit = map.phi1(map.phi2(map.phi1(dit)));
    edges.push_back(dit);

    map.splitVolume(edges);

    return x;
}


/************************************************************************************************
 *                				 Bisection Functions                                            *
 ************************************************************************************************/

template <typename PFP>
Dart edgeBisection(typename PFP::MAP& map, Dart d)
{
    //coupe l'arete en 2
    map.cutEdge(d);
    Dart e = map.phi1(d);

    Dart dit = e;
    do
    {
        map.splitFace(dit, map.phi1(map.phi1(dit)));
        dit = map.alpha2(dit);
    }
    while(dit != e);

    dit = e;
    std::vector<Dart> edges;
    do
    {
        if(!map.isBoundaryMarked3(dit))
        {
            edges.push_back(map.phi_1(dit));
            edges.push_back(map.phi_1(map.phi2(map.phi_1(edges[0]))));
            edges.push_back(map.phi1(map.phi2(dit)));
            map.splitVolume(edges);
            edges.clear();
        }
        dit = map.alpha2(dit);
    }
    while(dit != e);

    return e;
}



//namespace Tetgen
//{


//template <typename PFP>
//bool tetrahedralize(const typename PFP::MAP2& map2, const VertexAttribute<typename PFP::VEC3> position2,
//                    typename PFP::MAP3& map3, VertexAttribute<typename PFP::VEC3> position3,
//                    bool add_steiner_points_on_exterior_boundary, bool add_steiner_points_on_interior_boundary, double max_volume, double max_shape)
//{
//    //
//    // 1. map to tetgen
//    //

//    tetgenio surface;

//    // memory initialization
//    surface.initialize();

//    // 0-based indexing
//    surface.firstnumber = 0;

//    // input vertices
//    surface.numberofpoints = map2.nbOrbits<VERTEX>();
//    surface.pointlist = new REAL[surface.numberofpoints * 3];

//    //for each vertex
//    unsigned int i = 0;
//    TraversorV tv(map2);
//    for(Dart it = tv.begin() ; it != tv.end() ; it = tv.next())
//    {
//        surface.pointlist[i] = position2[it][0] ; i++ ; //x
//        surface.pointlist[i] = position2[it][1] ; i++ ; //y
//        surface.pointlist[i] = position2[it][2] ; i++ ; //z
//    }

//    tetgenio::facet* f ;
//    tetgenio::polygon* p ;
//    surface.numberoffacets = map2.nbOrbits<FACE>();
//    surface.facetlist = new tetgenio::facet[surface.numberoffacets] ;


//    //for each facet
//    i = 0;
//    TraversorF tf(map2);
//    for(Dart it = tf.begin() ; it != tf.end() ; it = tf.next())
//    {
//        f = &(surface.facetlist[i]) ;
//        f->numberofpolygons = 1 ;
//        f->polygonlist = new tetgenio::polygon[f->numberofpolygons] ;
//        p = f->polygonlist ;
//        p->numberofvertices = map2.faceDegree(it);
//        p->vertexlist = new int[p->numberofvertices] ;

//        unsigned int j = 0;
//        Dart dit = it;
//        do
//        {
//            p->vertexlist[j] = map2.getEmbedding<VERTEX>(dit);
//            dit = map.phi1(dit);
//            j++;
//        }while(dit != it);

//        f->numberofholes = 0 ;
//        f->holelist = nil ;
//        i++ ;
//    }

//    //
//    // 2. tetgen argument list
//    //
//    std::ostringstream s ;

//    // Q: Quiet: No terminal output except errors
//    // p: PLC : input data is surfacic
//    // n: output tet neighbors

//    // q: desired quality
//    if(max_volume > 0 && max_shape > 0.0)
//    {
//        s << "Qpna" << max_volume << "q"<< max_shape;
//    }
//    else if(max_volume > 0.0)
//    {
//        s << "Qpna" << max_volume ;
//    }
//    else if(max_shape > 0.0)
//    {
//        s << "Qpnq" << max_shape ;
//    }
//    else
//    {
//        s << "Qpn";
//    }

//    // YY: prohibit steiner points on boundaries
//    // (first Y for exterior boundary, second Y for the
//    // other ones).

//    if( add_steiner_points_on_exterior_boundary && !add_steiner_points_on_interior_boundary)
//    {
//       //Invalid combination of flags (do not preserve exterior boundary and preserve interior ones) - preserving exterior boundary as well
//        add_steiner_points_on_exterior_boundary = false ;
//    }

//    if(!add_steiner_points_on_exterior_boundary)
//    {
//        s << "Y" ;
//    }

//    if(!add_steiner_points_on_interior_boundary)
//    {
//        s << "Y" ;
//    }
//    std::string params = s.str() ;

//    //
//    // 3. tetrahedralization
//    //
//    tetgenio volume;
//    ::tetrahedralize(params.c_str(), &surface, &volume) ;


//    //
//    // 4. tetgen to map
//    //

//    //create vertices
//    double* p = volume.pointlist ;
//    std::vector<unsigned int> verticesID;
//    verticesID.reserve(volume.numberofpoints);
//    AttributeContainer& container = map3.template getAttributeContainer<VERTEX>() ;

//    for(unsigned int i = 0; i < volume.numberofpoints; i++)
//    {
//        typename PFP::VEC3 pos(p[0], p[1], p[2]);
//        unsigned int id = container.insertLine();

//        position3[id] = pos;
//        verticesID.push_back(id);

//        p += 3 ;
//    }

//    //create tetrahedrons
//    int* t = volume.tetrahedronlist ;
//    for(unsigned int i = 0; i < volume.numberoftetrahedra; i++)
//    {
//        Dart d = Algo::Surface::Modelisation::createTetrahedron<PFP>(map3, false);

//        for(unsigned int j = 0; j < 3; j++)
//        {
//            FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, verticesID[t[j] - volume.firstnumber]);
//            map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

////            //store darts per vertices to optimize reconstruction
////            Dart dd = d;
////            do
////            {
////                m.mark(dd) ;
////                vecDartsPerVertex[pt[2-j]].push_back(dd);
////                dd = map.phi1(map.phi2(dd));
////            } while(dd != d);

//            d = map.phi1(d);

//            set_cell_vertex(d, j, verticesID[t[j] - volume.firstnumber]) ;
//        }

//        t += 4 ;
//    }

//    //create adjacency
//    int* pn = volume.neighborlist ;
//    for(unsigned int i = 0; i < volume.numberoftetrahedra; i++)
//    {
//        for(int j=0; j<4; j++)
//        {
//            int adjacent = pn[j] ;

//            if(adjacent >= 0)
//            {
//                set_cell_adjacent( cells[i], j, cells[adjacent - volume.firstnumber]
//                ) ;
//            }
//        }
//        pn += 4 ;
//    }
//}

///**
// * generate tetrahedra based on an surface mesh object
// */
//template <typename PFP>
//bool process(const std::string& filename, typename PFP::MAP3& map3, bool add_steiner_points_on_exterior_boundary,
//             bool add_steiner_points_on_interior_boundary, double max_volume, double max_shape)
//{

//}





///**
// * generate tetrahedra based on an surface mesh object
// * -INT_MAX for surf/vol-id inherits from the mesh
// */
//template <typename PFP2, typename PFP3>
//typename PFP3::MAP process(typename PFP2::MAP& map, double volume, double quality, int volid = -INT_MAX, int surfid = -INT_MAX)
//{
//    //
//    // map to tetgen
//    //


//    tetgenio surface;
//    tetgenio::facet *f;
//    tetgenio::polygon *p;
//    int *elements, *surfaces;
//    int i, j, n, *ele, *tet;
//    char params [512];





//    // memory initialization
//    surface.initialize();

//    // 0-based indexing
//    surface.firstnumber = 0;

//    // input vertices
//    surface.numberofpoints =
//    surface.pointlist = new REAL[surface.numberofpoints * 3];


//    // input faces

//}

///**
// * generate tetrahedra based on an surface mesh object
// * -INT_MAX for surf/vol-id inherits from the mesh
// */
//template <typename PFP3>
//typename PFP3::MAP process(const std::string& filename, double volume, double quality, int volid = -INT_MAX, int surfid = -INT_MAX)
//{

//}

//} //namespace Tetgen



































///**
// * create a tetra based on the two triangles that have a common dart and phi2(dart)
// * return a new dart inside the tetra
// */
//template<typename PFP>
//Dart extractTetra(typename PFP::MAP& the_map, Dart d)
//{
//
//
//	Dart e = the_map.phi2(d);
//
//	//create the new faces
//	Dart dd = the_map.newFace(3);
//	Dart ee = the_map.newFace(3);
//
//	//update their sew
//	the_map.sewFaces(dd,ee);
//	the_map.sewFaces(the_map.phi3(dd),the_map.phi3(ee));
//
//	//add the two new faces in the mesh to obtain a tetra
//	Dart s2d = the_map.phi2(the_map.phi_1(d));
//	the_map.unsewFaces(the_map.phi_1(d));
//	the_map.sewFaces(the_map.phi_1(d),the_map.phi_1(dd));
//	the_map.sewFaces(s2d,the_map.phi3(the_map.phi_1(dd)));
//
//	Dart s2e = the_map.phi2(the_map.phi_1(e));
//	the_map.unsewFaces(the_map.phi_1(e));
//	the_map.sewFaces(the_map.phi_1(e),the_map.phi_1(ee));
//	the_map.sewFaces(s2e,the_map.phi3(the_map.phi_1(ee)));
//
//	Dart ss2d = the_map.phi2(the_map.phi1(d));
//	the_map.unsewFaces(the_map.phi1(d));
//	the_map.sewFaces(the_map.phi1(d),the_map.phi1(ee));
//	the_map.sewFaces(ss2d,the_map.phi3(the_map.phi1(ee)));
//
//	Dart ss2e = the_map.phi2(the_map.phi1(e));
//	the_map.unsewFaces(the_map.phi1(e));
//	the_map.sewFaces(the_map.phi1(e),the_map.phi1(dd));
//	the_map.sewFaces(ss2e,the_map.phi3(the_map.phi1(dd)));
//
//	//embed the coords
//	the_map.setVertexEmb(d,the_map.getVertexEmb(d));
//	the_map.setVertexEmb(e,the_map.getVertexEmb(e));
//	the_map.setVertexEmb(the_map.phi_1(d),the_map.getVertexEmb(the_map.phi_1(d)));
//	the_map.setVertexEmb(the_map.phi_1(e),the_map.getVertexEmb(the_map.phi_1(e)));
//
//	return dd;
//}
//
///**
// * tetrahedrization of the volume
// * @param the map
// * @param a dart of the volume
// * @param true if the faces are in CCW order
// * @return success of the tetrahedrization
// */
//template<typename PFP>
//bool smartVolumeTetrahedrization(typename PFP::MAP& the_map, Dart d, bool CCW=true)
//{
//
//	typedef typename PFP::EMB EMB;
//
//	bool ret=true;
//
//	if (!the_map.isTetrahedron(d))
//	{
//		//only works on a 3-map
//		assert(Dart::nbInvolutions()>=2 || "cannot be applied on this map, nbInvolutions must be at least 2");
//
//		if (Geometry::isConvex<PFP>(the_map,d,CCW))
//		{
//			the_map.tetrahedrizeVolume(d);
//		}
//		else
//		{
//
//			//get all the dart of the volume
//			std::vector<Dart> vStore;
//			FunctorStore fs(vStore);
//			the_map.foreach_dart_of_volume(d,fs);
//
//			if (vStore.size()==0)
//			{
//				if (the_map.phi1(d)==d)
//					CGoGNout << "plop" << CGoGNendl;
//				if (the_map.phi2(d)==d)
//					CGoGNout << "plip" << CGoGNendl;
//
//				CGoGNout << the_map.getVertexEmb(d)->getPosition() << CGoGNendl;
//				CGoGNout << "tiens tiens, c'est etrange" << CGoGNendl;
//			}
//			//prepare the list of embeddings of the current volume
//			std::vector<EMB *> lstEmb;
//
//			//get a marker
//			DartMarker m(the_map);
//
//			//all the darts from a vertex that can generate a tetra (3 adjacent faces)
//			std::vector<Dart> allowTetra;
//
//			//all the darts that are not in otherTetra
//			std::vector<Dart> otherTetra;
//
//			//for each dart of the volume
//			for (typename std::vector<Dart>::iterator it = vStore.begin() ; it != vStore.end() ; ++it )
//			{
//				Dart e = *it;
//				//if the vertex is not treated
//				if (!m.isMarked(e))
//				{
//					//store the embedding
//					lstEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(e)));
//					Dart ee=e;
//
//					//count the number of adjacent faces and mark the darts
//					int nbe=0;
//					do
//					{
//						nbe++;
//						m.markOrbit(DART,e);
//						ee=the_map.phi1(the_map.phi2(ee));
//					}
//					while (ee!=e);
//
//					//if 3 adjacents faces, we can create a tetra on this vertex
//					if (nbe==3)
//						allowTetra.push_back(e);
//					else
//						otherTetra.push_back(e);
//				}
//			}
//
//			//we haven't created a tetra yet
//			bool decoupe=false;
//
//			//if we have vertex that can be base
//			if (allowTetra.size()!=0)
//			{
//				//foreach possible vertex while we haven't done any cut
//				for (typename std::vector<Dart>::iterator it=allowTetra.begin();it!=allowTetra.end() && !decoupe ;++it)
//				{
//					//get the dart
//					Dart s=*it;
//					//store the emb
//					std::vector<EMB*> lstCurEmb;
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(s)));
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(the_map.phi1(s))));
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(the_map.phi_1(s))));
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(the_map.phi_1(the_map.phi2(s)))));
//
//					//store the coords of the point
//					gmtl::Vec3f points[4];
//					for (int i=0;i<4;++i)
//					{
//						points[i] = lstCurEmb[i]->getPosition();
//					}
//
//					//test if the future tetra is well oriented (concave case)
//					if (Geometry::isTetrahedronWellOriented(points,CCW))
//					{
//						//test if we haven't any point inside the future tetra
//						bool isEmpty=true;
//						for (typename std::vector<EMB *>::iterator iter = lstEmb.begin() ; iter != lstEmb.end() && isEmpty ; ++iter)
//						{
//							//we don't test the vertex that composes the new tetra
//							if (std::find(lstCurEmb.begin(),lstCurEmb.end(),*iter)==lstCurEmb.end())
//							{
//								isEmpty = !Geometry::isPointInTetrahedron(points, (*iter)->getPosition(), CCW);
//							}
//						}
//
//						//if no point inside the new tetra
//						if (isEmpty)
//						{
//							//cut the spike to make a tet
//							Dart dRes = the_map.cutSpike(*it);
//							decoupe=true;
//							//and continue with the rest of the volume
//							ret = ret && smartVolumeTetrahedrization<PFP>(the_map,the_map.phi3(dRes),CCW);
//						}
//					}
//				}
//			}
//
//			if (!decoupe)
//			{
//				//foreach other vertex while we haven't done any cut
//				for (typename std::vector<Dart>::iterator it=otherTetra.begin();it!=otherTetra.end() && !decoupe ;++it)
//				{
//					//get the dart
//					Dart s=*it;
//					//store the emb
//					std::vector<EMB*> lstCurEmb;
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(s)));
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(the_map.phi1(s))));
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(the_map.phi_1(s))));
//					lstCurEmb.push_back(reinterpret_cast<EMB*>(the_map.getVertexEmb(the_map.phi_1(the_map.phi2(s)))));
//
//					//store the coords of the point
//					gmtl::Vec3f points[4];
//					for (int i=0;i<4;++i)
//					{
//						points[i] = lstCurEmb[i]->getPosition();
//					}
//
//					//test if the future tetra is well oriented (concave case)
//					if (Geometry::isTetrahedronWellOriented(points,CCW))
//					{
//						//test if we haven't any point inside the future tetra
//						bool isEmpty=true;
//						for (typename std::vector<EMB *>::iterator iter = lstEmb.begin() ; iter != lstEmb.end() && isEmpty ; ++iter)
//						{
//							//we don't test the vertex that composes the new tetra
//							if (std::find(lstCurEmb.begin(),lstCurEmb.end(),*iter)==lstCurEmb.end())
//							{
//								isEmpty = !Geometry::isPointInTetrahedron(points, (*iter)->getPosition(), CCW);
//							}
//						}
//
//						//if no point inside the new tetra
//						if (isEmpty)
//						{
//							//cut the spike to make a tet
//							Dart dRes = extractTetra<PFP>(the_map,*it);
//							decoupe=true;
//							//and continue with the rest of the volume
//							smartVolumeTetrahedrization<PFP>(the_map,the_map.phi3(dRes),CCW);
//						}
//					}
//				}
//			}
//
//			if (!decoupe)
//				ret=false;
//		}
//	}
//	return ret;
//}

} // namespace Tetrahedralization

}

} // namespace Modelisation

} // namespace Algo

} // namespace CGoGN
