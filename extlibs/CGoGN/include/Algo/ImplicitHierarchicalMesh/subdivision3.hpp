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

#include "Algo/Geometry/centroid.h"
#include "Algo/Modelisation/subdivision.h"
#include "Algo/Modelisation/extrusion.h"
#include "Topology/generic/dartmarker.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace IHM
{



template <typename PFP>
void subdivideEdge(typename PFP::MAP& map, Dart d, typename AttributeHandler_Traits< typename PFP::VEC3, VERTEX, typename PFP::MAP >::Handler&position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int eLevel = map.edgeLevel(d) ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(eLevel) ;

    Dart dd = map.phi2(d) ;
    typename PFP::VEC3 p1 = position[d] ;
    typename PFP::VEC3 p2 = position[map.phi1(d)] ;

    map.setCurrentLevel(eLevel + 1) ;
    map.cutEdge(d) ;
    unsigned int eId = map.getEdgeId(d) ;
    map.setEdgeId(map.phi1(d), eId, EDGE) ;

    //mise a jour de l'id d'arrete sur chaque moitie d'arete
    map.setEdgeId(map.phi1(dd), eId, EDGE) ;
    map.setFaceId(EDGE, d) ; //mise a jour de l'id de face sur chaque brin de chaque moitie d'arete
    map.setFaceId(EDGE, dd) ;

    position[map.phi1(d)] = (p1 + p2) * typename PFP::REAL(0.5) ;
    //map.computeEdgeVertexFunctor(map.phi1(d));
    map.setCurrentLevel(cur) ;
//    map.checkEdgeAndFaceIDAttributes();
}

template <typename PFP>
void subdivideFace(typename PFP::MAP& map, Dart d, typename AttributeHandler_Traits< typename PFP::VEC3, VERTEX, typename PFP::MAP >::Handler& position, SubdivideType sType)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");
    typedef typename PFP::MAP MAP;
    const unsigned int fLevel = map.faceLevel(d) ;
    const unsigned int oldEmb = map.template getEmbedding<FACE>(d);
    Dart old = map.faceOldestDart(d) ;
    const unsigned int cur = map.getCurrentLevel() ;
    std::cerr << "currLvl is " << cur << ", subdividing a face of lvl " << fLevel << std::endl;
    map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

    std::vector< Dart > newFaces;
    newFaces.push_back(old);

    unsigned int degree = 0 ;
    typename PFP::VEC3 p ;
    Traversor2FE<typename PFP::MAP>  travE(map, old);
    for(Dart it = travE.begin(); it != travE.end() ; it = travE.next())
    {
        ++degree;
        p += position[it] ;
        if (!map.edgeIsSubdivided(it))
        {
            IHM::subdivideEdge<PFP>(map, it, position) ;// first cut the edges (if they are not already)
            // and compute the degree of the face
        } else
        {
        }
    }
    p /= typename PFP::REAL(degree) ;

//    map.checkEdgeAndFaceIDAttributes();
    map.setCurrentLevel(fLevel + 1) ;			// go to the next level to perform face subdivision

    if(degree == 3 && sType == IHM::S_TRI)	//subdiviser une face triangulaire
    {
        Dart dd = map.phi1(old) ;
        Dart e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;					// insert a new edge

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;
    }
    else
    {
        Dart dd = map.phi1(old) ;
        map.splitFace(dd, map.phi1(map.phi1(dd))) ;
//        std::cerr << "faceLevel(dd) " << map.faceLevel(dd) << std::endl;
//        std::cerr << "faceLevel(map.phi1(map.phi1(dd)) " << map.faceLevel(map.phi1(map.phi1(dd))) << std::endl;
        newFaces.push_back(dd);
        const Dart ne = map.phi2(map.phi_1(dd));
        Dart ne2 = map.phi2(ne);

        map.cutEdge(ne) ;
        const unsigned id = map.getNewEdgeId() ;
        map.setEdgeId(ne2, id, EDGE) ;

        position[map.phi1(ne)] = p ;

        dd = map.phi1(map.phi1(map.phi1(map.phi1(ne)))) ;
        while(dd != ne)
        {
            const Dart next = map.phi1(map.phi1(dd)) ;
            newFaces.push_back(map.phi1(ne));
            map.splitFace(map.phi1(ne), dd) ;

//            std::cerr << "faceLevel(map.phi1(ne)) " << map.faceLevel(map.phi1(ne)) << std::endl;
//            std::cerr << "faceLevel(dd) " << map.faceLevel(dd) << std::endl;
            dd = next ;
        }

    }

    // update the MR-attribute
    for (std::vector< Dart >::iterator dit = newFaces.begin() ; dit != newFaces.end() ; ++dit)
    {
        const Dart f = (*dit);
        const unsigned lvl1Face = map.template newCell<FACE>(); // the new emb for darts of lvl equals to flevel +1
        {
            map.setCurrentLevel(map.getMaxLevel());
            TraversorDartsOfOrbit< MAP, FACE > traDoF(map, f);
            for (Dart fit = traDoF.begin(); fit != traDoF.end() ; fit = traDoF.next())
            {
                map.setMaxFaceLevel(fit, fLevel + 1u);
                const unsigned int dl = map.getDartLevel(fit);
                if ( dl > fLevel)
                {
                    assert(dl == fLevel +1u);
                    map.setCurrentLevel(dl);
                    map.template setDartEmbedding<FACE>(fit, lvl1Face);
                    map.setCurrentLevel(map.getMaxLevel());
                }
            }
            map.setCurrentLevel(fLevel + 1) ;
        }
        map.template getAttributeContainer<FACE>().copyLine(lvl1Face, oldEmb);
        map.setFaceLevel(f, fLevel+1);
        map.checkEmbedding(FaceCell(f));
    }

    map.template checkAllEmbeddingsOfOrbit<FACE>();
    map.template checkAllEmbeddingsOfOrbit<VOLUME>();

//    map.checkEdgeAndFaceIDAttributes();
    map.setCurrentLevel(cur) ;
}

template <typename PFP>
Dart subdivideVolumeClassic(typename PFP::MAP& map, Dart d, typename AttributeHandler_Traits< typename PFP::VEC3, VERTEX, typename PFP::MAP >::Handler& position, bool OneLevelDifference)
{
    typedef typename PFP::MAP MAP;
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    std::cerr << "subdivideVolumeClassic called at lvl " << map.getCurrentLevel() << std::endl;
    const unsigned int vLevel = map.volumeLevel(d);
    std::cerr << "level of the volume is " << vLevel << std::endl;
    const Dart old = map.volumeOldestDart(d);

    const unsigned int cur = map.getCurrentLevel();
    map.setCurrentLevel(vLevel);


    /*
     * au niveau du volume courant i
     * stockage d'un brin de chaque face de celui-ci
     * avec calcul du centroid
     */

    DartMarker<MAP> mf(map);		// Lock a face marker to save one dart per face
    DartMarker<MAP> mv(map);

    typename PFP::VEC3 volCenter;
    unsigned count = 0 ;

    //Store faces that are traversed and start with the face of d
    std::vector<Dart> visitedFaces;
    visitedFaces.reserve(200);
    visitedFaces.push_back(old);

    //Store the edges before the cutEdge
    std::vector<Dart> oldEdges;
    oldEdges.reserve(20);

    mf.markOrbit(Face(old)) ;

    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
    {
        Dart e = visitedFaces[i] ;
        do
        {
            //add one old edge per vertex to the old edge list
            //compute volume centroid
            if(!mv.isMarked(e))
            {
                mv.markOrbit(Vertex(e));
                volCenter += position[e];
                ++count;
                oldEdges.push_back(e);
            }

            // add all face neighbours to the table
            Dart ee = map.phi2(e) ;
            if(!mf.isMarked(ee)) // not already marked
            {
                visitedFaces.push_back(ee) ;
                mf.markOrbit(Face(ee)) ;
            }

            e = map.phi1(e) ;
        } while(e != visitedFaces[i]) ;
    }

    assert(visitedFaces.size() == 6);

    volCenter /= typename PFP::REAL(count) ;

    /*
     * Subdivision
     */

    //Store the darts from quadrangulated faces
    std::vector<std::pair<Dart,Dart> > subdividedfaces;
    subdividedfaces.reserve(32);
//    map.checkEdgeAndFaceIDAttributes();
    //First step : subdivide edges and faces
    //creates a i+1 edge level and i+1 face level
    for (std::vector<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
    {
        Dart d = *face;

        //if needed subdivide face
        if(!map.faceIsSubdivided(d))
        {
            IHM::subdivideFace<PFP>(map, d, position, IHM::S_QUAD);
        }
        map.checkEdgeAndFaceIDAttributes();

        //save a dart from the subdivided face
        const unsigned int cur = map.getCurrentLevel() ;

        const unsigned int fLevel = map.faceLevel(d) + 1; //puisque dans tous les cas, la face est subdivisee
        map.setCurrentLevel(fLevel) ;

        //le brin est forcement du niveau cur
        Dart cf = map.phi1(d);
        Dart e = cf;
        do
        {
            subdividedfaces.push_back(std::pair<Dart,Dart>(e,map.phi2(e)));
            e = map.phi2(map.phi1(e));
        }while (e != cf);

        map.setCurrentLevel(cur);
    }

    map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

    std::vector<Dart> newEdges;	//save darts from inner edges
    newEdges.reserve(50);

    int next_edge = 0;
    Dart centralDart = NIL;

    //Second step : deconnect each corner, close each hole, subdivide each new face into 3
    for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
    {
        map.setCurrentLevel(vLevel + 1) ;

        next_edge++;
        Dart dit = *edge;
        std::vector<Dart> v ;
        do
        {

            //boucle sur les brinsinterieurs de la face
            const Dart de = map.phi1(dit);
            const Dart de_1 = map.phi_1(dit);
            map.setCurrentLevel(map.getMaxLevel());

            unsigned int edgeId = map.getEdgeId(de) ;
            const unsigned int eLevel = map.getDartLevel(de);

            bool finished = false;
            Dart it = de;
            do
            {
                v.push_back(it);

                it = map.phi1(it);
                if(it == de_1)
                    finished = true ;
                else if(map.getDartLevel(it) > eLevel)
                {
                    while(map.getEdgeId(it) != edgeId)
                        it = map.phi1(map.phi2(it));
                }

                edgeId = map.getEdgeId(it);
            } while(!finished);

            map.setCurrentLevel(vLevel+1);
            dit = map.phi2(map.phi_1(dit));
        }
        while(dit != *edge);

        map.setCurrentLevel(map.getMaxLevel());
        map.splitVolume(v) ;

        map.setCurrentLevel(vLevel+1);

        const Dart old = map.phi2(map.phi1(*edge));
        Dart dd = map.phi1(map.phi1(old)) ;
        map.splitFace(old,dd) ;

        unsigned int idface = map.getNewFaceId();
        map.setFaceId(dd,idface, FACE);

        Dart ne = map.phi1(map.phi1(old)) ;

        map.cutEdge(ne);
        position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
        centralDart = map.phi1(ne);
        newEdges.push_back(ne);
        newEdges.push_back(map.phi1(ne));

        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(ne, id, EDGE) ;

        const Dart stop = map.phi2(map.phi1(ne));
        ne = map.phi2(ne);
        do
        {
            dd = map.phi1(map.phi1(map.phi1(ne)));

            map.splitFace(ne, dd) ;
            unsigned int idface = map.getNewFaceId();
            map.setFaceId(dd,idface, FACE);

            newEdges.push_back(map.phi1(dd));

            ne = map.phi2(map.phi_1(ne));
            dd = map.phi1(map.phi1(dd));
        } while(dd != stop);
    }

    map.deleteVolume(map.phi3(map.phi2(map.phi1(oldEdges.front()))));

    //Third step : 3-sew internal faces
    for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
    {
        const Dart f1 = it->first;
        const Dart f2 = it->second;

        const Dart f1_2 = map.phi2(f1);


        if(map.isBoundaryFace(f1_2) && map.isBoundaryFace(map.phi2(f2)))
        {
            //id pour toutes les faces interieures
            map.setCurrentLevel(map.getMaxLevel());
            map.sewVolumes(f1_2, map.phi2(f2));

            unsigned int idface = map.getNewFaceId();
            map.setFaceId(f1_2,idface, FACE);
            map.setCurrentLevel(vLevel+1);
        }
    }

    map.setCurrentLevel(vLevel+1);
    {

        Traversor3VF< MAP > traF(map, centralDart, true);
        {
            for (FaceCell f = traF.begin(); f != traF.end() ; f = traF.next())
            {
                const unsigned lvl1Face = Algo::Topo::setOrbitEmbeddingOnNewCell<FACE>(map, f);
                {
                    map.setCurrentLevel(map.getMaxLevel());
                    TraversorDartsOfOrbit< MAP, FACE > traDoF(map, f);
                    for (Dart fit = traDoF.begin(); fit != traDoF.end() ; fit = traDoF.next())
                    {
                        map.setMaxFaceLevel(fit, vLevel + 1u);
                        const unsigned int dl = map.getDartLevel(fit);
                        map.setCurrentLevel(dl);
                        map.template setDartEmbedding<FACE>(fit, lvl1Face);
                        map.setCurrentLevel(map.getMaxLevel());
                    }
                    map.setCurrentLevel(vLevel + 1) ;
                }
                map.setFaceLevel(f, vLevel+1);
                map.checkEmbedding(f);
            }
        }

        map.setCurrentLevel(vLevel+1);
        Traversor3VW< MAP > traW(map, centralDart, true);
        for (VolumeCell it = traW.begin() ; it != traW.end() ; it = traW.next())
        {
            const unsigned newVolEmb = Algo::Topo::setOrbitEmbeddingOnNewCell<VOLUME>(map, it) ;
            map.setCurrentLevel(map.getMaxLevel());
            TraversorDartsOfOrbit< MAP, VOLUME > traDoW(map, it);
            unsigned oldEmb = EMBNULL;
            for(Dart wit = traDoW.begin() ; wit != traDoW.end() ; wit = traDoW.next())
            {
                map.setMaxVolumeLevel(wit, vLevel + 1u);
                const unsigned dl = map.getDartLevel(wit);
                if (dl > vLevel)
                {
                    map.setCurrentLevel(dl);
                    map.template setDartEmbedding<VOLUME>(wit, newVolEmb);
                    map.setCurrentLevel(map.getMaxLevel());
                } else
                {
                    oldEmb = map.MAP::ParentMap::template getEmbedding<VOLUME>(wit);
                }
            }
            assert(oldEmb != EMBNULL);
            map.template getAttributeContainer<VOLUME>().copyLine(newVolEmb, oldEmb);
            map.setCurrentLevel(vLevel+1);
            map.setVolumeLevel(it, vLevel + 1u);
        }
    }

    map.checkEdgeAndFaceIDAttributes();
    map.setCurrentLevel(cur) ;
    return centralDart;
}








template <typename PFP>
Dart subdivideVolumeClassic2(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, bool OneLevelDifference)
//Dart subdivideVolumeClassic(typename PFP::MAP& map, Dart d, Algo::Volume::IHM::AttributeHandler_IHM<typename PFP::VEC3, VERTEX>& position)
{
//	assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//	assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;
//	assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

//	unsigned int vLevel = map.volumeLevel(d);
//	Dart old = map.volumeOldestDart(d);

//	unsigned int cur = map.getCurrentLevel();
//	map.setCurrentLevel(vLevel);

//	if(OneLevelDifference)
//	{
//		Traversor3WF<typename PFP::MAP> traF(map, old);
//		for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
//		{
//			Dart nv = map.phi3(dit);
//			if(!map.isBoundaryMarked(3, nv))
//				if(map.volumeLevel(nv) == vLevel - 1)
//					subdivideVolumeClassic<PFP>(map,nv,position,OneLevelDifference);
//		}
//	}

//	/*
//	 * au niveau du volume courant i
//	 * stockage d'un brin de chaque face de celui-ci
//	 * avec calcul du centroid
//	 */

//	DartMarkerStore<typename PFP::MAP> mf(map);		// Lock a face marker to save one dart per face
//	DartMarkerStore<typename PFP::MAP> mv(map);

//	typename PFP::VEC3 volCenter;
//	unsigned count = 0 ;

//	//Store faces that are traversed and start with the face of d
//	std::vector<Dart> visitedFaces;
//	visitedFaces.reserve(1024);
//	visitedFaces.push_back(old);

//	//Store the edges before the cutEdge
//	std::vector<Dart> oldEdges;
//	oldEdges.reserve(20);

//	mf.markOrbit(Face(old)) ;

//	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
//	{
//		Dart e = visitedFaces[i] ;
//		do
//		{
//			//add one old edge per vertex to the old edge list
//			//compute volume centroid
//			if(!mv.isMarked(e))
//			{
//				mv.markOrbit(Vertex(e));
//				volCenter += position[e];
//				++count;
//				oldEdges.push_back(e);
//			}

//			// add all face neighbours to the table
//			Dart ee = map.phi2(e) ;
//			if(!mf.isMarked(ee)) // not already marked
//			{
//				visitedFaces.push_back(ee) ;
//				mf.markOrbit(Face(ee)) ;
//			}

//			e = map.phi1(e) ;
//		} while(e != visitedFaces[i]) ;
//	}

//	volCenter /= typename PFP::REAL(count) ;

//	/*
//	 * Subdivision
//	 */

//	//Store the darts from quadrangulated faces
//	std::vector<std::pair<Dart,Dart> > subdividedfaces;
//	subdividedfaces.reserve(25);

//	//First step : subdivide edges and faces
//	//creates a i+1 edge level and i+1 face level
//	for (std::vector<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
//	{
//		Dart d = *face;

//		//if needed subdivide face
//		if(!map.faceIsSubdivided(d))
//			IHM::subdivideFace<PFP>(map, d, position, IHM::S_QUAD);

//		//save a dart from the subdivided face
//		unsigned int cur = map.getCurrentLevel() ;

//		unsigned int fLevel = map.faceLevel(d) + 1; //puisque dans tous les cas, la face est subdivisee
//		map.setCurrentLevel(fLevel) ;

//		//le brin est forcement du niveau cur
//		Dart cf = map.phi1(d);
//		Dart e = cf;
//		do
//		{
//			subdividedfaces.push_back(std::pair<Dart,Dart>(e,map.phi2(e)));
//			e = map.phi2(map.phi1(e));
//		}while (e != cf);

//		map.setCurrentLevel(cur);
//	}

//	map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

//	std::vector<Dart> newEdges;	//save darts from inner edges
//	newEdges.reserve(50);


//	int next_edge = 0;
//	Dart centralDart = NIL;
//	//Second step : deconnect each corner, close each hole, subdivide each new face into 3
//	for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
//	{
//		next_edge++;

//		Dart dit = *edge;

//		std::cout << "edge = " << dit << std::endl;

//		std::vector<Dart> v ;
//		//boucle sur le sommet du coin
//		do
//		{
//			//boucle sur les brins interieurs de la face
//			Dart de = map.phi1(dit);
//			Dart de_1 = map.phi_1(dit);

//			map.setCurrentLevel(map.getMaxLevel());

//			unsigned int edgeId = map.getEdgeId(de) ;
//			unsigned int eLevel = map.getDartLevel(de);

//                edgeId = map.getEdgeId(it);

//            } while(!finished);
//            map.setCurrentLevel(vLevel+1);

//            dit = map.phi2(map.phi_1(dit));
//        }
//        while(dit != *edge);

////		if(v.size() > 9)
////		{
////			for(int k = 0 ; k < v.size() ; k++)
////				std::cout << "dart: " << v[k] << std::endl;

//				edgeId = map.getEdgeId(it);

//			} while(!finished);
//			map.setCurrentLevel(vLevel+1);

//			dit = map.phi2(map.phi_1(dit));
//		}
//		while(dit != *edge);

////		if(v.size() > 9)
////		{
////			for(int k = 0 ; k < v.size() ; k++)
////				std::cout << "dart: " << v[k] << std::endl;

////			std::cout << std::endl;
////			return NIL;
////		}

//        map.setCurrentLevel(map.getMaxLevel());
//        map.splitVolume(v) ;
//        map.setCurrentLevel(vLevel+1);

//        Dart old = map.phi2(map.phi1(*edge));
//        Dart dd = map.phi1(map.phi1(old)) ;

//        map.splitFace(old,dd) ;
//        unsigned int idface = map.getNewFaceId();
//        map.setFaceId(dd,idface, FACE);

//        Dart ne = map.phi1(map.phi1(old)) ;

//        map.cutEdge(ne);
//        position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
//        centralDart = map.phi1(ne);
//        newEdges.push_back(ne);
//        newEdges.push_back(map.phi1(ne));

//        unsigned int id = map.getNewEdgeId() ;
//        map.setEdgeId(ne, id, EDGE) ;

//        Dart stop = map.phi2(map.phi1(ne));
//        ne = map.phi2(ne);
//        do
//        {
//            dd = map.phi1(map.phi1(map.phi1(ne)));

//            map.splitFace(ne, dd) ;
//            unsigned int idface = map.getNewFaceId();
//            map.setFaceId(dd,idface, FACE);

//            newEdges.push_back(map.phi1(dd));

//            ne = map.phi2(map.phi_1(ne));
//            dd = map.phi1(map.phi1(dd));
//        }
//        while(dd != stop);

//    }

//     map.deleteVolume(map.phi3(map.phi2(map.phi1(oldEdges.front()))));

////int next_edge = 0;
//	//Third step : 3-sew internal faces
//	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
//	{
//		Dart f1 = (*it).first;
//		Dart f2 = (*it).second;
////		next_edge++;

////		if(next_edge == 10)
////			return NIL;

//        Dart f1_2 = map.phi2(f1);
////		Dart f2_2 = map.phi2(f2);

//        if(map.isBoundaryFace(f1_2) && map.isBoundaryFace(map.phi2(f2)))
//        {
//            std::cout << "f1 = " << f1 << std::endl;
//            std::cout << "f2 = "<< f2 << std::endl << std::endl;
//            //id pour toutes les faces interieures
//            map.setCurrentLevel(map.getMaxLevel());

//            map.sewVolumes(f1_2, map.phi2(f2));
//            map.setCurrentLevel(vLevel+1);


//            //Fais a la couture !!!!!
//            unsigned int idface = map.getNewFaceId();
//            map.setFaceId(map.phi2(f1),idface, FACE);
//        }

////		//FAIS a la couture !!!!!!!
////		//id pour toutes les aretes exterieurs des faces quadrangulees
////		unsigned int idedge = map.getEdgeId(f1);
////		map.setEdgeId(map.phi2(f1), idedge, DART);
////		map.setEdgeId( map.phi2(f2), idedge, DART);
//	}

///*
//    //LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
//    //id pour les aretes interieurs : (i.e. 6 pour un hexa)
//    DartMarkerStore<typename PFP::MAP> mne(map);
//    for(unsigned int i = 0; i < newEdges.size(); ++i)
//    {
//        if(!mne.isMarked(newEdges[i]))
//        {
//            unsigned int idedge = map.getNewEdgeId();
//            map.setEdgeId(newEdges[i], idedge, EDGE);
//            mne.markOrbit(Edge(newEdges[i]));
//        }
//    }
//*/
//    //	std::cout << map.getCurrentLevel() << std::endl;
//    //	//Second step : deconnect each corner, close each hole, subdivide each new face into 3
//    //    for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
//    //    {
//    //        Dart e = *edge;
//    //        if(e == 309)
//    //            std::cout << "plop" << std::endl;
//    //        //std::cout << "emb ? " << map.template getEmbedding<VERTEX>(e) << std::endl;
//    //        //if(map.template getEmbedding<VERTEX>(e) == EMBNULL)
//    //        map.computeVertexVertexFunctor(e);
//    //        //std::cout << "emb = " << map.template getEmbedding<VERTEX>(e) << " / dartlevel = " <<  map.getDartLevel(e) << std::endl;
//    //    }
//    //    std::cout << std::endl;

//    //    map.computerVolumeVertexFunctor(centralDart);

//    map.setCurrentLevel(cur) ;

//    return centralDart; //subdividedfaces.begin()->first;
    return NIL;
}





















template <typename PFP>
void subdivideEdgeGeom(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int eLevel = map.edgeLevel(d) ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(eLevel) ;

    Dart dd = map.phi2(d) ;
    typename PFP::VEC3 p1 = position[d] ;
    typename PFP::VEC3 p2 = position[map.phi1(d)] ;

    map.setCurrentLevel(eLevel + 1) ;

    map.cutEdge(d) ;
    unsigned int eId = map.getEdgeId(d) ;
    map.setEdgeId(map.phi1(d), eId, EDGE) ; //mise a jour de l'id d'arrete sur chaque moitie d'arete
    map.setEdgeId(map.phi1(dd), eId, EDGE) ;

    map.setFaceId(EDGE, d) ; //mise a jour de l'id de face sur chaque brin de chaque moitie d'arete
    map.setFaceId(EDGE, dd) ;

    //position[map.phi1(d)] = (p1 + p2) * typename PFP::REAL(0.5) ;
    map.computeEdgeVertexFunctor(map.phi1(d));

    map.setCurrentLevel(cur) ;
}

template <typename PFP>
void subdivideFaceGeom(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, SubdivideType sType)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int fLevel = map.faceLevel(d) ;
    Dart old = map.faceOldestDart(d) ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

    unsigned int vLevel = map.volumeLevel(old);
    //one level of subdivision in the neighbordhood
    //	Traversor3VW<typename PFP::MAP> trav3EW(map, old);
    //	for(Dart dit = trav3EW.begin() ; dit != trav3EW.end() ; dit = trav3EW.next())
    //	{
    //		Dart oldit = map.volumeOldestDart(dit);
    //
    //		//std::cout << "vLevel courant = " << map.volumeLevel(oldit) << std::endl;
    //
    //		if(((vLevel+1) - map.volumeLevel(oldit)) > 1)
    //				IHM::subdivideVolumeClassic<PFP>(map, oldit, position);
    //	}

    unsigned int degree = 0 ;
    typename PFP::VEC3 p ;
    Traversor2FE<typename PFP::MAP>  travE(map, old);
    for(Dart it = travE.begin(); it != travE.end() ; it = travE.next())
    {
        ++degree;
        p += position[it] ;

        if(!map.edgeIsSubdivided(it))							// first cut the edges (if they are not already)
            IHM::subdivideEdgeGeom<PFP>(map, it, position) ;	// and compute the degree of the face
    }
    p /= typename PFP::REAL(degree) ;


    map.setCurrentLevel(fLevel + 1) ;			// go to the next level to perform face subdivision

    Dart res;

    if(degree == 3 && sType == IHM::S_TRI)	//subdiviser une face triangulaire
    {
        Dart dd = map.phi1(old) ;
        Dart e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;					// insert a new edge
        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;		// set the edge id of the inserted edge to the next available id

        unsigned int idface = map.getFaceId(old);
        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;

        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;

        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

    }
    else
    {
        Dart dd = map.phi1(old) ;
        map.splitFace(dd, map.phi1(map.phi1(dd))) ;

        Dart ne = map.phi2(map.phi_1(dd));
        Dart ne2 = map.phi2(ne);

        map.cutEdge(ne) ;
        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(ne, id, EDGE) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(ne2, id, EDGE) ;

        //position[map.phi1(ne)] = p ;

        dd = map.phi1(map.phi1(map.phi1(map.phi1(ne)))) ;
        while(dd != ne)
        {
            Dart next = map.phi1(map.phi1(dd)) ;
            map.splitFace(map.phi1(ne), dd) ;
            Dart nne = map.phi2(map.phi_1(dd)) ;

            id = map.getNewEdgeId() ;
            map.setEdgeId(nne, id, EDGE) ;

            dd = next ;
        }

        unsigned int idface = map.getFaceId(old);
        //Dart e = dd;
        do
        {
            map.setFaceId(dd, idface, DART) ;
            map.setFaceId(map.phi2(dd), idface, DART) ;
            dd = map.phi2(map.phi1(dd));
        }
        while(dd != ne);

        map.computeFaceVertexFunctor(map.phi1(ne));
        //position[map.phi1(ne)] = p ;
    }

    map.setCurrentLevel(cur) ;
}

template <typename PFP>
Dart subdivideVolumeClassicGeom(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
//Dart subdivideVolumeClassic(typename PFP::MAP& map, Dart d, Algo::Volume::IHM::AttributeHandler_IHM<typename PFP::VEC3, VERTEX>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int vLevel = map.volumeLevel(d);
    Dart old = map.volumeOldestDart(d);

    unsigned int cur = map.getCurrentLevel();
    map.setCurrentLevel(vLevel);

    //	//one level of subdivision in the neighbordhood
    //	Traversor3VW<typename PFP::MAP> trav3EW(map, old);
    //	for(Dart dit = trav3EW.begin() ; dit != trav3EW.end() ; dit = trav3EW.next())
    //	{
    //		Dart oldit = map.volumeOldestDart(dit);
    //		if(((vLevel+1) - map.volumeLevel(oldit)) > 1)
    //			IHM::subdivideVolumeClassic<PFP>(map, oldit, position);
    //	}

    /*
     * au niveau du volume courant i
     * stockage d'un brin de chaque face de celui-ci
     * avec calcul du centroid
     */

    DartMarkerStore<typename PFP::MAP> mf(map);		// Lock a face marker to save one dart per face
    DartMarkerStore<typename PFP::MAP> mv(map);

    typename PFP::VEC3 volCenter;
    unsigned count = 0 ;

    //Store faces that are traversed and start with the face of d
    std::vector<Dart> visitedFaces;
    visitedFaces.reserve(200);
    visitedFaces.push_back(old);

    //Store the edges before the cutEdge
    std::vector<Dart> oldEdges;
    oldEdges.reserve(20);

    mf.markOrbit(Face(old)) ;

    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
    {
        Dart e = visitedFaces[i] ;
        do
        {
            //add one old edge per vertex to the old edge list
            //compute volume centroid
            if(!mv.isMarked(e))
            {
                mv.markOrbit(Vertex(e));
                volCenter += position[e];
                ++count;
                oldEdges.push_back(e);
            }

            // add all face neighbours to the table
            Dart ee = map.phi2(e) ;
            if(!mf.isMarked(ee)) // not already marked
            {
                visitedFaces.push_back(ee) ;
                mf.markOrbit(Face(ee)) ;
            }

            e = map.phi1(e) ;
        } while(e != visitedFaces[i]) ;
    }

    volCenter /= typename PFP::REAL(count) ;

    /*
     * Subdivision
     */

    //Store the darts from quadrangulated faces
    std::vector<std::pair<Dart,Dart> > subdividedfaces;
    subdividedfaces.reserve(25);

    //First step : subdivide edges and faces
    //creates a i+1 edge level and i+1 face level
    for (std::vector<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
    {
        Dart d = *face;

        //if needed subdivide face
        if(!map.faceIsSubdivided(d))
            IHM::subdivideFaceGeom<PFP>(map, d, position, IHM::S_QUAD);

        //save a dart from the subdivided face
        unsigned int cur = map.getCurrentLevel() ;

        unsigned int fLevel = map.faceLevel(d) + 1; //puisque dans tous les cas, la face est subdivisee
        map.setCurrentLevel(fLevel) ;

        //le brin est forcement du niveau cur
        Dart cf = map.phi1(d);
        Dart e = cf;
        do
        {
            subdividedfaces.push_back(std::pair<Dart,Dart>(e,map.phi2(e)));
            e = map.phi2(map.phi1(e));
        }while (e != cf);

        map.setCurrentLevel(cur);
    }

    map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

    std::vector<Dart> newEdges;	//save darts from inner edges
    newEdges.reserve(50);

    Dart centralDart = NIL;
    //Second step : deconnect each corner, close each hole, subdivide each new face into 3
    for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
    {
        Dart e = *edge;

        std::vector<Dart> v ;

        do
        {
            v.push_back(map.phi1(e));
            v.push_back(map.phi1(map.phi1(e)));

            e = map.phi2(map.phi_1(e));
        }
        while(e != *edge);

        map.splitVolume(v) ;

        Dart old = map.phi2(map.phi1(*edge));
        Dart dd = map.phi1(map.phi1(old)) ;
        map.splitFace(old,dd) ;

        unsigned int idface = map.getNewFaceId();
        map.setFaceId(dd,idface, FACE);

        Dart ne = map.phi1(map.phi1(old)) ;

        map.cutEdge(ne);
        position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
        centralDart = map.phi1(ne);
        newEdges.push_back(ne);
        newEdges.push_back(map.phi1(ne));

        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(ne, id, EDGE) ;

        Dart stop = map.phi2(map.phi1(ne));
        ne = map.phi2(ne);
        do
        {
            dd = map.phi1(map.phi1(map.phi1(ne)));

            map.splitFace(ne, dd) ;
            unsigned int idface = map.getNewFaceId();
            map.setFaceId(dd,idface, FACE);

            newEdges.push_back(map.phi1(dd));

            ne = map.phi2(map.phi_1(ne));
            dd = map.phi1(map.phi1(dd));
        }
        while(dd != stop);

        //map.computeVertexVertexFunctor(e);
    }


    map.deleteVolume(map.phi3(map.phi2(map.phi1(oldEdges.front()))));

    //Third step : 3-sew internal faces
    for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
    {
        Dart f1 = (*it).first;
        Dart f2 = (*it).second;

        if(map.isBoundaryFace(map.phi2(f1)) && map.isBoundaryFace(map.phi2(f2)))
        {
            //id pour toutes les faces interieures
            map.sewVolumes(map.phi2(f1), map.phi2(f2));

            //Fais a la couture !!!!!
            unsigned int idface = map.getNewFaceId();
            map.setFaceId(map.phi2(f1),idface, FACE);
        }

        //FAIS a la couture !!!!!!!
        //id pour toutes les aretes exterieurs des faces quadrangulees
        unsigned int idedge = map.getEdgeId(f1);
        map.setEdgeId(map.phi2(f1), idedge, DART);
        map.setEdgeId( map.phi2(f2), idedge, DART);
    }

    //LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
    //id pour les aretes interieurs : (i.e. 6 pour un hexa)
    DartMarkerStore<typename PFP::MAP> mne(map);
    for(unsigned int i = 0; i < newEdges.size(); ++i)
    {
        if(!mne.isMarked(newEdges[i]))
        {
            unsigned int idedge = map.getNewEdgeId();
            map.setEdgeId(newEdges[i], idedge, EDGE);
            mne.markOrbit(Edge(newEdges[i]));
        }
    }

    //std::cout << map.getCurrentLevel() << std::endl;
    //Second step : deconnect each corner, close each hole, subdivide each new face into 3
    for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
    {
        Dart e = *edge;
        //if(e == 309)
        //    std::cout << "plop" << std::endl;
        //std::cout << "emb ? " << map.template getEmbedding<VERTEX>(e) << std::endl;
        //if(map.template getEmbedding<VERTEX>(e) == EMBNULL)
        map.computeVertexVertexFunctor(e);
        //std::cout << "emb = " << map.template getEmbedding<VERTEX>(e) << " / dartlevel = " <<  map.getDartLevel(e) << std::endl;
    }
    std::cout << std::endl;

    map.computerVolumeVertexFunctor(centralDart);

    map.setCurrentLevel(cur) ;

    return centralDart; //subdividedfaces.begin()->first;
}













































































template <typename PFP>
void subdivideEdgeWrong(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int eLevel = map.edgeLevel(d) ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(eLevel) ;

    Dart dd = map.phi2(d) ;
    typename PFP::VEC3 p1 = position[d] ;
    typename PFP::VEC3 p2 = position[map.phi1(d)] ;

    map.setCurrentLevel(eLevel + 1) ;

    map.cutEdge(d) ;
    unsigned int eId = map.getEdgeId(d) ;
    map.setEdgeId(map.phi1(d), eId, EDGE) ; //mise a jour de l'id d'arrete sur chaque moitie d'arete
    map.setEdgeId(map.phi1(dd), eId, EDGE) ;

    map.setFaceId(EDGE, d) ; //mise a jour de l'id de face sur chaque brin de chaque moitie d'arete
    map.setFaceId(EDGE, dd) ;

    position[map.phi1(d)] = (p1 + p2) * typename PFP::REAL(0.5) ;

    map.setCurrentLevel(cur) ;
}

template <typename PFP>
void subdivideFaceWrong(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, SubdivideType sType)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int fLevel = map.faceLevel(d) ;
    Dart old = map.faceOldestDart(d) ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

    unsigned int vLevel = map.volumeLevel(old);
    //one level of subdivision in the neighbordhood
    //	Traversor3VW<typename PFP::MAP> trav3EW(map, old);
    //	for(Dart dit = trav3EW.begin() ; dit != trav3EW.end() ; dit = trav3EW.next())
    //	{
    //		Dart oldit = map.volumeOldestDart(dit);
    //
    //		//out << "vLevel courant = " << map.volumeLevel(oldit) << std::endl;
    //
    //		if(((vLevel+1) - map.volumeLevel(oldit)) > 1)
    //				IHM::subdivideVolumeClassic<PFP>(map, oldit, position);
    //	}

    unsigned int degree = 0 ;
    typename PFP::VEC3 p ;
    Traversor2FE<typename PFP::MAP>  travE(map, old);
    for(Dart it = travE.begin(); it != travE.end() ; it = travE.next())
    {
        ++degree;
        p += position[it] ;

        if(!map.edgeIsSubdivided(it))							// first cut the edges (if they are not already)
            IHM::subdivideEdgeWrong<PFP>(map, it, position) ;	// and compute the degree of the face
    }
    p /= typename PFP::REAL(degree) ;


    map.setCurrentLevel(fLevel + 1) ;			// go to the next level to perform face subdivision

    Dart res;

    if(degree == 3 && sType == IHM::S_TRI)	//subdiviser une face triangulaire
    {
        Dart dd = map.phi1(old) ;
        Dart e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;					// insert a new edge
        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;		// set the edge id of the inserted edge to the next available id

        unsigned int idface = map.getFaceId(old);
        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;

        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;

        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        Dart stop = map.phi2(map.phi1(old));
        Dart dit = stop;
        do
        {
            unsigned int dId = map.getEdgeId(map.phi_1(map.phi2(dit)));
            unsigned int eId = map.getEdgeId(map.phi1(map.phi2(dit)));

            unsigned int t = dId + eId;

            if(t == 0)
            {
                map.setEdgeId(dit, 1, EDGE) ;
                map.setEdgeId(map.phi2(dit), 1, EDGE) ;
            }
            else if(t == 1)
            {
                map.setEdgeId(dit, 2, EDGE) ;
                map.setEdgeId(map.phi2(dit), 2, EDGE) ;
            }
            else if(t == 2)
            {
                if(dId == eId)
                {
                    map.setEdgeId(dit, 0, EDGE) ;
                    map.setEdgeId(map.phi2(dit), 0, EDGE) ;
                }
                else
                {
                    map.setEdgeId(dit, 1, EDGE) ;
                    map.setEdgeId(map.phi2(dit), 1, EDGE) ;
                }
            }
            else if(t == 3)
            {
                map.setEdgeId(dit, 0, EDGE) ;
                map.setEdgeId(map.phi2(dit), 0, EDGE) ;
            }

            dit = map.phi1(dit);
        }while(dit != stop);

    }
    else
    {
        Dart dd = map.phi1(old) ;
        map.splitFace(dd, map.phi1(map.phi1(dd))) ;

        Dart ne = map.phi2(map.phi_1(dd));
        Dart ne2 = map.phi2(ne);

        map.cutEdge(ne) ;
        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(ne, id, EDGE) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(ne2, id, EDGE) ;

        position[map.phi1(ne)] = p ;

        dd = map.phi1(map.phi1(map.phi1(map.phi1(ne)))) ;
        while(dd != ne)
        {
            Dart next = map.phi1(map.phi1(dd)) ;
            map.splitFace(map.phi1(ne), dd) ;
            Dart nne = map.phi2(map.phi_1(dd)) ;

            id = map.getNewEdgeId() ;
            map.setEdgeId(nne, id, EDGE) ;

            dd = next ;
        }

        unsigned int idface = map.getFaceId(old);
        //Dart e = dd;
        do
        {
            map.setFaceId(dd, idface, DART) ;
            map.setFaceId(map.phi2(dd), idface, DART) ;
            dd = map.phi2(map.phi1(dd));
        }
        while(dd != ne);

        //position[map.phi1(ne)] = p ;
    }

    map.setCurrentLevel(cur) ;
}


template <typename PFP>
//Dart subdivideVolumeClassic(typename PFP::MAP& map, Dart d, AttributeHandler<typename PFP::VEC3, VERTEX>& position)
Dart subdivideVolumeClassicWrong(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int vLevel = map.volumeLevel(d);
    Dart old = map.volumeOldestDart(d);

    unsigned int cur = map.getCurrentLevel();
    map.setCurrentLevel(vLevel);

    //	//one level of subdivision in the neighbordhood
    //	Traversor3VW<typename PFP::MAP> trav3EW(map, old);
    //	for(Dart dit = trav3EW.begin() ; dit != trav3EW.end() ; dit = trav3EW.next())
    //	{
    //		Dart oldit = map.volumeOldestDart(dit);
    //		if(((vLevel+1) - map.volumeLevel(oldit)) > 1)
    //			IHM::subdivideVolumeClassic<PFP>(map, oldit, position);
    //	}

    /*
     * au niveau du volume courant i
     * stockage d'un brin de chaque face de celui-ci
     * avec calcul du centroid
     */

    DartMarkerStore<typename PFP::MAP> mf(map);		// Lock a face marker to save one dart per face
    DartMarkerStore<typename PFP::MAP> mv(map);

    typename PFP::VEC3 volCenter;
    unsigned count = 0 ;

    //Store faces that are traversed and start with the face of d
    std::vector<Dart> visitedFaces;
    visitedFaces.reserve(200);
    visitedFaces.push_back(old);

    //Store the edges before the cutEdge
    std::vector<Dart> oldEdges;
    oldEdges.reserve(20);

    mf.markOrbit<FACE>(old) ;

    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
    {
        Dart e = visitedFaces[i] ;
        do
        {
            //add one old edge per vertex to the old edge list
            //compute volume centroid
            if(!mv.isMarked(e))
            {
                mv.markOrbit<VERTEX>(e);
                volCenter += position[e];
                ++count;
                oldEdges.push_back(e);
            }

            // add all face neighbours to the table
            Dart ee = map.phi2(e) ;
            if(!mf.isMarked(ee)) // not already marked
            {
                visitedFaces.push_back(ee) ;
                mf.markOrbit<FACE>(ee) ;
            }

            e = map.phi1(e) ;
        } while(e != visitedFaces[i]) ;
    }

    volCenter /= typename PFP::REAL(count) ;

    /*
     * Subdivision
     */

    //Store the darts from quadrangulated faces
    std::vector<std::pair<Dart,Dart> > subdividedfaces;
    subdividedfaces.reserve(25);

    //First step : subdivide edges and faces
    //creates a i+1 edge level and i+1 face level
    for (std::vector<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
    {
        Dart d = *face;

        //if needed subdivide face
        if(!map.faceIsSubdivided(d))
            IHM::subdivideFaceWrong<PFP>(map, d, position, IHM::S_QUAD);

        //save a dart from the subdivided face
        unsigned int cur = map.getCurrentLevel() ;

        unsigned int fLevel = map.faceLevel(d) + 1; //puisque dans tous les cas, la face est subdivisee
        map.setCurrentLevel(fLevel) ;

        //le brin est forcement du niveau cur
        Dart cf = map.phi1(d);
        Dart e = cf;
        do
        {
            subdividedfaces.push_back(std::pair<Dart,Dart>(e,map.phi2(e)));
            e = map.phi2(map.phi1(e));
        }while (e != cf);

        map.setCurrentLevel(cur);
    }

    map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

    std::vector<Dart> newEdges;	//save darts from inner edges
    newEdges.reserve(50);

    Dart centralDart = NIL;
    //Second step : deconnect each corner, close each hole, subdivide each new face into 3
    for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
    {
        Dart e = *edge;

        std::vector<Dart> v ;

        do
        {
            v.push_back(map.phi1(e));
            v.push_back(map.phi1(map.phi1(e)));

            e = map.phi2(map.phi_1(e));
        }
        while(e != *edge);

        map.splitVolume(v) ;

        Dart old = map.phi2(map.phi1(*edge));
        Dart dd = map.phi1(map.phi1(old)) ;
        map.splitFace(old,dd) ;

        unsigned int idface = map.getNewFaceId();
        map.setFaceId(dd,idface, FACE);

        Dart ne = map.phi1(map.phi1(old)) ;

        map.cutEdge(ne);
        position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
        centralDart = map.phi1(ne);
        newEdges.push_back(ne);
        newEdges.push_back(map.phi1(ne));

        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(ne, id, EDGE) ;

        Dart stop = map.phi2(map.phi1(ne));
        ne = map.phi2(ne);
        do
        {
            dd = map.phi1(map.phi1(map.phi1(ne)));

            map.splitFace(ne, dd) ;
            unsigned int idface = map.getNewFaceId();
            map.setFaceId(dd,idface, FACE);

            newEdges.push_back(map.phi1(dd));

            ne = map.phi2(map.phi_1(ne));
            dd = map.phi1(map.phi1(dd));
        }
        while(dd != stop);
    }


    map.deleteVolume(map.phi3(map.phi2(map.phi1(oldEdges.front()))));

    //Third step : 3-sew internal faces
    for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
    {
        Dart f1 = (*it).first;
        Dart f2 = (*it).second;

        if(map.isBoundaryFace(map.phi2(f1)) && map.isBoundaryFace(map.phi2(f2)))
        {
            //id pour toutes les faces interieures
            map.sewVolumes(map.phi2(f1), map.phi2(f2));

            //Fais a la couture !!!!!
            unsigned int idface = map.getNewFaceId();
            map.setFaceId(map.phi2(f1),idface, FACE);
        }

        //FAIS a la couture !!!!!!!
        //id pour toutes les aretes exterieurs des faces quadrangulees
        unsigned int idedge = map.getEdgeId(f1);
        map.setEdgeId(map.phi2(f1), idedge, DART);
        map.setEdgeId( map.phi2(f2), idedge, DART);
    }

    //LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
    //id pour les aretes interieurs : (i.e. 6 pour un hexa)
    DartMarkerStore<typename PFP::MAP> mne(map);
    for(unsigned int i = 0; i < newEdges.size(); ++i)
    {
        if(!mne.isMarked(newEdges[i]))
        {
            unsigned int idedge = map.getNewEdgeId();
            map.setEdgeId(newEdges[i], idedge, EDGE);
            mne.markOrbit<EDGE>(newEdges[i]);
        }
    }

    map.setCurrentLevel(cur) ;

    return centralDart; //subdividedfaces.begin()->first;
}


/************************************************************************************************
 * 									Simplification												*
 ************************************************************************************************/


template <typename PFP>
void coarsenEdge(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(map.edgeCanBeCoarsened(d) || !"Trying to coarsen an edge that can not be coarsened") ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(cur + 1) ;
    map.uncutEdge(d) ;
    map.setCurrentLevel(cur) ;
}

template <typename PFP>
void coarsenFace(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, SubdivideType sType)
{
	assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
	assert(map.faceCanBeCoarsened(d) || !"Trying to coarsen a non-subdivided face or a more than once subdivided face") ;

	unsigned int cur = map.getCurrentLevel() ;

	unsigned int degree = 0 ;
	Dart fit = d ;
	do
	{
		++degree ;
		fit = map.phi1(fit) ;
	} while(fit != d) ;

	//	Dart d3 = map.phi3(d);

	if(degree == 3)
	{

	}
	else
	{
		map.setCurrentLevel(cur + 1) ;
		Dart centralV = map.phi1(map.phi1(d)) ;
		map.setCurrentLevel(map.getMaxLevel()) ;
		map.deleteVertexSubdividedFace(centralV);
		map.setCurrentLevel(cur) ;
	}

	fit = d ;
	do
	{
		if(map.edgeCanBeCoarsened(fit))
			IHM::coarsenEdge<PFP>(map, fit, position) ;
		fit = map.phi1(fit) ;
	} while(fit != d) ;
}

template <typename PFP>
void coarsenVolume(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(map.volumeIsSubdividedOnce(d) || !"Trying to coarsen a non-subdivided volume or a more than once subdivided volume") ;

    unsigned int cur = map.getCurrentLevel() ;

    /*
     * Deconnecter toutes les faces interieurs
     */
    map.setCurrentLevel(cur + 1) ;
    Dart nf = map.phi_1(map.phi2(map.phi1(d)));
    map.deleteVertex(nf);
    map.setCurrentLevel(cur) ;

    /*
     * simplify faces
     */
    Traversor3WF<typename PFP::MAP> trav3WF(map, d, true);
    std::vector< FaceCell > faces;
    faces.reserve(6);
    for(FaceCell dit = trav3WF.begin() ; dit != trav3WF.end() ; dit = trav3WF.next())
    {
        if (map.faceCanBeCoarsened(dit))
        {
            faces.push_back(dit);
        }
    }

    for(std::vector< FaceCell >::const_iterator fit = faces.begin(), end = faces.end() ; fit != end ; ++fit)
    {
        IHM::coarsenFace<PFP>(map, *fit, position, IHM::S_QUAD);
    }
}




/* **************************************************************************************
 *    							USE WITH CAUTION										*
 ****************************************************************************************/










//template <typename PFP>
//Dart subdivideVolumeClassic2(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
//{
//	assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//	assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;
//	assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

//	unsigned int vLevel = map.volumeLevel(d);
//	Dart old = map.volumeOldestDart(d);

//	unsigned int cur = map.getCurrentLevel();
//	map.setCurrentLevel(vLevel);

//	/*
//	 * Compute volume centroid
//	 */
//	typename PFP::VEC3 volCenter =  Algo::Surface::Geometry::volumeCentroid<PFP>(map, old, position);

//	Traversor3WV<typename PFP::MAP> traV(map, old);

//	/*
//	 * Subdivide Faces
//	 */
//	std::vector<std::pair<Dart,Dart> > subdividedFaces;
//	subdividedFaces.reserve(128);

//	Traversor3WF<typename PFP::MAP> traF(map, old);
//	for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
//	{
//		//if needed subdivide face
//		if(!map.faceIsSubdivided(dit))
//			IHM::subdivideFace<PFP>(map, dit, position, IHM::S_QUAD);

//		//save darts from the central vertex of each subdivided face
//		unsigned int cur = map.getCurrentLevel() ;
//		unsigned int fLevel = map.faceLevel(dit);
//		map.setCurrentLevel(fLevel + 1) ;
//		map.saveRelationsAroundVertex(map.phi2(map.phi1(dit)), subdividedFaces);
//		map.setCurrentLevel(cur);
//	}

//	cur = map.getCurrentLevel() ;
//	unsigned int fLevel = map.faceLevel(old);
//	map.setCurrentLevel(fLevel + 1) ;
//	map.unsewAroundVertex(subdividedFaces);
//	map.setCurrentLevel(cur);

//	/*
//	 * Create inside volumes
//	 */
//	std::vector<Dart> newEdges;	//save darts from inner edges
//	newEdges.reserve(50);
//	Dart centralDart = NIL;

//	//unsigned int degree = 0 ;
//	//typename PFP::VEC3 volCenter ;

//	map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

//	for(Dart dit = traV.begin(); dit != traV.end(); dit = traV.next())
//	{
//		//volCenter += position[d];
//		//++degree;

//		centralDart = map.quadranguleFace(dit);
//		position[centralDart] = volCenter;
//	}
//	//volCenter /= double(degree);

//	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedFaces.begin(); it != subdividedFaces.end(); ++it)
//	{
//		Dart f1 = map.phi2((*it).first);
//		Dart f2 = map.phi2((*it).second);

//		if(map.isBoundaryFace(f1) && map.isBoundaryFace(f2))
//		{
//			//std::cout << "plop" << std::endl;
//			map.sewVolumes(f1, f2, false);
//		}
//	}

//	position[centralDart] = volCenter;




//	//	//Third step : 3-sew internal faces
//	//	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
//	//	{
//	//		Dart f1 = (*it).first;
//	//		Dart f2 = (*it).second;
//	//
//	//
//	//		if(map.isBoundaryFace(map.phi2(f1)) && map.isBoundaryFace(map.phi2(f2)))
//	//		{
//	//			std::cout << "boundary" << std::endl;
//	//			//id pour toutes les faces interieures
//	//			map.sewVolumes(map.phi2(f1), map.phi2(f2));
//	//
//	////
//	////			//Fais a la couture !!!!!
//	////			unsigned int idface = map.getNewFaceId();
//	////			map.setFaceId(map.phi2(f1),idface, FACE);
//	//		}
//	////
//	////		//FAIS a la couture !!!!!!!
//	////		//id pour toutes les aretes exterieurs des faces quadrangulees
//	////		unsigned int idedge = map.getEdgeId(f1);
//	////		map.setEdgeId(map.phi2(f1), idedge, DART);
//	////		map.setEdgeId( map.phi2(f2), idedge, DART);
//	//	}

//	//LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
//	//id pour les aretes interieurs : (i.e. 6 pour un hexa)
//	DartMarker<typename PFP::MAP> mne(map);
//	for(unsigned int i = 0; i < newEdges.size(); ++i)
//	{
//		if(!mne.isMarked(newEdges[i]))
//		{
//			unsigned int idedge = map.getNewEdgeId();
//			map.setEdgeId(newEdges[i], idedge, EDGE);
//			mne.markOrbit<EDGE>(newEdges[i]);
//		}
//	}


//	//plonger a la fin de la boucle ????

//	map.setCurrentLevel(cur) ;

//	return subdividedFaces.front().first;

//}










template <typename PFP>
void subdivideEdgeLoop(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int eLevel = map.edgeLevel(d) ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(eLevel) ;

    Dart dd = map.phi2(d) ;
    typename PFP::VEC3 p1 = position[d] ;
    typename PFP::VEC3 p2 = position[map.phi1(d)] ;

    map.setCurrentLevel(eLevel + 1) ;

    map.cutEdge(d) ;
    unsigned int eId = map.getEdgeId(d) ;
    map.setEdgeId(map.phi1(d), eId, EDGE) ; //mise a jour de l'id d'arrete sur chaque moitie d'arete
    map.setEdgeId(map.phi1(dd), eId, EDGE) ;

    map.setFaceId(EDGE, d) ; //mise a jour de l'id de face sur chaque brin de chaque moitie d'arete
    map.setFaceId(EDGE, dd) ;

    position[map.phi1(d)] = (p1 + p2) * typename PFP::REAL(0.5) ;
    map.setCurrentLevel(cur) ;
}

template <typename PFP>
void subdivideFaceLoop(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, SubdivideType sType)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int fLevel = map.faceLevel(d) ;
    Dart old = map.faceOldestDart(d) ;

    unsigned int cur = map.getCurrentLevel() ;
    map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

    unsigned int vLevel = map.volumeLevel(old);
    //one level of subdivision in the neighbordhood
    //	Traversor3VW<typename PFP::MAP> trav3EW(map, old);
    //	for(Dart dit = trav3EW.begin() ; dit != trav3EW.end() ; dit = trav3EW.next())
    //	{
    //		Dart oldit = map.volumeOldestDart(dit);
    //
    //		//std::cout << "vLevel courant = " << map.volumeLevel(oldit) << std::endl;
    //
    //		if(((vLevel+1) - map.volumeLevel(oldit)) > 1)
    //				IHM::subdivideVolumeClassic<PFP>(map, oldit, position);
    //	}

    unsigned int degree = 0 ;
    typename PFP::VEC3 p ;
    Traversor2FE<typename PFP::MAP>  travE(map, old);
    for(Dart it = travE.begin(); it != travE.end() ; it = travE.next())
    {
        ++degree;
        p += position[it] ;

        if(!map.edgeIsSubdivided(it))							// first cut the edges (if they are not already)
            IHM::subdivideEdgeLoop<PFP>(map, it, position) ;	// and compute the degree of the face
    }
    p /= typename PFP::REAL(degree) ;


    map.setCurrentLevel(fLevel + 1) ;			// go to the next level to perform face subdivision

    Dart res;

    if(degree == 3 && sType == IHM::S_TRI)	//subdiviser une face triangulaire
    {
        Dart dd = map.phi1(old) ;
        Dart e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;					// insert a new edge
        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;		// set the edge id of the inserted edge to the next available id

        unsigned int idface = map.getFaceId(old);
        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;

        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        dd = e ;
        e = map.phi1(map.phi1(dd)) ;
        map.splitFace(dd, e) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(map.phi_1(dd), id, EDGE) ;

        map.setFaceId(dd, idface, FACE) ;
        map.setFaceId(e, idface, FACE) ;

        Dart stop = map.phi2(map.phi1(old));
        Dart dit = stop;
        do
        {
            unsigned int dId = map.getEdgeId(map.phi_1(map.phi2(dit)));
            unsigned int eId = map.getEdgeId(map.phi1(map.phi2(dit)));

            unsigned int t = dId + eId;

            if(t == 0)
            {
                map.setEdgeId(dit, 1, EDGE) ;
                map.setEdgeId(map.phi2(dit), 1, EDGE) ;
            }
            else if(t == 1)
            {
                map.setEdgeId(dit, 2, EDGE) ;
                map.setEdgeId(map.phi2(dit), 2, EDGE) ;
            }
            else if(t == 2)
            {
                if(dId == eId)
                {
                    map.setEdgeId(dit, 0, EDGE) ;
                    map.setEdgeId(map.phi2(dit), 0, EDGE) ;
                }
                else
                {
                    map.setEdgeId(dit, 1, EDGE) ;
                    map.setEdgeId(map.phi2(dit), 1, EDGE) ;
                }
            }
            else if(t == 3)
            {
                map.setEdgeId(dit, 0, EDGE) ;
                map.setEdgeId(map.phi2(dit), 0, EDGE) ;
            }

            dit = map.phi1(dit);
        }while(dit != stop);

    }
    else
    {
        Dart dd = map.phi1(old) ;
        map.splitFace(dd, map.phi1(map.phi1(dd))) ;

        Dart ne = map.phi2(map.phi_1(dd));
        Dart ne2 = map.phi2(ne);

        map.cutEdge(ne) ;
        unsigned int id = map.getNewEdgeId() ;
        map.setEdgeId(ne, id, EDGE) ;
        id = map.getNewEdgeId() ;
        map.setEdgeId(ne2, id, EDGE) ;

        position[map.phi1(ne)] = p ;

        dd = map.phi1(map.phi1(map.phi1(map.phi1(ne)))) ;
        while(dd != ne)
        {
            Dart next = map.phi1(map.phi1(dd)) ;
            map.splitFace(map.phi1(ne), dd) ;
            Dart nne = map.phi2(map.phi_1(dd)) ;

            id = map.getNewEdgeId() ;
            map.setEdgeId(nne, id, EDGE) ;

            dd = next ;
        }

        unsigned int idface = map.getFaceId(old);
        //Dart e = dd;
        do
        {
            map.setFaceId(dd, idface, DART) ;
            map.setFaceId(map.phi2(dd), idface, DART) ;
            dd = map.phi2(map.phi1(dd));
        }
        while(dd != ne);

        position[map.phi1(ne)] = p ;
    }

    map.setCurrentLevel(cur) ;
}









template <typename PFP>
void subdivideLoop(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;
    assert(!map.isBoundaryMarked(3,d) || !"Trying to subdivide a dart marked boundary");

    unsigned int vLevel = map.volumeLevel(d);
    Dart old = map.volumeOldestDart(d);

    unsigned int cur = map.getCurrentLevel();
    map.setCurrentLevel(vLevel);

    /*
     * Compute volume centroid
     */
    typename PFP::VEC3 volCenter =  Algo::Surface::Geometry::volumeCentroid<PFP>(map, old, position);

    /*
     * Subdivide Faces
     */
    std::vector<std::pair<Dart,Dart> > subdividedfaces;
    subdividedfaces.reserve(25);

    Traversor3WF<typename PFP::MAP> traF(map, old);
    for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
    {
        //if needed subdivide face
        if(!map.faceIsSubdivided(dit))
            IHM::subdivideFaceLoop<PFP>(map, dit, position, IHM::S_TRI);

        //save a dart from the subdivided face
        unsigned int cur = map.getCurrentLevel() ;

        unsigned int fLevel = map.faceLevel(d) + 1; //puisque dans tous les cas, la face est subdivisee
        map.setCurrentLevel(fLevel) ;

        //le brin est forcement du niveau cur
        Dart cf = map.phi1(d);
        Dart e = cf;
        do
        {
            subdividedfaces.push_back(std::pair<Dart,Dart>(e,map.phi2(e)));
            e = map.phi2(map.phi1(e));
        }while (e != cf);

        map.setCurrentLevel(cur);
    }

    /*
     * Create inside volumes
     */
    std::vector<Dart> newEdges;	//save darts from inner edges
    newEdges.reserve(50);
    Dart centralDart = NIL;

    //unsigned int degree = 0 ;
    //typename PFP::VEC3 volCenter ;
    Traversor3WV<typename PFP::MAP> traV(map, old);

    bool isNotTet = false;

    map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

    for(Dart dit = traV.begin(); dit != traV.end(); dit = traV.next())
    {
        //volCenter += position[d];
        //++degree;
        Dart f1 = map.phi1(dit);
        Dart e = dit;
        std::vector<Dart> v ;

        do
        {
            v.push_back(map.phi1(e));
            e = map.phi2(map.phi_1(e));
        }
        while(e != dit);

        map.splitVolume(v) ;

        for(std::vector<Dart>::iterator it = v.begin() ; it != v.end() ; ++it)
        {
            map.setEdgeId(*it,map.getEdgeId(*it), EDGE);
        }

        //if is not a tetrahedron
        unsigned int fdeg = map.faceDegree(map.phi2(f1));
        if(fdeg > 3)
        {
            isNotTet = true;
            Dart old = map.phi2(map.phi1(dit));
            Dart dd = map.phi1(old) ;
            map.splitFace(old,dd) ;

            Dart ne = map.phi1(old);

            map.cutEdge(ne);
            position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
            centralDart = map.phi1(ne);

            unsigned int idFace = map.getNewFaceId();
            map.setFaceId(ne, idFace, FACE);

            newEdges.push_back(ne);
            newEdges.push_back(map.phi1(ne));

            Dart stop = map.phi2(map.phi1(ne));
            ne = map.phi2(ne);
            do
            {
                dd = map.phi1(map.phi1(ne));

                map.splitFace(ne, dd) ;

                unsigned int idFace = map.getNewFaceId();
                map.setFaceId(ne, idFace, FACE);

                newEdges.push_back(map.phi1(dd));

                ne = map.phi2(map.phi_1(ne));
                dd = map.phi1(dd);
            }
            while(dd != stop);

            idFace = map.getNewFaceId();
            map.setFaceId(stop, idFace, FACE);

        }
        else
        {
            unsigned int idface = map.getNewFaceId();
            map.setFaceId(map.phi2(v.front()),idface, FACE);
        }
    }

    //switch inner faces
    if(isNotTet)
    {
        DartMarker<typename PFP::MAP> me(map);

        for(Dart dit = traV.begin(); dit != traV.end(); dit = traV.next())
        {
            Dart x = map.phi_1(map.phi2(map.phi1(dit)));
            Dart f = x;

            do
            {
                Dart f3 = map.phi3(f);

                if(!me.isMarked(f3))
                {
                    Dart tmp =  map.phi_1(map.phi2(map.phi_1(map.phi2(map.phi_1(f3))))); //future voisin par phi2

                    Dart f32 = map.phi2(f3);
                    map.swapEdges(f3, tmp);

                    unsigned int idface = map.getNewFaceId();
                    map.setFaceId(f3,idface, FACE);
                    idface = map.getNewFaceId();
                    map.setFaceId(f32,idface, FACE);

                    unsigned int idedge = map.getNewEdgeId();
                    map.setEdgeId(f3, idedge, EDGE);

                    map.setEdgeId(f3,map.getEdgeId(f3), EDGE);
                    map.setEdgeId(f32,map.getEdgeId(f32), EDGE);

                    me.markOrbit(Edge(f3));
                    me.markOrbit(Edge(f32));
                }

                f = map.phi2(map.phi_1(f));
            }while(f != x);
        }

        Algo::Topo::setOrbitEmbedding<VERTEX>(map,centralDart,map.template getEmbedding<VERTEX>(centralDart));

        //Third step : 3-sew internal faces
        for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
        {
            Dart f1 = (*it).first;
            Dart f2 = (*it).second;

            unsigned int idedge = map.getEdgeId(f1);
            map.setEdgeId(map.phi2(f1), idedge, EDGE);
            map.setEdgeId(map.phi2(f2), idedge, EDGE);
        }

    }

    map.setCurrentLevel(cur) ;
}


template <typename PFP>
Dart subdivideVolume(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;
    //assert(!map.neighborhoodLevelDiffersByOne(d) || !"Trying to subdivide a volume with neighborhood Level difference greater than 1");

    unsigned int vLevel = map.volumeLevel(d);
    Dart old = map.volumeOldestDart(d);

    unsigned int cur = map.getCurrentLevel();
    map.setCurrentLevel(vLevel);

    /*
     * au niveau du volume courant i
     * stockage d'un brin de chaque face de celui-ci
     * avec calcul du centroid
     */

    DartMarkerStore<typename PFP::MAP> mf(map);		// Lock a face marker to save one dart per face
    CellMarker<typename PFP::MAP, VERTEX> mv(map);

    typename PFP::VEC3 volCenter;
    unsigned count = 0 ;

    //Store faces that are traversed and start with the face of d
    std::vector<Dart> visitedFaces;
    visitedFaces.reserve(512);
    visitedFaces.push_back(old);

    //Store the edges before the cutEdge
    std::vector<Dart> oldEdges;
    oldEdges.reserve(20);

    mf.markOrbit<FACE>(old) ;

    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
    {
        Dart e = visitedFaces[i] ;
        do
        {
            //add one old edge per vertex to the old edge list
            //compute volume centroid
            if(!mv.isMarked(e))
            {
                mv.mark(e);
                volCenter += position[e];
                ++count;
                oldEdges.push_back(e);
            }

            // add all face neighbours to the table
            Dart ee = map.phi2(e) ;
            if(!mf.isMarked(ee)) // not already marked
            {
                visitedFaces.push_back(ee) ;
                mf.markOrbit<FACE>(ee) ;
            }

            e = map.phi1(e) ;
        } while(e != visitedFaces[i]) ;
    }

    volCenter /= typename PFP::REAL(count) ;

    /*
     * Subdivision
     */
    //type 'q' : quad et plus
    //type 't' : tri
    std::vector<std::pair<char, std::pair<Dart,Dart> > > subdividedfaces;

    //First step : subdivide edges and faces
    //creates a i+1 edge level and i+1 face level
    for (std::vector<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
    {
        Dart d = *face;

        //if needed subdivide face
        if(!map.faceIsSubdivided(d))
        {
            IHM::subdivideFace<PFP>(map, d, position);
        }


        //save a dart from the subdivided face
        unsigned int cur = map.getCurrentLevel() ;

        unsigned int fLevel = map.faceLevel(d) + 1; //puisque dans tous les cas, la face est subdivisee
        map.setCurrentLevel(fLevel) ;

        //test si la face est triangulaire ou non
        if(map.phi1(map.phi1(map.phi1(d))) == d)
        {
            //std::cout << "trian" << std::endl;
            Dart cf = map.phi2(map.phi1(d));
            Dart e = cf;
            do
            {
                std::pair<Dart,Dart> pd(e,map.phi2(e));
                subdividedfaces.push_back(std::pair<char, std::pair<Dart,Dart> > ('t',pd));
                e = map.phi1(e);
            }while (e != cf);
        }
        else
        {
            //std::cout << "quad" << std::endl;
            Dart cf = map.phi1(d);
            Dart e = cf;
            do
            {
                std::pair<Dart,Dart> pd(e,map.phi2(e));
                subdividedfaces.push_back(std::pair<char, std::pair<Dart,Dart> >('q',pd));
                e = map.phi2(map.phi1(e));
            }while (e != cf);
        }

        map.setCurrentLevel(cur);

    }

    map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

    std::vector<Dart> newEdges;	//save darts from inner edges
    newEdges.reserve(50);

    bool isocta = false;
    bool ishex = false;
    bool isprism = false;
    bool ispyra = false;

    Dart centralDart = NIL;

    //Second step : deconnect each corner, close each hole, subdivide each new face into 3
    for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
    {
        Dart e = *edge;
        std::vector<Dart> v ;

        Dart f1 = map.phi1(*edge);
        //Dart f2 = map.phi2(f1);

        do
        {
            if(map.phi1(map.phi1(map.phi1(e))) != e)
                v.push_back(map.phi1(map.phi1(e)));

            v.push_back(map.phi1(e));

            e = map.phi2(map.phi_1(e));
        }
        while(e != *edge);

        map.splitVolume(v) ;

        //fonction qui calcule le degree max des faces atour d'un sommet
        unsigned int fdeg = map.faceDegree(map.phi2(f1));

        if(fdeg == 3)
        {
            unsigned int idface = map.getNewFaceId();
            map.setFaceId(map.phi2(f1),idface, FACE);
        }
        else if(fdeg == 4)
        {
            std::cout << "== 4" << std::endl;

            if(map.PFP::MAP::ParentMap::vertexDegree(*edge) == 3)
            {
                isocta = false;
                ispyra = true;

                std::cout << "pyra" << std::endl;

                Dart it = *edge;
                if((map.faceDegree(it) == 3) && (map.faceDegree(map.phi2(it))) == 3)
                {
                    it = map.phi2(map.phi_1(it));
                }
                else if((map.faceDegree(it) == 3) && (map.faceDegree(map.phi2(it)) == 4))
                {
                    it = map.phi1(map.phi2(it));
                }

                Dart old = map.phi2(map.phi1(it));
                Dart dd = map.phi1(map.phi1(old));

                map.splitFace(old,dd) ;
                centralDart = old;

                newEdges.push_back(map.phi_1(old));
            }
            else
            {
                isocta = true;

                std::cout << "octa" << std::endl;

                Dart old = map.phi2(map.phi1(*edge));
                Dart dd = map.phi1(old) ;
                map.splitFace(old,dd) ;

                Dart ne = map.phi1(old);

                map.cutEdge(ne);
                position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
                centralDart = map.phi1(ne);
                newEdges.push_back(ne);
                newEdges.push_back(map.phi1(ne));

                Dart stop = map.phi2(map.phi1(ne));
                ne = map.phi2(ne);
                do
                {
                    dd = map.phi1(map.phi1(ne));

                    map.splitFace(ne, dd) ;

                    newEdges.push_back(map.phi1(dd));

                    ne = map.phi2(map.phi_1(ne));
                    dd = map.phi1(dd);
                }
                while(dd != stop);
            }
        }
        else if(fdeg == 5)
        {
            std::cout << "== 5" << std::endl;

            isprism = true;

            Dart it = *edge;
            if(map.faceDegree(it) == 3)
            {
                it = map.phi2(map.phi_1(it));
            }
            else if(map.faceDegree(map.phi2(map.phi_1(*edge))) == 3)
            {
                it = map.phi2(map.phi_1(map.phi2(map.phi_1(it))));
            }

            Dart old = map.phi2(map.phi1(it));
            Dart dd = map.phi_1(map.phi_1(old));

            map.splitFace(old,dd) ;

            newEdges.push_back(map.phi_1(dd));
        }
        else if(fdeg == 6)
        {
            std::cout << "== 6" << std::endl;

            ishex = true;

            Dart old = map.phi2(map.phi1(e));
            Dart dd = map.phi1(map.phi1(old)) ;
            map.splitFace(old,dd) ;

            Dart ne = map.phi1(map.phi1(old)) ;

            map.cutEdge(ne);
            position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
            newEdges.push_back(ne);
            newEdges.push_back(map.phi1(ne));


            Dart stop = map.phi2(map.phi1(ne));
            ne = map.phi2(ne);
            do
            {
                dd = map.phi1(map.phi1(map.phi1(ne)));

                //A Verifier !!
                map.splitFace(ne, dd) ;

                newEdges.push_back(map.phi1(dd));

                ne = map.phi2(map.phi_1(ne));
                dd = map.phi1(map.phi1(dd));
            }
            while(dd != stop);
        }

        break;
    }



    if(ishex)
    {
        map.deleteVolume(map.phi3(map.phi2(map.phi1(oldEdges.front()))));

        //Third step : 3-sew internal faces
        for (std::vector<std::pair<char, std::pair<Dart,Dart> > >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
        {
            Dart f1 = map.phi2( ((*it).second).first );
            Dart f2 = map.phi2( ((*it).second).second );


            //if(map.phi3(map.phi2(f1)) == map.phi2(f1) && map.phi3(map.phi2(f2)) == map.phi2(f2))
            //if(map.phi3(f1) == f1 && map.phi3(f2) == f2)
            if(map.isBoundaryFace(f1) && map.isBoundaryFace(f2))
            {
                std::cout << "sewvolumes" << std::endl;

                map.sewVolumes(f1, f2);
                //id pour toutes les faces interieures
                //map.sewVolumes(map.phi2(f1), map.phi2(f2));

                //Fais a la couture !!!!!
                unsigned int idface = map.getNewFaceId();
                map.setFaceId(f1,idface, FACE);
            }

            //FAIS a la couture !!!!!!!
            //id pour toutes les aretes exterieurs des faces quadrangulees
            unsigned int idedge = map.getEdgeId(map.phi2(f1));
            map.setEdgeId(f1, idedge, DART);
            map.setEdgeId(f2, idedge, DART);

        }

        //LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
        //id pour les aretes interieurs : (i.e. 16 pour un octa)
        DartMarker<typename PFP::MAP> mne(map);
        for(std::vector<Dart>::iterator it = newEdges.begin() ; it != newEdges.end() ; ++it)
        {
            if(!mne.isMarked(*it))
            {
                unsigned int idedge = map.getNewEdgeId();
                map.setEdgeId(*it, idedge, EDGE);
                mne.markOrbit<EDGE>(*it);
            }
        }

    }

    if(isocta)
    {

        DartMarker<typename PFP::MAP> me(map);

        for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
        {
            Dart x = map.phi_1(map.phi2(map.phi1(*edge)));
            Dart f = x;

            do
            {
                Dart f3 = map.phi3(f);

                if(!me.isMarked(f3))
                {
                    Dart tmp =  map.phi_1(map.phi2(map.phi_1(map.phi2(map.phi_1(f3))))); //future voisin par phi2

                    Dart f32 = map.phi2(f3);
                    map.swapEdges(f3, tmp);

                    unsigned int idface = map.getNewFaceId();
                    map.setFaceId(f3,idface, FACE);
                    idface = map.getNewFaceId();
                    map.setFaceId(f32,idface, FACE);

                    unsigned int idedge = map.getNewEdgeId();
                    map.setEdgeId(f3, idedge, EDGE);

                    map.setEdgeId(f3,map.getEdgeId(f3), EDGE);
                    map.setEdgeId(f32,map.getEdgeId(f32), EDGE);

                    me.markOrbit<EDGE>(f3);
                    me.markOrbit<EDGE>(f32);
                }

                f = map.phi2(map.phi_1(f));
            }while(f != x);
        }

        //		map.template setOrbitEmbedding<VERTEX>(centralDart, map.template getEmbedding<VERTEX>(centralDart));
        Algo::Topo::setOrbitEmbedding<VERTEX>(map,centralDart,map.template getEmbedding<VERTEX>(centralDart));

        //Third step : 3-sew internal faces
        for (std::vector<std::pair<char, std::pair<Dart,Dart> > >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
        {
            Dart f1 = map.phi2( ((*it).second).first );
            Dart f2 = map.phi2( ((*it).second).second );
            unsigned int idedge = map.getEdgeId(f1);
            map.setEdgeId(f1, idedge, EDGE);
            map.setEdgeId(f2, idedge, EDGE);
        }

    }


    if(isprism)
    {
        //map.deleteVolume(map.phi3(map.phi2(map.phi1(oldEdges.front()))));
        //		for (std::vector<std::pair<char, std::pair<Dart,Dart> > >::iterator it = subdividedfaces.begin(); it != subdividedfaces.end(); ++it)
        //		{
        //			Dart f1 = map.phi2( ((*it).second).first );
        //			Dart f2 = map.phi2( ((*it).second).second );
        //		}
    }

    if(ispyra)
    {

    }


    //	{
    //	//Third step : 3-sew internal faces
    //	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfacesT.begin(); it != subdividedfacesT.end(); ++it)
    //	{
    //		Dart f1 = (*it).first;
    //		Dart f2 = (*it).second;
    //
    //		//FAIS a la couture !!!!!!!
    //		//id pour toutes les aretes exterieurs des faces quadrangulees
    //		unsigned int idedge = map.getEdgeId(f1);
    //		map.setEdgeId(map.phi2(f1), idedge, DART);
    //		map.setEdgeId( map.phi2(f2), idedge, DART);
    //
    //	}
    //
    //	//LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
    //	//id pour les aretes interieurs : (i.e. 16 pour un octa)
    //	DartMarker mne(map);
    //	for(std::vector<Dart>::iterator it = newEdges.begin() ; it != newEdges.end() ; ++it)
    //	{
    //		if(!mne.isMarked(*it))
    //		{
    //			unsigned int idedge = map.getNewEdgeId();
    //			map.setEdgeId(*it, idedge, EDGE);
    //			mne.markOrbit<EDGE>(*it);
    //		}
    //	}
    //	}
    //
    //	{
    //	//Third step : 3-sew internal faces
    //	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfacesQ.begin(); it != subdividedfacesQ.end(); ++it)
    //	{
    //		Dart f1 = (*it).first;
    //		Dart f2 = (*it).second;
    //
    //		if(map.phi3(map.phi2(f1)) == map.phi2(f1) && map.phi3(map.phi2(f2)) == map.phi2(f2))
    //		{
    //			//id pour toutes les faces interieures
    //			map.sewVolumes(map.phi2(f1), map.phi2(f2));
    //
    //			//Fais a la couture !!!!!
    //			unsigned int idface = map.getNewFaceId();
    //			map.setFaceId(map.phi2(f1),idface, FACE);
    //		}
    //
    //		//FAIS a la couture !!!!!!!
    //		//id pour toutes les aretes exterieurs des faces quadrangulees
    //		unsigned int idedge = map.getEdgeId(f1);
    //		map.setEdgeId(map.phi2(f1), idedge, DART);
    //		map.setEdgeId( map.phi2(f2), idedge, DART);
    //	}
    //
    //	//LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
    //	//id pour les aretes interieurs : (i.e. 6 pour un hexa)
    //	DartMarker mne(map);
    //	for(std::vector<Dart>::iterator it = newEdges.begin() ; it != newEdges.end() ; ++it)
    //	{
    //		if(!mne.isMarked(*it))
    //		{
    //			unsigned int idedge = map.getNewEdgeId();
    //			map.setEdgeId(*it, idedge, EDGE);
    //			mne.markOrbit<EDGE>(*it);
    //		}
    //	}
    //	}

    map.setCurrentLevel(cur) ;

    //	return subdividedfacesQ.begin()->first;
    return NIL;
}


//		if(fdeg > 4)
//		{
//			std::cout << "> 4" << std::endl;
//
//			ishex = true;
//
//			Dart old = map.phi2(map.phi1(e));
//			Dart dd = map.phi1(map.phi1(old)) ;
//			map.splitFace(old,dd) ;
//
//			Dart ne = map.phi1(map.phi1(old)) ;
//
//			map.cutEdge(ne);
//			position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
//			newEdges.push_back(ne);
//			newEdges.push_back(map.phi1(ne));
//
//
//			Dart stop = map.phi2(map.phi1(ne));
//			ne = map.phi2(ne);
//			do
//			{
//				dd = map.phi1(map.phi1(map.phi1(ne)));
//
//				//A Verifier !!
//				map.splitFace(ne, dd) ;
//
//				newEdges.push_back(map.phi1(dd));
//
//				ne = map.phi2(map.phi_1(ne));
//				dd = map.phi1(map.phi1(dd));
//			}
//			while(dd != stop);
//		}
//		else if(fdeg > 3)
//		{
//			std::cout << "> 3" << std::endl;
//
//			//map.closeHole(f2);
//			//map.sewVolumes(map.phi2(f1),map.phi2(f2));
//			istet = false;
//
//			Dart old = map.phi2(map.phi1(*edge));
//			Dart dd = map.phi1(old) ;
//			map.splitFace(old,dd) ;
//
//			Dart ne = map.phi1(old);
//
//			map.cutEdge(ne);
//			position[map.phi1(ne)] = volCenter; //plonger a la fin de la boucle ????
//			newEdges.push_back(ne);
//			newEdges.push_back(map.phi1(ne));
//
//			Dart stop = map.phi2(map.phi1(ne));
//			ne = map.phi2(ne);
//			do
//			{
//				dd = map.phi1(map.phi1(ne));
//
//				map.splitFace(ne, dd) ;
//
//				newEdges.push_back(map.phi1(dd));
//
//				ne = map.phi2(map.phi_1(ne));
//				dd = map.phi1(dd);
//			}
//			while(dd != stop);
//		}
//		else
//		{
//			//map.closeHole(f2);
//			//map.sewVolumes(map.phi2(f1),map.phi2(f2));
//
//			unsigned int idface = map.getNewFaceId();
//			map.setFaceId(map.phi2(f1),idface, FACE);
//		}


template <typename PFP>
Dart subdivideVolumeGen(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;

    unsigned int vLevel = map.volumeLevel(d);
    Dart old = map.volumeOldestDart(d);

    unsigned int cur = map.getCurrentLevel();
    map.setCurrentLevel(vLevel);

    /*
     * au niveau du volume courant i
     * stockage d'un brin de chaque face de celui-ci
     * avec calcul du centroid
     */

    DartMarkerStore<typename PFP::MAP> mf(map);		// Lock a face marker to save one dart per face
    CellMarker<typename PFP::MAP, VERTEX> mv(map);

    typename PFP::VEC3 volCenter;
    unsigned count = 0 ;

    //Store faces that are traversed and start with the face of d
    std::vector<Dart> visitedFaces;
    visitedFaces.reserve(512);
    visitedFaces.push_back(old);

    //Store the edges before the cutEdge
    std::vector<Dart> oldEdges;
    oldEdges.reserve(512);

    mf.markOrbit<FACE>(old) ;

    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
    {
        Dart e = visitedFaces[i] ;
        do
        {
            //add one old edge per vertex to the old edge list
            //compute volume centroid
            if(!mv.isMarked(e))
            {
                mv.mark(e);
                volCenter += position[e];
                ++count;
                oldEdges.push_back(e);
            }

            // add all face neighbours to the table
            Dart ee = map.phi2(e) ;
            if(!mf.isMarked(ee)) // not already marked
            {
                visitedFaces.push_back(ee) ;
                mf.markOrbit<FACE>(ee) ;
            }

            e = map.phi1(e) ;
        } while(e != visitedFaces[i]) ;
    }

    volCenter /= typename PFP::REAL(count) ;

    /*
     * Subdivision
     */
    //Store the darts from quadrangulated faces
    std::vector<std::pair<Dart,Dart> > subdividedfacesQ;
    subdividedfacesQ.reserve(25);

    std::vector<std::pair<Dart,Dart> > subdividedfacesT;
    subdividedfacesT.reserve(25);


    //First step : subdivide edges and faces
    //creates a i+1 edge level and i+1 face level
    for (std::vector<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
    {
        Dart d = *face;

        //if needed subdivide face
        if(!map.faceIsSubdivided(d))
            IHM::subdivideFace<PFP>(map, d, position);


        //save a dart from the subdivided face
        unsigned int cur = map.getCurrentLevel() ;

        unsigned int fLevel = map.faceLevel(d) + 1; //puisque dans tous les cas, la face est subdivisee
        map.setCurrentLevel(fLevel) ;

        //test si la face est triangulaire ou non
        if(map.phi1(map.phi1(map.phi1(d))) == d)
        {
            //std::cout << "trian" << std::endl;
            Dart cf = map.phi2(map.phi1(d));
            Dart e = cf;
            do
            {
                subdividedfacesT.push_back(std::pair<Dart,Dart>(e,map.phi2(e)));
                e = map.phi1(e);
            }while (e != cf);
        }
        else
        {
            //std::cout << "quad" << std::endl;
            Dart cf = map.phi1(d);
            Dart e = cf;
            do
            {
                subdividedfacesQ.push_back(std::pair<Dart,Dart>(e,map.phi2(e)));
                e = map.phi2(map.phi1(e));
            }while (e != cf);


        }

        map.setCurrentLevel(cur);
    }

    map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

    std::vector<Dart> newEdges;	//save darts from inner edges
    newEdges.reserve(50);

//    //Second step : deconnect each corner, close each hole, subdivide each new face into 3
//    for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
//    {
//        Dart e = *edge;

//        std::vector<Dart> v ;

//        do
//        {
//            if(map.phi1(map.phi1(map.phi1(e))) != e)
//                v.push_back(map.phi1(map.phi1(e))); //remplacer par une boucle qui découd toute la face et non juste une face carre (jusqu'a phi_1(e))

//            v.push_back(map.phi1(e));

//            e = map.phi2(map.phi_1(e));
//        }
//        while(e != *edge);

//        map.splitVolume(v) ;

//        //degree du sommet exterieur
//        unsigned int cornerDegree = map.Map2::vertexDegree(*edge);

//        //tourner autour du sommet pour connaitre le brin d'un sommet de valence < cornerDegree
//        bool found = false;
//        Dart stop = e;
//        do
//        {

//            if(map.Map2::vertexDegree(map.phi2(map.phi1(e))) < cornerDegree)
//            {
//                stop = map.phi2(map.phi1(e));
//                found = true;
//            }

//            e = map.phi2(map.phi_1(e));
//        }
//        while(!found && e != *edge);

//        //si il existe un sommet de degre inferieur au degree du coin
//        if(found)
//        {
//            //chercher le brin de faible degree suivant
//            bool found2 = false;
//            Dart dd = map.phi1(stop);

//            do
//            {
//                if(map.Map2::vertexDegree(dd) < cornerDegree)
//                    found2 = true;
//                else
//                    dd = map.phi1(dd);
//            }
//            while(!found2);

//            //cas de la pyramide
//            if(dd == stop)
//            {
//                //std::cout << "pyramide" << std::endl;
//                map.splitFace(dd, map.phi1(map.phi1(dd)));
//            }
//            else
//            {
//                map.splitFace(dd, stop);

//                //calcul de la taille des faces de chaque cote de stop
//                if(!( (map.Map2::faceDegree(map.phi_1(stop)) == 3 && map.Map2::faceDegree(map.phi2(map.phi_1(stop))) == 4) ||
//                      (map.Map2::faceDegree(map.phi_1(stop)) == 4 && map.Map2::faceDegree(map.phi2(map.phi_1(stop))) == 3) ))
//                {
//                    //std::cout << "octaedre ou hexaedre" << std::endl;

//                    Dart ne = map.phi_1(stop) ;
//                    map.cutEdge(ne);
//                    position[map.phi1(ne)] = volCenter;
//                    stop = map.phi2(map.phi1(ne));

//                    bool finished = false;
//                    Dart it = map.phi2(ne);

//                    do
//                    {
//                        //chercher le brin de faible degree suivant
//                        bool found2 = false;
//                        Dart dd = map.phi1(it);

//                        do
//                        {
//                            if(dd == stop)
//                                finished = true;
//                            else if(map.Map2::vertexDegree(dd) < cornerDegree)
//                                found2 = true;
//                            else
//                                dd = map.phi1(dd);
//                        }
//                        while(!found2 & !finished);

//                        if(found2)
//                        {
//                            map.splitFace(it,dd);
//                        }

//                        it = map.phi_1(dd);

//                        if(it == stop)
//                            finished = true;

//                    }
//                    while(!finished);

//                }
//                else
//                {
//                    //std::cout << "prisme" << std::endl;
//                    //tester si besoin de fermer f2 (par exemple pas besoin pour hexa... mais pour tet, octa, prisme oui)
//                    //map.closeHole(f2);
//                }

//            }

//        }
//        //sinon cas du tetraedre
//        else
//        {
//            //std::cout << "tetraedre" << std::endl;
//            //tester si besoin de fermer f2 (par exemple pas besoin pour hexa... mais pour tet, octa, prisme oui)
//            //map.closeHole(f2);
//        }

//    }




//    //std::cout << "1ere etape finished" << std::endl;

//    CellMarker<typename PFP::MAP, FACE> mtf(map);

//    //Etape 2
//    for (std::vector<std::pair<Dart,Dart> >::iterator edges = subdividedfacesT.begin(); edges != subdividedfacesT.end(); ++edges)
//    {
//        //		Dart f1 = (*edges).first;
//        Dart f2 = (*edges).second;

//        //si ce n'est pas un tetrahedre
//        if( !( (map.Map2::faceDegree(f2) == 3 && map.Map2::faceDegree(map.phi2(f2)) == 3 &&
//                map.Map2::faceDegree(map.phi2(map.phi_1(f2))) == 3) && map.Map2::vertexDegree(f2) == 3))
//        {

//            //map.deleteVolume(map.phi3(map.phi2(map.phi1(oldEdges.front()))));

//            //			if(!mtf.isMarked(f1))
//            //			{
//            //				mtf.mark(f1);
//            //
//            //				map.closeHole(f1);
//            //
//            //				if(map.Map2::faceDegree(map.phi2(f2)) == 3)
//            //				{
//            //					//std::cout << "ajout d'un tetraedre" << std::endl;
//            //					Dart x = Algo::Modelisation::trianguleFace<PFP>(map, map.phi2(f1));
//            //					position[x] = volCenter;
//            //				}
//            //				else
//            //				{
//            //					//std::cout << "ajout d'un prisme" << std::endl;
//            //					//Dart x = Algo::Modelisation::extrudeFace<PFP>(map,position,map.phi2(f1),5.0);
//            //					Dart c = Algo::Modelisation::trianguleFace<PFP>(map, map.phi2(f1));
//            //
//            //					Dart cc = c;
//            //					// cut edges
//            //					do
//            //					{
//            //
//            //						typename PFP::VEC3 p1 = position[cc] ;
//            //						typename PFP::VEC3 p2 = position[map.phi1(cc)] ;
//            //
//            //						map.cutEdge(cc);
//            //
//            //						position[map.phi1(cc)] = (p1 + p2) * typename PFP::REAL(0.5) ;
//            //
//            //						cc = map.phi2(map.phi_1(cc));
//            //					}while (cc != c);
//            //
//            //					// cut faces
//            //					do
//            //					{
//            //						Dart d1 = map.phi1(cc);
//            //						Dart d2 = map.phi_1(cc);
//            //						map.splitFace(d1,d2);
//            //						cc = map.phi2(map.phi_1(cc));//map.Map2::alpha1(cc);
//            //					}while (cc != c);
//            //
//            //					//merge central faces by removing edges
//            //					bool notFinished=true;
//            //					do
//            //					{
//            //						Dart d1 = map.Map2::alpha1(cc);
//            //						if (d1 == cc)			// last edge is pending edge inside of face
//            //							notFinished = false;
//            //						map.deleteFace(cc);
//            //						cc = d1;
//            //					} while (notFinished);
//            //
//            //
//            //					map.closeHole(map.phi1(map.phi1(map.phi2(f1))));
//            //
//            //				}
//            //			}

//        }
//    }

    //std::cout << "2e etape finished" << std::endl;


    //	{
    //		//Third step : 3-sew internal faces
    //		for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfacesT.begin(); it != subdividedfacesT.end(); ++it)
    //		{
    //			Dart f1 = (*it).first;
    //			Dart f2 = (*it).second;
    //
    //
    //
    //			if(map.phi3(map.phi2(f1)) == map.phi2(f1) && map.phi3(map.phi2(f2)) == map.phi2(f2))
    //			{
    //				if(map.getEmbedding<VERTEX>(map.phi_1(map.phi2(f2))) == map.getEmbedding<VERTEX>(map.phi_1(map.phi2(f1))))
    //				{
    //					map.Map3::sewVolumes(map.phi2(f2), map.phi2(f1));
    //				}
    //				else
    //				{
    //
    //				//id pour toutes les faces interieures
    //				map.sewVolumes(map.phi2(f2), map.phi2(f1));
    //
    //
    //				}
    //
    //				//Fais a la couture !!!!!
    //				unsigned int idface = map.getNewFaceId();
    //				map.setFaceId(map.phi2(f1),idface, FACE);
    //			}
    //
    //
    //			//FAIS a la couture !!!!!!!
    //			//id pour toutes les aretes exterieurs des faces quadrangulees
    //			unsigned int idedge = map.getEdgeId(f1);
    //			map.setEdgeId(map.phi2(f1), idedge, DART);
    //			map.setEdgeId( map.phi2(f2), idedge, DART);
    //
    //		}
    //
    //		//LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
    //		//id pour les aretes interieurs : (i.e. 16 pour un octa)
    //		DartMarker mne(map);
    //		for(std::vector<Dart>::iterator it = newEdges.begin() ; it != newEdges.end() ; ++it)
    //		{
    //			if(!mne.isMarked(*it))
    //			{
    //				unsigned int idedge = map.getNewEdgeId();
    //				map.setEdgeId(*it, idedge, EDGE);
    //				mne.markOrbit<EDGE>(*it);
    //			}
    //		}
    //	}
    //
    //	{
    //		//Third step : 3-sew internal faces
    //		for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfacesQ.begin(); it != subdividedfacesQ.end(); ++it)
    //		{
    //			Dart f1 = (*it).first;
    //			Dart f2 = (*it).second;
    //
    //			if(map.phi3(map.phi2(f1)) == map.phi2(f1) && map.phi3(map.phi2(f2)) == map.phi2(f2))
    //			{
    //				//id pour toutes les faces interieures
    //				map.sewVolumes(map.phi2(f2), map.phi2(f1));
    //
    //				//Fais a la couture !!!!!
    //				unsigned int idface = map.getNewFaceId();
    //				map.setFaceId(map.phi2(f1),idface, FACE);
    //			}
    //
    //			//FAIS a la couture !!!!!!!
    //			//id pour toutes les aretes exterieurs des faces quadrangulees
    //			unsigned int idedge = map.getEdgeId(f1);
    //			map.setEdgeId(map.phi2(f1), idedge, DART);
    //			map.setEdgeId( map.phi2(f2), idedge, DART);
    //		}
    //		//LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
    //		//id pour les aretes interieurs : (i.e. 16 pour un octa)
    //		DartMarker mne(map);
    //		for(std::vector<Dart>::iterator it = newEdges.begin() ; it != newEdges.end() ; ++it)
    //		{
    //			if(!mne.isMarked(*it))
    //			{
    //				unsigned int idedge = map.getNewEdgeId();
    //				map.setEdgeId(*it, idedge, EDGE);
    //				mne.markOrbit<EDGE>(*it);
    //			}
    //		}
    //	}
    //
    //	//cas tordu pour le prisme
    //	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedfacesT.begin(); it != subdividedfacesT.end(); ++it)
    //	{
    //		Dart f1 = (*it).first;
    //		Dart f2 = (*it).second;
    //
    //		if(  !(map.Map2::faceDegree(f2) == 3 && map.Map2::faceDegree(map.phi2(f2)) == 3 &&
    //				map.Map2::faceDegree(map.phi2(map.phi1(f2))) == 3 && map.Map2::faceDegree(map.phi2(map.phi_1(f2))) == 3))
    //		{
    //
    //
    //			if(map.phi2(map.phi1(map.phi1(map.phi2(f1)))) == map.phi3(map.phi2(map.phi1(map.phi1(map.phi2(f1))))) &&
    //					map.phi2(map.phi3(map.phi2(map.phi3(map.phi2(map.phi3(map.phi2(map.phi2(map.phi1(map.phi1(map.phi2(f1))))))))))) ==
    //							map.phi3(map.phi2(map.phi3(map.phi2(map.phi3(map.phi2(map.phi3(map.phi2(map.phi2(map.phi1(map.phi1(map.phi2(f1))))))))))))
    //			)
    //			{
    //				map.sewVolumes(map.phi2(map.phi1(map.phi1(map.phi2(f1)))),
    //						map.phi2(map.phi3(map.phi2(map.phi3(map.phi2(map.phi3(map.phi1(map.phi1(map.phi2(f1))))))))));
    //			}
    //
    //		}
    //	}

    map.setCurrentLevel(cur) ;

    return subdividedfacesQ.begin()->first;
}



/***********************************************************************************
 *												Raffinement
 ***********************************************************************************/
template <typename PFP>
void splitVolume(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
    assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    assert(!map.volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;

    unsigned int cur = map.getCurrentLevel() ;
    unsigned int vLevel = map.volumeLevel(d) ;
    map.setCurrentLevel(vLevel) ;

    // first cut the edges (if they are not already)
    Dart t = d;
    do
    {
        if(!map.edgeIsSubdivided(map.phi1(map.phi2(t))))
            IHM::subdivideEdge<PFP>(map, map.phi1(map.phi2(t)), position) ;
        t = map.phi1(t);
    }
    while(t != d);

    unsigned int fLevel = map.faceLevel(map.phi1(d));
    map.setCurrentLevel(fLevel+1) ;		// go to the level of the face to subdivide its edges
    Dart neighboordVolume = map.phi1(map.phi1(map.phi2(d)));
    //map.setCurrentLevel(cur) ; // go to the next level to perform volume subdivision

    //Split the faces and open the midlle
    do
    {
        Dart t2 = map.phi2(t);

        unsigned int fLevel = map.faceLevel(t2) ;
        map.setCurrentLevel(fLevel+1) ;		// go to the level of the face to subdivide its edges

        Dart face2 = map.phi1(map.phi1(t2));
        map.splitFace(map.phi_1(t2), face2);
        map.unsewFaces(map.phi1(map.phi1(t2)));

        //id de face pour les 2 nouveaux brins
        unsigned int idface = map.getFaceId(t2);
        map.setFaceId(map.phi1(map.phi1(t2)), idface, DART);
        map.setFaceId(map.phi_1(face2), idface, DART);

        t = map.phi1(t);
    }
    while(t != d);

    //close the middle to create volumes & sew them
    map.setCurrentLevel(vLevel + 1) ; // go to the next level to perform volume subdivision

    map.closeHole(map.phi1(map.phi1(map.phi2(d))));
    map.closeHole(map.phi_1(neighboordVolume));
    map.sewVolumes(map.phi2(map.phi1(map.phi1(map.phi2(d)))), map.phi2(map.phi_1(neighboordVolume)));

    unsigned int idface = map.getNewFaceId();
    map.setFaceId(map.phi2(map.phi_1(neighboordVolume)), idface, FACE);

    do
    {
        Dart t211 = map.phi1(map.phi1(map.phi2(t)));

        unsigned int idedge = map.getNewEdgeId();
        map.setEdgeId(t211, idedge, EDGE);
        t = map.phi1(t);
    }
    while(t != d);

    map.setCurrentLevel(cur) ;
}

} //namespace IHM
} //Volume
} //namespace Algo
} //namespace CGoGN


