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
//#include "Algo/Modelisation/extrusion.h"
#include "Geometry/intersection.h"
//#include "NL/nl.h"
//#include "Algo/LinearSolving/basic.h"
#include "Algo/Geometry/laplacian.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Modelisation
{

template <typename PFP>
bool isHexahedron(typename PFP::MAP& the_map, Dart d, unsigned int thread)
{
    unsigned int nbFaces = 0;

    //Test the number of faces end its valency
    Traversor3WF<typename PFP::MAP> travWF(the_map, d, false, thread);
    for(Dart dit = travWF.begin() ; dit != travWF.end(); dit = travWF.next())
    {
        //increase the number of faces
        nbFaces++;
        if(nbFaces > 6)	//too much faces
            return false;

        //test the valency of this face
        if(the_map.faceDegree(dit) != 4)
            return false;
    }

    return true;
}

template <typename PFP>
Dart cut3Ear(typename PFP::MAP& map, Dart d)
{
	Dart e = d;
	int nb = 0;
	Dart dNew;

	Dart dRing;
	Dart dRing2;

	//count the valence of the vertex
	do
	{
		nb++;
		e = map.phi1(map.phi2(e));
	} while (e != d);

	if(nb < 3)
	{
		CGoGNout << "Warning : cannot cut 2 volumes without creating a degenerated face " << CGoGNendl;
		return d;
	}
	else
	{
		std::vector<Dart> vPath;

		//triangulate around the vertex
		do
		{
			Dart dN = map.phi1(map.phi2(e));
			if(map.template phi<111>(e) != e)
				map.splitFace(map.phi_1(e), map.phi1(e));

			dRing = map.phi2(map.phi1(e));

			vPath.push_back(dRing); //remember all darts from the ring

			e = dN;
		} while (e != d);

		map.splitVolume(vPath);
	}

	return map.phi2(dRing);
}

template <typename PFP>
Dart sliceConvexVolume(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, Dart d, Geom::Plane3D<typename PFP::REAL > pl)
{
    typedef typename PFP::MAP MAP;
	Dart dRes=NIL;
	unsigned int nbInter = 0;
	unsigned int nbVertices = 0;
    CellMarkerStore< MAP, VERTEX> vs(map);			//marker for new vertices from edge cut
    CellMarkerStore< MAP, FACE> cf(map);
	Dart dPath;

	MarkerForTraversor<typename PFP::MAP::ParentMap, EDGE > mte(map);
	MarkerForTraversor<typename PFP::MAP::ParentMap, FACE > mtf(map);

	//search edges and vertices crossing the plane
	Traversor3WE<typename PFP::MAP::ParentMap > te(map,d);
	for(Dart dd = te.begin() ;dd != te.end() ; dd = te.next())
	{
		if(!mte.isMarked(dd))
		{
			if(fabs(pl.distance(position[dd]))<0.000001f)
			{
				nbVertices++;
				vs.mark(dd); //mark vertex on slicing path
				mte.mark(dd);
			}
			else
			{
				typename PFP::VEC3 interP;
				typename PFP::VEC3 vec(Surface::Geometry::vectorOutOfDart<PFP>(map,dd,position));
				Geom::Intersection inter = Geom::intersectionLinePlane<typename PFP::VEC3, typename Geom::Plane3D<typename PFP::REAL > >(position[dd],vec,pl,interP);

				if(inter==Geom::FACE_INTERSECTION)
				{
					Dart dOp = map.phi1(dd);
					typename PFP::VEC3 v2(interP-position[dd]);
					typename PFP::VEC3 v3(interP-position[dOp]);
					if(vec.norm2()>v2.norm2() && vec.norm2()>v3.norm2())
					{
						nbInter++;

						cf.mark(dd);			//mark face and opposite face to split
						cf.mark(map.phi2(dd));

						map.cutEdge(dd);
						Dart dN = map.phi1(dd);

						mte.mark(dN);

						vs.mark(dN);			//mark vertex for split
						position[dN] = interP; 	//place
					}
				}
			}
		}
	}

//	std::cout << "edges cut: " << nbInter << std::endl;
	unsigned int nbSplit=0;

	//slice when at least two edges are concerned
	if(nbInter>1)
	{
		Traversor3WF<typename PFP::MAP::ParentMap > tf(map,d);
		for(Dart dd = tf.begin() ; dd != tf.end() ; dd = tf.next())
		{
			//for faces with a new vertex
			if(cf.isMarked(dd))
			{
				cf.unmark(dd);

				Dart dS = dd;
				bool split=false;

				do
				{
					//find the new vertex
					if(vs.isMarked(dS))
					{
						Dart dSS = map.phi1(dS);
						//search an other new vertex (or an existing vertex intersected with the plane) in order to split the face
						do
						{
							if(vs.isMarked(dSS))
							{
								nbSplit++;
								map.splitFace(dS,dSS);
								dPath=map.phi_1(dS);
								split=true;
							}
							dSS = map.phi1(dSS);
						} while(!split && dSS!=dS);
					}
					dS = map.phi1(dS);
				} while(!split && dS!=dd);
			}
		}

//		std::cout << "face split " << nbSplit << std::endl;

		//define the path to split
		std::vector<Dart> vPath;
		vPath.reserve((nbSplit+nbVertices)+1);
		vPath.push_back(dPath);
		for(std::vector<Dart>::iterator it = vPath.begin() ;it != vPath.end() ; ++it)
		{
			Dart dd = map.phi1(*it);

			Dart ddd = map.phi1(map.phi2(dd));

			while(!vs.isMarked(map.phi1(ddd)) && ddd!=dd)
				ddd = map.phi1(map.phi2(ddd));

			if(vs.isMarked(map.phi1(ddd)) && !map.sameVertex(ddd,*vPath.begin()))
				vPath.push_back(ddd);
		}

		assert(vPath.size()>2);
		map.splitVolume(vPath);
		dRes = map.phi2(*vPath.begin());
	}

	return dRes;
}

template <typename PFP>
Dart sliceConvexVolume(typename PFP::MAP& map, VertexAttribute<typename PFP::MAP, typename PFP::VEC3>& position, Dart d, CellMarker<typename PFP::MAP, EDGE>& edgesToCut, CellMarker<typename PFP::MAP, VERTEX>& verticesToSplit)
{
    typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

	Dart dRes;
	unsigned int nbInter = 0;
	unsigned int nbVertices = 0;
    CellMarkerStore<MAP, VERTEX> vs(map);			//marker for new vertices from edge cut
    CellMarkerStore<MAP, FACE> cf(map);
	Dart dPath;

	MarkerForTraversor<typename PFP::MAP::ParentMap, EDGE > mte(map);
	MarkerForTraversor<typename PFP::MAP::ParentMap, FACE > mtf(map);

	//search edges and vertices crossing the plane
	Traversor3WE<typename PFP::MAP::ParentMap > te(map,d);
	for(Dart dd = te.begin() ;dd != te.end() ; dd = te.next())
	{
		if(!mte.isMarked(dd) && edgesToCut.isMarked(dd))
		{
			nbInter++;
			VEC3 p = (position[dd]+position[map.phi1(dd)])*0.5f;
			cf.mark(dd);			//mark face and opposite face to split
			cf.mark(map.phi2(dd));

			map.cutEdge(dd);
			Dart dN = map.phi1(dd);

			mte.mark(dN);

			vs.mark(dN);		//mark vertex for split
			position[dN] = p;
		}
	}

//	std::cout << "edges cut: " << nbInter << std::endl;
	unsigned int nbSplit=0;

	//at least two edges are concerned
	assert(nbInter>1);

	Traversor3WF<typename PFP::MAP::ParentMap > tf(map,d);
	for(Dart dd = tf.begin() ; dd != tf.end() ; dd = tf.next())
	{
		//for faces with a new vertex
		if(cf.isMarked(dd))
		{
			cf.unmark(dd);

			Dart dS = dd;
			bool split=false;

			do {
				//find the new vertex
				if(vs.isMarked(dS) || verticesToSplit.isMarked(dS))
				{
					Dart dSS = map.phi1(dS);
					//search an other new vertex (or an existing vertex intersected with the plane) in order to split the face
					do {
						if(vs.isMarked(dSS) || verticesToSplit.isMarked(dSS))
						{
							nbSplit++;
							map.splitFace(dS,dSS);
							dPath=map.phi_1(dS);
							split=true;
						}
						dSS = map.phi1(dSS);
					} while(!split && dSS!=dS);
				}
				dS = map.phi1(dS);
			} while(!split && dS!=dd);
		}

		//define the path to split
		std::vector<Dart> vPath;
		vPath.reserve((nbSplit+nbVertices)+1);
		vPath.push_back(dPath);
		for(std::vector<Dart>::iterator it = vPath.begin() ;it != vPath.end() ; ++it)
		{
			Dart dd = map.phi1(*it);

			Dart ddd = map.phi1(map.phi2(dd));

			while(!vs.isMarked(map.phi1(ddd)) && ddd!=dd)
				ddd = map.phi1(map.phi2(ddd));

			if(vs.isMarked(map.phi1(ddd)) && !map.sameVertex(ddd,*vPath.begin()))
				vPath.push_back(ddd);
		}

		assert(vPath.size()>2);
		map.splitVolume(vPath);
		dRes = map.phi2(*vPath.begin());
	}

	return dRes;
}

template <typename PFP>
std::vector<Dart> sliceConvexVolumes(typename PFP::MAP& map, VertexAttribute<typename PFP::MAP, typename PFP::VEC3>& position,CellMarker<typename PFP::MAP, VOLUME>& volumesToCut, CellMarker<typename PFP::MAP, EDGE>& edgesToCut, CellMarker<typename PFP::MAP, VERTEX>& verticesToSplit)
{
    typedef typename PFP::MAP MAP;
    std::vector<Dart> vRes;

    typedef typename PFP::VEC3 VEC3;
    CellMarker<MAP, VERTEX> localVerticesToSplit(map); //marker for new vertices from edge cut

    //Step 1: Cut the edges and mark the resulting vertices as vertices to be face-split
    TraversorE<typename PFP::MAP> te(map);
    CellMarkerStore<MAP, FACE> cf(map);

    for(Dart d = te.begin(); d != te.end(); d=te.next()) //cut all edges
    {
        if(edgesToCut.isMarked(d))
        {
            VEC3 p = (position[d]+position[map.phi1(d)])*0.5f;

            //turn around the edge and mark for future split face
            Traversor3EF<typename PFP::MAP> t3ef(map,d);
            for(Dart dd = t3ef.begin() ; dd != t3ef.end() ; dd = t3ef.next())
            	cf.mark(dd);			//mark face to split

            map.cutEdge(d);
            Dart dN = map.phi1(d);

            localVerticesToSplit.mark(dN);		//mark vertex for split
            position[dN] = p;
        }
    }

    //Step 2: Split faces with cut edges
    TraversorF<typename PFP::MAP> tf(map);
    for(Dart d = tf.begin(); d != tf.end(); d=tf.next())
    {
        if(cf.isMarked(d))
        {
            cf.unmark(d);
            Dart dS = d;
            bool split=false;
            do
            {
                //find the new vertex
                if(localVerticesToSplit.isMarked(dS) || verticesToSplit.isMarked(dS))
                {
                	//start from phi1(phi1()) to avoid the creation of faces of degree 2
                    Dart dSS = map.phi1(map.phi1(dS));
                    //search an other new vertex (or an existing vertex to split) in order to split the face

                    do
                    {
                        if((localVerticesToSplit.isMarked(dSS) || verticesToSplit.isMarked(dSS))
                        		&& !map.sameVertex(dS,dSS))
                        {
                            map.splitFace(dS,dSS);
                            split=true;
                        }
                        dSS = map.phi1(dSS);
                    } while(!split && dSS!=dS);
                    split=true; //go out of the first loop if no split case has been found
                }
                dS = map.phi1(dS);
            } while(!split && dS!=d);
        }
    }

    //Step 3 : Find path and split volumes
    TraversorW<typename PFP::MAP> tw(map);
    for(Dart d = tw.begin(); d != tw.end(); d=tw.next()) //Parcours des volumes
    {
        if(volumesToCut.isMarked(d))
        {
            Traversor3WV<typename PFP::MAP> t3wv(map,d);
            Dart dPath;
            bool found=false;

            //find a vertex of the volume to start the path to split
            for(Dart dd = t3wv.begin(); dd != t3wv.end() && !found; dd=t3wv.next())
            {
                if(localVerticesToSplit.isMarked(dd) || verticesToSplit.isMarked(dd))
                {
                    Dart ddd = dd;
                    while(!localVerticesToSplit.isMarked(map.phi1(ddd))
                    		&& !verticesToSplit.isMarked(map.phi1(ddd)))
                        ddd = map.phi1(map.phi2(ddd));
                    found=true;
                    dPath=ddd;
                }
            }
            //define the path to split
            std::vector<Dart> vPath;
            vPath.reserve(32);
            vPath.push_back(dPath);
            CellMarker<MAP, FACE> cmf(map);


            //define the path to split for the whole volume
            bool pathFound=false;
            for(std::vector<Dart>::iterator it = vPath.begin() ; !pathFound && it != vPath.end(); ++it)
            {
                Dart dd = map.phi1(*it);

                if(map.sameVertex(dd,*vPath.begin()))
                	pathFound=true;
                else
                {
                	Dart ddd = map.phi1(map.phi2(dd));

                	while(!localVerticesToSplit.isMarked(map.phi1(ddd)) && !verticesToSplit.isMarked(map.phi1(ddd)))
                		ddd = map.phi1(map.phi2(ddd));

                	vPath.push_back(ddd);
                }
            }

            assert(vPath.size()>2);
            map.splitVolume(vPath);
            vRes.push_back(map.phi2(*vPath.begin()));
        }
    }

    return vRes;
}

template <typename PFP, typename EMBV>
void catmullClarkVol(typename PFP::MAP& map, EMBV& attributs)
{
    typedef typename PFP::MAP MAP;
	typedef typename EMBV::DATA_TYPE EMB;

	//std::vector<Dart> l_centers;
	std::vector<Dart> l_vertices;

	//pre-computation : compute the centroid of all volume
    VolumeAutoAttribute<MAP, EMB> attBary(map);
	Volume::Geometry::computeCentroidVolumes<PFP>(map, const_cast<const EMBV&>(attributs), attBary);

	//subdivision
	//1. cut edges
    DartMarkerNoUnmark<MAP> mv(map);
	TraversorE<typename PFP::MAP> travE(map);
	for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
	{
		//memorize each vertices per volumes
		if( !mv.isMarked(d))
		{
			l_vertices.push_back(d);
			mv.markOrbit<PFP::MAP::VERTEX_OF_PARENT>(d);
		}

		Dart f = map.phi1(d);
		map.cutEdge(d) ;
		Dart e = map.phi1(d) ;

		attributs[e] =  attributs[d];
		attributs[e] += attributs[f];
		attributs[e] *= 0.5;

		travE.skip(d) ;
		travE.skip(e) ;
	}

	//2. split faces - quadrangule faces
	TraversorF<typename PFP::MAP> travF(map) ;
	for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
	{
		EMB center = Surface::Geometry::faceCentroid<PFP,EMBV>(map,d,attributs);

		Dart dd = map.phi1(d) ;
		Dart next = map.phi1(map.phi1(dd)) ;
		map.splitFace(dd, next) ;

		Dart ne = map.phi2(map.phi_1(dd)) ;
		map.cutEdge(ne) ;
		travF.skip(dd) ;

		attributs[map.phi1(ne)] = center;

		dd = map.phi1(map.phi1(next)) ;
		while(dd != ne)
		{
			Dart tmp = map.phi1(ne) ;
			map.splitFace(tmp, dd) ;
			travF.skip(tmp) ;
			dd = map.phi1(map.phi1(dd)) ;
		}

		travF.skip(ne) ;
	}

	//3. create inside volumes

	std::vector<std::pair<Dart, Dart> > subdividedFaces;
	subdividedFaces.reserve(2048);
	for (std::vector<Dart>::iterator it = l_vertices.begin(); it != l_vertices.end(); ++it)
	{
		Dart e = *it;
		std::vector<Dart> v ;

		do
		{
			v.push_back(map.phi1(e));
			v.push_back(map.phi1(map.phi1(e)));

			if(!map.PFP::MAP::ParentMap::isBoundaryEdge(map.phi1(e)))
				subdividedFaces.push_back(std::pair<Dart,Dart>(map.phi1(e),map.phi2(map.phi1(e))));

			if(!map.PFP::MAP::ParentMap::isBoundaryEdge(map.phi1(map.phi1(e))))
				subdividedFaces.push_back(std::pair<Dart,Dart>(map.phi1(map.phi1(e)),map.phi2(map.phi1(map.phi1(e)))));

			e = map.phi2(map.phi_1(e));
		}
		while(e != *it);

		//
		// SplitSurfaceInVolume
		//

		std::vector<Dart> vd2 ;
		vd2.reserve(v.size());

		// save the edge neighbors darts
		for(std::vector<Dart>::iterator it2 = v.begin() ; it2 != v.end() ; ++it2)
		{
			vd2.push_back(map.phi2(*it2));
		}

		assert(vd2.size() == v.size());

		map.PFP::MAP::ParentMap::splitSurface(v, true, false);

		// follow the edge path a second time to embed the vertex, edge and volume orbits
		for(unsigned int i = 0; i < v.size(); ++i)
		{
			Dart dit = v[i];
			Dart dit2 = vd2[i];

			// embed the vertex embedded from the origin volume to the new darts
			if(map.template isOrbitEmbedded<VERTEX>())
			{
				map.template copyDartEmbedding<VERTEX>(map.phi2(dit), map.phi1(dit));
				map.template copyDartEmbedding<VERTEX>(map.phi2(dit2), map.phi1(dit2));
			}

			// embed the edge embedded from the origin volume to the new darts
			if(map.template isOrbitEmbedded<EDGE>())
			{
				unsigned int eEmb = map.template getEmbedding<EDGE>(dit) ;
				map.template setDartEmbedding<EDGE>(map.phi2(dit), eEmb);
				map.template setDartEmbedding<EDGE>(map.phi2(dit2), eEmb);
			}

			// embed the volume embedded from the origin volume to the new darts
			if(map.template isOrbitEmbedded<VOLUME>())
			{
				map.template copyDartEmbedding<VOLUME>(map.phi2(dit), dit);
				map.template copyDartEmbedding<VOLUME>(map.phi2(dit2), dit2);
			}
		}

		//
		//
		//

		Dart dd = map.phi2(map.phi1(*it));
		Dart next = map.phi1(map.phi1(dd)) ;
		map.PFP::MAP::ParentMap::splitFace(dd, next);

		if (map.template isOrbitEmbedded<VERTEX>())
		{
			map.template copyDartEmbedding<VERTEX>(map.phi_1(next), dd) ;
			map.template copyDartEmbedding<VERTEX>(map.phi_1(dd), next) ;
		}

		Dart ne = map.phi2(map.phi_1(dd));
		map.PFP::MAP::ParentMap::cutEdge(ne);

//		dd = map.phi1(map.phi1(next)) ;
//		while(dd != ne)
//		{
//			Dart tmp = map.phi1(ne) ;
//			map.PFP::MAP::ParentMap::splitFace(tmp, dd);
//
//			if (map.isOrbitEmbedded<VERTEX>())
//			{
//				map.copyDartEmbedding<VERTEX>(map.phi_1(dd), tmp) ;
//				map.copyDartEmbedding<VERTEX>(map.phi_1(tmp), dd) ;
//			}
//
//			dd = map.phi1(map.phi1(dd)) ;
//		}
//
	}

//		setCurrentLevel(getMaxLevel()) ;
//		//4 couture des relations precedemment sauvegarde
//		for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedFaces.begin(); it != subdividedFaces.end(); ++it)
//		{
//			Dart f1 = phi2((*it).first);
//			Dart f2 = phi2((*it).second);
//
//			//if(isBoundaryFace(f1) && isBoundaryFace(f2))
//			if(phi3(f1) == f1 && phi3(f2) == f2)
//				sewVolumes(f1, f2, false);
//		}
//		setOrbitEmbedding<VERTEX>(centralDart, getEmbedding<VERTEX>(centralDart));
		//attributs[map.phi1(ne)] = attBary[*it];
//
//		setCurrentLevel(getMaxLevel() - 1) ;
//	}
//
//	//A optimiser
//
//	TraversorE<typename PFP::MAP> travE2(map);
//	for (Dart d = travE2.begin(); d != travE2.end(); d = travE2.next())
//	{
//		map.setOrbitEmbedding<VERTEX>(map.phi1(d), map.getEmbedding<VERTEX>(map.phi1(d)));
//	}
//
//	TraversorF<typename PFP::MAP> travF2(map) ;
//	for (Dart d = travF2.begin(); d != travF2.end(); d = travF2.next())
//	{
//		map.setOrbitEmbedding<VERTEX>(map.phi2(map.phi1(d)), map.getEmbedding<VERTEX>(map.phi2(map.phi1(d))));
//	}
}

inline double sqrt3_K(unsigned int n)
{
	switch(n)
	{
		case 1: return 0.333333 ;
		case 2: return 0.555556 ;
		case 3: return 0.5 ;
		case 4: return 0.444444 ;
		case 5: return 0.410109 ;
		case 6: return 0.388889 ;
		case 7: return 0.375168 ;
		case 8: return 0.365877 ;
		case 9: return 0.359328 ;
		case 10: return 0.354554 ;
		case 11: return 0.350972 ;
		case 12: return 0.348219 ;
		default:
			double t = cos((2.0*M_PI)/double(n)) ;
			return (4.0 - t) / 9.0 ;
	}
}

template <typename PFP>
void sqrt3Vol(typename PFP::MAP& map, VertexAttribute<typename PFP::MAP, typename PFP::VEC3>& position)
{
    typedef typename PFP::MAP MAP;
    DartMarkerStore<MAP> m(map);

    DartMarkerStore<MAP> newBoundaryV(map);

	//
	// 1-4 flip of all tetrahedra
	//
	TraversorW<typename PFP::MAP> tW(map);
	for(Dart dit = tW.begin() ; dit != tW.end() ; dit = tW.next())
	{
		Traversor3WF<typename PFP::MAP> tWF(map, dit);
		for(Dart ditWF = tWF.begin() ; ditWF != tWF.end() ; ditWF = tWF.next())
		{
			if(!map.isBoundaryFace(ditWF) && !m.isMarked(ditWF))
				m.markOrbit<FACE>(ditWF);
		}

		typename PFP::VEC3 volCenter(0.0);
		volCenter += position[dit];
		volCenter += position[map.phi1(dit)];
		volCenter += position[map.phi_1(dit)];
		volCenter += position[map.phi_1(map.phi2(dit))];
		volCenter /= 4;

		Dart dres = Volume::Modelisation::Tetrahedralization::flip1To4<PFP>(map, dit);
		position[dres] = volCenter;
	}

	//
	// 2-3 swap of all old interior faces
	//
	TraversorF<typename PFP::MAP> tF(map);
	for(Dart dit = tF.begin() ; dit != tF.end() ; dit = tF.next())
	{
		if(m.isMarked(dit))
		{
			m.unmarkOrbit<FACE>(dit);
			Volume::Modelisation::Tetrahedralization::swap2To3<PFP>(map, dit);
		}
	}

	//
	// 1-3 flip of all boundary tetrahedra
	//
	TraversorW<typename PFP::MAP> tWb(map);
	for(Dart dit = tWb.begin() ; dit != tWb.end() ; dit = tWb.next())
	{
		if(map.isBoundaryVolume(dit))
		{
			Traversor3WE<typename PFP::MAP> tWE(map, dit);
			for(Dart ditWE = tWE.begin() ; ditWE != tWE.end() ; ditWE = tWE.next())
			{
				if(map.isBoundaryEdge(ditWE) && !m.isMarked(ditWE))
					m.markOrbit<EDGE>(ditWE);
			}

			typename PFP::VEC3 faceCenter(0.0);
			faceCenter += position[dit];
			faceCenter += position[map.phi1(dit)];
			faceCenter += position[map.phi_1(dit)];
			faceCenter /= 3;

			Dart dres = Volume::Modelisation::Tetrahedralization::flip1To3<PFP>(map, dit);
			position[dres] = faceCenter;

			newBoundaryV.markOrbit<VERTEX>(dres);
		}
	}

/*
	TraversorV<typename PFP::MAP> tVg(map);
	for(Dart dit = tVg.begin() ; dit != tVg.end() ; dit = tVg.next())
	{
		if(map.isBoundaryVertex(dit) && !newBoundaryV.isMarked(dit))
		{
			Dart db = map.findBoundaryFaceOfVertex(dit);

			typename PFP::VEC3 P = position[db] ;
			typename PFP::VEC3 newP(0) ;
			unsigned int val = 0 ;
			Dart vit = db ;
			do
			{
				//newP += position[map.phi_1(map.phi2(map.phi1(vit)))] ;
				newP += position[map.phi2(vit)];
				++val ;
				vit = map.phi2(map.phi_1(vit)) ;
			} while(vit != db) ;
			typename PFP::REAL K = sqrt3_K(val) ;
			newP *= typename PFP::REAL(3) ;
			newP -= typename PFP::REAL(val) * P ;
			newP *= K / typename PFP::REAL(2 * val) ;
			newP += (typename PFP::REAL(1) - K) * P ;
			position[db] = newP ;
		}
	}
*/

/*
	//
	// edge-removal on all old boundary edges
	//
	TraversorE<typename PFP::MAP> tE(map);
	for(Dart dit = tE.begin() ; dit != tE.end() ; dit = tE.next())
	{
		if(m.isMarked(dit))
		{
			m.unmarkOrbit<EDGE>(dit);
			Dart d = map.phi2(map.phi3(map.findBoundaryFaceOfEdge(dit)));
			Volume::Modelisation::Tetrahedralization::swapGen3To2<PFP>(map, d);
		}
	}

*/



//	TraversorV<typename PFP::MAP> tVg(map,selected);
//	for(Dart dit = tVg.begin() ; dit != tVg.end() ; dit = tVg.next())
//	{
//		if(map.isBoundaryVertex(dit) && !newBoundaryV.isMarked(dit))
//		{
//			Dart db = map.findBoundaryFaceOfVertex(dit);
//
//			typename PFP::VEC3 P = position[db] ;
//			typename PFP::VEC3 newP(0) ;
//			unsigned int val = 0 ;
//			Dart vit = db ;
//			do
//			{
//				newP += position[map.phi_1(map.phi2(map.phi1(vit)))] ;
//				++val ;
//				vit = map.phi2(map.phi_1(vit)) ;
//			} while(vit != db) ;
//			typename PFP::REAL K = sqrt3_K(val) ;
//			newP *= typename PFP::REAL(3) ;
//			newP -= typename PFP::REAL(val) * P ;
//			newP *= K / typename PFP::REAL(2 * val) ;
//			newP += (typename PFP::REAL(1) - K) * P ;
//			position[db] = newP ;
//		}
//	}

	//AutoVertexAttribute laplacian qui est une copie de position

//	TraversorV<typename PFP::MAP> tVg2(map,selected);
//	for(Dart dit = tVg2.begin() ; dit != tVg2.end() ; dit = tVg2.next())
//	{
//		if(!map.isBoundaryVertex(dit))
//		{
//			//modifie laplacian
//		}
//	}

	//echange lapaclian et position


//	VertexAutoAttribute<typename PFP::VEC3> diffCoord(map, "diffCoord");
//	Algo::Volume::Geometry::computeLaplacianTopoVertices<PFP>(map, position, diffCoord) ;
//
//	VertexAutoAttribute<unsigned int> vIndex(map, "vIndex");
//
//	unsigned int nb_vertices = map.template computeIndexCells<VERTEX>(vIndex);
//
//
//	CellMarker<VERTEX> lockingMarker(map);
//
//	TraversorV<typename PFP::MAP> tv(map);
//	for(Dart dit = tv.begin() ; dit != tv.end() ; dit = tv.next())
//	{
//		if(map.isBoundaryVertex(dit))
//			lockingMarker.mark(dit);
//	}
//
//
//	NLContext nlContext = nlNewContext();
//	nlSolverParameteri(NL_NB_VARIABLES, nb_vertices);
//	nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
//	nlSolverParameteri(NL_SOLVER, NL_CHOLMOD_EXT);
//
//
//	nlMakeCurrent(nlContext);
//	if(nlGetCurrentState() == NL_STATE_INITIAL)
//		nlBegin(NL_SYSTEM) ;
//
//	for(int coord = 0; coord < 3; ++coord)
//	{
//		LinearSolving::setupVariables<PFP>(map, vIndex, lockingMarker, position, coord) ;
//		nlBegin(NL_MATRIX) ;
//		LinearSolving::addRowsRHS_Laplacian_Topo<PFP>(map, vIndex, diffCoord, coord) ;
////		LinearSolving::addRowsRHS_Laplacian_Cotan<PFP>(*map, perMap->vIndex, perMap->edgeWeight, perMap->vertexArea, perMap->diffCoord, coord) ;
//		nlEnd(NL_MATRIX) ;
//		nlEnd(NL_SYSTEM) ;
//		nlSolve() ;
//		LinearSolving::getResult<PFP>(map, vIndex, position, coord) ;
//		nlReset(NL_TRUE) ;
//	}
//
//	nlDeleteContext(nlContext);
}


// solving Ax = b

//template <typename PFP>
//void relaxation(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position)
//{
//	VertexAttribute<unsigned int> indexV = map.template getAttribute<unsigned int, VERTEX>("indexV");
//	if(!indexV.isValid())
//		indexV = map.template addAttribute<unsigned int, VERTEX>("indexV");

//	unsigned int nb_vertices = map.template computeIndexCells<VERTEX>(indexV);

//	//uniform weight
//	float weight = 1.0;

//	NLContext nlContext = nlNewContext();
//	nlSolverParameteri(NL_NB_VARIABLES, nb_vertices);
//	nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
//	nlSolverParameteri(NL_SOLVER, NL_CHOLMOD_EXT);

////	nlMakeCurrent(nlContext);
//	if(nlGetCurrentState() == NL_STATE_INITIAL)
//		nlBegin(NL_SYSTEM) ;

//	for(unsigned int coord = 0; coord < 3; ++coord)
//	{
//		std::cout << "coord " << coord << std::flush;
//		//setup variables
//		TraversorV<typename PFP::MAP> tv(map);
//		for(Dart dit = tv.begin() ; dit != tv.end() ; dit = tv.next())
//		{
//			nlSetVariable(indexV[dit], (position[dit])[coord]);

//			if(map.isBoundaryVertex(dit))
//				nlLockVariable(indexV[dit]);
//		}

//		std::cout << "... variables set... " << std::flush;

//		nlBegin(NL_MATRIX) ;

//		nlEnable(NL_NORMALIZE_ROWS) ;

//		TraversorV<typename PFP::MAP> tv2(map);
//		for(Dart dit = tv2.begin() ; dit != tv2.end() ; dit = tv2.next())
//		{
//			if(!map.isBoundaryVertex(dit))
//			{
//				nlRowParameterd(NL_RIGHT_HAND_SIDE, 0) ; //b[i]
//				//nlRowParameterd(NL_ROW_SCALING, weight) ;

//				nlBegin(NL_ROW) ;

//				float sum = 0;
//				Traversor3VVaE<typename PFP::MAP> tvvae(map, dit);
//				for(Dart ditvvae = tvvae.begin() ; ditvvae != tvvae.end() ; ditvvae = tvvae.next())
//				{
//					nlCoefficient(indexV[ditvvae], weight);
//					sum += weight;
//				}

//				nlCoefficient(indexV[dit], -sum) ;
//				nlEnd(NL_ROW) ;
//			}
//		}

//		nlDisable(NL_NORMALIZE_ROWS) ;

//		nlEnd(NL_MATRIX) ;

//		nlEnd(NL_SYSTEM) ;
//		std::cout << "... system built... " << std::flush;

//		nlSolve();
//		std::cout << "... system solved... " << std::flush;

//		//results
//		TraversorV<typename PFP::MAP> tv3(map);
//		for(Dart dit = tv3.begin() ; dit != tv3.end() ; dit = tv3.next())
//		{
//			position[dit][coord] = nlGetVariable(indexV[dit]);
//		}

//		nlReset(NL_TRUE) ;
//		std::cout << "... done" << std::endl;
//	}

//	nlDeleteContext(nlContext);
//}

template <typename PFP>
void computeDual(typename PFP::MAP& map, VertexAttribute<typename PFP::MAP, typename PFP::VEC3>& position)
{
	// VolumeAttribute -> after dual new VertexAttribute
    VolumeAttribute<typename PFP::MAP, typename PFP::VEC3> positionV  = map.template getAttribute<typename PFP::MAP, typename PFP::VEC3, VOLUME>("position") ;
	if(!positionV.isValid())
		positionV = map.template addAttribute<typename PFP::VEC3, VOLUME>("position") ;

	// Compute Centroid for the volumes
	Algo::Volume::Geometry::computeCentroidVolumes<PFP>(map, position, positionV) ;

	// Compute the Dual mesh
	map.computeDual();
	position = positionV ;
}


} // namespace Modelisation

} // namespace volume

} // namespace Algo

} // namespace CGoGN

