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

#include "Algo/Multiresolution/Map3MR/map3MR_PrimalAdapt.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MR
{

namespace Primal
{

namespace Adaptive
{

template <typename PFP>
Map3MR<PFP>::Map3MR(typename PFP::MAP& map) :
	m_map(map),
	shareVertexEmbeddings(false),
	vertexVertexFunctor(NULL),
	edgeVertexFunctor(NULL),
	faceVertexFunctor(NULL),
	volumeVertexFunctor(NULL)
{

}

/*! @name Cells informations
 *************************************************************************/
template <typename PFP>
unsigned int Map3MR<PFP>::edgeLevel(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"edgeLevel : called with a dart inserted after current level") ;

	// the level of an edge is the maximum of the
	// insertion levels of its darts
//	unsigned int r = 0;
//	Dart dit = d;
//	do
//	{
		unsigned int ld = m_map.getDartLevel(d) ;
		unsigned int ldd = m_map.getDartLevel(m_map.phi2(d)) ;
		unsigned int max = ld > ldd ? ld : ldd;

//		r =  r < max ? max : r ;

//		dit = m_map.alpha2(dit);
//	} while(dit != d);

//	return r;
	return max;
}

template <typename PFP>
unsigned int Map3MR<PFP>::faceLevel(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceLevel : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == 0)
		return 0 ;

	Dart it = d ;
	unsigned int min1 = m_map.getDartLevel(it) ;		// the level of a face is the second minimum of the
	it = m_map.phi1(it) ;
	unsigned int min2 = m_map.getDartLevel(it) ;		// insertion levels of its darts

	if(min2 < min1)
	{
		unsigned int tmp = min1 ;
		min1 = min2 ;
		min2 = tmp ;
	}

	it = m_map.phi1(it) ;
	while(it != d)
	{
		unsigned int dl = m_map.getDartLevel(it) ;
		if(dl < min2)
		{
			if(dl < min1)
			{
				min2 = min1 ;
				min1 = dl ;
			}
			else
				min2 = dl ;
		}
		it = m_map.phi1(it) ;
	}

	return min2 ;
}

template <typename PFP>
unsigned int Map3MR<PFP>::volumeLevel(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"volumeLevel : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == 0)
		return 0 ;

	unsigned int vLevel = std::numeric_limits<unsigned int>::max(); //hook sioux

	//First : the level of a volume is the minimum of the levels of its faces
	Traversor3WF<typename PFP::MAP> travF(m_map, d);
	for (Dart dit = travF.begin(); dit != travF.end(); dit = travF.next())
	{
		// in a first time, the level of a face
		//the level of the volume is the minimum of the
		//levels of its faces
		unsigned int fLevel = faceLevel(dit);
		vLevel = fLevel < vLevel ? fLevel : vLevel ;
	}

	//Second : the case of all faces regularly subdivided but not the volume itself

	return vLevel;
}

template <typename PFP>
Dart Map3MR<PFP>::faceOldestDart(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceOldestDart : called with a dart inserted after current level") ;

	Dart it = d ;
	Dart oldest = it ;
	unsigned int l_old = m_map.getDartLevel(oldest) ;
	do
	{
		unsigned int l = m_map.getDartLevel(it) ;
		if(l == 0)
			return it ;
		if(l < l_old)
		{
			oldest = it ;
			l_old = l ;
		}
		it = m_map.phi1(it) ;
	} while(it != d) ;
	return oldest ;
}

template <typename PFP>
Dart Map3MR<PFP>::volumeOldestDart(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"volumeOldestDart : called with a dart inserted after current level") ;

	Dart oldest = d;
	Traversor3WF<typename PFP::MAP> travF(m_map, d);
	for (Dart dit = travF.begin(); dit != travF.end(); dit = travF.next())
	{
		//for every dart in this face
		Dart old = faceOldestDart(dit);
		if(m_map.getDartLevel(old) < m_map.getDartLevel(oldest))
			oldest = old;
	}

	return oldest;
}

template <typename PFP>
bool Map3MR<PFP>::edgeIsSubdivided(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"edgeIsSubdivided : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		return false ;

	Dart d2 = m_map.phi2(d) ;
	m_map.incCurrentLevel() ;
	Dart d2_l = m_map.phi2(d) ;
	m_map.decCurrentLevel() ; ;
	if(d2 != d2_l)
		return true ;
	else
		return false ;
}

template <typename PFP>
bool Map3MR<PFP>::faceIsSubdivided(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceIsSubdivided : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		return false ;

	// a face whose level in the current level map is lower than
	// the current level can be subdivided to higher levels
	unsigned int fLevel = faceLevel(d) ;
	if(fLevel < m_map.getCurrentLevel())
		return false ;

	bool subd = false ;

	//Une face dont toute les aretes sont subdivise mais pas la face elle meme renvoie false .. sinon c'est true
	Dart dit = d;
	bool edgesAreSubdivided = true;
	do
	{
		edgesAreSubdivided &= edgeIsSubdivided(dit);
		dit = m_map.phi1(dit);
	}while(dit != d);

	if(edgesAreSubdivided)
	{
		m_map.incCurrentLevel() ;
		if(m_map.getDartLevel(m_map.phi1(m_map.phi1(d))) == m_map.getCurrentLevel()) //TODO a vérifier le phi1(phi1())
			subd = true ;
		m_map.decCurrentLevel() ;
	}
	else
		return false;

	return subd ;
}

template <typename PFP>
bool Map3MR<PFP>::volumeIsSubdivided(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"volumeIsSubdivided : called with a dart inserted after current level") ;

	unsigned int vLevel = volumeLevel(d);
	if(vLevel <= m_map.getCurrentLevel())
		return false;

	//Test if all faces are subdivided
	bool facesAreSubdivided = faceIsSubdivided(d) ;
	Traversor3WF<typename PFP::MAP> travF(m_map, d);
	for (Dart dit = travF.begin(); dit != travF.end(); dit = travF.next())
	{
		facesAreSubdivided &= faceIsSubdivided(dit) ;
	}

	//But not the volume itself
	bool subd = false;
	m_map.incCurrentLevel();
	if(facesAreSubdivided && m_map.getDartLevel(m_map.phi2(m_map.phi1(m_map.phi1(d)))) == m_map.getCurrentLevel())
		subd = true;
	m_map.decCurrentLevel() ;
	return subd;
}

/*! @name Topological helping functions
 *************************************************************************/
template <typename PFP>
void Map3MR<PFP>::swapEdges(Dart d, Dart e)
{
	if(!m_map.PFP::MAP::ParentMap::isBoundaryEdge(d) && !m_map.PFP::MAP::ParentMap::isBoundaryEdge(e))
	{
		Dart d2 = m_map.phi2(d);
		Dart e2 = m_map.phi2(e);

		m_map.PFP::MAP::ParentMap::swapEdges(d,e);

//		m_map.PFP::MAP::ParentMap::unsewFaces(d);
//		m_map.PFP::MAP::ParentMap::unsewFaces(e);
//
//		m_map.PFP::MAP::ParentMap::sewFaces(d, e);
//		m_map.PFP::MAP::ParentMap::sewFaces(d2, e2);

		if(m_map.template isOrbitEmbedded<VERTEX>())
		{
			m_map.template copyDartEmbedding<VERTEX>(d, m_map.phi2(m_map.phi_1(d)));
			m_map.template copyDartEmbedding<VERTEX>(e, m_map.phi2(m_map.phi_1(e)));
			m_map.template copyDartEmbedding<VERTEX>(d2, m_map.phi2(m_map.phi_1(d2)));
			m_map.template copyDartEmbedding<VERTEX>(e2, m_map.phi2(m_map.phi_1(e2)));
		}

		if(m_map.template isOrbitEmbedded<EDGE>())
		{

		}

//		if(m_map.template isOrbitEmbedded<VOLUME>())
//			m_map.template setOrbitEmbeddingOnNewCell<VOLUME>(d);


		m_map.duplicateDart(d);
		m_map.duplicateDart(d2);
		m_map.duplicateDart(e);
		m_map.duplicateDart(e2);
	}
}


template <typename PFP>
Dart Map3MR<PFP>::cutEdge(Dart d)
{
	Dart dit = d;
	do
	{
		Dart dd = m_map.phi2(dit) ;
		Dart d1 = m_map.phi1(dit);
		Dart dd1 = m_map.phi1(dd);

		m_map.duplicateDart(dit);
		m_map.duplicateDart(dd);
		m_map.duplicateDart(d1);
		m_map.duplicateDart(dd1);

		dit = m_map.alpha2(dit);
	}while(dit != d);

	Dart nd = m_map.cutEdge(d);

	return nd;
}

template <typename PFP>
void Map3MR<PFP>::splitFace(Dart d, Dart e)
{
	Dart dprev = m_map.phi_1(d) ;
	Dart eprev = m_map.phi_1(e) ;

	m_map.duplicateDart(d);
	m_map.duplicateDart(e);
	m_map.duplicateDart(dprev);
	m_map.duplicateDart(eprev);

	m_map.duplicateDart(m_map.phi3(d));
	m_map.duplicateDart(m_map.phi3(e));
	m_map.duplicateDart(m_map.phi3(dprev));
	m_map.duplicateDart(m_map.phi3(eprev));

	m_map.splitFace(d,e);
}

template <typename PFP>
void Map3MR<PFP>::splitVolume(std::vector<Dart>& vd)
{
	m_map.splitVolume(vd);

	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		Dart dit = *it;
		m_map.duplicateDart(dit);

		Dart dit2 = m_map.phi2(dit);
		m_map.duplicateDart(dit2);

		Dart dit23 = m_map.phi3(dit2);
		m_map.duplicateDart(dit23);

		Dart dit232 = m_map.phi2(dit23);
		m_map.duplicateDart(dit232);
	}
}


/*! @name Subdivision
 *************************************************************************/
template <typename PFP>
void Map3MR<PFP>::subdivideEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideEdge : called with a dart inserted after current level") ;
	assert(!edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;

	assert(m_map.getCurrentLevel() == edgeLevel(d) || !"Trying to subdivide an edge on a bad current level") ;

	m_map.incCurrentLevel();

	Dart nd = cutEdge(d);

	(*edgeVertexFunctor)(nd) ;

	m_map.decCurrentLevel() ;
}

template <typename PFP>
void Map3MR<PFP>::coarsenEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideEdge : called with a dart inserted after current level") ;
	//assert(edgeCanBeCoarsened(d) || !"Trying to subdivide an already subdivided edge") ;

}

template <typename PFP>
void Map3MR<PFP>::subdivideFace(Dart d, bool triQuad)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideFace : called with a dart inserted after current level") ;
	assert(!faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int fLevel = faceLevel(d) ;
	Dart old = faceOldestDart(d) ;

	m_map.pushLevel() ;
	m_map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

	unsigned int degree = 0 ;
	Dart it = old ;
	do
	{
		++degree ;						// compute the degree of the face

		if(!edgeIsSubdivided(it))
			subdivideEdge(it) ;			// and cut the edges (if they are not already)
		it = m_map.phi1(it) ;
	} while(it != old) ;

	m_map.setCurrentLevel(fLevel + 1) ;	// go to the next level to perform face subdivision

    if(triQuad && degree == 3)	// if subdividing a triangle
    {
        Dart dd = m_map.phi1(old) ;
        Dart e = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(e) ;
        e = m_map.phi1(e) ;
        splitFace(dd, e) ;

        dd = e ;
        e = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(e) ;
        e = m_map.phi1(e) ;
        splitFace(dd, e) ;

        dd = e ;
        e = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(e) ;
        e = m_map.phi1(e) ;
        splitFace(dd, e) ;
    }
    else							// if subdividing a polygonal face
    {
        Dart dd = m_map.phi1(old) ;
        Dart next = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(next) ;
        next = m_map.phi1(next) ;
        splitFace(dd, next) ;			// insert a first edge
        Dart ne = m_map.phi2(m_map.phi_1(dd));

        cutEdge(ne) ;					// cut the new edge to insert the central vertex

        dd = m_map.phi1(next) ;
        (*vertexVertexFunctor)(dd) ;
        dd = m_map.phi1(dd) ;
        while(dd != ne)					// turn around the face and insert new edges
        {								// linked to the central vertex
            splitFace(m_map.phi1(ne), dd) ;
            dd = m_map.phi1(dd) ;
            (*vertexVertexFunctor)(dd) ;
            dd = m_map.phi1(dd) ;
        }

        (*faceVertexFunctor)(m_map.phi1(ne)) ;
    }

	m_map.popLevel() ;
}

template <typename PFP>
unsigned int Map3MR<PFP>::subdivideVolume(Dart d, bool triQuad, bool OneLevelDifference)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideVolume : called with a dart inserted after current level") ;
	assert(!volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int vLevel = volumeLevel(d);
	Dart old = volumeOldestDart(d);

	m_map.pushLevel() ;
	m_map.setCurrentLevel(vLevel) ;		// go to the level of the volume to subdivide its faces

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		m_map.addLevelBack() ;

	//
	// Subdivide Faces and Edges
	//
	Traversor3WF<typename PFP::MAP> traF(m_map, old);
	for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
	{
		//if needed subdivide face
		if(!faceIsSubdivided(dit))
			subdivideFace(dit,triQuad);
	}

	//
	// Create inside volumes
	//
	std::vector<std::pair<Dart, Dart> > subdividedFaces;
	subdividedFaces.reserve(128);

	bool isocta = false;
	bool ishex = false;
	bool isprism = false;
	bool ispyra = false;

	Dart centralDart = NIL;
	Traversor3WV<typename PFP::MAP> traWV(m_map, d);
	for(Dart ditWV = traWV.begin(); ditWV != traWV.end(); ditWV = traWV.next())
	{
		m_map.incCurrentLevel() ;

		Dart e = ditWV;
		std::vector<Dart> v ;

		do
		{
			v.push_back(m_map.phi1(e));

			if(m_map.phi1(m_map.phi1(m_map.phi1(e))) != e)
				v.push_back(m_map.phi1(m_map.phi1(e)));

			if(!m_map.PFP::MAP::ParentMap::isBoundaryEdge(m_map.phi1(e)))
				subdividedFaces.push_back(std::pair<Dart,Dart>(m_map.phi1(e),m_map.phi2(m_map.phi1(e))));

			if(m_map.phi1(m_map.phi1(m_map.phi1(e))) != e)
				if(!m_map.PFP::MAP::ParentMap::isBoundaryEdge(m_map.phi1(m_map.phi1(e))))
					subdividedFaces.push_back(std::pair<Dart,Dart>(m_map.phi1(m_map.phi1(e)),m_map.phi2(m_map.phi1(m_map.phi1(e)))));

			e = m_map.phi2(m_map.phi_1(e));
		}
		while(e != ditWV);

		m_map.splitVolume(v);

		unsigned int fdeg = m_map.faceDegree(m_map.phi2(m_map.phi1(ditWV)));

		if(fdeg == 4)
		{
			if(m_map.PFP::MAP::ParentMap::vertexDegree(ditWV) == 3)
			{
				isocta = false;
				ispyra = true;

				Dart it = ditWV;
				if((m_map.faceDegree(it) == 3) && (m_map.faceDegree(m_map.phi2(it))) == 3)
				{
					it = m_map.phi2(m_map.phi_1(it));
				}
				else if((m_map.faceDegree(it) == 3) && (m_map.faceDegree(m_map.phi2(it)) == 4))
				{
					it = m_map.phi1(m_map.phi2(it));
				}

				Dart old = m_map.phi2(m_map.phi1(it));
				Dart dd = m_map.phi1(m_map.phi1(old));

				m_map.splitFace(old,dd) ;
				centralDart = old;
			}
			else
			{
				if(ispyra)
					isocta = false;
				else
					isocta = true;

				Dart old = m_map.phi2(m_map.phi1(ditWV));
				Dart dd = m_map.phi1(old) ;
				m_map.splitFace(old,dd) ;

				Dart ne = m_map.phi1(old);

				m_map.cutEdge(ne);
				centralDart = m_map.phi1(ne);

				Dart stop = m_map.phi2(m_map.phi1(ne));
				ne = m_map.phi2(ne);
				do
				{
					dd = m_map.phi1(m_map.phi1(ne));

					m_map.splitFace(ne, dd) ;

					ne = m_map.phi2(m_map.phi_1(ne));
					dd = m_map.phi1(dd);
				}
				while(dd != stop);
			}
		}
		else if(fdeg == 5)
		{
			isprism = true;

			Dart it = ditWV;
			if(m_map.faceDegree(it) == 3)
			{
				it = m_map.phi2(m_map.phi_1(it));
			}
			else if(m_map.faceDegree(m_map.phi2(m_map.phi_1(ditWV))) == 3)
			{
				it = m_map.phi2(m_map.phi_1(m_map.phi2(m_map.phi_1(it))));
			}

			Dart old = m_map.phi2(m_map.phi1(it));
			Dart dd = m_map.phi_1(m_map.phi_1(old));

			m_map.splitFace(old,dd) ;
		}
		if(fdeg == 6)
		{
			ishex = true;

			Dart dd = m_map.phi2(m_map.phi1(ditWV));;
			Dart next = m_map.phi1(m_map.phi1(dd)) ;
			m_map.splitFace(dd, next) ;		// insert a first edge

			Dart ne = m_map.phi2(m_map.phi_1(dd)) ;
			m_map.cutEdge(ne) ;				// cut the new edge to insert the central vertex
			centralDart = m_map.phi1(ne);

			dd = m_map.phi1(m_map.phi1(next)) ;
			while(dd != ne)				// turn around the face and insert new edges
			{							// linked to the central vertex
				Dart tmp = m_map.phi1(ne) ;
				m_map.splitFace(tmp, dd) ;
				dd = m_map.phi1(m_map.phi1(dd)) ;
			}
		}

		m_map.decCurrentLevel() ;
	}

	if(ishex)
	{
		m_map.incCurrentLevel();

		m_map.deleteVolume(m_map.phi3(m_map.phi2(m_map.phi1(d))));

		for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedFaces.begin(); it != subdividedFaces.end(); ++it)
		{
			Dart f1 = m_map.phi2((*it).first);
			Dart f2 = m_map.phi2((*it).second);

			if(m_map.isBoundaryFace(f1) && m_map.isBoundaryFace(f2))
			{
					m_map.sewVolumes(f1, f2);//, false);
			}
		}

		//replonger l'orbit de ditV.
		m_map.template setOrbitEmbedding<VERTEX>(centralDart, m_map.template getEmbedding<VERTEX>(centralDart));
		(*volumeVertexFunctor)(centralDart) ;

		m_map.decCurrentLevel() ;
	}

	if(ispyra)
	{
		isocta = false;

		Dart ditV = d;

		Traversor3WV<typename PFP::MAP> traWV(m_map, d);
		for(Dart ditWV = traWV.begin(); ditWV != traWV.end(); ditWV = traWV.next())
		{
			if(m_map.PFP::MAP::ParentMap::vertexDegree(ditWV) == 4)
				ditV = ditWV;
		}

		m_map.incCurrentLevel();
		Dart x = m_map.phi_1(m_map.phi2(m_map.phi1(ditV)));
		std::vector<Dart> embVol;

		Dart f = x;
		do
		{
			Dart f3 = m_map.phi3(f);
			Dart tmp =  m_map.phi_1(m_map.phi2(m_map.phi_1(m_map.phi2(m_map.phi_1(f3))))); //future voisin par phi2
			swapEdges(f3, tmp);
			embVol.push_back(tmp);

			f = m_map.phi2(m_map.phi_1(f));
		}while(f != x);


		for(std::vector<Dart>::iterator it = embVol.begin() ; it != embVol.end() ; ++it)
		{
			Dart dit = *it;

			m_map.template setOrbitEmbeddingOnNewCell<VOLUME>(dit);
			m_map.template copyCell<VOLUME>(dit, ditV);
		}

		//embed the volumes around swapEdges



		//replonger l'orbit de ditV.
		m_map.template setOrbitEmbedding<VERTEX>(m_map.phi2(m_map.phi3(x)), m_map.template getEmbedding<VERTEX>(m_map.phi2(m_map.phi3(x))));
		//m_map.template setOrbitEmbedding<VERTEX>(centralDart, m_map.template getEmbedding<VERTEX>(centralDart));
		//(*volumeVertexFunctor)(x) ;

		m_map.decCurrentLevel() ;
	}

	if(isocta)
	{
		Traversor3WV<typename PFP::MAP> traWV(m_map, d);

		for(Dart ditWV = traWV.begin(); ditWV != traWV.end(); ditWV = traWV.next())
		{
			m_map.incCurrentLevel();
			Dart x = m_map.phi_1(m_map.phi2(m_map.phi1(ditWV)));
			std::vector<Dart> embVol;

			if(!Algo::Volume::Modelisation::Tetrahedralization::isTetrahedron<PFP>(m_map,x))
			{
				DartMarkerStore me(m_map);

				Dart f = x;

				do
				{
					Dart f3 = m_map.phi3(f);

					if(!me.isMarked(f3))
					{
						Dart tmp =  m_map.phi_1(m_map.phi2(m_map.phi_1(m_map.phi2(m_map.phi_1(f3))))); //future voisin par phi2

						Dart f32 = m_map.phi2(f3);
						swapEdges(f3, tmp);

						me.markOrbit<EDGE>(f3);
						me.markOrbit<EDGE>(f32);

						embVol.push_back(tmp);
					}

					f = m_map.phi2(m_map.phi_1(f));
				}while(f != x);

			}

			for(std::vector<Dart>::iterator it = embVol.begin() ; it != embVol.end() ; ++it)
			{
				Dart dit = *it;

				m_map.template setOrbitEmbeddingOnNewCell<VOLUME>(dit);
				m_map.template copyCell<VOLUME>(dit, d);
			}


			m_map.template setOrbitEmbedding<VERTEX>(x, m_map.template getEmbedding<VERTEX>(x));
			(*volumeVertexFunctor)(x) ;

			m_map.decCurrentLevel() ;
		}
	}

	if(isprism)
	{
		m_map.incCurrentLevel();

		Dart ditWV = d;
		if(m_map.faceDegree(d) == 3)
		{
			ditWV = m_map.phi2(m_map.phi_1(d));
		}
		else if(m_map.faceDegree(m_map.phi2(m_map.phi_1(d))) == 3)
		{
			ditWV = m_map.phi1(m_map.phi2(d));
		}

		ditWV = m_map.phi3(m_map.phi_1(m_map.phi2(m_map.phi1(ditWV))));


		std::vector<Dart> path;

		Dart dtemp = ditWV;
		do
		{
			//future voisin par phi2
			Dart sF1 = m_map.phi1(m_map.phi2(m_map.phi3(dtemp)));
			Dart wrongVolume = m_map.phi3(sF1);
			Dart sF2 = m_map.phi3(m_map.phi2(wrongVolume));
			Dart tmp =  m_map.phi3(m_map.phi2(m_map.phi1(sF2)));
			swapEdges(dtemp, tmp);

			m_map.deleteVolume(wrongVolume);
			m_map.sewVolumes(sF1,sF2);

			path.push_back(dtemp);
			dtemp = m_map.phi_1(m_map.phi2(m_map.phi_1(dtemp)));


		}while(dtemp != ditWV);

		m_map.splitVolume(path);

		m_map.decCurrentLevel() ;
	}

	m_map.incCurrentLevel();
	m_map.popLevel() ;

	return vLevel;
}

template <typename PFP>
unsigned int Map3MR<PFP>::subdivideHexa(Dart d, bool OneLevelDifference)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideVolume : called with a dart inserted after current level") ;
	assert(!volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int vLevel = volumeLevel(d);
	Dart old = volumeOldestDart(d);

	m_map.pushLevel() ;
	m_map.setCurrentLevel(vLevel) ;		// go to the level of the volume to subdivide its faces

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		m_map.addLevelBack() ;

	if(OneLevelDifference)
	{
		Traversor3WF<typename PFP::MAP> traF(m_map, old);
		for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
		{
			Dart nv = m_map.phi3(dit);
			if(!m_map.isBoundaryMarked3(nv))
				if(volumeLevel(nv) == vLevel - 1)
					subdivideHexa(nv,false);
		}
	}
	std::vector<std::pair<Dart, Dart> > subdividedFaces;
	subdividedFaces.reserve(128);

	Traversor3WF<typename PFP::MAP> traF(m_map, old);
	for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
	{
		//if needed subdivide face
		if(!faceIsSubdivided(dit))
			subdivideFace(dit,false);

		//save a dart from the subdivided face
		m_map.pushLevel();

		unsigned int fLevel = faceLevel(dit); //puisque dans tous les cas, la face est subdivisee
		m_map.setCurrentLevel(fLevel + 1) ;

		//le brin est forcement du niveau cur
		Dart cf = m_map.phi1(dit);
		Dart e = cf;
		do
		{
			subdividedFaces.push_back(std::pair<Dart,Dart>(e,m_map.phi2(e)));
			e = m_map.phi2(m_map.phi1(e));
		}while (e != cf);

		m_map.popLevel();
	}


	Dart centralDart = NIL;

	Traversor3WV<typename PFP::MAP> traWV(m_map, old);
	for(Dart ditWV = traWV.begin(); ditWV != traWV.end(); ditWV = traWV.next())
	{
		m_map.pushLevel();

		m_map.setCurrentLevel(vLevel + 1) ;

		(*vertexVertexFunctor)(ditWV) ;

		Dart e = ditWV;
		std::vector<Dart> v ;

		do
		{
			v.push_back(m_map.phi1(e));
			v.push_back(m_map.phi1(m_map.phi1(e)));

//			if(!m_map.PFP::MAP::ParentMap::isBoundaryEdge(m_map.phi1(e)))
//				subdividedFaces.push_back(std::pair<Dart,Dart>(m_map.phi1(e),m_map.phi2(m_map.phi1(e))));

//			//if(m_map.phi1(m_map.phi1(m_map.phi1(e))) != e)
//			if(!m_map.PFP::MAP::ParentMap::isBoundaryEdge(m_map.phi1(m_map.phi1(e))))
//				subdividedFaces.push_back(std::pair<Dart,Dart>(m_map.phi1(m_map.phi1(e)),m_map.phi2(m_map.phi1(m_map.phi1(e)))));

			e = m_map.phi2(m_map.phi_1(e));
		}
		while(e != ditWV);

		m_map.splitVolume(v);

		Dart dd = m_map.phi2(m_map.phi1(ditWV));;
		Dart next = m_map.phi1(m_map.phi1(dd)) ;
		m_map.splitFace(dd, next) ;		// insert a first edge

		Dart ne = m_map.phi2(m_map.phi_1(dd)) ;
		m_map.cutEdge(ne) ;				// cut the new edge to insert the central vertex
		centralDart = m_map.phi1(ne);

		dd = m_map.phi1(m_map.phi1(next)) ;
		while(dd != ne)				// turn around the face and insert new edges
		{							// linked to the central vertex
			m_map.splitFace(m_map.phi1(ne), dd) ;
			dd = m_map.phi1(m_map.phi1(dd)) ;
		}

		m_map.popLevel() ;
	}

	m_map.setCurrentLevel(vLevel + 1) ;

	m_map.deleteVolume(m_map.phi3(m_map.phi2(m_map.phi1(old))));

	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedFaces.begin(); it != subdividedFaces.end(); ++it)
	{
		Dart f1 = m_map.phi2((*it).first);
		Dart f2 = m_map.phi2((*it).second);

		if(m_map.isBoundaryFace(f1) && m_map.isBoundaryFace(f2))
		{
			m_map.sewVolumes(f1, f2);//, false);
		}
	}

	(*volumeVertexFunctor)(centralDart) ;

	m_map.popLevel() ;

	return vLevel;
}

template <typename PFP>
void Map3MR<PFP>::subdivideFace2(Dart d, bool triQuad)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideFace : called with a dart inserted after current level") ;
	assert(!faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int fLevel = faceLevel(d) ;
	Dart old = faceOldestDart(d) ;

	m_map.pushLevel() ;
	m_map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

	std::cout << "face Level = " << fLevel << std::endl;

	unsigned int degree = 0 ;
	Dart it = old ;
	do
	{
		++degree ;						// compute the degree of the face

		std::cout << "edge Level = " << edgeLevel(it) << std::endl;
		if(!edgeIsSubdivided(it))
		{
			std::cout << "cutEdge" << std::endl;
			subdivideEdge(it) ;			// and cut the edges (if they are not already)
		}

		it = m_map.phi1(it) ;
	} while(it != old) ;

//	m_map.setCurrentLevel(fLevel + 1) ;	// go to the next level to perform face subdivision

//    if(triQuad && degree == 3)	// if subdividing a triangle
//    {
//        Dart dd = m_map.phi1(old) ;
//        Dart e = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(e) ;
//        e = m_map.phi1(e) ;
//        splitFace(dd, e) ;

//        dd = e ;
//        e = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(e) ;
//        e = m_map.phi1(e) ;
//        splitFace(dd, e) ;

//        dd = e ;
//        e = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(e) ;
//        e = m_map.phi1(e) ;
//        splitFace(dd, e) ;
//    }
//    else							// if subdividing a polygonal face
//    {
//        Dart dd = m_map.phi1(old) ;
//        Dart next = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(next) ;
//        next = m_map.phi1(next) ;
//        splitFace(dd, next) ;			// insert a first edge
//        Dart ne = m_map.phi2(m_map.phi_1(dd));

//        cutEdge(ne) ;					// cut the new edge to insert the central vertex

//        dd = m_map.phi1(next) ;
//        (*vertexVertexFunctor)(dd) ;
//        dd = m_map.phi1(dd) ;
//        while(dd != ne)					// turn around the face and insert new edges
//        {								// linked to the central vertex
//            splitFace(m_map.phi1(ne), dd) ;
//            dd = m_map.phi1(dd) ;
//            (*vertexVertexFunctor)(dd) ;
//            dd = m_map.phi1(dd) ;
//        }

//        (*faceVertexFunctor)(m_map.phi1(ne)) ;
//    }

	m_map.popLevel() ;
}

template <typename PFP>
unsigned int Map3MR<PFP>::subdivideHexa2(Dart d, bool OneLevelDifference)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideVolume : called with a dart inserted after current level") ;
	assert(!volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int vLevel = volumeLevel(d);
	Dart old = volumeOldestDart(d);

	m_map.pushLevel() ;
	m_map.setCurrentLevel(vLevel) ;		// go to the level of the volume to subdivide its faces

	std::cout << "volume Level = " << vLevel << std::endl;

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
	{
		std::cout << "addLevel ?" << std::endl;
		m_map.addLevelBack() ;
	}

	Traversor3WF<typename PFP::MAP> traF(m_map, old);
	for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
	{
		std::cout << "face" << std::endl;

		//if needed subdivide face
		if(!faceIsSubdivided(dit))
		{
			std::cout << "subdivide Face"<< std::endl;
			subdivideFace2(dit,false);
		}
	}

	m_map.popLevel() ;

	std::cout << "plop" << std::endl;

	return vLevel;
}


template <typename PFP>
void Map3MR<PFP>::subdivideVolumeTetOcta(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideVolumeTetOcta : called with a dart inserted after current level") ;
	assert(!volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int vLevel = volumeLevel(d);
	Dart old = volumeOldestDart(d);

	m_map.pushLevel() ;
	m_map.setCurrentLevel(vLevel) ;		// go to the level of the face to subdivide its edges

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		m_map.addLevelBack() ;

	unsigned int j = 0;

	//
	// Subdivide Faces and Edges
	//
	Traversor3WF<typename PFP::MAP> traF(m_map, old);
	for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
	{
		std::cout << "CurrentLevel = " << m_map.getCurrentLevel() << std::endl;
		std::cout << "face level = " << faceLevel(dit) << std::endl;
		//std::cout << "d = " << dit << " is Subdivided ? " << faceIsSubdivided(dit) << std::endl;

		//if needed subdivide face
		if(!faceIsSubdivided(dit))
		{
			std::cout << "subdivide face = " << dit << std::endl;
			subdivideFace(dit, true);
			++j;
		}
	}

	//
	// Create inside volumes
	//
	Dart centralDart = NIL;
	bool isNotTet = false;
	Traversor3WV<typename PFP::MAP> traV(m_map, old);
	m_map.setCurrentLevel(vLevel + 1) ;
	for(Dart dit = traV.begin(); dit != traV.end(); dit = traV.next())
	{
		//m_map.template setOrbitEmbedding<VERTEX>(dit, EMBNULL);
		(*vertexVertexFunctor)(dit) ;

		Dart f1 = m_map.phi1(dit);
		Dart e = dit;
		std::vector<Dart> v ;

		do
		{
			v.push_back(m_map.phi1(e));
			e = m_map.phi2(m_map.phi_1(e));
		}
		while(e != dit);

		std::cout << "v size = " << v.size() << std::endl;

		splitVolume(v) ;

		//if is not a tetrahedron
		unsigned int fdeg = m_map.faceDegree(m_map.phi2(f1));
		if(fdeg > 3)
		{
			isNotTet = true;
			Dart old = m_map.phi2(m_map.phi1(dit));
			Dart dd = m_map.phi1(old) ;
			splitFace(old,dd) ;

			Dart ne = m_map.phi1(old);

			cutEdge(ne);
			centralDart = m_map.phi1(ne);
			//(*volumeVertexFunctor)(centralDart) ;
			//propagateOrbitEmbedding<VERTEX>(centralDart) ;

			Dart stop = m_map.phi2(m_map.phi1(ne));
			ne = m_map.phi2(ne);
			do
			{
				dd = m_map.phi1(m_map.phi1(ne));

				splitFace(ne, dd) ;

				ne = m_map.phi2(m_map.phi_1(ne));
				dd = m_map.phi1(dd);
			}
			while(dd != stop);
		}
	}

	//switch inner faces
	if(isNotTet)
	{
		DartMarkerStore me(m_map);

		for(Dart dit = traV.begin(); dit != traV.end(); dit = traV.next())
		{
			Dart x = m_map.phi_1(m_map.phi2(m_map.phi1(dit)));
			Dart f = x;

			do
			{
				Dart f3 = m_map.phi3(f);

				if(!me.isMarked(f3))
				{
					Dart tmp =  m_map.phi_1(m_map.phi2(m_map.phi_1(m_map.phi2(m_map.phi_1(f3))))); //future voisin par phi2
					//centralDart = tmp;

					Dart f32 = m_map.phi2(f3);
					swapEdges(f3, tmp);

					me.markOrbit<EDGE>(f3);
					me.markOrbit<EDGE>(f32);
				}

				f = m_map.phi2(m_map.phi_1(f));
			}while(f != x);
		}

		//m_map.template setOrbitEmbedding<VERTEX>(centralDart, EMBNULL);
		m_map.template setOrbitEmbedding<VERTEX>(centralDart, m_map.template getEmbedding<VERTEX>(centralDart));
		(*volumeVertexFunctor)(centralDart) ;
		//propagateOrbitEmbedding<VERTEX>(centralDart) ;
	}

	m_map.popLevel();
}


} // namespace Adaptive

} // namespace Primal

} // namespace MR

} // namespace Volume

} // namespace Algo

} // namespace CGoGN
