///*******************************************************************************
//* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional m_maps  *
//* version 0.1                                                                  *
//* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
//*                                                                              *
//* This library is free software; you can redistribute it and/or modify it      *
//* under the terms of the GNU Lesser General Public License as published by the *
//* Free Software Foundation; either version 2.1 of the License, or (at your     *
//* option) any later version.                                                   *
//*                                                                              *
//* This library is distributed in the hope that it will be useful, but WITHOUT  *
//* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
//* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
//* for more details.                                                            *
//*                                                                              *
//* You should have received a copy of the GNU Lesser General Public License     *
//* along with this library; if not, write to the Free Software Foundation,      *
//* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
//*                                                                              *
//* Web site: http://cgogn.unistra.fr/                                           *
//* Contact information: cgogn@unistra.fr                                        *
//*                                                                              *
//*******************************************************************************/

//#include "Algo/Multiresolution/IHM3/ihm3_PrimalAdapt.h"
//#include "Topology/generic/traversor/traversor2.h"

//namespace CGoGN
//{

//namespace Algo
//{

//namespace Volume
//{

//namespace MR
//{

//namespace Primal
//{

//namespace Adaptive
//{

//template <typename PFP>
//IHM3<PFP>::IHM3(MAP& map) :
//	m_map(map),
//    shareVertexEmbeddings(true),
//    vertexVertexFunctor(NULL),
//    edgeVertexFunctor(NULL),
//    faceVertexFunctor(NULL)
//{

//}

///***************************************************
// *               CELLS INFORMATION                 *
// ***************************************************/

//template <typename PFP>
//inline unsigned int IHM3<PFP>::edgeLevel(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

//	// the level of an edge is the maximum of the
//	// insertion levels of its darts
//    return std::max(m_map.getDartLevel(d),
//                    m_map.getDartLevel(m_map.phi2(d)));
//}

//template <typename PFP>
//unsigned int IHM3<PFP>::faceLevel(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceLevel : called with a dart inserted after current level") ;

//	if(m_map.getCurrentLevel() == 0)
//        return 0 ;

//    Dart it = d ;
//    Dart old = it ;
//    unsigned int l_old = m_map.getDartLevel(old) ;
//    unsigned int fLevel = edgeLevel(it) ;
//    do
//    {
//        it = m_map.phi1(it) ;
//        unsigned int dl = m_map.getDartLevel(it) ;
//        if(dl < l_old)							// compute the oldest dart of the face
//        {										// in the same time
//            old = it ;
//            l_old = dl ;
//        }										// in a first time, the level of a face
//        unsigned int l = edgeLevel(it) ;		// is the minimum of the levels
//        fLevel = l < fLevel ? l : fLevel ;		// of its edges
//    } while(it != d) ;

//    unsigned int cur = m_map.getCurrentLevel() ;
//    m_map.setCurrentLevel(fLevel) ;

//    unsigned int nbSubd = 0 ;
//    it = old ;
//    unsigned int eId = m_map.getEdgeId(old) ;			// the particular case of a face
//    do											// with all neighboring faces regularly subdivided
//    {											// but not the face itself
//        ++nbSubd ;								// is treated here
//        it = m_map.phi1(it) ;
//	} while(m_map.getEdgeId(it) == eId) ;

//    while(nbSubd > 1)
//    {
//        nbSubd /= 2 ;
//        --fLevel ;
//    }

//    m_map.setCurrentLevel(cur) ;

//    return fLevel ;
//}

//template <typename PFP>
//unsigned int IHM3<PFP>::volumeLevel(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

//	if(m_map.getCurrentLevel() == 0)
//        return 0 ;

////	Dart oldest = d ;
////	unsigned int vLevel = std::numeric_limits<unsigned int>::max(); //hook sioux
////	//First : the level of a volume is the minimum of the levels of its faces
////	Traversor3WF<typename PFP::MAP> travF(m_map, d);
////	for (Dart dit = travF.begin(); dit != travF.end(); dit = travF.next())
////	{
////		// in a first time, the level of a face
////		//the level of the volume is the minimum of the
////		//levels of its faces
////		unsigned int fLevel = faceLevel(dit);
////		vLevel = fLevel < vLevel ? fLevel : vLevel ;
////	}


//    //First : the level of a volume is the
//    //minimum of the levels of its faces

//	DartMarkerStore<MAP> mark(m_map);		// Lock a marker

//    std::vector<Dart> visitedFaces;		// Faces that are traversed
//    visitedFaces.reserve(512);
//    visitedFaces.push_back(d);			// Start with the face of d
//    std::vector<Dart>::iterator face;

//    Dart oldest = d ;
//    unsigned int vLevel = std::numeric_limits<unsigned int>::max() ; //hook de ouf

//    //parcours les faces du volume au niveau courant
//    //on cherche le brin de niveau le plus bas de la hierarchie
//    //on note le niveau le plus bas de la hierarchie
//	mark.markOrbit(Face(d)) ;
//    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
//    {
//        Dart e = visitedFaces[i] ;

//        // in a first time, the level of a face
//        //the level of the volume is the minimum of the
//        //levels of its faces

//        //
//        // Compute the level of this face
//        // and the oldest dart
//        //
//        Dart it = e ;
//        Dart old = it ;
//        unsigned int l_old = m_map.getDartLevel(old) ;
//        unsigned int fLevel = edgeLevel(it) ;
//        do
//        {
//			it = m_map.phi1(it) ;
//            unsigned int dl = m_map.getDartLevel(it) ;
//            if(dl < l_old)							// compute the oldest dart of the face
//            {										// in the same time
//                old = it ;
//                l_old = dl ;
//            }										// in a first time, the level of a face
//            unsigned int l = edgeLevel(it) ;		// is the minimum of the levels
//            fLevel = l < fLevel ? l : fLevel ;		// of its edges
//        } while(it != e) ;

//        unsigned int cur = m_map.getCurrentLevel() ;
//        m_map.setCurrentLevel(fLevel) ;

//        unsigned int nbSubd = 0 ;
//        it = old ;
//        unsigned int eId =  m_map.getEdgeId(old) ;			// the particular case of a face
//        do											// with all neighboring faces regularly subdivided
//        {											// but not the face itself
//            ++nbSubd ;								// is treated here
//            it = m_map.phi1(it) ;
//        } while( m_map.getEdgeId(it) == eId) ;

//        while(nbSubd > 1)
//        {
//            nbSubd /= 2 ;
//            --fLevel ;
//        }

//        m_map.setCurrentLevel(cur) ;

//        //
//        // compute the minimum level of the volume
//        // if the level of this face is lower than the saved volume level
//        //
//        vLevel = fLevel < vLevel ? fLevel : vLevel ;

//        //
//        // compute the oldest dart from the volume
//        // if the oldest dart from this face is oldest than the oldest saved dart
//        //
//		if(m_map.getDartLevel(old) < m_map.getDartLevel(oldest))
//            oldest = old ;

//        //
//        // add all face neighbours to the table
//        //
//        do
//        {
//            Dart ee = m_map.phi2(e) ;
//            if(!mark.isMarked(ee)) // not already marked
//            {
//                visitedFaces.push_back(ee) ;
//				mark.markOrbit(Face(ee)) ;
//            }
//            e = m_map.phi1(e) ;
//        } while(e != visitedFaces[i]) ;
//    }



//    //Second : the case of all faces regularly subdivided but not the volume itself
//    unsigned int cur = m_map.getCurrentLevel() ;
//    m_map.setCurrentLevel(vLevel) ;

//    unsigned int nbSubd = 0 ;
//    Dart it = oldest ;
//    unsigned int eId = m_map.getEdgeId(oldest) ;

//    do
//    {
//        ++nbSubd ;
//        it = m_map.phi1(it) ;
//	} while(m_map.getEdgeId(it) == eId) ;


//    while(nbSubd > 1)
//    {
//        nbSubd /= 2 ;
//        --vLevel ;
//    }

//    m_map.setCurrentLevel(cur) ;

//    return vLevel;
//}

//template <typename PFP>
//Dart IHM3<PFP>::faceOldestDart(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//    Dart it = d ;
//    Dart oldest = it ;
//    unsigned int l_old = m_map.getDartLevel(oldest) ;
//    do
//    {
//        unsigned int l = m_map.getDartLevel(it) ;
//        if(l == 0)
//            return it ;
//        if(l < l_old)
//        //if(l < l_old || (l == l_old && it < oldest))
//        {
//            oldest = it ;
//            l_old = l ;
//        }
//        it = m_map.phi1(it) ;
//    } while(it != d) ;
//    return oldest ;
//}

//template <typename PFP>
//Dart IHM3<PFP>::volumeOldestDart(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

//    Dart oldest = d;
//    unsigned int l_old = m_map.getDartLevel(oldest);

//	Traversor3WF<typename PFP::MAP> trav3WF(m_map, oldest);
//    for(Dart dit = trav3WF.begin() ; dit != trav3WF.end() ; dit = trav3WF.next())
//    {
//        Dart old = faceOldestDart(dit);
//        unsigned int l = m_map.getDartLevel(old);
//        if(l < l_old)
//        {
//            oldest = old;
//            l_old = l;
//        }
//    }

//    return oldest;
//}

//template <typename PFP>
//bool IHM3<PFP>::edgeIsSubdivided(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

//	//TODO replace by phi1
//	Dart d2 = m_map.phi2(d) ;
//	m_map.incCurrentLevel();
//	Dart d2_l = m_map.phi2(d) ;
//	m_map.decCurrentLevel() ;
//	if(d2 != d2_l)
//        return true ;
//    else
//        return false ;
//}

//template <typename PFP>
//bool IHM3<PFP>::edgeCanBeCoarsened(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

//    bool subd = false ;
//    bool subdOnce = true ;
//    bool degree2 = false ;

//    if(edgeIsSubdivided(d))
//    {
//        subd = true ;
//        m_map.incCurrentLevel() ;

//		if(m_map.vertexDegree(m_map.phi1(d)) == 2)
//        {
//            degree2 = true ;
//            if(edgeIsSubdivided(d))
//                subdOnce = false ;
//        }
//        m_map.decCurrentLevel() ;
//    }
//    return subd && degree2 && subdOnce ;
//}

//template <typename PFP>
//bool IHM3<PFP>::faceIsSubdivided(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

////	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
////		return false ;

//	unsigned int fLevel = faceLevel(d) ;
//	if(fLevel < m_map.getCurrentLevel())
//        return false ;

//    bool subd = false ;
//	m_map.incCurrentLevel();
//	if(m_map.getDartLevel(m_map.phi1(d)) == m_map.getCurrentLevel() && m_map.getEdgeId(m_map.phi1(d)) != m_map.getEdgeId(d))
//        subd = true ;
//	m_map.decCurrentLevel();

//    return subd ;
//}

//template <typename PFP>
//bool IHM3<PFP>::faceCanBeCoarsened(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

//    bool subd = false;
//    bool subdOnce = true;
//    bool subdNeighborhood = false; //deux volumes voisins de la face ne sont pas subdivise

//    if(faceIsSubdivided(d))
//    {
//        subd = true;
//        Dart d3 = m_map.phi3(d);

//        //tester si le volume voisin est subdivise
//        if(d3 != d && volumeIsSubdivided(d3))
//            subdNeighborhood = true;

//		unsigned int cur = m_map.getCurrentLevel();
//		m_map.setCurrentLevel(cur + 1) ;
//		//tester si la face subdivise a des faces subdivise
//        Dart cf = m_map.phi1(d);

//        do
//        {
//            if(faceIsSubdivided(cf))
//                subdOnce = false;

//			cf = m_map.phi2(m_map.phi1(cf));
//        }
//        while(subdOnce && cf != m_map.phi1(d));

//		m_map.setCurrentLevel(cur) ;
//	}

//    return subd && !subdNeighborhood && subdOnce;
//}

//template <typename PFP>
//bool IHM3<PFP>::volumeIsSubdivided(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

//	unsigned int vLevel = volumeLevel(d);
//	if(vLevel < m_map.getCurrentLevel())
//        return false;

////	bool subd = false;

////	unsigned int cur = m_map.getCurrentLevel();
////	m_map.setCurrentLevel(cur + 1) ;
////	if(m_map.getDartLevel(m_map.phi2(m_map.phi1(m_map.phi1(d)))) == m_map.getCurrentLevel() && m_map.getFaceId(m_map.phi2(m_map.phi1(m_map.phi1(d)))) != m_map.getFaceId(d))
////		subd = true;
////	m_map.setCurrentLevel(cur) ;

////	std::cout << "volume is subdivided ? " << ( subd ? "true" : "false" ) << std::endl;

////	return subd;

//	bool facesAreSubdivided = faceIsSubdivided(d) ;
//	//bool facesAreSubdivided = true ;

//	Traversor3WF<MAP> trav3WF(m_map, d);
//	for(Dart dit = trav3WF.begin() ; dit != trav3WF.end() ; dit = trav3WF.next())
//	{
//		// in a first time, the level of a face
//		//the level of the volume is the minimum of the
//		//levels of its faces

//		facesAreSubdivided &= faceIsSubdivided(dit) ;
//	}

//	//but not the volume itself
//	bool subd = false;
//	m_map.incCurrentLevel() ;
//	if(facesAreSubdivided && m_map.getDartLevel(m_map.phi2(m_map.phi1(m_map.phi1(d)))) == m_map.getCurrentLevel() && m_map.getFaceId(m_map.phi2(m_map.phi1(m_map.phi1(d)))) != m_map.getFaceId(d))
//		subd = true;
//	m_map.decCurrentLevel() ;

//	return subd;

//}

//template <typename PFP>
//bool IHM3<PFP>::volumeIsSubdividedOnce(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//    unsigned int vLevel = volumeLevel(d);
//    if(vLevel < m_map.getCurrentLevel())
//        return false;

//    bool subd = false ;
//    bool subdOnce = true ;

//    m_map.incCurrentLevel() ;
//	if(m_map.getDartLevel(m_map.phi2(m_map.phi1(m_map.phi1(d)))) == m_map.getCurrentLevel() && m_map.getFaceId(m_map.phi2(m_map.phi1(m_map.phi1(d)))) != m_map.getFaceId(d))
//    {
//        subd = true;
//        m_map.incCurrentLevel() ;
//		Dart dcenter = m_map.phi_1(m_map.phi2(m_map.phi1(d)));
//		Traversor3VW<ImplicitHierarchicalMap3> trav3(m_map, dcenter);
//        for(Dart dit = trav3.begin() ; subdOnce && dit != trav3.end() && subdOnce; dit = trav3.next())
//        {
//			if(m_map.getDartLevel(m_map.phi2(m_map.phi1(m_map.phi1(dit)))) == m_map.getCurrentLevel() && m_map.getFaceId(m_map.phi2(m_map.phi1(m_map.phi1(dit)))) != m_map.getFaceId(dit))
//                subdOnce = false;
//        }
//        m_map.decCurrentLevel() ;
//    }
//    m_map.decCurrentLevel() ;
//    return subd && subdOnce;

////	//si le volume est subdivise
////
////	//test si toutes les faces sont subdivisee
////	DartMarkerStore<MAP> mark(m_map);		// Lock a marker
////
////	std::vector<Dart> visitedFaces;		// Faces that are traversed
////	visitedFaces.reserve(512);
////	visitedFaces.push_back(d);			// Start with the face of d
////	std::vector<Dart>::iterator face;
////
////	bool facesAreSubdivided = faceIsSubdivided(d) ;
////
////	//parcours les faces du volume au niveau courant
////	//on cherche le brin de niveau le plus bas de la hierarchie
////	//on note le niveau le plus bas de la hierarchie
////	mark.markOrbit<FACE>(d) ;
////	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
////	{
////		Dart e = visitedFaces[i] ;
////
////		// in a first time, the level of a face
////		//the level of the volume is the minimum of the
////		//levels of its faces
////
////		facesAreSubdivided &= faceIsSubdivided(e) ;
////
////		do	// add all face neighbours to the table
////		{
////			Dart ee = phi2(e) ;
////			if(!mark.isMarked(ee)) // not already marked
////			{
////				visitedFaces.push_back(ee) ;
////				mark.markOrbit<FACE>(ee) ;
////			}
////			e = phi1(e) ;
////		} while(e != visitedFaces[i]) ;
////	}

//}

///***************************************************
// *               SUBDIVISION                       *
// ***************************************************/

//template <typename PFP>
//void IHM3<PFP>::subdivideEdge(Dart d)
//{
//    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//	assert(!edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;

//    unsigned int eLevel = edgeLevel(d) ;

//    unsigned int cur = m_map.getCurrentLevel() ;
//    m_map.setCurrentLevel(eLevel) ;

//    Dart dd = m_map.phi2(d) ;

//    m_map.setCurrentLevel(eLevel + 1) ;

//    m_map.cutEdge(d) ;
//    unsigned int eId = m_map.getEdgeId(d) ;
//	m_map.setEdgeId(m_map.phi1(d), eId) ; //mise a jour de l'id d'arrete sur chaque moitie d'arete
//	m_map.setEdgeId(m_map.phi1(dd), eId) ;

//    m_map.setFaceId(EDGE, d) ; //mise a jour de l'id de face sur chaque brin de chaque moitie d'arete
//    m_map.setFaceId(EDGE, dd) ;

//    (*edgeVertexFunctor)(m_map.phi1(d)) ;

//    m_map.setCurrentLevel(cur) ;
//}

//template <typename PFP>
//void IHM3<PFP>::coarsenEdge(Dart d)
//{
//    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//    assert(edgeCanBeCoarsened(d) || !"Trying to coarsen an edge that can not be coarsened") ;

//    unsigned int cur = m_map.getCurrentLevel() ;
//    m_map.setCurrentLevel(cur + 1) ;
//    m_map.uncutEdge(d) ;
//    m_map.setCurrentLevel(cur) ;
//}

//template <typename PFP>
//unsigned int IHM3<PFP>::subdivideFace(Dart d, bool triQuad)
//{
//    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//	assert(!faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

//    unsigned int fLevel = faceLevel(d) ;
//    Dart old = faceOldestDart(d) ;

//    unsigned int cur = m_map.getCurrentLevel() ;
//    m_map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

//    unsigned int degree = 0 ;
//	Traversor3FE<typename PFP::MAP>  travE(m_map, old);
//    for(Dart it = travE.begin(); it != travE.end() ; it = travE.next())
//    {
//        ++degree ;						// compute the degree of the face

//		if(!edgeIsSubdivided(it))							// first cut the edges (if they are not already)
//			subdivideEdge(it) ;	// and compute the degree of the face
//    }

//    m_map.setCurrentLevel(fLevel + 1) ;			// go to the next level to perform face subdivision

//    if((degree == 3) && triQuad)					// if subdividing a triangle
//    {
//        Dart dd = m_map.phi1(old) ;
//        Dart e = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(e) ;
//		e = m_map.phi1(e) ;

//        m_map.splitFace(dd, e) ;					// insert a new edge
//		unsigned int id = m_map.getNewEdgeId() ;
////		unsigned int id = m_map.triRefinementEdgeId(m_map.phi_1(dd));
//		m_map.setEdgeId(m_map.phi_1(dd), id) ;		// set the edge id of the inserted edge to the next available id

//        unsigned int idface = m_map.getFaceId(old);
//        m_map.setFaceId(dd, idface, FACE) ;
//        m_map.setFaceId(e, idface, FACE) ;

//        dd = e ;
//        e = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(e) ;
//        e = m_map.phi1(dd);
//        m_map.splitFace(dd, e) ;
//		id = m_map.getNewEdgeId() ;
////		id = m_map.triRefinementEdgeId(m_map.phi_1(dd));
//		m_map.setEdgeId(m_map.phi_1(dd), id) ;

//        m_map.setFaceId(dd, idface, FACE) ;
//        m_map.setFaceId(e, idface, FACE) ;

//        dd = e ;
//        e = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(e) ;
//        e = m_map.phi1(dd);
//        m_map.splitFace(dd, e) ;
//		id = m_map.getNewEdgeId() ;
////		id = m_map.triRefinementEdgeId(m_map.phi_1(dd));
//		m_map.setEdgeId(m_map.phi_1(dd), id) ;

//        m_map.setFaceId(dd, idface, FACE) ;
//        m_map.setFaceId(e, idface, FACE) ;
//    }
//    else
//    {
//        Dart dd = m_map.phi1(old) ;
//        Dart next = m_map.phi1(dd) ;
//        (*vertexVertexFunctor)(next) ;
//        next = m_map.phi1(next);
//        m_map.splitFace(dd, next) ;
//        Dart ne = m_map.phi2(m_map.phi_1(dd));
//        Dart ne2 = m_map.phi2(ne);

//        m_map.cutEdge(ne) ;
//		unsigned int id = m_map.getNewEdgeId() ;
////		unsigned int id = m_map.getQuadRefinementEdgeId(m_map.phi2(ne));
//		m_map.setEdgeId(ne, id) ;
//		id = m_map.getNewEdgeId() ;
////		id = m_map.getQuadRefinementEdgeId(m_map.phi2(ne2));
//		m_map.setEdgeId(ne2, id) ;

//		dd = m_map.phi1(next) ;
//        (*vertexVertexFunctor)(dd) ;
//		dd = m_map.phi1(dd);
//        while(dd != ne)
//        {
//            m_map.splitFace(m_map.phi1(ne), dd) ;
//            Dart nne = m_map.phi2(m_map.phi_1(dd)) ;
//			id = m_map.getNewEdgeId() ;
////			id = m_map.getQuadRefinementEdgeId(m_map.phi2(nne));
//			m_map.setEdgeId(nne, id) ;
//			dd = m_map.phi1(dd) ;
//			(*vertexVertexFunctor)(dd) ;
//			dd = m_map.phi1(dd) ;
//        }

//        unsigned int idface = m_map.getFaceId(old);
//        //Dart e = dd;
//        do
//        {
//            m_map.setFaceId(dd, idface, DART) ;
//            m_map.setFaceId(m_map.phi2(dd), idface, DART) ;
//            dd = m_map.phi2(m_map.phi1(dd));
//        }
//        while(dd != ne);

//        (*faceVertexFunctor)(m_map.phi1(ne)) ;
//    }

//    m_map.setCurrentLevel(cur) ;
//    return cur;
//}

//template <typename PFP>
//void IHM3<PFP>::coarsenFace(Dart d)
//{
//    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//    assert(faceCanBeCoarsened(d) || !"Trying to coarsen a non-subdivided face or a more than once subdivided face") ;

//    unsigned int cur = m_map.getCurrentLevel() ;

//    unsigned int degree = 0 ;
//    Dart fit = d ;
//    do
//    {
//        ++degree ;
//        fit = m_map.phi1(fit) ;
//    } while(fit != d) ;

////	Dart d3 = m_map.phi3(d);

//    if(degree == 3)
//    {

//    }
//    else
//    {
//        m_map.setCurrentLevel(cur + 1) ;
//        m_map.deleteVertexSubdividedFace(d);
//        m_map.setCurrentLevel(cur) ;

////		Dart centralV = m_map.phi1(m_map.phi1(d));
////		m_map.m_map2::deleteVertex(centralV);
////
////		//demarking faces from border to delete .... fucking shit
////		Dart it = d ;
////		do
////		{
////			if (m_map.boundaryUnmark(it))
////				return true ;
////			it = m_map.phi2(m_map.phi_1(it)) ;
////		} while (it != d) ;
////
////		m_map.m_map2::deleteVertex(m_map.phi1(m_map.phi1(d3)));

//    }

//    fit = d ;
//    do
//    {
//        if(edgeCanBeCoarsened(fit))
//            coarsenEdge(fit) ;
//        fit = m_map.phi1(fit) ;
//    } while(fit != d) ;
//}

//template <typename PFP>
//Dart IHM3<PFP>::subdivideVolume(Dart d, bool triQuad, bool OneLevelDifference)
//{
//    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//    assert(!volumeIsSubdivided(d) || !"Trying to subdivide an already subdivided volume") ;

//    unsigned int vLevel = volumeLevel(d);
//    Dart old = volumeOldestDart(d);

//    unsigned int cur = m_map.getCurrentLevel();
//    m_map.setCurrentLevel(vLevel);

//	std::cout << "current Level = " << m_map.getCurrentLevel() << std::endl;

////	if(OneLevelDifference)
////	{
////		Traversor3WF<typename PFP::MAP> traF(m_map, old);
////		for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
////		{
////			Dart nv = m_map.phi3(dit);
////			if(!m_map.isBoundaryMarked(3, nv))
////				if(volumeLevel(nv) == vLevel - 1)
////					subdivideVolume(nv,triQuad,OneLevelDifference);
////		}
////	}


//	//Store the edges before the cutEdge
//	std::vector<Dart> oldEdges;
//	oldEdges.reserve(20);

//	Traversor3WV<typename PFP::MAP> traV(m_map, old);
//	for(Dart dit = traV.begin(); dit != traV.end(); dit = traV.next())
//	{
//		oldEdges.push_back(dit);
//	}

//	std::vector<std::pair<Dart, Dart> > subdividedFaces;
//	subdividedFaces.reserve(128);

//	Traversor3WF<typename PFP::MAP> traF(m_map, old);
//	for(Dart dit = traF.begin(); dit != traF.end(); dit = traF.next())
//	{
//		//if needed subdivide face
//		if(!faceIsSubdivided(dit))
//			subdivideFace(dit, triQuad);

//		//save a dart from the subdivided face
////		m_map.incCurrentLevel();
//		unsigned int cur = m_map.getCurrentLevel();

//		unsigned int fLevel = faceLevel(dit); //puisque dans tous les cas, la face est subdivisee
//		m_map.setCurrentLevel(fLevel + 1) ;

//		//le brin est forcement du niveau cur
//		Dart cf = m_map.phi1(dit);
//		Dart e = cf;
//		do
//		{
//			subdividedFaces.push_back(std::pair<Dart,Dart>(e,m_map.phi2(e)));
//			e = m_map.phi2(m_map.phi1(e));
//		}while (e != cf);

////		m_map.decCurrentLevel();
//		m_map.setCurrentLevel(cur);
//	}

//	Dart centralDart = NIL;

//    std::vector<Dart> newEdges;	//save darts from inner edges
//    newEdges.reserve(50);

//	m_map.setCurrentLevel(vLevel + 1) ;

//    //Second step : deconnect each corner, close each hole, subdivide each new face into 3
////	Traversor3WV<typename PFP::MAP> traWV(m_map, old);
////	for(Dart ditWV = traWV.begin(); ditWV != traWV.end(); ditWV = traWV.next())
////	{
//	for (std::vector<Dart>::iterator edge = oldEdges.begin(); edge != oldEdges.end(); ++edge)
//	{
//		Dart e = *edge;

//        std::vector<Dart> v ;

//        do
//        {
//			v.push_back(m_map.phi1(m_map.phi1(e)));
//            v.push_back(m_map.phi1(e));
//            e = m_map.phi2(m_map.phi_1(e));
//        }
//		while(e != *edge);

//		m_map.splitVolume(v) ;

//		Dart old = m_map.phi2(m_map.phi1(*edge));
//		Dart dd = m_map.phi1(m_map.phi1(old)) ;
//		m_map.splitFace(old,dd) ;

//		unsigned int idface = m_map.getNewFaceId();
//		m_map.setFaceId(dd,idface, FACE);

//		Dart ne = m_map.phi1(m_map.phi1(old)) ;

//		m_map.cutEdge(ne);
//		centralDart = m_map.phi1(ne);
//		newEdges.push_back(ne);
//		newEdges.push_back(m_map.phi1(ne));

//		unsigned int id = m_map.getNewEdgeId() ;
//		m_map.setEdgeId(ne, id) ;

//		Dart stop = m_map.phi2(m_map.phi1(ne));
//		ne = m_map.phi2(ne);
//		do
//		{
//			dd = m_map.phi1(m_map.phi1(m_map.phi1(ne)));

//			m_map.splitFace(ne, dd) ;
//			unsigned int idface = m_map.getNewFaceId();
//			m_map.setFaceId(dd,idface, FACE);

//			newEdges.push_back(m_map.phi1(dd));

//			ne = m_map.phi2(m_map.phi_1(ne));
//			dd = m_map.phi1(m_map.phi1(dd));
//		}
//		while(dd != stop);
//    }

//	if(vLevel < 1)
//	{
//	 m_map.deleteVolume(m_map.phi3(m_map.phi2(m_map.phi1(oldEdges.front()))));

//	//Third step : 3-sew internal faces
//	for (std::vector<std::pair<Dart,Dart> >::iterator it = subdividedFaces.begin(); it != subdividedFaces.end(); ++it)
//	{
//		Dart f1 = (*it).first;
//		Dart f2 = (*it).second;

//		if(m_map.isBoundaryFace(m_map.phi2(f1)) && m_map.isBoundaryFace(m_map.phi2(f2)))
//		{
//			//id pour toutes les faces interieures
//			m_map.sewVolumes(m_map.phi2(f1), m_map.phi2(f2));

//			//Fais a la couture !!!!!
//			unsigned int idface = m_map.getNewFaceId();
//			m_map.setFaceId(m_map.phi2(f1),idface, FACE);
//		}

//		//FAIS a la couture !!!!!!!
//		//id pour toutes les aretes exterieurs des faces quadrangulees
//		unsigned int idedge = m_map.getEdgeId(f1);
//		m_map.setDartEdgeId(m_map.phi2(f1), idedge);
//		m_map.setDartEdgeId( m_map.phi2(f2), idedge);
//	}

//    //replonger l'orbit de ditV.
////	Algo::Topo::setOrbitEmbedding<VERTEX>(m_map,centralDart, m_map.template getEmbedding<VERTEX>(centralDart));

//    //LA copie de L'id est a gerer avec le sewVolumes normalement !!!!!!
//    //id pour les aretes interieurs : (i.e. 6 pour un hexa)
//	DartMarkerStore<typename PFP::MAP> mne(m_map);
//	for(unsigned int i = 0; i < newEdges.size(); ++i)
//	{
//		if(!mne.isMarked(newEdges[i]))
//		{
//			unsigned int idedge = m_map.getNewEdgeId();
//			m_map.setEdgeId(newEdges[i], idedge);
//			mne.markOrbit(Edge(newEdges[i]));
//		}
//	}

//	(*volumeVertexFunctor)(centralDart) ;

////    m_map.setCurrentLevel(cur) ;

//	return centralDart;
//	}
//	else
//		return NIL;
//}

//template <typename PFP>
//void IHM3<PFP>::coarsenVolume(Dart d)
//{
//    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
//	assert(volumeIsSubdividedOnce(d) || !"Trying to coarsen a non-subdivided volume or a more than once subdivided volume") ;

//    unsigned int cur = m_map.getCurrentLevel() ;

//    /*
//     * Deconnecter toutes les faces interieurs
//     */
//    m_map.setCurrentLevel(cur + 1) ;
//    Dart nf = m_map.phi_1(m_map.phi2(m_map.phi1(d)));
//    m_map.deleteVertex(nf);
//    m_map.setCurrentLevel(cur) ;

//    /*
//     * simplifier les faces
//     */
//    Traversor3WF<typename PFP::m_map> trav3WF(m_map, d, true);
//    for(Dart dit = trav3WF.begin() ; dit != trav3WF.end() ; dit = trav3WF.next())
//    {
//        if(faceCanBeCoarsened(dit))
//            coarsenFace(dit);
//    }
//}

//} // namespace Adaptive

//} // namespace Primal

//} // namespace MR

//} // namespace Volume

//} // namespace Algo

//} // namespace CGoGN
