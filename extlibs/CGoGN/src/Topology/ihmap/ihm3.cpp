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

#include <cmath>
#include <limits>
#include "Topology/ihmap/ihm3.h"
#include "Topology/generic/traversor/traversor3.h"


namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace IHM
{

ImplicitHierarchicalMap3::ImplicitHierarchicalMap3() : ParentMap()
{
//    m_dartLevel = this->addAttribute<unsigned int, DART, MAP, HandlerAccessorPolicy >("dartLevel") ;
//    this->setDartLvlAttribute(m_dartLevel.getDataVector());
//    m_edgeId = this->addAttribute<unsigned int, DART, MAP, HandlerAccessorPolicy >("edgeId") ;
//    m_faceId = this->addAttribute<unsigned int, DART, MAP, HandlerAccessorPolicy >("faceId") ;
//    m_nextLevelCell = m_attribs[VERTEX].addAttribute<unsigned int>("nextLevelCell") ;
//    for(unsigned int i = m_attribs[VERTEX].begin(); i < m_attribs[VERTEX].end(); m_attribs[VERTEX].next(i))
//        m_nextLevelCell->operator[](i) = EMBNULL ;
}

ImplicitHierarchicalMap3::~ImplicitHierarchicalMap3()
{

}

void ImplicitHierarchicalMap3::clear(bool removeAttrib)
{
    Parent::clear(removeAttrib) ;
	if (removeAttrib)
    {
        for(unsigned int i = m_attribs[DART].begin(); i < m_attribs[DART].end(); m_attribs[DART].next(i))
            m_nextLevelCell->operator[](i) = EMBNULL ;
    }
}


//void ImplicitHierarchicalMap3::deleteVolume(Dart d)
//{
//	unsigned int emb = getEmbedding<VERTEX>(d);
//	Dart dr = phi1(phi1(d));

//	EmbeddedMap3::deleteVolume(d);

//	if(isOrbitEmbedded<VERTEX>())
//	{
//		setOrbitEmbedding<VERTEX>(dr,emb);
//	}
//}

void ImplicitHierarchicalMap3::swapEdges(Dart d, Dart e)
{
	if(!Map2::isBoundaryEdge(d) && !Map2::isBoundaryEdge(e))
	{
	Dart d2 = phi2(d);
	Dart e2 = phi2(e);

	Map2::unsewFaces(d);
	Map2::unsewFaces(e);

	Map2::sewFaces(d, e);
	Map2::sewFaces(d2, e2);

	if(isOrbitEmbedded<VERTEX>())
	{
		copyDartEmbedding<VERTEX>(d, phi2(phi_1(d)));
		copyDartEmbedding<VERTEX>(e, phi2(phi_1(e)));
		copyDartEmbedding<VERTEX>(d2, phi2(phi_1(d2)));
		copyDartEmbedding<VERTEX>(e2, phi2(phi_1(e2)));
	}

	if(isOrbitEmbedded<EDGE>())
	{

	}

	if(isOrbitEmbedded<VOLUME>())
		Algo::Topo::setOrbitEmbeddingOnNewCell<VOLUME>(*this, d);
	}
}

void ImplicitHierarchicalMap3::saveRelationsAroundVertex(Dart d, std::vector<std::pair<Dart, Dart> >& vd)
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;

	//le brin est forcement du niveau cur
	Dart dit = d;

	do
	{
		vd.push_back(std::pair<Dart,Dart>(dit,phi2(dit)));

		dit = phi2(phi_1(dit));

	}while(dit != d);
}

void ImplicitHierarchicalMap3::unsewAroundVertex(std::vector<std::pair<Dart, Dart> >& vd)
{
	//unsew the edge path
	for(std::vector<std::pair<Dart, Dart> >::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		Dart dit = (*it).first;
		Dart dit2 = (*it).second;

		Map2::unsewFaces(dit);

		if(isOrbitEmbedded<VERTEX>())
		{
			copyDartEmbedding<VERTEX>(phi2(dit2), dit);
			copyDartEmbedding<VERTEX>(phi2(dit), dit2);
		}

		if(isOrbitEmbedded<EDGE>())
		{

		}
	}
}

Dart ImplicitHierarchicalMap3::quadranguleFace(Dart d)
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;

	Dart centralDart = NIL;
	Map2::fillHole(phi1(d));

	Dart old = phi2(phi1(d));
	Dart bc = newBoundaryCycle(faceDegree(old));
	sewVolumes(old, bc, false);

	if (isOrbitEmbedded<VERTEX>())
	{
		Dart it = bc;
		do
		{
			//copyDartEmbedding<VERTEX>(it, phi1(phi3(it)));
			Algo::Topo::setOrbitEmbedding<VERTEX>(*this, it, getEmbedding<VERTEX>(phi1(phi3(it))));
			it = phi1(it) ;
		} while(it != bc) ;
	}


	Dart dd = phi1(phi1(old)) ;
	splitFace(old,dd) ;

	unsigned int idface = getNewFaceId();
	setFaceId(dd,idface, FACE);

	Dart ne = phi1(phi1(old)) ;

	cutEdge(ne);
	centralDart = phi1(ne);

	//newEdges.push_back(ne);
	//newEdges.push_back(map.phi1(ne));

	unsigned int id = getNewEdgeId() ;
	setEdgeId(ne, id, EDGE) ;

	Dart stop = phi2(phi1(ne));
	ne = phi2(ne);
	do
	{
		dd = phi1(phi1(phi1(ne)));

		splitFace(ne, dd) ;

		unsigned int idface = getNewFaceId();
		setFaceId(dd,idface, FACE);

		//newEdges.push_back(map.phi1(dd));

		ne = phi2(phi_1(ne));
		dd = phi1(phi1(dd));
	}
	while(dd != stop);

	return centralDart;
}


void ImplicitHierarchicalMap3::deleteVertexSubdividedFace(Dart d)
{
    assert(this->volumeOldestDart(d) == d);

	Dart centralV = phi1(phi1(d));
	Dart res = NIL;
	Dart vit = centralV ;
	do
	{
		if(res == NIL && phi1(phi1(centralV)) != centralV)
			res = phi1(centralV) ;

		Dart f = phi_1(phi2(vit)) ;
		phi1sew(vit, f) ;

		vit = phi2(phi_1(vit)) ;
	} while(vit != centralV) ;
	Map1::deleteCycle(centralV) ;

	Dart d3 = phi1(phi3(centralV));
	res = NIL;
	vit = d3 ;
	do
	{
		if(res == NIL && phi1(phi1(d3)) != d3)
			res = phi1(d3) ;

		Dart f = phi_1(phi2(vit)) ;
		phi1sew(vit, f) ;

		vit = phi2(phi_1(vit)) ;
	} while(vit != d3) ;
	Map1::deleteCycle(d3) ;

}

void ImplicitHierarchicalMap3::initEdgeId()
{
    DartMarker<ParentMap> edgeMark(*this) ;
    for(Dart d = this->beginMaxLvl(); d != this->endMaxLvl(); this->nextMaxLvl(d))
	{
		if(!edgeMark.isMarked(d))
		{
			Dart e = d;
			do
			{
                Parent::setEdgeId(e, m_edgeIdCount);
				edgeMark.mark(e);

                Parent::setEdgeId(phi2MaxLvl(e), m_edgeIdCount);
                edgeMark.mark(phi2MaxLvl(e));

                e = this->alpha2MaxLvl(e);
			} while(e != d);

			m_edgeIdCount++;
		}
	}
}

void ImplicitHierarchicalMap3::initFaceId()
{
    DartMarker<ParentMap> faceMark(*this) ;
    for(Dart d = this->beginMaxLvl(); d != this->endMaxLvl(); this->nextMaxLvl(d))
	{
		if(!faceMark.isMarked(d))
		{
			Dart e = d;
			do
			{
                Parent::setFaceId(e, m_faceIdCount);
				faceMark.mark(e);

                Dart e3 = phi3MaxLvl(e);
                Parent::setFaceId(e3, m_faceIdCount);
				faceMark.mark(e3);

                e = phi1MaxLvl(e);
			} while(e != d);

			m_faceIdCount++;
		}
	}
}

unsigned int ImplicitHierarchicalMap3::faceLevel(Dart d)
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;

    if(getCurrentLevel() == 0)
        return 0 ;

    Dart it = d ;
    Dart old = it ;
    unsigned int l_old = getDartLevel(old) ;
    unsigned int fLevel = edgeLevel(it) ;
    do
    {
        it = phi1(it) ;
        unsigned int dl = getDartLevel(it) ;
        if(dl < l_old)							// compute the oldest dart of the face
        {										// in the same time
            old = it ;
            l_old = dl ;
        }										// in a first time, the level of a face
        unsigned int l = edgeLevel(it) ;		// is the minimum of the levels
        fLevel = l < fLevel ? l : fLevel ;		// of its edges
    } while(it != d) ;

    unsigned int cur = getCurrentLevel() ;
    setCurrentLevel(fLevel);

    unsigned int nbSubd = 0 ;
    it = old ;
    unsigned int eId = getEdgeId(old);			// the particular case of a face
    do											// with all neighboring faces regularly subdivided
    {											// but not the face itself
        ++nbSubd ;								// is treated here
        it = phi1(it) ;
    } while(getEdgeId(it) == eId && (getDartLevel(it) != l_old)) ;

    while(nbSubd > 1)
    {
        nbSubd /= 2 ;
        --fLevel ;
    }

    setCurrentLevel(cur);
    return fLevel ;
}

unsigned int ImplicitHierarchicalMap3::volumeLevel(Dart d)
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;

    if(getCurrentLevel() == 0)
        return 0 ;

    Dart oldest = d ;
    unsigned int l_oldest=getDartLevel(d);
    unsigned int vLevel = std::numeric_limits<unsigned int>::max(); //hook sioux
//	//First : the level of a volume is the minimum of the levels of its faces
    Traversor3WF<ImplicitHierarchicalMap3> travF(*this, d);
    for (Dart dit = travF.begin(); dit != travF.end(); dit = travF.next())
    {
        // in a first time, the level of a face
        //the level of the volume is the minimum of the
        //levels of its faces
        unsigned int fLevel = faceLevel(dit);
        vLevel = fLevel < vLevel ? fLevel : vLevel ;
        Dart old =faceOldestDart(dit);
        unsigned int l_old=getDartLevel(old);
        if(l_old < l_oldest)
        {
            l_oldest=l_old;
            oldest = old ;
        }
    }

    //Second : the case of all faces regularly subdivided but not the volume itself
    unsigned int cur = getCurrentLevel() ;
    setCurrentLevel(vLevel) ;

    unsigned int nbSubd = 0 ;
    Dart it = oldest ;
    unsigned int eId = getEdgeId(oldest) ;
    unsigned int fId = getFaceId(oldest);

    do
    {
        ++nbSubd ;
        it = phi1(it) ;
        while(getEdgeId(it)!=eId && getFaceId(it) == fId  && getDartLevel(it) != l_oldest)
        {
            it=phi1(phi2(it));
        }
    } while(getFaceId(it) == fId  && getDartLevel(it) != l_oldest) ;


    while(nbSubd > 1)
    {
        nbSubd /= 2 ;
        --vLevel ;
    }

    setCurrentLevel(cur) ;

    return vLevel;
}

Dart ImplicitHierarchicalMap3::edgeNewestDart(Dart d) const
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;

    unsigned int l_new = getDartLevel(d);
    if (l_new == getCurrentLevel())
    {
        return d;
    }

    const Dart phi2d = phi2(d);
    const unsigned int l2 = getDartLevel(phi2(d));
    if (l2 == getCurrentLevel())
    {
        return phi2d;
    }

    if (l2 > l_new)
    {
        d = phi2d;
        l_new = l2;
    }


    Dart newest = d;
    Dart dit = alpha2(d);
    do
    {
        const unsigned l = getDartLevel(dit);
        if (l == this->getCurrentLevel())
            return dit ;

        if (l > l_new )
        {
            l_new = l;
            newest = dit;
        }
        dit = alpha2(dit);
    } while(dit != d);
    return newest;
}

Dart ImplicitHierarchicalMap3::faceOldestDart(Dart d) const
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;
	Dart it = d ;
	Dart oldest = it ;
    unsigned int l_old = getDartLevel(oldest) ;
	do
	{
        unsigned int l = getDartLevel(it) ;
		if(l == 0)
			return it ;
		if(l < l_old)
		//if(l < l_old || (l == l_old && it < oldest))
		{
			oldest = it ;
			l_old = l ;
		}
		it = phi1(it) ;
	} while(it != d) ;
	return oldest ;
}

Dart ImplicitHierarchicalMap3::faceNewestDart(Dart d) const {
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;
    Dart it = d ;
    Dart newest = it ;
    unsigned int l_new = getDartLevel(newest) ;
    do
    {
        const unsigned int l = getDartLevel(it) ;
        if(l == this->getCurrentLevel())
            return it ;
        if(l > l_new)
        {
            newest = it ;
            l_new = l ;
        }
        it = phi1(it) ;
    } while(it != d) ;
    return newest;
}

Dart ImplicitHierarchicalMap3::volumeOldestDart(Dart d)
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;

	Dart oldest = d;
    unsigned int l_old = getDartLevel(oldest);

	Traversor3WF<ImplicitHierarchicalMap3> trav3WF(*this, oldest);
	for(Dart dit = trav3WF.begin() ; dit != trav3WF.end() ; dit = trav3WF.next())
	{
		Dart old = faceOldestDart(dit);
        unsigned int l = getDartLevel(old);
		if(l < l_old)
		{
			oldest = old;
			l_old = l;
		}
	}

    return oldest;
}

Dart ImplicitHierarchicalMap3::volumeNewestDart(Dart d) const
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;

    Dart newest = d;
    unsigned int l_new = getDartLevel(newest);

    Traversor3WF<ImplicitHierarchicalMap3> trav3WF(*this, newest);
    for(Dart dit = trav3WF.begin(), end = trav3WF.end() ; dit != end; dit = trav3WF.next())
    {
        Dart old = faceOldestDart(dit);
        const unsigned int l = getDartLevel(old);

        if(l == this->getCurrentLevel())
            return dit ;

        if(l > l_new)
        {
            newest = old;
            l_new = l;
        }
    }

    return newest;
}

bool ImplicitHierarchicalMap3::edgeIsSubdivided(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;

	//Dart d2 = phi2(d) ;
	Dart d1 = phi1(d) ;
    ++m_curLevel ;
	//Dart d2_l = phi2(d) ;
	Dart d1_l = phi1(d) ;
    --m_curLevel ;
	//if(d2 != d2_l)
	if(d1 != d1_l)
		return true ;
	else
		return false ;
}

bool ImplicitHierarchicalMap3::edgeCanBeCoarsened(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;

	bool subd = false ;
	bool subdOnce = true ;
	bool degree2 = false ;

	if(edgeIsSubdivided(d))
	{
		subd = true ;
        ++m_curLevel ;

		if(vertexDegree(phi1(d)) == 2)
		{
			degree2 = true ;
			if(edgeIsSubdivided(d))
				subdOnce = false ;
		}
        --m_curLevel ;
	}
	return subd && degree2 && subdOnce ;
}

bool ImplicitHierarchicalMap3::faceIsSubdivided(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
	unsigned int fLevel = faceLevel(d) ;
    if(fLevel < m_curLevel)
		return false ;

	bool subd = false ;
    ++m_curLevel ;
//	if(m_dartLevel[phi1(d)] == m_curLevel && m_edgeId[phi1(d)] != m_edgeId[d])
    if (fLevel > faceLevel(d))
		subd = true ;
    --m_curLevel ;

	return subd ;
}

bool ImplicitHierarchicalMap3::faceCanBeCoarsened(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;

	bool subd = false;
	bool subdOnce = true;
	bool subdNeighborhood = false; //deux volumes voisins de la face ne sont pas subdivise

	if(faceIsSubdivided(d))
	{
		subd = true;
		Dart d3 = phi3(d);

		//tester si le volume voisin est subdivise
		if(d3 != d && volumeIsSubdivided(d3))
			subdNeighborhood = true;

        ++m_curLevel;
		//tester si la face subdivise a des faces subdivise
		Dart cf = phi1(d);

		do
		{
			if(faceIsSubdivided(cf))
				subdOnce = false;

			cf = phi2(phi1(cf));
		}
		while(subdOnce && cf != phi1(d));

        --m_curLevel;
	}

	return subd && !subdNeighborhood && subdOnce;
}



bool ImplicitHierarchicalMap3::volumeIsSubdivided(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    unsigned int vLevel = volumeLevel(d);
    if(vLevel < m_curLevel)
        return false;

    bool subd = false;

    ++m_curLevel;
//    if(m_dartLevel[phi2(phi1(phi1(d)))] == m_curLevel && m_faceId[phi2(phi1(phi1(d)))] != m_faceId[d])
    if(volumeLevel(d)>vLevel) //test par thomas
        subd = true;
    --m_curLevel;

    return subd;
}


bool ImplicitHierarchicalMap3::volumeIsSubdividedOnce(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    unsigned int vLevel = volumeLevel(d);
    if(vLevel < m_curLevel)
        return false;

    bool subd = false ;
    bool subdOnce = true ;

    ++m_curLevel;
//    if(m_dartLevel[phi2(phi1(phi1(d)))] == m_curLevel && m_faceId[phi2(phi1(phi1(d)))] != m_faceId[d])
    if(volumeLevel(d)>vLevel)
    {
        subd = true;
        ++m_curLevel;
        Dart dcenter = phi_1(phi2(phi1(d)));
        Traversor3VW<ImplicitHierarchicalMap3> trav3(*this, dcenter);
        for(Dart dit = trav3.begin() ; subdOnce && dit != trav3.end() && subdOnce; dit = trav3.next())
        {
//            if(m_dartLevel[phi2(phi1(phi1(dit)))] == m_curLevel && m_faceId[phi2(phi1(phi1(dit)))] != m_faceId[dit])
            if(volumeLevel(dit)>vLevel+1)
                subdOnce = false;
        }
        --m_curLevel;
    }
    --m_curLevel;
    return subd && subdOnce;

}


bool ImplicitHierarchicalMap3::neighborhoodLevelDiffersMoreThanOne(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;

	int vLevel = volumeLevel(d);
	bool isMoreThanOne = false;

//	std::cout << "niveau du courant : " << vLevel << std::endl;
	Traversor3WWaV<ImplicitHierarchicalMap3> trav3WWaV(*this, d);
	for(Dart dit = trav3WWaV.begin() ; !isMoreThanOne && dit != trav3WWaV.end() ; dit = trav3WWaV.next())
	{
		//Dart oldit = volumeOldestDart(dit);
//		std::cout << "niveau du voisin : " << volumeLevel(dit) << std::endl;
//		std::cout << "difference de niveau avec voisin : " << abs((volumeLevel(dit) - vLevel)) << std::endl;
		if(abs((int(volumeLevel(dit)) - vLevel)) > 1)
			isMoreThanOne = true;
	}

	return isMoreThanOne;

//	Traversor3EW<ImplicitHierarchicalMap3> trav3EW(*this, old);
//	for(Dart dit = trav3EW.begin() ; dit != trav3EW.end() ; dit = trav3EW.next())
//	{
//		Dart oldit = volumeOldestDart(dit);
//		if((volumeLevel(oldit) - vLevel) > 1)
//			overOne = false;
//	}

}

bool ImplicitHierarchicalMap3::coarsenNeighborhoodLevelDiffersMoreThanOne(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
//	assert(m_curLevel > 0 || !"Coarsen a volume at level 0");

	int vLevel = volumeLevel(d)-1;
	bool isMoreThanOne = false;

//	std::cout << "niveau du courant : " << vLevel << std::endl;
	Traversor3WWaV<ImplicitHierarchicalMap3> trav3WWaV(*this, d);
	for(Dart dit = trav3WWaV.begin() ; !isMoreThanOne && dit != trav3WWaV.end() ; dit = trav3WWaV.next())
	{
		//Dart oldit = volumeOldestDart(dit);
//		std::cout << "niveau du voisin : " << volumeLevel(dit) << std::endl;
//		std::cout << "difference de niveau avec voisin : " << abs((volumeLevel(dit) - vLevel)) << std::endl;
		if(abs((int(volumeLevel(dit)) - vLevel)) > 1)
			isMoreThanOne = true;
	}

	return isMoreThanOne;

//	Traversor3EW<ImplicitHierarchicalMap3> trav3EW(*this, old);
//	for(Dart dit = trav3EW.begin() ; dit != trav3EW.end() ; dit = trav3EW.next())
//	{
//		if(!faceIsSubdividedOnce(dit))
//			OverOne = true;
//	}

	//unsigned int vLevel = volumeLevel(d);

//	DartMarkerStore mf(*this);		// Lock a face marker to save one dart per face
//
//	//Store faces that are traversed and start with the face of d
//	std::vector<Dart> visitedFaces;
//	visitedFaces.reserve(512);
//	visitedFaces.push_back(old);
//
//	mf.markOrbit<FACE>(old) ;
//
//	for(unsigned int i = 0; !found && i < visitedFaces.size(); ++i)
//	{
//		Dart e = visitedFaces[i] ;
//		do
//		{
//			// add all face neighbours to the table
//
//			if(faceIsSubdivided(e))
//			{
//				++m_curLevel;
//
//				if(faceIsSubdividedOnce(e))
//					found = true;
//
//				--m_curLevel;
//			}
//			Dart ee = phi2(e) ;
//			if(!mf.isMarked(ee)) // not already marked
//			{
//				visitedFaces.push_back(ee) ;
//				mf.markOrbit<FACE>(ee) ;
//			}
//
//			e = phi1(e) ;
//		} while(e != visitedFaces[i]) ;
//	}
//
    //	return found;
}

void ImplicitHierarchicalMap3::sewVolumes(Dart d, Dart e, bool withBoundary)
{
    if (!withBoundary)
    {
        Map3::sewVolumes(d, e, false) ;
        return ;
    }

    Map3::sewVolumes(d, e, withBoundary);

    // embed the vertex orbits from the oriented face with dart e
    // with vertex orbits value from oriented face with dart d
    if (isOrbitEmbedded<VERTEX>())
    {
        Dart it = d ;
        do
        {
            Algo::Topo::setOrbitEmbedding<VERTEX>(*this, it, getEmbedding<VERTEX>(it)) ;
            it = phi1(it) ;
        } while(it != d) ;
    }

    // embed the new edge orbit with the old edge orbit value
    // for all the face
    if (isOrbitEmbedded<EDGE>())
    {
        Dart it = d ;
        do
        {
            Algo::Topo::setOrbitEmbedding<EDGE>(*this, it, getEmbedding<EDGE>(it)) ;
            it = phi1(it) ;
        } while(it != d) ;
    }

    // embed the face orbit from the volume sewn
    if (isOrbitEmbedded<FACE>())
    {
        Algo::Topo::setOrbitEmbedding<FACE>(*this, e, getEmbedding<FACE>(d)) ;
    }
}

void ImplicitHierarchicalMap3::splitVolume(std::vector<Dart> &vd)
{
    //        std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
    Map3::splitVolume(vd);

    // follow the edge path a second time to embed the vertex, edge and volume orbits
    for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
    {
        Dart dit = *it;
        Dart dit23 = phi3(phi2(dit));

        // embed the vertex embedded from the origin volume to the new darts
        if(isOrbitEmbedded<VERTEX>())
        {
            copyDartEmbedding<VERTEX>(dit23, dit);
            copyDartEmbedding<VERTEX>(phi2(dit), phi1(dit));
        }

        // embed the edge embedded from the origin volume to the new darts
        if(isOrbitEmbedded<EDGE2>())
        {
            Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE2>(*this, dit23) ;
            copyCell<EDGE2>(getEmbedding<EDGE2>(dit23), getEmbedding<EDGE2>(dit)) ;

            copyDartEmbedding<EDGE2>(phi2(dit), dit);
        }

        // embed the edge embedded from the origin volume to the new darts
        if(isOrbitEmbedded<EDGE>())
        {
            unsigned int eEmb = getEmbedding<EDGE>(dit) ;
            setDartEmbedding<EDGE>(dit23, eEmb);
            setDartEmbedding<EDGE>(phi2(dit), eEmb);
        }

        // embed the volume embedded from the origin volume to the new darts
        if(isOrbitEmbedded<VOLUME>())
        {
            copyDartEmbedding<VOLUME>(phi2(dit), dit);
        }
    }

    if (isOrbitEmbedded<FACE>()) {
        Algo::Topo::initOrbitEmbeddingOnNewCell<FACE>(*this, phi2(vd.front())) ;
    }


    if(isOrbitEmbedded<VOLUME>())
    {
        Dart v = vd.front() ;
        Dart v23 = phi3(phi2(v));
        Algo::Topo::setOrbitEmbeddingOnNewCell<VOLUME>(*this, v23) ;
        Algo::Topo::copyCellAttributes<VOLUME>(*this, v23, v);
//        Algo::Topo::setOrbitEmbedding<VOLUME>(*this,v, this->getEmbedding<VOLUME>(v));
    }
    //    assert(this->template checkEmbeddings<VOLUME>());
}

void ImplicitHierarchicalMap3::splitFace(Dart d, Dart e)
{
    Dart dd = phi1(phi3(d));
    Dart ee = phi1(phi3(e));
    Map3::splitFace(d, e);

    if(isOrbitEmbedded<VERTEX>())
    {
        unsigned int vEmb1 = getEmbedding<VERTEX>(d) ;
        unsigned int vEmb2 = getEmbedding<VERTEX>(e) ;
        setDartEmbedding<VERTEX>(phi_1(e), vEmb1);
        setDartEmbedding<VERTEX>(phi_1(ee), vEmb1);
        setDartEmbedding<VERTEX>(phi_1(d), vEmb2);
        setDartEmbedding<VERTEX>(phi_1(dd), vEmb2);
    }

    if(isOrbitEmbedded<EDGE>())
    {
        Algo::Topo::initOrbitEmbeddingOnNewCell<EDGE>(*this,phi_1(d)) ;
    }

    if(isOrbitEmbedded<FACE2>())
    {
        copyDartEmbedding<FACE2>(phi_1(d), d) ;
        Algo::Topo::setOrbitEmbeddingOnNewCell<FACE2>(*this, e) ;
        Algo::Topo::copyCellAttributes<FACE2>(*this, e, d) ;

        copyDartEmbedding<FACE2>(phi_1(dd), dd) ;
        Algo::Topo::setOrbitEmbeddingOnNewCell<FACE2>(*this, ee) ;
        Algo::Topo::copyCellAttributes<FACE2>(*this, ee, dd) ;
    }

    if(isOrbitEmbedded<FACE>())
    {
        unsigned int fEmb = getEmbedding<FACE>(d) ;
        setDartEmbedding<FACE>(phi_1(d), fEmb) ;
        setDartEmbedding<FACE>(phi_1(ee), fEmb) ;
        Algo::Topo::setOrbitEmbeddingOnNewCell<FACE>(*this, e);
        Algo::Topo::copyCellAttributes<FACE>(*this, e, d);

    }

    if(isOrbitEmbedded<VOLUME>())
    {
        unsigned int vEmb1 = getEmbedding<VOLUME>(d) ;
        setDartEmbedding<VOLUME>(phi_1(d),  vEmb1);
        setDartEmbedding<VOLUME>(phi_1(e),  vEmb1);

        unsigned int vEmb2 = getEmbedding<VOLUME>(dd) ;
        setDartEmbedding<VOLUME>(phi_1(dd),  vEmb2);
        setDartEmbedding<VOLUME>(phi_1(ee),  vEmb2);
    }
}

Dart ImplicitHierarchicalMap3::cutEdge(Dart d)
{
    Dart nd = Map3::cutEdge(d);

    if(isOrbitEmbedded<VERTEX>())
    {
        Algo::Topo::initOrbitEmbeddingOnNewCell<VERTEX>(*this, nd) ;
    }

    if(isOrbitEmbedded<EDGE>())
    {
        // embed the new darts created in the cut edge
        Algo::Topo::setOrbitEmbedding<EDGE>(*this, d, getEmbedding<EDGE>(d)) ;
        // embed a new cell for the new edge and copy the attributes' line (c) Lionel
        Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, nd) ;
        Algo::Topo::copyCellAttributes<EDGE>(*this, nd, d) ;
    }

    if(isOrbitEmbedded<FACE2>())
    {
        Dart f = d;
        do
        {
            Dart f1 = phi1(f) ;

            copyDartEmbedding<FACE2>(f1, f);
            Dart e = phi3(f1);
            copyDartEmbedding<FACE2>(phi1(e), e);
            f = alpha2(f);
        } while(f != d);
    }

    if(isOrbitEmbedded<FACE>())
    {
        Dart f = d;
        do
        {
            unsigned int fEmb = getEmbedding<FACE>(f) ;
            setDartEmbedding<FACE>(phi1(f), fEmb);
            setDartEmbedding<FACE>(phi3(f), fEmb);
            f = alpha2(f);
        } while(f != d);
    }

    if(isOrbitEmbedded<VOLUME>())
    {
        Dart f = d;
        do
        {
            unsigned int vEmb = getEmbedding<VOLUME>(f) ;
            setDartEmbedding<VOLUME>(phi1(f), vEmb);
            setDartEmbedding<VOLUME>(phi2(f), vEmb);
            f = alpha2(f);
        } while(f != d);
    }

    return nd ;
}

} // namespace IHM

} // namespace Volume

} // namespace Algo

} // namespace CGoGN







//bool ImplicitHierarchicalMap3::faceIsSubdividedOnce(Dart d)
//{
//	assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
//	unsigned int fLevel = faceLevel(d) ;
//	if(fLevel < m_curLevel)		// a face whose level in the current level map is lower than
//		return false ;			// the current level can not be subdivided to higher levels
//
//	unsigned int degree = 0 ;
//	bool subd = false ;
//	bool subdOnce = true ;
//	Dart fit = d ;
//	do
//	{
//		++m_curLevel ;
//		if(m_dartLevel[phi1(fit)] == m_curLevel && m_edgeId[phi1(fit)] != m_edgeId[fit])
//		{
//			subd = true ;
//			++m_curLevel ;
//			if(m_dartLevel[phi1(fit)] == m_curLevel && m_edgeId[phi1(fit)] != m_edgeId[fit])
//				subdOnce = false ;
//			--m_curLevel ;
//		}
//		--m_curLevel ;
//		++degree ;
//		fit = phi1(fit) ;
//
//	} while(subd && subdOnce && fit != d) ;
//
//	if(degree == 3 && subd)
//	{
//		++m_curLevel ;
//		Dart cf = phi2(phi1(d)) ;
//		++m_curLevel ;
//		if(m_dartLevel[phi1(cf)] == m_curLevel && m_edgeId[phi1(cf)] != m_edgeId[cf])
//			subdOnce = false ;
//		--m_curLevel ;
//		--m_curLevel ;
//	}
//
//	return subd && subdOnce ;
//}




//Dart ImplicitHierarchicalMap3::cutEdge(Dart d)
//{
//        Dart resV = EmbeddedMap3::cutEdge(d);
//
//        unsigned int eId = getEdgeId(d);
//        Dart dit = d;
//        do
//        {
//        	//EdgeId
//        	m_edgeId[phi1(dit)] = eId;
//        	m_edgeId[phi3(dit)] = eId;
//
//        	//FaceId
//        	unsigned int fId = getFaceId(dit);
//        	m_faceId[phi1(dit)] = fId;
//        	m_edgeId[phi3(dit)] = fId;
//
//            dit = alpha2(dit);
//        }
//        while(dit != d);
//
//        return resV;
//}
//
//bool ImplicitHierarchicalMap3::uncutEdge(Dart d)
//{
//       return EmbeddedMap3::uncutEdge(d);
//}
//
//void ImplicitHierarchicalMap3::splitFace(Dart d, Dart e)
//{
//        EmbeddedMap3::splitFace(d,e);
//
//        unsigned int eId = getNewEdgeId();
//        unsigned int fId = getFaceId(d);
//
//        Dart ne = phi_1(d);
//        Dart ne3 = phi3(ne);
//
//        m_edgeId[ne] = eId;
//        m_edgeId[phi2(ne)] = eId;
//        m_edgeId[ne3] = eId;
//        m_edgeId[phi2(ne3)] = eId;
//
//        m_faceId[ne] = fId;
//        m_faceId[phi2(ne)] = fId;
//        m_faceId[ne3] = fId;
//        m_faceId[phi2(ne3)] = fId;
//}
//
//void ImplicitHierarchicalMap3::sewVolumes(Dart d, Dart e, bool withBoundary)
//{
//        EmbeddedMap3::sewVolumes(d,e);
//
//        unsigned int fId;
//
//        if(m_faceId[d] < m_faceId[phi3(d)])
//        	fId = m_faceId[d] ;
//        else
//        	fId = m_edgeId[phi3(d)];
//
//        Dart dit = d;
//        do
//        {
//                //EdgeId
////                if(m_edgeId[dit] < m_edgeId[phi3(dit)])
////                	m_edgeId[phi3(dit)] = m_edgeId[dit] ;
////                else
////                	m_edgeId[dit] = m_edgeId[phi3(dit)];
//
//                //FaceId
//                m_faceId[dit] = fId;
//                m_faceId[phi3(dit)] = fId;
//
//                dit = phi1(dit);
//        }
//        while(dit != d);
//}
//
//void ImplicitHierarchicalMap3::splitVolume(std::vector<Dart>& vd)
//{
//        EmbeddedMap3::splitVolume(vd);
//
//        unsigned int fId = getNewFaceId();
//
//        for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
//        {
//                Dart dit = *it;
//
//                //Edge Id
//                m_edgeId[phi2(dit)] = m_edgeId[dit];
//
//                //Face Id
//                m_faceId[phi2(dit)] = fId;
//        }
//}
//
//Dart ImplicitHierarchicalMap3::beginSplittingPath(Dart d, DartMarker& m)
//{
//	Dart dres = NIL;
//	Dart dit = d;
//	bool found = false;
//
//	// Recherche d'un brin de depart du chemin d'arete
//	do
//	{
//		Dart eit = phi1(dit);
//
//		if(!m.isMarked(eit) && getDartLevel(eit) == getCurrentLevel())
//		{
//			found = true;
//			dres = eit;
//		}
//
//		dit = phi2(phi_1(dit));
//	}
//	while(!found && dit != d);
//
//	return dres;
//}
//
//void ImplicitHierarchicalMap3::constructSplittingPath(Dart d, std::vector<Dart>& v, DartMarker& m)
//{
//
//	//Construction du chemin d'arete
//	Dart cit = d;
//
//	v.push_back(cit);
//	m.markOrbit<EDGE>(cit);
//
//	do
//	{
//
//		if(std::min(getDartLevel(phi1(cit)),getDartLevel(phi2(phi1(cit))))  == getDartLevel(d))
//		{
//			if(m.isMarked(phi1(cit)))
//			{
//				cit = phi1(phi2(phi1(cit)));
//				std::cout << "1_1" << std::endl;
//			}
//		}
//		else if(std::min(getDartLevel(phi1(cit)),getDartLevel(phi2(phi1(cit)))) < getDartLevel(d))
//		{
//			cit = phi1(phi2(phi1(cit)));
//			std::cout << "2" << std::endl;
//		}
//		else
//			cit = phi1(cit);
//
//		v.push_back(cit);
//		m.markOrbit<EDGE>(cit);
//
//
//	}
//	while(cit != d);
//
////	do
////	{
////		v.push_back(cit);
////		m.markOrbit<EDGE>(cit);
////
////		cit = phi1(cit);
////
////		//std::cout << "cit = " << cit << std::endl;
////
////		if(std::min(getDartLevel(cit), getDartLevel(phi2(cit))) == getDartLevel(d))
////		{
////			if(m.isMarked(cit))
////			{
////				cit = phi1(phi2(cit));
////				//std::cout << "1_1" << std::endl;
////			}
////		}
////		else if(std::min(getDartLevel(cit),getDartLevel(phi2(cit))) < getDartLevel(d))
////		{
////			cit = phi1(phi2(cit));
////			//std::cout << "2" << std::endl;
////		}
////
////	}while(cit != d);
//
//}

