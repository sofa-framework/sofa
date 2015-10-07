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

ImplicitHierarchicalMap3::ImplicitHierarchicalMap3() :
    ParentMap()
{
    a_volumeLevel = this->template addAttribute< unsigned, VOLUME, MAP, NonVertexAttributeAccessorCPHMap< unsigned, VOLUME > >("volumeLvl");
    a_faceLevel = this->template addAttribute< unsigned, FACE, MAP, NonVertexAttributeAccessorCPHMap< unsigned, FACE > >("faceLvl");
    a_maxVolumeLevel = m_attribs[DART].addAttribute<unsigned int>("maxVolumeLevel") ;
    a_maxFaceLevel = m_attribs[DART].addAttribute<unsigned int>("maxFaceLevel") ;
    m_faceAttributeBrowser = new FaceAttributeBrowser(this);
    m_volumeAttributeBrowser = new VolumeAttributeBrowser(this);
    m_vertexAttributeBrowser = new VertexAttributeBrowser(this);
    m_attribs[VOLUME].setContainerBrowser(m_volumeAttributeBrowser);
    m_attribs[FACE].setContainerBrowser(m_faceAttributeBrowser);
    m_attribs[VERTEX].setContainerBrowser(m_vertexAttributeBrowser);
}

ImplicitHierarchicalMap3::~ImplicitHierarchicalMap3()
{
    delete m_volumeAttributeBrowser;
    delete m_faceAttributeBrowser;
    delete m_vertexAttributeBrowser;
}

void ImplicitHierarchicalMap3::clear(bool removeAttrib)
{
    if (removeAttrib)
    {
        m_attribs[FACE].removeAttribute< unsigned >("faceLvl");
        m_attribs[VOLUME].removeAttribute< unsigned >("volumeLvl");
        m_attribs[DART].removeAttribute< unsigned >("maxVolumeLevel");
        m_attribs[DART].removeAttribute< unsigned >("maxFaceLevel");
    }

    Parent::clear(removeAttrib) ;

    if (removeAttrib)
    {
        a_volumeLevel = this->template addAttribute< unsigned, VOLUME, MAP, NonVertexAttributeAccessorCPHMap< unsigned, VOLUME > >("volumeLvl");
        a_faceLevel = this->template addAttribute< unsigned, FACE, MAP, NonVertexAttributeAccessorCPHMap< unsigned, FACE > >("faceLvl");
        a_maxVolumeLevel = m_attribs[DART].addAttribute<unsigned int>("maxVolumeLevel") ;
        a_maxFaceLevel = m_attribs[DART].addAttribute<unsigned int>("maxFaceLevel") ;
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
    const Dart old =phi1(phi1(d));
    assert(getDartLevel(old) <= getCurrentLevel() -1u);
    Dart d3 = phi1(phi3(d));
    Dart res = NIL;
    Dart vit = d ;
    do
    {
        if(res == NIL && phi1(phi1(d)) != d)
            res = phi1(d) ;

        Dart f = phi_1(phi2(vit)) ;
        phi1sew(vit, f) ;

        vit = phi2(phi_1(vit)) ;
    } while(vit != d) ;
    Map1::deleteCycle(d) ;

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

    {
        const unsigned int currLVL = getCurrentLevel();
        unsigned goodEmb = ParentMap::getEmbedding< FACE >(old);

        setCurrentLevel(getMaxLevel());
        TraversorDartsOfOrbit< MAP, FACE> traDoO(*this, old);
        for (Dart doo = traDoO.begin() ; doo != traDoO.end() ; doo = traDoO.next())
        {
            const unsigned dl = getDartLevel(doo);
            if ( dl > currLVL -1u)
            {
                setCurrentLevel(dl);
                this->setDartEmbedding<FACE>(doo, goodEmb) ;
                setCurrentLevel(getMaxLevel());
            }
        }
        setCurrentLevel(currLVL);
    }
}

Dart ImplicitHierarchicalMap3::deleteVertex(Dart d)
{
    const VolumeCell res = VolumeCell(Map3::deleteVertex(d)); // the new volume of lvl currLVL -1
    if (res != NIL)
    {
        const unsigned int currLVL = getCurrentLevel();
        setCurrentLevel(getMaxLevel());
        unsigned goodEmb = std::numeric_limits<unsigned >::max();
        TraversorDartsOfOrbit< MAP, VOLUME> traDoO(*this, res);

        for (Dart doo = traDoO.begin() ; doo != traDoO.end() ; doo = traDoO.next())
        {
            if (getDartLevel(doo) == currLVL -1u)
            {
                goodEmb = ParentMap::getEmbedding< VOLUME >(doo);
                break;
            }
        }

        for (Dart doo = traDoO.begin() ; doo != traDoO.end() ; doo = traDoO.next())
        {
            const unsigned dl = getDartLevel(doo);
            if ( dl > currLVL -1u)
            {
                setCurrentLevel(dl);
                this->setDartEmbedding<VOLUME>(doo, goodEmb) ;
                setCurrentLevel(getMaxLevel());
            }
        }

        setCurrentLevel(currLVL);
    }
    return res;
}

void ImplicitHierarchicalMap3::initEdgeId()
{
    m_edgeIdCount = 0u;
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

void ImplicitHierarchicalMap3::initMaxCellLevel()
{
    for(Dart d = this->beginMaxLvl(), end = this->endMaxLvl() ; d != end ; this->nextMaxLvl(d))
    {
        setMaxFaceLevel(d, 0u);
        setMaxVolumeLevel(d, 0u);
    }
}

void ImplicitHierarchicalMap3::updateMaxLevelVolume(VolumeCell w)
{
    const unsigned currLVL = getCurrentLevel();
    setCurrentLevel(getMaxLevel());
    const unsigned maxVolumeLevel = volumeLevel(w);
    setCurrentLevel(currLVL);
    TraversorDartsOfOrbit< MAP, VOLUME > traDoW(*this, w);
    for (Dart doo = traDoW.begin(), end = traDoW.end(); doo != end ; doo = traDoW.next())
    {
        setMaxVolumeLevel(doo, maxVolumeLevel);
    }
    setCurrentLevel(currLVL);
}

void ImplicitHierarchicalMap3::updateMaxLevelFace(FaceCell f)
{
    const unsigned currLVL = getCurrentLevel();
    setCurrentLevel(getMaxLevel());
    const unsigned maxFaceLevel = faceLevel(f);
    setCurrentLevel(currLVL);
    TraversorDartsOfOrbit< MAP, FACE > traDoF(*this, f);
    for (Dart doo = traDoF.begin(), end = traDoF.end(); doo != end ; doo = traDoF.next())
    {
        setMaxFaceLevel(doo, maxFaceLevel);
    }
    setCurrentLevel(currLVL);
}

void ImplicitHierarchicalMap3::initFaceId()
{
    m_faceIdCount = 0u;
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

unsigned int ImplicitHierarchicalMap3::faceLevel(Dart d) const
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    return a_faceLevel[FaceCell(d)];
}

unsigned int ImplicitHierarchicalMap3::volumeLevel(Dart d) const
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;
    return a_volumeLevel[VolumeCell(d)];
}


Dart ImplicitHierarchicalMap3::edgeNewestDart(Dart d) const
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;


    Dart newest = d ;
    unsigned int l_new = getDartLevel(newest) ;
    if (l_new == getCurrentLevel())
    {
        return d;
    }

    const Dart phi2d = phi2(d);
    if (getDartLevel(phi2d) > l_new)
    {
        newest = phi2d;
    }

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
    if (l_new == getCurrentLevel())
    {
        return d;
    }

    do
    {
        const unsigned int l = getDartLevel(it) ;
        if (l == getCurrentLevel())
        {
            return it;
        } else
        {
            if (l > l_new)
            {
                newest = it ;
                l_new = l ;
            }
        }
        it = phi1(it) ;
    } while(it != d) ;
    return newest;
}

Dart ImplicitHierarchicalMap3::dartOfMaxFaceLevel(FaceCell d) const {
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;
    const unsigned int maxFaceLvl = a_maxFaceLevel->operator [](dartIndex(d));
    Dart it = d ;
    Dart newest = it ;
    unsigned int l_new = getDartLevel(newest) ;
    if (l_new >= maxFaceLvl)
    {
        return d;
    }

    do
    {
        const unsigned int l = getDartLevel(it) ;
        if (l >= maxFaceLvl)
        {
            return it;
        } else
        {
            if(l > l_new )
            {
                newest = it ;
                l_new = l ;
            }
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

    Traversor3WF<ImplicitHierarchicalMap3> trav3WF(*this, oldest, true);
	for(Dart dit = trav3WF.begin() ; dit != trav3WF.end() ; dit = trav3WF.next())
	{
        const Dart old = faceOldestDart(dit);
        const unsigned int l = getDartLevel(old);
		if(l < l_old)
		{
			oldest = old;
			l_old = l;
            if (l== 0u)
            {
                break;
            }
		}
	}

    assert(!oldest.isNil());
    return oldest;
}

Dart ImplicitHierarchicalMap3::volumeNewestDart(Dart d) const
{
    const unsigned dld = getDartLevel(d);
    const unsigned curr = getCurrentLevel();
    assert(dld <= curr || !"Access to a dart introduced after current level") ;

    Dart newest = d;
    unsigned int l_new = getDartLevel(newest);
    if (l_new == getCurrentLevel())
    {
        return d;
    }

    Traversor3WF<ImplicitHierarchicalMap3> trav3WF(*this, newest, true);
    for(Dart dit = trav3WF.begin(), end = trav3WF.end() ; (dit != end) && (l_new < getCurrentLevel()); dit = trav3WF.next())
    {
        const Dart newDart = faceNewestDart(dit);
        const unsigned int l = getDartLevel(newDart);
        if (l == getCurrentLevel())
        {
            return newDart;
        } else
        {
            if( (l > l_new) /*|| (l == l_new && ParentMap::getEmbedding<VOLUME>(newDart) != EMBNULL)*/ )
            {
                newest = newDart;
                l_new = l;
            }
        }
    }

    return newest;
}


Dart ImplicitHierarchicalMap3::dartOfMaxVolumeLevel(VolumeCell d) const
{
    const unsigned dld = getDartLevel(d);
    const unsigned curr = getCurrentLevel();
    assert(dld <= curr || !"Access to a dart introduced after current level") ;

    const unsigned int maxVolumeLvl = a_maxVolumeLevel->operator [](dartIndex(d));
    Dart newest = d;
    unsigned int l_new = getDartLevel(newest);
    if (l_new >= maxVolumeLvl)
    {
        return d;
    }

    TraversorDartsOfOrbit< ImplicitHierarchicalMap3, VOLUME > traDoW(*this, d);
    for (Dart dit = traDoW.begin(), end = traDoW.end() ; dit != end; dit = traDoW.next())
    {
       const unsigned int dl = getDartLevel(dit);
       if (dl >= maxVolumeLvl)
       {
           return dit;
       } else
       {
           if( (dl > l_new) )
           {
               newest = dit;
               l_new = dl;
           }
       }
    }
    return newest;
}

bool ImplicitHierarchicalMap3::edgeIsSubdivided(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;

	//Dart d2 = phi2(d) ;
    const Dart d1 = phi1(d) ;
    setCurrentLevel(getCurrentLevel() + 1) ;
	//Dart d2_l = phi2(d) ;
    const Dart d1_l = phi1(d) ;
    setCurrentLevel(getCurrentLevel() - 1) ;

    return (d1 != d1_l) ;

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
        setCurrentLevel(getCurrentLevel() + 1);

		if(vertexDegree(phi1(d)) == 2)
		{
			degree2 = true ;
			if(edgeIsSubdivided(d))
				subdOnce = false ;
		}
        setCurrentLevel(getCurrentLevel() - 1);
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
    setCurrentLevel(getCurrentLevel() + 1);
    //    if (fLevel < faceLevel(d))
    if( getDartLevel(phi1(d)) == getCurrentLevel() && getEdgeId(phi1(d)) != getEdgeId(d) )
    {
        subd = true ;
    }
    setCurrentLevel(getCurrentLevel() - 1);
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
        const Dart d3 = phi3(d);

		//tester si le volume voisin est subdivise
        if(!isBoundaryMarkedCurrent(d3) )
        {
            if (volumeIsSubdivided(d3))
            {
                return false;
                //            subdNeighborhood = true;
            }
        }


        setCurrentLevel(getCurrentLevel() + 1);
		//tester si la face subdivise a des faces subdivise
        const Dart phi1d = phi1(d);
        Dart cf = phi1d;

		do
		{
			if(faceIsSubdivided(cf))
            {
                setCurrentLevel(getCurrentLevel() - 1);
                return false;
//                subdOnce = false;
            }

			cf = phi2(phi1(cf));
		}
        while(subdOnce && cf != phi1d);

        setCurrentLevel(getCurrentLevel() - 1);
    } else {
//        std::cerr << "face not subdivided : " << d << std::endl;
    }

	return subd && !subdNeighborhood && subdOnce;
}



bool ImplicitHierarchicalMap3::volumeIsSubdivided(Dart d)
{
    assert(getDartLevel(d) <= getCurrentLevel() || !"Access to a dart introduced after current level") ;
    const unsigned int vLevel = volumeLevel(d);
    if(vLevel < getCurrentLevel())
        return false;

    bool subd = false;

    setCurrentLevel(getCurrentLevel() + 1);
    if(volumeLevel(d) > vLevel)
        subd = true;
    setCurrentLevel(getCurrentLevel() - 1);
    return subd;
}


bool ImplicitHierarchicalMap3::volumeIsSubdividedOnce(Dart d)
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    const unsigned int vLevel = volumeLevel(d);
    if(vLevel < getCurrentLevel())
        return false;

    bool subd = false ;
    bool subdOnce = true ;

    setCurrentLevel(getCurrentLevel() + 1);
    if(volumeLevel(d) > vLevel)
    {
        subd = true;
        setCurrentLevel(getCurrentLevel() + 1);
        Dart dcenter = phi_1(phi2(phi1(d)));
        Traversor3VW<ImplicitHierarchicalMap3> trav3(*this, dcenter);
        for(Dart dit = trav3.begin() ; subdOnce && dit != trav3.end() && subdOnce; dit = trav3.next())
        {
            if(volumeLevel(dit) > vLevel+1)
            {
                subdOnce = false;
            }
        }
        setCurrentLevel(getCurrentLevel() - 1);
    }
    setCurrentLevel(getCurrentLevel() - 1);
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
//				setCurrentLevel(getCurrentLevel() + 1);
//
//				if(faceIsSubdividedOnce(e))
//					found = true;
//
//				setCurrentLevel(getCurrentLevel() - 1);
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
    const unsigned curr = getCurrentLevel();
    if (!withBoundary)
    {
        Map3::sewVolumes(d, e, false) ;
        return ;
    }

    Map3::sewVolumes(d, e, withBoundary);

    this->setCurrentLevel(getMaxLevel());
    setFaceId(d, getFaceId(d), FACE);
    {
        Dart it = d ;
        do
        {
            setEdgeId(it, getEdgeId(it), EDGE);
            it = phi1MaxLvl(it) ;
        } while(it != d) ;
    }

    // embed the vertex orbits from the oriented face with dart e
    // with vertex orbits value from oriented face with dart d
    if (isOrbitEmbedded<VERTEX>())
    {
        Dart it = d ;
        do
        {
            this->setCurrentLevel(getDartLevel(it));
            const unsigned emb = ParentMap::getEmbedding<VERTEX>(it);
            TraversorDartsOfOrbit< MAP, VERTEX> traDoO(*this, VertexCell(it));
            for (Dart doo = traDoO.begin() ; doo != traDoO.end() ; doo = traDoO.next())
            {
                this->setDartEmbedding<VERTEX>(doo, emb) ;
            }
            this->setCurrentLevel(curr);
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
            this->setCurrentLevel(getDartLevel(it));
            setDartEmbedding<EDGE>(it, ParentMap::getEmbedding<EDGE>(phi3MaxLvl(it))) ;
            this->setCurrentLevel(curr);
            it = phi1(it) ;
        } while(it != d) ;
    }

    // embed the face orbit from the volume sewn
//    if (isOrbitEmbedded<FACE>())
//    {
        {
            Dart it = d ;
            do
            {
                this->setCurrentLevel(getDartLevel(it));
                setDartEmbedding<FACE>(it,  ParentMap::getEmbedding<FACE>(phi3MaxLvl(it)));
                this->setCurrentLevel(getMaxLevel());
                it = phi1(it) ;
            } while(it != d) ;
        }
//    }
    this->setCurrentLevel(curr);
}

void ImplicitHierarchicalMap3::splitVolume(std::vector<Dart> &vd)
{
//    std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
//    std::cerr << "splitVolume Called. Current volume Level is : " << this->volumeLevel(vd.front()) << std::endl;
    const unsigned curr = getCurrentLevel();


    std::vector<Dart> phi2vd = vd;
    for(std::vector<Dart>::iterator it = phi2vd.begin() ; it != phi2vd.end() ; ++it)
    {
        *it = phi2MaxLvl(*it);
    }


//    std::transform(phi2vd.begin(), phi2vd.end(), phi2vd.begin(), bl::bind(&ImplicitHierarchicalMap3::phi2MaxLvl,this, bl::_1));
    const unsigned oldVolLvl = volumeLevel(vd.front());

    Map3::splitVolume(vd);

    const unsigned fid = this->getNewFaceId();
    setFaceId(phi2(vd.front()), fid, FACE);


    // follow the edge path a second time to embed the vertex, edge and volume orbits
    this->setCurrentLevel(getMaxLevel());
    for (unsigned i = 0u; i < vd.size(); ++i)
    {
        const Dart ditvd = vd[i];
        const Dart ditphi2vd = phi2vd[i];
        setDartLevel(phi2MaxLvl(ditvd), getDartLevel(ditphi2vd));
        setDartLevel(phi2MaxLvl(ditphi2vd), getDartLevel(ditvd));
        setEdgeId(ditvd, getEdgeId(ditvd), EDGE);
        setMaxVolumeLevel(phi2MaxLvl(ditvd), oldVolLvl);
        setMaxVolumeLevel(phi2MaxLvl(ditphi2vd), oldVolLvl);
    }

    for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
    {
        const Dart dit = *it;
        const Dart dit1 = phi1MaxLvl(dit);
        const Dart dit2 = phi2MaxLvl(dit);
        const Dart dit23 = phi3MaxLvl(dit2);
        setCurrentLevel(getDartLevel(dit));

        // embed the vertex embedded from the origin volume to the new darts
        if(isOrbitEmbedded<VERTEX>())
        {
            setCurrentLevel(getDartLevel(dit23));
            copyDartEmbedding<VERTEX>(dit23, dit);
            setCurrentLevel(getDartLevel(dit2));
            copyDartEmbedding<VERTEX>(dit2, dit1);
        }

        // embed the edge embedded from the origin volume to the new darts
        if(isOrbitEmbedded<EDGE>())
        {
            const unsigned int eEmb = ParentMap::getEmbedding<EDGE>(dit) ;
            assert(eEmb != EMBNULL);
            setCurrentLevel(getDartLevel(dit23));
            setDartEmbedding<EDGE>(dit23, eEmb);
            setCurrentLevel(getDartLevel(dit2));
            setDartEmbedding<EDGE>(dit2, eEmb);
        }

        // embed the volume embedded from the origin volume to the new darts
        if(isOrbitEmbedded<VOLUME>())
        {
            setCurrentLevel(getDartLevel(dit2));
            copyDartEmbedding<VOLUME>(dit2, dit);
        }
    }

    if (isOrbitEmbedded<FACE>()) {
        this->setCurrentLevel(getMaxLevel());
        const unsigned int newFaceEmb = Algo::Topo::initOrbitEmbeddingOnNewCell<FACE>(*this, phi2(vd.front())) ;
        this->setCurrentLevel(getMaxLevel());
        {
            TraversorDartsOfOrbit< MAP, FACE > traDoF(*this, phi2(vd.front()));
            for(Dart fit = traDoF.begin() ; fit != traDoF.end() ; fit = traDoF.next())
            {
                this->setCurrentLevel(getDartLevel(fit));
                setDartEmbedding<FACE>(fit, newFaceEmb);
                this->setCurrentLevel(getMaxLevel());
            }
        }
    }

    setCurrentLevel(curr);
}

void ImplicitHierarchicalMap3::splitFace(Dart d, Dart e)
{
    const Dart dd = phi1(phi3(d));
    const Dart ee = phi1(phi3(e));
    const Dart old = this->faceOldestDart(d) ;

    const unsigned int volEmb = ParentMap::getEmbedding<VOLUME>(volumeNewestDart(d)) ;
    const unsigned int neighVolEmb = getEmbedding<VOLUME>(volumeNewestDart(dd)) ;

    Map3::splitFace(d, e);


    const unsigned int id = this->getNewEdgeId() ;
    this->setEdgeId(this->phi_1MaxLvl(d), id, EDGE) ;		// set the edge id of the inserted edge to the next available id
    unsigned int idface = this->getFaceId(old);
    this->setFaceId(d, idface, FACE) ;
    this->setFaceId(e, idface, FACE) ;
//    idface = this->getNewFaceId();

    if(isOrbitEmbedded<VERTEX>())
    {
        const unsigned int vEmb1 = ParentMap::getEmbedding<VERTEX>(d) ;
        const unsigned int vEmb2 = ParentMap::getEmbedding<VERTEX>(e) ;
        setDartEmbedding<VERTEX>(phi_1(e), vEmb1);
        setDartEmbedding<VERTEX>(phi_1(ee), vEmb1);
        setDartEmbedding<VERTEX>(phi_1(d), vEmb2);
        setDartEmbedding<VERTEX>(phi_1(dd), vEmb2);
    }

    if(isOrbitEmbedded<EDGE>())
    {
        const unsigned int lvl1Edge = Algo::Topo::initOrbitEmbeddingOnNewCell<EDGE>(*this, phi_1(d)) ;
        const unsigned currLVL = this->getCurrentLevel();
        {
            setCurrentLevel(getMaxLevel());
            TraversorDartsOfOrbit< MAP, EDGE > traDoE(*this, phi_1MaxLvl(d));
            for (Dart eit = traDoE.begin(); eit != traDoE.end() ; eit = traDoE.next())
            {
                assert(ParentMap::getEmbedding<EDGE>(eit) == lvl1Edge);
            }
        }
        this->setCurrentLevel(currLVL);
    }

    if(isOrbitEmbedded<FACE>())
    {
        const unsigned int fEmb = ParentMap::getEmbedding<FACE>(d) ;
        assert (fEmb != EMBNULL);
        setDartEmbedding<FACE>(phi_1(d), fEmb) ;
        setDartEmbedding<FACE>(phi_1(e), fEmb) ;
        setDartEmbedding<FACE>(phi_1(dd), fEmb) ;
        setDartEmbedding<FACE>(phi_1(ee), fEmb) ;
    }

    if(isOrbitEmbedded<VOLUME>())
    {
        assert(getDartLevel(phi_1(d)) == getDartLevel(phi_1(e)) && getDartLevel(phi_1(d)) == getDartLevel(phi_1(dd)) &&getDartLevel(phi_1(d)) == getDartLevel(phi_1(ee)) && getDartLevel(phi_1(d)) == getCurrentLevel());
        setDartEmbedding<VOLUME>(phi_1(d),  volEmb);
        setDartEmbedding<VOLUME>(phi_1(e),  volEmb);
        setDartEmbedding<VOLUME>(phi_1(dd),  neighVolEmb);
        setDartEmbedding<VOLUME>(phi_1(ee),  neighVolEmb);
    }
}

Dart ImplicitHierarchicalMap3::cutEdge(Dart d)
{
    const Dart dd = this->phi2(d) ;

    std::vector< Dart > volumesNewestDart;
    {
        Dart dit = d;
        do
        {
            if (!isBoundaryMarkedCurrent(dit))
            {
                volumesNewestDart.push_back(volumeNewestDart(dit));
            }
            dit = alpha2(dit);
        } while(dit != d);
    }

    const Dart nd = Map3::cutEdge(d);

    const unsigned int eId = this->getEdgeId(d) ;
    this->setEdgeId(this->phi1MaxLvl(dd), eId, EDGE) ;
    this->setEdgeId(this->phi1MaxLvl(d), eId, EDGE) ;

    this->setFaceId(EDGE, d) ; //mise a jour de l'id de face sur chaque brin de chaque moitie d'arete
    this->setFaceId(EDGE, dd) ;


    if(isOrbitEmbedded<VERTEX>())
    {
        Algo::Topo::initOrbitEmbeddingOnNewCell<VERTEX>(*this, nd) ;
    }

    if(isOrbitEmbedded<EDGE>())
    {

        const Dart phi1dd = phi1MaxLvl(dd);
        if (getDartLevel(d) < getDartLevel(phi1dd))
        {
            const unsigned int oldEdgeEmb = ParentMap::template getEmbedding<EDGE>(d);
            const unsigned lvl1EdgeEmb = Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, phi1dd) ;
            const unsigned lvl1EdgeEmb2 = Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, phi1(d)) ;
            assert(oldEdgeEmb != lvl1EdgeEmb);
            this->template getAttributeContainer<EDGE>().copyLine(lvl1EdgeEmb,oldEdgeEmb);
            this->template getAttributeContainer<EDGE>().copyLine(lvl1EdgeEmb2,oldEdgeEmb);
        } else
        {
            Algo::Topo::setOrbitEmbedding<EDGE>(*this, d, ParentMap::template getEmbedding<EDGE>(d)) ;
            Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, nd) ;
            Algo::Topo::copyCellAttributes<EDGE>(*this, nd, d) ;
        }


        {
            const unsigned currLVL = getCurrentLevel();
            const unsigned newEdgeEmb = Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, dd) ;
            setCurrentLevel(getMaxLevel());
            TraversorDartsOfOrbit< MAP, EDGE > traDoE(*this, nd);
            for (Dart eit = traDoE.begin(); eit != traDoE.end() ; eit = traDoE.next())
            {
                assert (ParentMap::template getEmbedding<EDGE>(eit) == newEdgeEmb || (getDartLevel(eit) < currLVL));
            }
            setCurrentLevel(currLVL);

        }
        Algo::Topo::copyCellAttributes<EDGE>(*this, dd, d) ;
    }


    if(isOrbitEmbedded<FACE>())
    {
        Dart f = d;
        do
        {
            if (!isBoundaryMarkedCurrent(f))
            {
                setDartEmbedding<FACE>(phi1(f),  ParentMap::getEmbedding<FACE>(f));
                setDartEmbedding<FACE>(phi2(f),  ParentMap::getEmbedding<FACE>(phi2(phi1(f))));

                setMaxFaceLevel(phi1(f), getMaxFaceLevel(f));
                setMaxFaceLevel(phi2(f), getMaxFaceLevel(phi2(phi1(f))));
            }
            f = alpha2(f);
        } while(f != d);
    }

    if(isOrbitEmbedded<VOLUME>())
    {
        Dart w = d;
        unsigned int index = 0u;
        do
        {
            if (!isBoundaryMarkedCurrent(w))
            {
                const unsigned int wEmb = ParentMap::getEmbedding<VOLUME>(volumesNewestDart[index]);
                setDartEmbedding<VOLUME>(phi1(w), wEmb);
                setDartEmbedding<VOLUME>(phi2(w), wEmb);

                setMaxVolumeLevel(phi1(w), getMaxVolumeLevel(w));
                setMaxVolumeLevel(phi2(w), getMaxVolumeLevel(w));
                ++index;
            }
            w = alpha2(w);
        } while(w != d);
    }

    return nd ;
}

bool ImplicitHierarchicalMap3::checkCounters()
{
    bool res = true;

    if (this->isOrbitEmbedded<VERTEX>())
    {
        const unsigned int currLVL = this->getCurrentLevel();
        setCurrentLevel(getMaxLevel());
        CGoGN::TraversorCell<ImplicitHierarchicalMap3, VERTEX, FORCE_DART_MARKING> traV(*this, true);
        AttributeContainer& cont = this->template getAttributeContainer<VERTEX>();
        for (Cell<VERTEX> cit = traV.begin(), end = traV.end(); cit != end ; cit = traV.next())
        {
            const unsigned emb = this->template getEmbedding(cit);
            TraversorDartsOfOrbit<ImplicitHierarchicalMap3, VERTEX > traDoo(*this, cit);
            unsigned int nbDarts = 0u;
            for (Dart dit = traDoo.begin(), dend = traDoo.end() ; dit != dend ; dit = traDoo.next())
            {
                ++nbDarts;
            }
            const unsigned nbRefs = cont.nbRefs(emb);
            if (nbDarts + 1u != nbRefs)
            {
                res = false;
                std::cerr << "checkCounters failed with nbdarts = " << nbDarts << " and nbrefs = " <<  nbRefs << " (cell "<<  cit << ")" << std::endl;
                break;
            }
        }
        setCurrentLevel(currLVL);
    }
    return res;
}



} // namespace IHM

} // namespace Volume

} // namespace Algo

} // namespace CGoGN
