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

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace IHM
{

/***************************************************
 *  	   ATTRIBUTES MANAGEMENT    	           	 *
 ***************************************************/

template <typename T, unsigned int ORBIT>
AttributeHandler_IHM<T, ORBIT> ImplicitHierarchicalMap3::addAttribute(const std::string& nameAttr)
{
	bool addNextLevelCell = false ;
	if(!isOrbitEmbedded<ORBIT>())
		addNextLevelCell = true ;

	AttributeHandler<T, ORBIT> h = Map3::addAttribute<T, ORBIT>(nameAttr) ;

	if(addNextLevelCell)
	{
		AttributeContainer& cellCont = m_attribs[ORBIT] ;
		AttributeMultiVector<unsigned int>* amv = cellCont.addAttribute<unsigned int>("nextLevelCell") ;
		m_nextLevelCell[ORBIT] = amv ;
		for(unsigned int i = cellCont.begin(); i < cellCont.end(); cellCont.next(i))
			amv->operator[](i) = EMBNULL ;
	}

	return AttributeHandler_IHM<T, ORBIT>(this, h.getDataVector()) ;
}

template <typename T, unsigned int ORBIT>
AttributeHandler_IHM<T, ORBIT> ImplicitHierarchicalMap3::getAttribute(const std::string& nameAttr)
{
	AttributeHandler<T, ORBIT> h = Map3::getAttribute<T, ORBIT>(nameAttr) ;
	return AttributeHandler_IHM<T, ORBIT>(this, h.getDataVector()) ;
}


inline void ImplicitHierarchicalMap3::update_topo_shortcuts()
{
	Map3::update_topo_shortcuts();
	m_dartLevel = Map3::getAttribute<unsigned int, DART>("dartLevel") ;
	m_faceId = Map3::getAttribute<unsigned int, DART>("faceId") ;
	m_edgeId = Map3::getAttribute<unsigned int, DART>("edgeId") ;

	//AttributeContainer& cont = m_attribs[DART] ;
	//m_nextLevelCell = cont.getDataVector<unsigned int>(cont.getAttributeIndex("nextLevelCell")) ;
}


/***************************************************
 *     		 	    MAP TRAVERSAL         		   *
 ***************************************************/

inline Dart ImplicitHierarchicalMap3::newDart()
{
	Dart d = Map3::newDart() ;
	m_dartLevel[d] = m_curLevel ;
	if(m_curLevel > m_maxLevel)			// update max level
		m_maxLevel = m_curLevel ;		// if needed
	return d ;
}

inline Dart ImplicitHierarchicalMap3::phi1(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
	bool finished = false ;

	unsigned int edgeId = m_edgeId[d] ;
	Dart it = d ;

	do
	{

		it = Map3::phi1(it) ;
		if(m_dartLevel[it] <= m_curLevel)
			finished = true ;
		else
		{
			while(m_edgeId[it] != edgeId)
				it = Map3::phi1(phi2bis(it));

		}
	} while(!finished) ;

	return it ;
}

inline Dart ImplicitHierarchicalMap3::phi_1(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
	bool finished = false ;

	Dart it = Map3::phi_1(d) ;
	unsigned int edgeId = m_edgeId[it] ;
	do
	{
		if(m_dartLevel[it] <= m_curLevel)
			finished = true ;
		else
		{
			it = Map3::phi_1(it) ;
			while(m_edgeId[it] != edgeId)
				it = Map3::phi_1(phi2bis(it));
		}
	} while(!finished) ;

	return it ;
}

inline Dart ImplicitHierarchicalMap3::phi2bis(Dart d)
{
	unsigned int faceId = m_faceId[d];
	Dart it = d;

	it = Map3::phi2(it) ;

	/* du cote des volumes non subdivise (subdiv adapt) */
	if(m_faceId[it] == faceId)
		return it;
	else
	{
		do
		{
			it = Map3::phi2(Map3::phi3(it));
		}
		while(m_faceId[it] != faceId);

		return it;
	}
}

inline Dart ImplicitHierarchicalMap3::phi2(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

	return Map3::phi2(Map3::phi_1(phi1(d))) ;
}

inline Dart ImplicitHierarchicalMap3::phi3(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

	if(Map3::phi3(d) == d)
		return d;

	return Map3::phi3(Map3::phi_1(phi1(d)));
}

inline Dart ImplicitHierarchicalMap3::alpha0(Dart d)
{
	return phi3(d) ;
}

inline Dart ImplicitHierarchicalMap3::alpha1(Dart d)
{
	//assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

	return phi3(phi_1(d)) ;
}

inline Dart ImplicitHierarchicalMap3::alpha2(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

	return phi3(phi2(d));
}

inline Dart ImplicitHierarchicalMap3::alpha_2(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

	return phi2(phi3(d));
}

inline Dart ImplicitHierarchicalMap3::begin() const
{
	Dart d = Map3::begin() ;
	while(m_dartLevel[d] > m_curLevel)
		Map3::next(d) ;
	return d ;
}

inline Dart ImplicitHierarchicalMap3::end() const
{
	return Map3::end() ;
}

inline void ImplicitHierarchicalMap3::next(Dart& d) const
{
	do
	{
		Map3::next(d) ;
	} while(d != Map3::end() && m_dartLevel[d] > m_curLevel) ;
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread)
{
	DartMarkerStore mv(*this, thread);	// Lock a marker
	bool found = false;					// Last functor return value

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; !found && i < darts.size(); ++i)
	{
		// add phi21 and phi23 successor if they are not marked yet
		Dart d2 = phi2(darts[i]);
		Dart d21 = phi1(d2); // turn in volume
		Dart d23 = phi3(d2); // change volume

		if(!mv.isMarked(d21))
		{
			darts.push_back(d21);
			mv.mark(d21);
		}
		if(!mv.isMarked(d23))
		{
			darts.push_back(d23);
			mv.mark(d23);
		}

		found = f(darts[i]);
	}
	return found;
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int thread)
{
	Dart dNext = d;
	do {
		if (foreach_dart_of_edge2(dNext, f, thread))
			return true;
		dNext = alpha2(dNext);
	} while (dNext != d);
	return false;
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_oriented_face(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	Dart dNext = d ;
	do
	{
		if (f(dNext))
			return true ;
		dNext = phi1(dNext) ;
	} while (dNext != d) ;
	return false ;
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_face(Dart d, FunctorType& f, unsigned int thread)
{
	return foreach_dart_of_oriented_face(d, f, thread) || foreach_dart_of_oriented_face(phi3(d), f, thread);
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_oriented_volume(Dart d, FunctorType& f, unsigned int thread)
{
	DartMarkerStore mark(*this, thread);	// Lock a marker
	bool found = false;				// Last functor return value

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d

	// For every face added to the list
	for(unsigned int i = 0; !found && i < visitedFaces.size(); ++i)
	{
		if (!mark.isMarked(visitedFaces[i]))	// Face has not been visited yet
		{
			// Apply functor to the darts of the face
			found = foreach_dart_of_oriented_face(visitedFaces[i], f);

			// If functor returns false then mark visited darts (current face)
			// and add non visited adjacent faces to the list of face
			if (!found)
			{
				Dart e = visitedFaces[i] ;
				do
				{
					mark.mark(e);				// Mark
					Dart adj = phi2(e);			// Get adjacent face
					if (!mark.isMarked(adj))
						visitedFaces.push_back(adj);	// Add it
					e = phi1(e);
				} while(e != visitedFaces[i]);
			}
		}
	}
	return found;
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_volume(Dart d, FunctorType& f, unsigned int thread)
{
	return foreach_dart_of_oriented_volume(d, f, thread) ;
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread)
{
	DartMarkerStore mark(*this, thread);	// Lock a marker
	bool found = false;				// Last functor return value

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d

	// For every face added to the list
	for(unsigned int i = 0; !found && i < visitedFaces.size(); ++i)
	{
		if (!mark.isMarked(visitedFaces[i]))	// Face has not been visited yet
		{
			// Apply functor to the darts of the face
			found = foreach_dart_of_face(visitedFaces[i], f);

			// If functor returns false then mark visited darts (current face)
			// and add non visited adjacent faces to the list of face
			if (!found)
			{
				Dart e = visitedFaces[i] ;
				do
				{
					mark.mark(e);				// Mark
					Dart adj = phi2(e);			// Get adjacent face
					if (!mark.isMarked(adj))
						visitedFaces.push_back(adj);	// Add it
					e = phi1(e);
				} while(e != visitedFaces[i]);
			}
		}
	}
	return found;


//	//return foreach_dart_of_oriented_volume(d, f) ;
//	DartMarkerStore mv(*this,thread);	// Lock a marker
//	bool found = false;					// Last functor return value
//
//	std::vector<Dart> darts;	// Darts that are traversed
//	darts.reserve(1024);
//	darts.push_back(d);			// Start with the dart d
//	mv.mark(d);
//
//	for(unsigned int i = 0; !found && i < darts.size(); ++i)
//	{
//		// add all successors if they are not marked yet
//		Dart d2 = phi1(darts[i]); // turn in face
//		Dart d3 = phi2(darts[i]); // change face
//		Dart d4 = phi3(darts[i]); // change volume
//
//		if (!mv.isMarked(d2))
//		{
//			darts.push_back(d2);
//			mv.mark(d2);
//		}
//		if (!mv.isMarked(d3))
//		{
//			darts.push_back(d2);
//			mv.mark(d2);
//		}
//		if (!mv.isMarked(d4))
//		{
//			darts.push_back(d4);
//			mv.mark(d4);
//		}
//
//		found = f(darts[i]);
//	}
//	return found;
}


inline bool ImplicitHierarchicalMap3::foreach_dart_of_vertex2(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	Dart dNext = d;
	do
	{
		if (f(dNext))
			return true;
		dNext = phi2(phi_1(dNext));
 	} while (dNext != d);
 	return false;
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_edge2(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	if (f(d))
		return true;
	return f(phi2(d));
}

inline bool ImplicitHierarchicalMap3::foreach_dart_of_face2(Dart d, FunctorType& f, unsigned int thread)
{
	return foreach_dart_of_oriented_face(d,f,thread);
}

/***************************************************
 *              LEVELS MANAGEMENT                  *
 ***************************************************/

inline void ImplicitHierarchicalMap3::incCurrentLevel()
{
	if(m_curLevel < m_maxLevel)
		++m_curLevel ;
	else
		CGoGNout << "incCurrentLevel : already at maximum resolution level" << CGoGNendl ;
}

inline void ImplicitHierarchicalMap3::decCurrentLevel()
{
	if(m_curLevel > 0)
		--m_curLevel ;
	else
		CGoGNout << "decCurrentLevel : already at minimum resolution level" << CGoGNendl ;
}



inline unsigned int ImplicitHierarchicalMap3::getCurrentLevel()
{
	return m_curLevel ;
}

inline void ImplicitHierarchicalMap3::setCurrentLevel(unsigned int l)
{
	m_curLevel = l ;
}

inline unsigned int ImplicitHierarchicalMap3::getMaxLevel()
{
	return m_maxLevel ;
}

inline unsigned int ImplicitHierarchicalMap3::getDartLevel(Dart d)
{
	return m_dartLevel[d] ;
}

inline void ImplicitHierarchicalMap3::setDartLevel(Dart d, unsigned int l)
{
	m_dartLevel[d] = l ;
}

/***************************************************
 *             EDGE ID MANAGEMENT                  *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap3::getNewEdgeId()
{
	return m_edgeIdCount++ ;
}


inline unsigned int ImplicitHierarchicalMap3::getEdgeId(Dart d)
{
	return m_edgeId[d] ;
}

inline void ImplicitHierarchicalMap3::setEdgeId(Dart d, unsigned int i, unsigned int orbit)
{
	if(orbit == EDGE)
	{
		Dart e = d;

		do
		{
			m_edgeId[e] = i;
			m_edgeId[Map3::phi2(e)] = i;

			e = Map3::alpha2(e);
		} while(e != d);
	}
	else if(orbit == DART)
	{
		m_edgeId[d] = i;
	}
}

/***************************************************
 *             FACE ID MANAGEMENT                  *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap3::getNewFaceId()
{
	return m_faceIdCount++;
}

inline unsigned int ImplicitHierarchicalMap3::getFaceId(Dart d)
{
	return m_faceId[d] ;
}

inline void ImplicitHierarchicalMap3::setFaceId(unsigned int orbit, Dart d)
{
	//Mise a jour de l'id de face pour les brins autour d'une arete
	if(orbit == EDGE)
	{
		Dart e = d;

		do
		{
			m_faceId[Map3::phi1(e)] = m_faceId[e];

			e = Map3::alpha2(e);
		}while(e != d);

	}
}

inline void ImplicitHierarchicalMap3::setFaceId(Dart d, unsigned int i, unsigned int orbit)
{

	//Mise a jour de l'id de face pour les brins autour de la face
	if(orbit == FACE)
	{
		Dart e = d;

		do
		{
			m_faceId[e] = i;

			Dart e3 = Map3::phi3(e);
			if(e3 != e)
				m_faceId[e3] = i;

			e = Map3::phi1(e);
		} while(e != d);
	}
	else if(orbit == DART)
	{
		m_faceId[d] = i;

		if(Map3::phi3(d) != d)
			m_faceId[Map3::phi3(d)] = i;
	}
}

/***************************************************
 *               CELLS INFORMATION                 *
 ***************************************************/

//TODO
inline unsigned int ImplicitHierarchicalMap3::vertexInsertionLevel(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
	return m_dartLevel[d] ;
}

inline unsigned int ImplicitHierarchicalMap3::edgeLevel(Dart d)
{
	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

	// the level of an edge is the maximum of the
	// insertion levels of its darts

	unsigned int r = 0;

	Dart e = d;
	do
	{
		unsigned int ld = m_dartLevel[e] ;
		unsigned int ldd = m_dartLevel[phi2(e)] ;
		unsigned int temp = ld < ldd ? ldd : ld;

		r =  r < temp ? temp : r ;

		e = alpha2(e);
	} while(e != d);

	return r;
}

template <unsigned int ORBIT>
inline unsigned int ImplicitHierarchicalMap3::getEmbedding(Dart d)
{
	unsigned int nbSteps = m_curLevel - vertexInsertionLevel(d);
	unsigned int index = EmbeddedMap3::getEmbedding<ORBIT>(d);

	unsigned int step = 0;
	while(step < nbSteps)
	{
		step++;
		unsigned int next = m_nextLevelCell[ORBIT]->operator[](index);
		//index = next;
		if(next != EMBNULL) index = next;
		else break;
	}

	return index;
}

inline bool ImplicitHierarchicalMap3::isWellEmbedded()
{
	//std::cout << "yeahhh ? " << std::endl;
	TraversorV<ImplicitHierarchicalMap3> tv(*this);

	for(Dart dv = tv.begin() ; dv != tv.end() ; dv = tv.next())
	{
		unsigned int curem = this->getEmbedding<VERTEX>(dv);
		//std::cout << "current emb = " << curem << std::endl;

		unsigned int count = 0;
		TraversorDartsOfOrbit<ImplicitHierarchicalMap3, VERTEX> to(*this, dv);
		for(Dart dit = to.begin() ; dit != to.end() ; dit = to.next())
		{
			//std::cout << getDartLevel(dit) << std::endl;

			if(curem != this->getEmbedding<VERTEX>(dit))
			{
				std::cout << "erreur dart #" << dit;
				std::cout << " / curem = " << curem;
				std::cout << " / em = " << this->getEmbedding<VERTEX>(dit);
				std::cout << std::endl;
			}

			std::cout << EmbeddedMap3::getEmbedding<VERTEX>(dit) << "(" << this->getEmbedding<VERTEX>(dit) << ")" << " / ";
			++count;
		}
		std::cout << " / vertex degree = " << count << std::endl;

	}

	return true;
}


/***************************************************
 *               ATTRIBUTE HANDLER                 *
 ***************************************************/

template <typename T, unsigned int ORBIT>
T& AttributeHandler_IHM<T, ORBIT>::operator[](Dart d)
{
	ImplicitHierarchicalMap3* m = reinterpret_cast<ImplicitHierarchicalMap3*>(this->m_map) ;
	assert(m->m_dartLevel[d] <= m->m_curLevel || !"Access to a dart introduced after current level") ;
	assert(m->vertexInsertionLevel(d) <= m->m_curLevel || !"Access to the embedding of a vertex inserted after current level") ;

//	std::cout << std::endl << "vertexInsertionLevel[" << d <<"] = " << m->vertexInsertionLevel(d) << "\t";

	unsigned int nbSteps = m->m_curLevel - m->vertexInsertionLevel(d) ;
	unsigned int index = m->EmbeddedMap3::getEmbedding<ORBIT>(d) ;

//	std::cout << " m->vertexInsertionLevel(d) = " <<  m->vertexInsertionLevel(d) << std::endl;
//	std::cout << "m_curLevel = " << m->m_curLevel << std::endl;
//	std::cout << " nbSteps = " <<  nbSteps << std::endl;
//	std::cout << "index EmbMap3 = " << index << std::endl;

	if(index == EMBNULL)
	{
		index = m->setOrbitEmbeddingOnNewCell<ORBIT>(d) ;
		m->m_nextLevelCell[ORBIT]->operator[](index) = EMBNULL ;
	}

	AttributeContainer& cont = m->getAttributeContainer<ORBIT>() ;
	unsigned int step = 0 ;
	while(step < nbSteps)
	{
		step++ ;
		unsigned int nextIdx = m->m_nextLevelCell[ORBIT]->operator[](index) ;
		if (nextIdx == EMBNULL)
		{
			nextIdx = m->newCell<ORBIT>() ;
			m->copyCell<ORBIT>(nextIdx, index) ;
			m->m_nextLevelCell[ORBIT]->operator[](index) = nextIdx ;
			m->m_nextLevelCell[ORBIT]->operator[](nextIdx) = EMBNULL ;
			cont.refLine(index) ;
		}
		index = nextIdx ;
	}

//	std::cout << "emb = " << index << std::endl;

//	std::cout << "index IHM = " << index << std::endl;
//	if(index != EMBNULL)
//		std::cout << " emb = " << this->m_attrib->operator[](index) << std::endl << std::endl;

	return this->m_attrib->operator[](index);
}

template <typename T, unsigned int ORBIT>
const T& AttributeHandler_IHM<T, ORBIT>::operator[](Dart d) const
{
	ImplicitHierarchicalMap3* m = reinterpret_cast<ImplicitHierarchicalMap3*>(this->m_map) ;
	assert(m->m_dartLevel[d] <= m->m_curLevel || !"Access to a dart introduced after current level") ;
	assert(m->vertexInsertionLevel(d) <= m->m_curLevel || !"Access to the embedding of a vertex inserted after current level") ;

	unsigned int nbSteps = m->m_curLevel - m->vertexInsertionLevel(d) ;
	unsigned int index = m->EmbeddedMap3::getEmbedding<ORBIT>(d) ;

//	std::cout << "(const) m->vertexInsertionLevel(d) = " <<  m->vertexInsertionLevel(d) << std::endl;
//	std::cout << "(const) m_curLevel = " << m->m_curLevel << std::endl;
//	std::cout << "(const) nbSteps = " <<  nbSteps << std::endl;
//	std::cout << "(const) index EmbMap3 = " << index << std::endl;

	unsigned int step = 0 ;
	while(step < nbSteps)
	{
		step++ ;
		unsigned int nextIdx = m->m_nextLevelCell[ORBIT]->operator[](index) ;
		if(nextIdx != EMBNULL) index = nextIdx ;
		else break ;
	}

//	if(index != EMBNULL)
//		std::cout << "(const) emb = " << this->m_attrib->operator[](index) << std::endl << std::endl;

	return this->m_attrib->operator[](index);
}


} //namespace IHM
} // Volume
} //namespace Algo
} //namespace CGoGN
