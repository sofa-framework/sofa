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

#include "Topology/generic/dartmarker.h"
//#include "Topology/generic/traversor/traversorFactory.h"

namespace CGoGN
{
/****************************************
 *           BUFFERS MANAGEMENT           *
 ****************************************/

inline std::vector<Dart>* GenericMap::askDartBuffer(unsigned int thread)
{
	if (s_vdartsBuffers[thread].empty())
	{
		std::vector<Dart>* vd = new std::vector<Dart>;
		vd->reserve(128);
		return vd;
	}
	std::vector<Dart>* vd = s_vdartsBuffers[thread].back();
	s_vdartsBuffers[thread].pop_back();
//    std::cerr << "current number of vec in the dart buffer of thread n " << thread << " :: " << s_vdartsBuffers[thread].size() << std::endl;
	return vd;
}

inline void GenericMap::releaseDartBuffer(std::vector<Dart>* vd, unsigned int thread)
{
//	if (vd->capacity()>1024)
//	{
//		std::vector<Dart> v;
//		vd->swap(v);
//		vd->reserve(128);
//	}
	vd->clear();
	s_vdartsBuffers[thread].push_back(vd);
//    std::cerr << "current number of vec in the dart buffer of thread n " << thread << " :: " << s_vdartsBuffers[thread].size() << std::endl;
}


inline std::vector<unsigned int>* GenericMap::askUIntBuffer(unsigned int thread)
{
	if (s_vintsBuffers[thread].empty())
	{
        std::vector<unsigned int>* vui = new std::vector<unsigned int>;
        vui->reserve(128);
		return vui;
	}

	std::vector<unsigned int>* vui = s_vintsBuffers[thread].back();
	s_vintsBuffers[thread].pop_back();
	return vui;
}

inline void GenericMap::releaseUIntBuffer(std::vector<unsigned int>* vui, unsigned int thread)
{
//	if (vui->capacity()>1024)
//	{
//        std::vector<unsigned int> v;
//        vui->swap(v);
//        vui->reserve(128);
//	}
    vui->clear();
	s_vintsBuffers[thread].push_back(vui);
}



/****************************************
 *           DARTS MANAGEMENT           *
 ****************************************/

inline Dart GenericMap::newDart()
{
	unsigned int di = m_attribs[DART].insertLine();		// insert a new dart line
	for(unsigned int i = 0; i < NB_ORBITS; ++i)
	{
		if (m_embeddings[i])							// set all its embeddings
			(*m_embeddings[i])[di] = EMBNULL ;			// to EMBNULL
	}

	return Dart::create(di) ;
}

inline void GenericMap::deleteDartLine(unsigned int index)
{
	m_attribs[DART].removeLine(index) ;	// free the dart line

	for(unsigned int orbit = 0; orbit < NB_ORBITS; ++orbit)
	{
		if (m_embeddings[orbit])									// for each embedded orbit
		{
			unsigned int emb = (*m_embeddings[orbit])[index] ;		// get the embedding of the dart
			if(emb != EMBNULL)
				m_attribs[orbit].unrefLine(emb);					// and unref the corresponding line
		}
	}
}

inline unsigned int GenericMap::copyDartLine(unsigned int index)
{
	unsigned int newindex = m_attribs[DART].insertLine() ;	// create a new dart line
	m_attribs[DART].copyLine(newindex, index) ;				// copy the given dart line
	for(unsigned int orbit = 0; orbit < NB_ORBITS; ++orbit)
	{
		if (m_embeddings[orbit])
		{
			unsigned int emb = (*m_embeddings[orbit])[newindex] ;	// add a ref to the cells pointed
			if(emb != EMBNULL)										// by the new dart line
				m_attribs[orbit].refLine(emb) ;
		}
	}
	return newindex ;
}

//inline bool GenericMap::isDartValid(Dart d)
//{
//	return !d.isNil() && m_attribs[DART].used(dartIndex(d)) ;
//}

/****************************************
 *         EMBEDDING MANAGEMENT         *
 ****************************************/

template <unsigned int ORBIT>
inline bool GenericMap::isOrbitEmbedded() const
{
	return (ORBIT == DART) || (m_embeddings[ORBIT] != NULL) ;
}

inline bool GenericMap::isOrbitEmbedded(unsigned int orbit) const
{
	return (orbit == DART) || (m_embeddings[orbit] != NULL) ;
}

template <unsigned int ORBIT>
inline unsigned int GenericMap::newCell()
{
	assert(isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
	return m_attribs[ORBIT].insertLine();
}

template <unsigned int ORBIT>
inline void GenericMap::copyCell(unsigned int i, unsigned int j)
{
	assert(isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
    assert(i != EMBNULL);
    assert(j != EMBNULL);
	m_attribs[ORBIT].copyLine(i, j) ;
}


template <unsigned int ORBIT>
inline void GenericMap::initCell(unsigned int i)
{
	assert(isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
	m_attribs[ORBIT].initLine(i) ;
}

/****************************************
 *   ATTRIBUTES CONTAINERS MANAGEMENT   *
 ****************************************/

inline unsigned int GenericMap::getNbCells(unsigned int orbit) const
{
	return m_attribs[orbit].size() ;
}

template <unsigned int ORBIT>
inline AttributeContainer& GenericMap::getAttributeContainer()
{
	return m_attribs[ORBIT] ;
}

template <unsigned int ORBIT>
inline const AttributeContainer& GenericMap::getAttributeContainer() const
{
	return m_attribs[ORBIT] ;
}

inline AttributeContainer& GenericMap::getAttributeContainer(unsigned int orbit)
{
	return m_attribs[orbit] ;
}

inline const AttributeContainer& GenericMap::getAttributeContainer(unsigned int orbit) const
{
	return m_attribs[orbit] ;
}

inline AttributeMultiVectorGen* GenericMap::getAttributeVectorGen(unsigned int orbit, const std::string& nameAttr)
{
	return m_attribs[orbit].getVirtualDataVector(nameAttr) ;
}


template <unsigned int ORBIT>
AttributeMultiVector<MarkerBool>* GenericMap::askMarkVector(unsigned int thread)
{
	assert(isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded") ;

	if (!m_markVectors_free[ORBIT][thread].empty())
	{
		AttributeMultiVector<MarkerBool>* amv = m_markVectors_free[ORBIT][thread].back();
		m_markVectors_free[ORBIT][thread].pop_back();
		return amv;
	}
	else
	{
        boost::mutex::scoped_lock lockMV(m_MarkerStorageMutex[ORBIT]);

		unsigned int x=m_nextMarkerId++;
		std::string number("___");
		number[2]= '0'+x%10;
		x = x/10;
		number[1]= '0'+x%10;
		x = x/10;
		number[0]= '0'+x%10;

        AttributeMultiVector<MarkerBool>* amv = m_attribs[ORBIT].addAttribute<MarkerBool>("marker_" + orbitName<ORBIT>() + number);
		return amv;
	}
}


template <unsigned int ORBIT>
inline void GenericMap::releaseMarkVector(AttributeMultiVector<MarkerBool>* amv, unsigned int thread)
{
	assert(isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded") ;
    amv->allFalse();
	m_markVectors_free[ORBIT][thread].push_back(amv);
}



template <unsigned int ORBIT>
inline AttributeMultiVector<unsigned int>* GenericMap::getEmbeddingAttributeVector()
{
	return m_embeddings[ORBIT] ;
}

template <typename R>
bool GenericMap::registerAttribute(const std::string &nameType)
{
	RegisteredBaseAttribute* ra = new RegisteredAttribute<R>;
	if (ra == NULL)
	{
		CGoGNerr << "Erreur enregistrement attribut" << CGoGNendl;
		return false;
	}

	ra->setTypeName(nameType);

	m_attributes_registry_map->insert(std::pair<std::string, RegisteredBaseAttribute*>(nameType,ra));
	return true;
}

/****************************************
 *   EMBEDDING ATTRIBUTES MANAGEMENT    *
 ****************************************/

template <unsigned int ORBIT>
void GenericMap::addEmbedding()
{
	if (!isOrbitEmbedded<ORBIT>())
	{
		std::ostringstream oss;
		oss << "EMB_" << ORBIT;

		AttributeContainer& dartCont = m_attribs[DART] ;
		AttributeMultiVector<unsigned int>* amv = dartCont.addAttribute<unsigned int>(oss.str()) ;
		m_embeddings[ORBIT] = amv ;

		// set new embedding to EMBNULL for all the darts of the map
		for(unsigned int i = dartCont.begin(); i < dartCont.end(); dartCont.next(i))
			(*amv)[i] = EMBNULL ;
	}
}

/****************************************
 *          ORBITS TRAVERSALS           *
 ****************************************/

/****************************************
 *  TOPOLOGICAL ATTRIBUTES MANAGEMENT   *
 ****************************************/

inline AttributeMultiVector<Dart>* GenericMap::addRelation(const std::string& name)
{
	AttributeContainer& cont = m_attribs[DART] ;
	AttributeMultiVector<Dart>* amv = cont.addAttribute<Dart>(name) ;

	// set new relation to fix point for all the darts of the map
	for(unsigned int i = cont.begin(); i < cont.end(); cont.next(i))
		(*amv)[i] = Dart(i) ;

	return amv ;
}

inline AttributeMultiVector<Dart>* GenericMap::getRelation(const std::string& name)
{
	AttributeContainer& cont = m_attribs[DART] ;
	AttributeMultiVector<Dart>* amv = cont.getDataVector<Dart>(cont.getAttributeIndex(name)) ;
	return amv ;
}

} //namespace CGoGN
