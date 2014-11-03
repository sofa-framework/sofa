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
#include <cassert>
#include "Container/registered.h"
#include "Utils/nameTypes.h"

namespace CGoGN
{

inline unsigned int AttributeContainer::getOrbit() const
{
	return m_orbit ;
}

inline void AttributeContainer::setOrbit(unsigned int orbit)
{
	m_orbit = orbit ;
	for(unsigned int i = 0; i < m_tableAttribs.size(); ++i)
	{
		if (m_tableAttribs[i] != NULL)
			m_tableAttribs[i]->setOrbit(orbit);
	}
}

inline void AttributeContainer::setRegistry(std::map< std::string, RegisteredBaseAttribute* >* re)
{
	m_attributes_registry_map = re;
}

/**************************************
 *          BASIC FEATURES            *
 **************************************/

template <typename T>
AttributeMultiVector<T>* AttributeContainer::addAttribute(const std::string& attribName)
{
	// first check if attribute already exist
	unsigned int index ;
	if (attribName != "")
	{
		index = getAttributeIndex(attribName) ;
		if (index != UNKNOWN)
		{
			std::cout << "attribute " << attribName << " already found.." << std::endl ;
			return NULL ;
		}
	}

	// create the new attribute
	std::string typeName = nameOfType(T()) ;
	AttributeMultiVector<T>* amv = new AttributeMultiVector<T>(attribName, typeName) ;

	if(!m_freeIndices.empty())
	{
		index = m_freeIndices.back() ;
		m_freeIndices.pop_back() ;
		m_tableAttribs[index] = amv ;
	}
	else
	{
		index = m_tableAttribs.size() ;
		m_tableAttribs.push_back(amv) ;
	}

	amv->setOrbit(m_orbit) ;
	amv->setIndex(index) ;

	// generate a name for the attribute if no one was given
	if (attribName == "")
	{
		std::stringstream ss ;
		ss << "unknown" << m_nbUnknown++ ;
		amv->setName(ss.str()) ;
	}

	// update the memory cost of a line
	m_lineCost += sizeof(T) ;

	// resize the new attribute so that it has the same size than others
	amv->setNbBlocks(m_holesBlocks.size()) ;

	m_nbAttributes++ ;

	return amv ;
}

template <typename T>
void AttributeContainer::addAttribute(const std::string& attribName, const std::string& nametype, unsigned int index)
{
	// first check if attribute already exist
	if (attribName != "")
	{
		unsigned int i = getAttributeIndex(attribName) ;
		if (i != UNKNOWN)
			return ;
	}

	// create the new attribute
	AttributeMultiVector<T>* amv = new AttributeMultiVector<T>(attribName, nametype);

	m_tableAttribs[index] = amv;
	amv->setOrbit(m_orbit) ;
	amv->setIndex(index) ;

	// generate a name for the attribute if no one was given
	if (attribName == "")
	{
		std::stringstream ss;
		ss << "unknown" << m_nbUnknown++;
		amv->setName(ss.str());
	}

	// update the memory cost of a line
	m_lineCost += sizeof(T) ;

	// resize the new attribute so that it has the same size than others
	amv->setNbBlocks(m_holesBlocks.size()) ;

	m_nbAttributes++;
}

template <typename T>
bool AttributeContainer::removeAttribute(const std::string& attribName)
{
	unsigned int index = getAttributeIndex(attribName) ;

	if (index == UNKNOWN)
	{
		std::cerr << "removeAttribute by name: attribute not found (" << attribName << ")"<< std::endl ;
		return false ;
	}

	// delete the attribute
	delete m_tableAttribs[index] ;
	m_tableAttribs[index] = NULL ;

	if (index == m_tableAttribs.size() - 1)
		m_tableAttribs.pop_back() ;
	else
		m_freeIndices.push_back(index) ;

	--m_nbAttributes ;
	m_lineCost -= sizeof(T);

	return true ;
}

template <typename T>
bool AttributeContainer::removeAttribute(unsigned int index)
{
	if(m_tableAttribs[index] == NULL)
	{
		std::cerr << "removeAttribute by index: attribute not found" << std::endl ;
		return false ;
	}

	// delete the attribute
	delete m_tableAttribs[index] ;
	m_tableAttribs[index] = NULL ;

	if(index == m_tableAttribs.size() - 1)
		m_tableAttribs.pop_back() ;
	else
		m_freeIndices.push_back(index) ;

	--m_nbAttributes ;
	m_lineCost -= sizeof(T);

	return true ;
}


/**************************************
 *      INFO ABOUT THE CONTAINER      *
 **************************************/

inline unsigned int AttributeContainer::getNbAttributes() const
{
	return m_nbAttributes;
}

inline unsigned int AttributeContainer::size() const
{
	return m_size;
}

inline unsigned int AttributeContainer::capacity() const
{
	return m_holesBlocks.size() * _BLOCKSIZE_;
}

inline unsigned int AttributeContainer::memoryTotalSize() const
{
	return capacity() * (m_lineCost + 8);
}

inline unsigned int AttributeContainer::memorySize() const
{
	return size() * (m_lineCost + 8);
}

inline bool AttributeContainer::used(unsigned int index) const
{
	return m_holesBlocks[index / _BLOCKSIZE_]->used(index % _BLOCKSIZE_);
}

/**************************************
 *         CONTAINER TRAVERSAL        *
 **************************************/

//inline unsigned int AttributeContainer::begin() const
//{
//	unsigned int it = 0;
//	while ((it < m_maxSize) && (!used(it)))
//		++it;
//	return it;
//}

//inline unsigned int AttributeContainer::end() const
//{
//	return m_maxSize;
//}

//inline void AttributeContainer::next(unsigned int &it) const
//{
//	do
//	{
//		++it;
//	} while ((it < m_maxSize) && (!used(it)));
//}

inline unsigned int AttributeContainer::begin() const
{
	if (m_currentBrowser != NULL)
		return m_currentBrowser->begin();
	return AttributeContainer::realBegin();
}

inline unsigned int AttributeContainer::end() const
{
	if (m_currentBrowser != NULL)
		return m_currentBrowser->end();
	return AttributeContainer::realEnd();
}

inline void AttributeContainer::next(unsigned int &it) const
{
	if (m_currentBrowser != NULL)
		m_currentBrowser->next(it);
	else
		AttributeContainer::realNext(it);
}

inline unsigned int AttributeContainer::realBegin() const
{
	unsigned int it = 0;
	while ((it < m_maxSize) && (!used(it)))
		++it;
	return it;
}

inline unsigned int AttributeContainer::realEnd() const
{
	return m_maxSize;
}

inline void AttributeContainer::realNext(unsigned int &it) const
{
	do
	{
		++it;
	} while ((it < m_maxSize) && (!used(it)));
}

/**************************************
 *          LINES MANAGEMENT          *
 **************************************/

inline void AttributeContainer::initLine(unsigned int index)
{
	for(unsigned int i = 0; i < m_tableAttribs.size(); ++i)
	{
		if (m_tableAttribs[i] != NULL)
			m_tableAttribs[i]->initElt(index);
	}
}

inline void AttributeContainer::copyLine(unsigned int dstIndex, unsigned int srcIndex)
{
	for(unsigned int i = 0; i < m_tableAttribs.size(); ++i)
	{
		if (m_tableAttribs[i] != NULL)
			m_tableAttribs[i]->copyElt(dstIndex, srcIndex);
	}
}

inline void AttributeContainer::refLine(unsigned int index)
{
	m_holesBlocks[index / _BLOCKSIZE_]->ref(index % _BLOCKSIZE_);
}

inline bool AttributeContainer::unrefLine(unsigned int index)
{
	if (m_holesBlocks[index / _BLOCKSIZE_]->unref(index % _BLOCKSIZE_))
	{
		--m_size;
		return true;
	}
	return false;
}

inline unsigned int AttributeContainer::getNbRefs(unsigned int index) const
{
	unsigned int bi = index / _BLOCKSIZE_;
	unsigned int j = index % _BLOCKSIZE_;

	return m_holesBlocks[bi]->nbRefs(j);
}

inline void AttributeContainer::setNbRefs(unsigned int index, unsigned int nb)
{
	m_holesBlocks[index / _BLOCKSIZE_]->setNbRefs(index % _BLOCKSIZE_, nb);
}

/**************************************
 *       ATTRIBUTES MANAGEMENT        *
 **************************************/

inline bool AttributeContainer::copyAttribute(unsigned int index_dst, unsigned int index_src)
{
	return m_tableAttribs[index_dst]->copy(m_tableAttribs[index_src]);
}

inline bool AttributeContainer::swapAttributes(unsigned int index1, unsigned int index2)
{
	return m_tableAttribs[index1]->swap(m_tableAttribs[index2]);
}

/**************************************
 *       ATTRIBUTES DATA ACCESS       *
 **************************************/

template <typename T>
AttributeMultiVector<T>* AttributeContainer::getDataVector(unsigned int attrIndex)
{
	assert(attrIndex < m_tableAttribs.size() || !"getDataVector: attribute index out of bounds");
	assert(m_tableAttribs[attrIndex] != NULL || !"getDataVector: attribute does not exist");

	AttributeMultiVector<T>* atm = dynamic_cast<AttributeMultiVector<T>*>(m_tableAttribs[attrIndex]);
	assert((atm != NULL) || !"getDataVector: wrong type");
	return atm;
}

inline AttributeMultiVectorGen* AttributeContainer::getVirtualDataVector(unsigned int attrIndex)
{
	return m_tableAttribs[attrIndex];
}

template <typename T>
AttributeMultiVector<T>* AttributeContainer::getDataVector(const std::string& attribName)
{
	unsigned int index = getAttributeIndex(attribName) ;
	if(index == UNKNOWN)
		return NULL ;

	AttributeMultiVector<T>* atm = dynamic_cast<AttributeMultiVector<T>*>(m_tableAttribs[index]);
	assert((atm != NULL) || !"getDataVector: wrong type");
	return atm;
}

inline AttributeMultiVectorGen* AttributeContainer::getVirtualDataVector(const std::string& attribName)
{
	unsigned int index = getAttributeIndex(attribName) ;
	if(index == UNKNOWN)
		return NULL ;
	else
		return m_tableAttribs[index];
}

template <typename T>
inline T& AttributeContainer::getData(unsigned int attrIndex, unsigned int eltIndex)
{
	assert(eltIndex < m_maxSize || !"getData: element index out of bounds");
	assert(m_holesBlocks[eltIndex / _BLOCKSIZE_]->used(eltIndex % _BLOCKSIZE_) || !"getData: element does not exist");
	assert((m_tableAttribs[attrIndex] != NULL) || !"getData: attribute does not exist");

	AttributeMultiVector<T>* atm = dynamic_cast<AttributeMultiVector<T>*>(m_tableAttribs[attrIndex]);
	assert((atm != NULL) || !"getData: wrong type");

	return atm->operator[](eltIndex);
}

template <typename T>
inline const T& AttributeContainer::getData(unsigned int attrIndex, unsigned int eltIndex) const
{
	assert(eltIndex < m_maxSize || !"getData: element index out of bounds");
	assert(m_holesBlocks[eltIndex / _BLOCKSIZE_]->used(eltIndex % _BLOCKSIZE_) || !"getData: element does not exist");
	assert((m_tableAttribs[attrIndex] != NULL) || !"getData: attribute does not exist");

	AttributeMultiVector<T>* atm = dynamic_cast<AttributeMultiVector<T>*>(m_tableAttribs[attrIndex]);
	assert((atm != NULL) || !"getData: wrong type");

	return atm->operator[](eltIndex);
}

template <typename T>
inline void AttributeContainer::setData(unsigned int attrIndex, unsigned int eltIndex, const T& data)
{
	assert(eltIndex < m_maxSize || !"getData: element index out of bounds");
	assert(m_holesBlocks[eltIndex / _BLOCKSIZE_]->used(eltIndex % _BLOCKSIZE_) || !"getData: element does not exist");
	assert((m_tableAttribs[attrIndex] != NULL) || !"getData: attribute does not exist");

	AttributeMultiVector<T>* atm = dynamic_cast<AttributeMultiVector<T>*>(m_tableAttribs[attrIndex]);
	assert((atm != NULL) || !"getData: wrong type");

	atm->operator[](eltIndex) = data;
}

} // namespace CGoGN
