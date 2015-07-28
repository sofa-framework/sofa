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

#include <typeinfo>
#include <stdio.h>
#include <string.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#include <iostream>

#include "Container/attributeContainer.h"
#include "Container/attributeMultiVector.h"
#include "Topology/generic/dart.h"

namespace CGoGN
{

AttributeContainer::AttributeContainer() :
    m_currentBrowser(NULL),
    m_orbit(0),
    m_nbAttributes(0),
    m_nbUnknown(0),
    m_size(0),
    m_maxSize(0),
    m_lineCost(0),
    m_attributes_registry_map(NULL)
{
    m_holesBlocks.reserve(512);
}

AttributeContainer::~AttributeContainer()
{
    for (unsigned int index = 0; index < m_tableAttribs.size(); ++index)
    {
        if (m_tableAttribs[index] != NULL)
            delete m_tableAttribs[index];
    }

    for (unsigned int index = 0; index < m_holesBlocks.size(); ++index)
    {
        if (m_holesBlocks[index] != NULL)
            delete m_holesBlocks[index];
    }

}

/**************************************
 *       INFO ABOUT ATTRIBUTES        *
 **************************************/

unsigned int AttributeContainer::getAttributeIndex(const std::string& attribName) const
{
    unsigned int index ;
    bool found = false ;
    for (index = 0; index < m_tableAttribs.size() && !found; ++index)
    {
        if(m_tableAttribs[index] != NULL && m_tableAttribs[index]->getName() == attribName)
            found = true ;
    }

    if (!found)
        return UNKNOWN;
    else
        return index - 1 ;
}

const std::string& AttributeContainer::getAttributeName(unsigned int index) const
{
    assert(index < m_tableAttribs.size() || !"getAttributeName: attribute index out of bounds");
    assert(m_tableAttribs[index] != NULL || !"getAttributeName: attribute does not exist");

    return m_tableAttribs[index]->getName() ;
}

template <typename T>
unsigned int AttributeContainer::getAttributeBlocksPointers(unsigned int attrIndex, std::vector<T*>& vect_ptr, unsigned int& byteBlockSize)
{
    assert(attrIndex < m_tableAttribs.size() || !"getAttributeBlocksPointers: attribute index out of bounds");
    assert(m_tableAttribs[attrIndex] != NULL || !"getAttributeBlocksPointers: attribute does not exist");

    AttributeMultiVector<T>* atm = dynamic_cast<AttributeMultiVector<T>*>(m_tableAttribs[attrIndex]);
    assert((atm != NULL) || !"getAttributeBlocksPointers: wrong type");
    return atm->getBlocksPointers(vect_ptr, byteBlockSize);
}

unsigned int AttributeContainer::getAttributesNames(std::vector<std::string>& names) const
{
    names.clear() ;
    names.reserve(m_nbAttributes) ;

    for (unsigned int i = 0; i < m_tableAttribs.size(); ++i)
    {
        if(m_tableAttribs[i] != NULL)
            names.push_back(m_tableAttribs[i]->getName()) ;
    }

    return m_nbAttributes ;
}

unsigned int AttributeContainer::getAttributesTypes(std::vector<std::string>& types)
{
    types.clear() ;
    types.reserve(m_nbAttributes) ;

    for (unsigned int i = 0; i < m_tableAttribs.size(); ++i)
    {
        if(m_tableAttribs[i] != NULL)
            types.push_back(m_tableAttribs[i]->getTypeName()) ;
    }

    return m_nbAttributes ;
}

/**************************************
 *        CONTAINER MANAGEMENT        *
 **************************************/

void AttributeContainer::swap(AttributeContainer& cont)
{
    // swap everything but the orbit

	m_tableAttribs.swap(cont.m_tableAttribs);
	m_tableMarkerAttribs.swap(cont.m_tableMarkerAttribs);
	m_freeIndices.swap(cont.m_freeIndices);
	m_holesBlocks.swap(cont.m_holesBlocks);
	m_tableBlocksWithFree.swap(cont.m_tableBlocksWithFree);
	m_tableBlocksEmpty.swap(cont.m_tableBlocksEmpty);

    unsigned int temp = m_nbAttributes;
    m_nbAttributes = cont.m_nbAttributes;
    cont.m_nbAttributes = temp;

    temp = m_nbUnknown ;
    m_nbUnknown = cont.m_nbUnknown ;
    cont.m_nbUnknown = temp ;

    temp = m_size;
    m_size = cont.m_size;
    cont.m_size = temp;

    temp = m_maxSize;
    m_maxSize = cont.m_maxSize;
    cont.m_maxSize = temp;

    temp = m_lineCost;
    m_lineCost = cont.m_lineCost;
    cont.m_lineCost = temp;
}

void AttributeContainer::clear(bool removeAttrib)
{
    m_size = 0;
    m_maxSize = 0;

    // raz des cases libres
    for (std::vector<HoleBlockRef*>::iterator it = m_holesBlocks.begin(); it != m_holesBlocks.end(); ++it)
        delete (*it);

    { // add bracket just for scope of temporary vectors
        std::vector<HoleBlockRef*> bf;
        m_holesBlocks.swap(bf);

        std::vector<unsigned int> bwf;
        m_tableBlocksWithFree.swap(bwf);
    }

    // detruit les données
    for (std::vector<AttributeMultiVectorGen*>::iterator it = m_tableAttribs.begin(); it != m_tableAttribs.end(); ++it)
    {
        if ((*it) != NULL)
            (*it)->clear();
    }

    // on enleve les attributs ?
    if (removeAttrib)
    {
        // nb a zero
        m_nbAttributes = 0;

        // detruit tous les attributs
        for (std::vector<AttributeMultiVectorGen*>::iterator it = m_tableAttribs.begin(); it != m_tableAttribs.end(); ++it)
        {
            if ((*it) != NULL)
                delete (*it);
        }
        std::vector<AttributeMultiVectorGen*> amg;
        m_tableAttribs.swap(amg);

        // detruit tous les attributs MarkerBool
        for (std::vector<AttributeMultiVector<MarkerBool>*>::iterator it = m_tableMarkerAttribs.begin(); it != m_tableMarkerAttribs.end(); ++it)
        {
            if ((*it) != NULL)
                delete (*it);
        }
        std::vector<AttributeMultiVector<MarkerBool>*> amgb;
        m_tableMarkerAttribs.swap(amgb);

        std::vector<unsigned int> fi;
        m_freeIndices.swap(fi);
    }
}

bool AttributeContainer::hasMarkerAttribute() const
{
	// not only size() != 0 because of BoundaryMarkers !
    for (std::vector<AttributeMultiVector<MarkerBool>*>::const_iterator it = m_tableMarkerAttribs.begin(); it != m_tableMarkerAttribs.end(); ++it)
	{
		std::string strMarker = (*it)->getName().substr(0,6);
		if (strMarker=="marker")
		return true;
	}
	return false;

}


void AttributeContainer::compact(std::vector<unsigned int>& mapOldNew)
{
    printFreeIndices();
	mapOldNew.clear();
	mapOldNew.resize(realEnd(),0xffffffff);

//VERSION THAT PRESERVE ORDER OF ELEMENTS ?
//	unsigned int down = 0;
//	for (unsigned int occup = realBegin(); occup != realEnd(); next(occup))
//	{
//		mapOldNew[occup] = down;
//		copyLine(down,occup);
//		// copy ref counter
//		setNbRefs(down,getNbRefs(occup));
//		down++;
//	}

	// fill the holes with data & create the map Old->New
	unsigned int up = realRBegin();
	unsigned int down = 0;

	while (down < up)
	{
		if (!used(down))
		{
			mapOldNew[up] = down;
			// copy data
			copyLine(down,up);
			// copy ref counter
			setNbRefs(down,getNbRefs(up));
			// set next element to catch for hole filling
			realRNext(up);
		}
		down++;
	}

	// end of table = nb elements
	m_maxSize = m_size;

	// no more blocks empty
	m_tableBlocksEmpty.clear();

	// only the last block has free indices
	m_tableBlocksWithFree.clear();

	// compute nb block full
	unsigned int nbb = m_size / _BLOCKSIZE_;
	// update holeblock
	for (unsigned int i=0; i<nbb; ++i)
		m_holesBlocks[i]->compressFull(_BLOCKSIZE_);

	//update last holeblock
	unsigned int nbe = m_size % _BLOCKSIZE_;
	if (nbe != 0)
	{
		m_holesBlocks[nbb]->compressFull(nbe);
		m_tableBlocksWithFree.push_back(nbb);
		nbb++;
	}

	// free memory and resize
	for (int i = m_holesBlocks.size() - 1; i > int(nbb); --i)
		delete m_holesBlocks[i];
	m_holesBlocks.resize(nbb);


	// release unused data memory
	for(unsigned int j = 0; j < m_tableAttribs.size(); ++j)
	{
		if (m_tableAttribs[j] != NULL)
			m_tableAttribs[j]->setNbBlocks(m_holesBlocks.size());
	}
}

void AttributeContainer::printFreeIndices() {
    std::cerr << "begin printFreeIndices()" << std::endl;
    for (unsigned i = 0u ; i < m_tableBlocksWithFree.size() ; ++i) {
        std::cerr << "tableBlock number " << m_tableBlocksWithFree[i] << std::endl;
        HoleBlockRef* const block = m_holesBlocks[m_tableBlocksWithFree[i]];
        block->printTableFree();
    }
    std::cerr << "end printFreeIndices()" << std::endl;
}



/**************************************
 *          LINES MANAGEMENT          *
 **************************************/

//unsigned int AttributeContainer::insertLine()
//{
//	// if no more rooms
//	if (m_tableBlocksWithFree.empty())
//	{
//		HoleBlockRef* ptr = new HoleBlockRef();					// new block
//		m_tableBlocksWithFree.push_back(m_holesBlocks.size());	// add its future position to block_free
//		m_holesBlocks.push_back(ptr);							// and add it to block table

//		for(unsigned int i = 0; i < m_tableAttribs.size(); ++i)
//		{
//			if (m_tableAttribs[i] != NULL)
//				m_tableAttribs[i]->addBlock();					// add a block to every attribute
//		}
//	}

//	// get the first free block index (last in vector)
//	unsigned int bf = m_tableBlocksWithFree.back();
//	// get the block
//	HoleBlockRef* block = m_holesBlocks[bf];

//	// add new element in block and compute index
//	unsigned int ne = block->newRefElt(m_maxSize);
//	unsigned int index = _BLOCKSIZE_ * bf + ne;

//	// if no more room in block remove it from free_blocks
//	if (block->full())
//		m_tableBlocksWithFree.pop_back();

//	++m_size;

////	initLine(index);

//	return index;
//}

void AttributeContainer::removeFromFreeIndices(unsigned int index) {
    const unsigned int bi = index / _BLOCKSIZE_;
    const unsigned int j = index % _BLOCKSIZE_;
    HoleBlockRef* const block = m_holesBlocks[bi];
    const bool res = block->removeFromFreeElts(j) ;
    if (res )
        if (block->full()) {
            // should be the front index
            unsigned int i = 0u;
            while (m_tableBlocksWithFree[i] != bi) { ++i; }
            std::swap(m_tableBlocksWithFree[i], m_tableBlocksWithFree.back());
            m_tableBlocksWithFree.pop_back();
        }
    //        ++m_size;
}


//void AttributeContainer::updateHole(unsigned int index) {
//    //    std::cerr << "updateHole called for orbit " << m_orbit << ", with index = " << index << std::endl;
//    //    std::cerr << "###############################################################" << std::endl;
//    //    std::cerr << "m_tableBlocksWithFree :";
//    //    for (int i = 0 ; i < m_tableBlocksWithFree.size() ; ++i) {
//    //        std::cerr << " " << m_tableBlocksWithFree[i] ;
//    //    }
//    //    std::cerr << std::endl;
//    //    std::cerr << "###############################################################" << std::endl;

//    const unsigned int bi = index / _BLOCKSIZE_;
//    const unsigned int j = index % _BLOCKSIZE_;
//    HoleBlockRef* const block = m_holesBlocks[bi];
//    const bool  blockWasFull = std::find(m_tableBlocksWithFree.begin(), m_tableBlocksWithFree.end(),bi) == m_tableBlocksWithFree.end();
//    //    std::cerr << "update hole called  on index " << index << " bi = " << bi << " j = " << j << std::endl;
//    const bool lineRemoved = block->updateHole(j);
//    if (lineRemoved) {
//        if (blockWasFull) {
//            m_tableBlocksWithFree.push_back(bi);
//        } else {
//            if (block->empty())		// block is empty after removal
//                m_tableBlocksEmpty.push_back(bi);
//        }

//        std::sort(m_tableBlocksWithFree.begin(), m_tableBlocksWithFree.end(), std::greater<unsigned int>());
//    }
//    ////                DEBUG
//    //        std::cerr << "###############################################################" << std::endl;
//    //        std::cerr << "m_tableBlocksWithFree :";
//    //        for (int i = 0 ; i < m_tableBlocksWithFree.size() ; ++i) {
//    //            std::cerr << " " << m_tableBlocksWithFree[i] ;
//    //        }
//    //        std::cerr << std::endl;
//    //        std::cerr << "###############################################################" << std::endl;
//    ////                END DEBUG

//}

unsigned int AttributeContainer::insertLine()
{
    // if no more rooms
    if (m_tableBlocksWithFree.empty())
    {
        HoleBlockRef* ptr = new HoleBlockRef();					// new block
        unsigned int numBlock = m_holesBlocks.size();
        m_tableBlocksWithFree.push_back(numBlock);	// add its future position to block_free
        m_holesBlocks.push_back(ptr);							// and add it to block table

        for(unsigned int i = 0; i < m_tableAttribs.size(); ++i)
        {
            if (m_tableAttribs[i] != NULL)
                m_tableAttribs[i]->addBlock();					// add a block to every attribute
        }

		for(unsigned int i = 0; i < m_tableMarkerAttribs.size(); ++i)
		{
			if (m_tableMarkerAttribs[i] != NULL)
				m_tableMarkerAttribs[i]->addBlock();					// add a block to every attribute
		}

        // add new element in block and compute index

		// inc nb of elements
		++m_size;

		// add new element in block and compute index

		unsigned int ne = ptr->newRefElt(m_maxSize);
		return _BLOCKSIZE_ * numBlock + ne;
	}
	// else

	// get the first free block index (last in vector)
	unsigned int bf = m_tableBlocksWithFree.back();
	// get the block
	HoleBlockRef* block = m_holesBlocks[bf];

	// add new element in block and compute index
    unsigned int ne = block->newRefElt(m_maxSize);
	unsigned int index = _BLOCKSIZE_ * bf + ne;

	if (ne == _BLOCKSIZE_-1)
	{
		if (bf == (m_holesBlocks.size()-1))
		{
			// we are filling the last line of capacity
			HoleBlockRef* ptr = new HoleBlockRef();					// new block
			unsigned int numBlock = m_holesBlocks.size();
			m_tableBlocksWithFree.back() = numBlock;
			m_tableBlocksWithFree.push_back(bf);
			m_holesBlocks.push_back(ptr);

			for(unsigned int i = 0; i < m_tableAttribs.size(); ++i)
			{
				if (m_tableAttribs[i] != NULL)
					m_tableAttribs[i]->addBlock();					// add a block to every attribute
			}
			for(unsigned int i = 0; i < m_tableMarkerAttribs.size(); ++i)
			{
				if (m_tableMarkerAttribs[i] != NULL)
					m_tableMarkerAttribs[i]->addBlock();					// add a block to every attribute
			}
		}
	}

	// if no more room in block remove it from free_blocks
	if (block->full())
		m_tableBlocksWithFree.pop_back();

	++m_size;
	return index;
}


void AttributeContainer::removeLine(unsigned int index)
{
    unsigned int bi = index / _BLOCKSIZE_;
    unsigned int j = index % _BLOCKSIZE_;

    HoleBlockRef* block = m_holesBlocks[bi];

    if (block->used(j))
    {
        if (block->full())		// block has no free elements before removal
            m_tableBlocksWithFree.push_back(bi);

        block->removeElt(j);

        --m_size;

        if (block->empty())		// block is empty after removal
            m_tableBlocksEmpty.push_back(bi);
    }
    else
    {
        std::cerr << "Error removing non existing index " << index << std::endl;
    }

}

/**************************************
 *            SAVE & LOAD             *
 **************************************/


void AttributeContainer::saveBin(CGoGNostream& fs, unsigned int id) const
{
	std::vector<AttributeMultiVectorGen*> bufferamv;
	bufferamv.reserve(m_tableAttribs.size());

//	for(std::vector<AttributeMultiVectorGen*>::const_iterator it = m_tableAttribs.begin(); it != m_tableAttribs.end(); ++it)
//	{
//		if (*it != NULL)
//		{
//			const std::string& attName = (*it)->getName();
//			std::string markName = attName.substr(0,7);
//			if (markName != "marker_")
//				bufferamv.push_back(*it);
//		}
//	}
	for(std::vector<AttributeMultiVectorGen*>::const_iterator it = m_tableAttribs.begin(); it != m_tableAttribs.end(); ++it)
	{
		if (*it != NULL)
			bufferamv.push_back(*it);
	}

	// en ascii id et les tailles

	std::vector<unsigned int> bufferui;
	bufferui.reserve(10);

	bufferui.push_back(id);
	bufferui.push_back(_BLOCKSIZE_);
	bufferui.push_back(m_holesBlocks.size());
	bufferui.push_back(m_tableBlocksWithFree.size());
	bufferui.push_back(bufferamv.size());
	bufferui.push_back(m_size);
	bufferui.push_back(m_maxSize);
	bufferui.push_back(m_orbit);
	bufferui.push_back(m_nbUnknown);

	// count attribute of boundary markers and increase nb of saved attributes
	for(std::vector<AttributeMultiVector<MarkerBool>*>::const_iterator it = m_tableMarkerAttribs.begin(); it != m_tableMarkerAttribs.end(); ++it)
	{
		const std::string& attName = (*it)->getName();
		if (attName[0] == 'B') // for BoundaryMark0/1
			bufferui[4]++;
	}

	fs.write(reinterpret_cast<const char*>(&bufferui[0]), bufferui.size()*sizeof(unsigned int));

	unsigned int i = 0;

	for(std::vector<AttributeMultiVector<MarkerBool>*>::const_iterator it = m_tableMarkerAttribs.begin(); it != m_tableMarkerAttribs.end(); ++it)
	{
		const std::string& attName = (*it)->getName();
		if (attName[0] == 'B') // for BoundaryMark0/1
			(*it)->saveBin(fs, i++);
	}

	for(std::vector<AttributeMultiVectorGen*>::const_iterator it = bufferamv.begin(); it != bufferamv.end(); ++it)
	{
		(*it)->saveBin(fs, i++);
	}

	//en binaire les blocks de ref
	for (std::vector<HoleBlockRef*>::const_iterator it = m_holesBlocks.begin(); it != m_holesBlocks.end(); ++it)
		(*it)->saveBin(fs);

	// les indices des blocks libres
	fs.write(reinterpret_cast<const char*>(&m_tableBlocksWithFree[0]), m_tableBlocksWithFree.size() * sizeof(unsigned int));
}

unsigned int AttributeContainer::loadBinId(CGoGNistream& fs)
{
	unsigned int id;
	fs.read(reinterpret_cast<char*>(&id), sizeof(unsigned int));
	return id;
}

bool AttributeContainer::loadBin(CGoGNistream& fs)
{
	if (m_attributes_registry_map == NULL)
	{
		CGoGNerr << "Attribute Registry non initialized"<< CGoGNendl;
		return false;
	}

	std::vector<unsigned int> bufferui;
	bufferui.resize(256);

	fs.read(reinterpret_cast<char*>(&(bufferui[0])), 8*sizeof(unsigned int));	//WARNING 9 hard coded

	unsigned int bs, szHB, szBWF, nbAtt;
	bs = bufferui[0];
	szHB = bufferui[1];
	szBWF = bufferui[2];
	nbAtt = bufferui[3];
	m_size = bufferui[4];
	m_maxSize = bufferui[5];
	m_orbit = bufferui[6];
	m_nbUnknown = bufferui[7];


	if (bs != _BLOCKSIZE_)
	{
		CGoGNerr << "Loading unavailable, different block sizes: "<<_BLOCKSIZE_<<" / " << bs << CGoGNendl;
		return false;
	}


	for (unsigned int j = 0; j < nbAtt; ++j)
	{
		std::string nameAtt;
		std::string typeAtt;
		/*unsigned int id = */AttributeMultiVectorGen::loadBinInfos(fs,nameAtt, typeAtt);

		std::map<std::string, RegisteredBaseAttribute*>::iterator itAtt = m_attributes_registry_map->find(typeAtt);
		if (itAtt == m_attributes_registry_map->end())
		{
			CGoGNout << "Skipping non registred attribute of type name"<< typeAtt <<CGoGNendl;
			AttributeMultiVectorGen::skipLoadBin(fs);
		}
		else
		{
			if (typeAtt == "MarkerBool")
			{
				assert(j<m_tableMarkerAttribs.size());
				m_tableMarkerAttribs[j]->loadBin(fs); // use j because BM are saved first
			}
			else
			{
				RegisteredBaseAttribute* ra = itAtt->second;
				AttributeMultiVectorGen* amvg = ra->addAttribute(*this, nameAtt);
				amvg->loadBin(fs);
			}
		}
	}

	m_holesBlocks.resize(szHB);

	// blocks
	for (unsigned int i = 0; i < szHB; ++i)
	{
		m_holesBlocks[i] = new HoleBlockRef;
		m_holesBlocks[i]->loadBin(fs);
	}

	// les indices des blocks libres
	m_tableBlocksWithFree.resize(szBWF);
	fs.read(reinterpret_cast<char*>(&(m_tableBlocksWithFree[0])), szBWF*sizeof(unsigned int));

	return true;
}

void  AttributeContainer::copyFrom(const AttributeContainer& cont)
{
// 	clear is done from the map

	m_size = cont.m_size;
	m_maxSize = cont.m_maxSize;
	m_orbit = cont.m_orbit;
	m_nbUnknown = cont.m_nbUnknown;
	m_nbAttributes = cont.m_nbAttributes;
	m_lineCost = cont.m_lineCost;

	// blocks
	unsigned int sz = cont.m_holesBlocks.size();
	m_holesBlocks.resize(sz);
	for (unsigned int i = 0; i < sz; ++i)
		m_holesBlocks[i] = new HoleBlockRef(*(cont.m_holesBlocks[i]));

	//  free indices
	sz = cont.m_freeIndices.size();
	m_freeIndices.resize(sz);
	for (unsigned int i = 0; i < sz; ++i)
		m_freeIndices[i] = cont.m_freeIndices[i];

	// blocks with free
	sz = cont.m_tableBlocksWithFree.size();
	m_tableBlocksWithFree.resize(sz);
	for (unsigned int i = 0; i < sz; ++i)
		m_tableBlocksWithFree[i] = cont.m_tableBlocksWithFree[i];

	// empty blocks
	sz = cont.m_tableBlocksEmpty.size();
	m_tableBlocksEmpty.resize(sz);
	for (unsigned int i = 0; i < sz; ++i)
		m_tableBlocksEmpty[i] = cont.m_tableBlocksEmpty[i];

	//attributes (warning attribute can have different numbers than in original)
	m_tableAttribs.reserve(m_nbAttributes);
	sz = cont.m_tableAttribs.size();
	for (unsigned int i = 0; i < sz; ++i)
	{
		if (cont.m_tableAttribs[i] != NULL)
		{
			AttributeMultiVectorGen* ptr = cont.m_tableAttribs[i]->new_obj();
			ptr->setName(cont.m_tableAttribs[i]->getName());
			ptr->setOrbit(cont.m_tableAttribs[i]->getOrbit());
			ptr->setIndex(m_tableAttribs.size());
			ptr->setNbBlocks(cont.m_tableAttribs[i]->getNbBlocks());
			ptr->copy(cont.m_tableAttribs[i]);
			m_tableAttribs.push_back(ptr);
		}
	}
	sz = cont.m_tableMarkerAttribs.size();
	for (unsigned int i = 0; i < sz; ++i)
	{
		AttributeMultiVector<MarkerBool>* ptr = new AttributeMultiVector<MarkerBool>;
		ptr->setTypeName(cont.m_tableMarkerAttribs[i]->getTypeName());
		ptr->setName(cont.m_tableMarkerAttribs[i]->getName());
		ptr->setOrbit(cont.m_tableMarkerAttribs[i]->getOrbit());
		ptr->setIndex(m_tableMarkerAttribs.size());
		ptr->setNbBlocks(cont.m_tableMarkerAttribs[i]->getNbBlocks());
		ptr->copy(cont.m_tableMarkerAttribs[i]);
		m_tableMarkerAttribs.push_back(ptr);
	}
}

void AttributeContainer::dumpCSV() const
{
	CGoGNout << "Name ; ;";
	for (unsigned int i = 0; i < m_tableAttribs.size(); ++i)
	{
		if (m_tableAttribs[i] != NULL)
		{
			CGoGNout << m_tableAttribs[i]->getName() << " ; ";
		}
	}
	for (unsigned int i = 0; i < m_tableMarkerAttribs.size(); ++i)
	{
		CGoGNout << m_tableMarkerAttribs[i]->getName() << " ; ";
	}
	CGoGNout << CGoGNendl;
	CGoGNout << "Type  ; ;";
	for (unsigned int i = 0; i < m_tableAttribs.size(); ++i)
	{
		if (m_tableAttribs[i] != NULL)
		{
			CGoGNout << m_tableAttribs[i]->getTypeName() << " ; ";
		}
	}
	for (unsigned int i = 0; i < m_tableMarkerAttribs.size(); ++i)
	{
		CGoGNout << m_tableMarkerAttribs[i]->getTypeName() << " ; ";
	}
	CGoGNout << CGoGNendl;
	CGoGNout << "line ; refs ;";
	for (unsigned int i = 0; i < m_tableAttribs.size(); ++i)
	{
		if (m_tableAttribs[i] != NULL)
		{
			CGoGNout << "value;";
		}
	}
	for (unsigned int i = 0; i < m_tableMarkerAttribs.size(); ++i)
	{
		CGoGNout << "value;";
	}
	CGoGNout << CGoGNendl;

	for (unsigned int l=this->begin(); l!= this->end(); this->next(l))
	{
		CGoGNout << l << " ; "<< this->getNbRefs(l)<< " ; ";
		for (unsigned int i = 0; i < m_tableAttribs.size(); ++i)
		{
			if (m_tableAttribs[i] != NULL)
			{
				m_tableAttribs[i]->dump(l);
				CGoGNout << " ; ";
			}
		}
		for (unsigned int i = 0; i < m_tableMarkerAttribs.size(); ++i)
		{
				m_tableMarkerAttribs[i]->dump(l);
				CGoGNout << " ; ";
		}

		CGoGNout << CGoGNendl;
	}
	CGoGNout << CGoGNendl;
}

void AttributeContainer::dumpByLines() const
{
	CGoGNout << "Container of "<<orbitName(this->getOrbit())<< CGoGNendl;
	for (unsigned int i = 0; i < m_tableAttribs.size(); ++i)
	{
		if (m_tableAttribs[i] != NULL)
		{
			CGoGNout << "Name: "<< m_tableAttribs[i]->getName();
			CGoGNout << " / Type: "<< m_tableAttribs[i]->getTypeName();
			for (unsigned int l=this->begin(); l!= this->end(); this->next(l))
			{
				CGoGNout << l << " ; ";
				m_tableAttribs[i]->dump(l);
				CGoGNout << CGoGNendl;
			}
		}
	}
	for (unsigned int i = 0; i < m_tableMarkerAttribs.size(); ++i)
	{
		CGoGNout << "Name: "<< m_tableMarkerAttribs[i]->getName();
		CGoGNout << " / Type: "<< m_tableMarkerAttribs[i]->getTypeName();
		for (unsigned int l=this->begin(); l!= this->end(); this->next(l))
		{
			CGoGNout << l << " ; ";
			m_tableMarkerAttribs[i]->dump(l);
			CGoGNout << CGoGNendl;
		}
	}
}

AttributeMultiVectorGen* AttributeContainer::addAttribute(const std::string& typeName, const std::string& attribName)
{
	// first check if attribute already exist
	unsigned int index = UNKNOWN ;
	if (attribName != "")
	{
		index = getAttributeIndex(attribName) ;
		if (index != UNKNOWN)
		{
			CGoGNerr << "attribute " << attribName << " already found.." << CGoGNendl ;
			return NULL ;
		}
	}

	// create the new attribute
	std::map<std::string, RegisteredBaseAttribute*>::iterator itAtt = m_attributes_registry_map->find(typeName);
	if (itAtt == m_attributes_registry_map->end())
	{
		CGoGNerr << "type " << typeName << " not registred.." << CGoGNendl ;
		return NULL ;
	}

	RegisteredBaseAttribute* ra = itAtt->second;
	AttributeMultiVectorGen* amv = ra->addAttribute(*this, attribName);

	return amv ;
}


AttributeMultiVector<MarkerBool>* AttributeContainer::addMarkerAttribute(const std::string& attribName)
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
	AttributeMultiVector<MarkerBool>* amv = new AttributeMultiVector<MarkerBool>(attribName, "MarkerBool") ;

	index = m_tableMarkerAttribs.size() ;
	m_tableMarkerAttribs.push_back(amv) ;

	amv->setOrbit(m_orbit) ;
	amv->setIndex(index) ;

	// resize the new attribute so that it has the same size than others
	amv->setNbBlocks(m_holesBlocks.size()) ;

	return amv ;
}


bool AttributeContainer::removeMarkerAttribute(const std::string& attribName)
{
	unsigned int index ;
	bool found = false ;
	for (index = 0; index < m_tableAttribs.size() && !found; ++index)
	{
		if (m_tableMarkerAttribs[index]->getName() == attribName)
			found = true ;
	}

	if (!found)
		return false;

	index--; // because of for loop

	delete m_tableMarkerAttribs[index] ;
	m_tableMarkerAttribs[index] = m_tableMarkerAttribs.back() ;
	m_tableMarkerAttribs.pop_back() ;

	return true;
}

} //namespace CGoGN
