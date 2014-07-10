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

#include "Topology/generic/mapImpl/mapMulti.h"

namespace CGoGN
{

/****************************************
 *     RESOLUTION LEVELS MANAGEMENT     *
 ****************************************/

void MapMulti::printMR()
{
	std::cout << std::endl ;

	for(unsigned int j = 0; j < m_mrNbDarts.size(); ++j)
		std::cout << m_mrNbDarts[j] << " / " ;
	std::cout << std::endl << "==========" << std::endl ;

	for(unsigned int i = m_mrattribs.begin(); i != m_mrattribs.end(); m_mrattribs.next(i))
	{
		std::cout << i << " : " << (*m_mrLevels)[i] << " / " ;
		for(unsigned int j = 0; j < m_mrDarts.size(); ++j)
			std::cout << (*m_mrDarts[j])[i] << " ; " ;
		std::cout << std::endl ;
	}
}

void MapMulti::initMR()
{
	m_mrattribs.clear(true) ;
	m_mrattribs.setRegistry(m_attributes_registry_map) ;

	m_mrDarts.clear() ;
	m_mrDarts.reserve(16) ;
	m_mrNbDarts.clear();
	m_mrNbDarts.reserve(16);
	m_mrLevelStack.clear() ;
	m_mrLevelStack.reserve(16) ;

	m_mrLevels = m_mrattribs.addAttribute<unsigned int>("MRLevel") ;

	AttributeMultiVector<unsigned int>* newAttrib = m_mrattribs.addAttribute<unsigned int>("MRdart_0") ;
	m_mrDarts.push_back(newAttrib) ;
	m_mrNbDarts.push_back(0) ;

	setCurrentLevel(0) ;
}

void MapMulti::addLevelBack()
{
	//AttributeMultiVector<unsigned int>* newAttrib = addLevel();

	unsigned int newLevel = m_mrDarts.size() ;
	std::stringstream ss ;
	ss << "MRdart_"<< newLevel ;
	AttributeMultiVector<unsigned int>* newAttrib = m_mrattribs.addAttribute<unsigned int>(ss.str()) ;

	m_mrDarts.push_back(newAttrib) ;
	m_mrNbDarts.push_back(0) ;

	if(m_mrDarts.size() > 1 )
	{
		// copy the indices of previous level into new level
		AttributeMultiVector<unsigned int>* prevAttrib = m_mrDarts[newLevel - 1];
		m_mrattribs.copyAttribute(newAttrib->getIndex(), prevAttrib->getIndex()) ;
	}
}

void MapMulti::addLevelFront()
{
	//AttributeMultiVector<unsigned int>* newAttrib = addLevel();

	unsigned int newLevel = m_mrDarts.size() ;
	std::stringstream ss ;
	ss << "MRdart_"<< newLevel ;
	AttributeMultiVector<unsigned int>* newAttrib = m_mrattribs.addAttribute<unsigned int>(ss.str()) ;
	AttributeMultiVector<unsigned int>* prevAttrib = m_mrDarts[0];

	// copy the indices of previous level into new level
	m_mrattribs.copyAttribute(newAttrib->getIndex(), prevAttrib->getIndex()) ;

	m_mrDarts.insert(m_mrDarts.begin(), newAttrib) ;
	m_mrNbDarts.insert(m_mrNbDarts.begin(), 0) ;
}

void MapMulti::removeLevelBack()
{
	unsigned int maxL = getMaxLevel() ;
	if(maxL > 0)
	{
		AttributeMultiVector<unsigned int>* maxMR = m_mrDarts[maxL] ;
		AttributeMultiVector<unsigned int>* prevMR = m_mrDarts[maxL - 1] ;
		for(unsigned int i = m_mrattribs.begin(); i != m_mrattribs.end(); m_mrattribs.next(i))
		{
			unsigned int idx = (*maxMR)[i] ;
			if((*m_mrLevels)[i] == maxL)	// if the MRdart was introduced on the level we're removing
			{
				deleteDartLine(idx) ;		// delete the pointed dart line
				m_mrattribs.removeLine(i) ;	// delete the MRdart line
			}
			else							// if the dart was introduced on a previous level
			{
				if(idx != (*prevMR)[i])		// delete the pointed dart line only if
					deleteDartLine(idx) ;	// it is not shared with previous level
			}
		}

		m_mrattribs.removeAttribute<unsigned int>(maxMR->getIndex()) ;
		m_mrDarts.pop_back() ;
		m_mrNbDarts.pop_back() ;

		if(m_mrCurrentLevel == maxL)
			--m_mrCurrentLevel ;
	}
}

void MapMulti::removeLevelFront()
{
	unsigned int maxL = getMaxLevel() ;
	if(maxL > 0) //must have at min 2 levels (0 and 1) to remove the front one
	{
		AttributeMultiVector<unsigned int>* minMR = m_mrDarts[0] ;
//		AttributeMultiVector<unsigned int>* firstMR = m_mrDarts[1] ;
		for(unsigned int i = m_mrattribs.begin(); i != m_mrattribs.end(); m_mrattribs.next(i))
		{
//			unsigned int idx = (*minMR)[i] ;
			if((*m_mrLevels)[i] != 0)	// if the MRdart was introduced after the level we're removing
			{
				--(*m_mrLevels)[i]; //decrement his level of insertion
			}
			else							// if the dart was introduced on a this level and not used after
			{
//				if(idx != (*firstMR)[i])		// delete the pointed dart line only if
//					deleteDartLine(idx) ;	// it is not shared with next level
			}
		}

		m_mrNbDarts[1] += m_mrNbDarts[0];

		m_mrattribs.removeAttribute<unsigned int>(minMR->getIndex()) ;
		m_mrDarts.erase(m_mrDarts.begin()) ;
		m_mrNbDarts.erase(m_mrNbDarts.begin()) ;

		--m_mrCurrentLevel ;
	}
}

void MapMulti::copyLevel(unsigned int level)
{
	AttributeMultiVector<unsigned int>* newAttrib = m_mrDarts[level] ;
	AttributeMultiVector<unsigned int>* prevAttrib = m_mrDarts[level - 1];

	// copy the indices of previous level into new level
	m_mrattribs.copyAttribute(newAttrib->getIndex(), prevAttrib->getIndex()) ;
}

void MapMulti::duplicateDarts(unsigned int newlevel)
{
//	AttributeMultiVector<unsigned int>* attrib = m_mrDarts[level] ;  //is a copy of the mrDarts at level-1 or level+1

//	for(unsigned int i = m_mrattribs.begin(); i != m_mrattribs.end(); m_mrattribs.next(i))
//	{
//		unsigned int oldi = (*attrib)[i] ;	// get the index of the dart in previous level
//		(*attrib)[i] = copyDartLine(oldi) ;	// copy the dart and affect it to the new level
//	}

	AttributeMultiVector<unsigned int>* attrib = m_mrDarts[newlevel] ;  //is a copy of the mrDarts at level-1 or level+1
	AttributeMultiVector<unsigned int>* prevAttrib = m_mrDarts[newlevel - 1] ;      // copy the indices of

	for(unsigned int i = m_mrattribs.begin(); i != m_mrattribs.end(); m_mrattribs.next(i))
	{
		unsigned int oldi = (*prevAttrib)[i] ;	// get the index of the dart in previous level
		(*attrib)[i] = copyDartLine(oldi) ;	// copy the dart and affect it to the new level
	}
}

/****************************************
 *             SAVE & LOAD              *
 ****************************************/

bool MapMulti::saveMapBin(const std::string& filename) const
{
	CGoGNostream fs(filename.c_str(), std::ios::out|std::ios::binary);
	if (!fs)
	{
		CGoGNerr << "Unable to open file for writing: " << filename << CGoGNendl;
		return false;
	}

	// Entete
	char* buff = new char[256];
	for (int i = 0; i < 256; ++i)
		buff[i] = char(255);

	memcpy(buff, "CGoGN_MRMap", 12);

	std::string mt = mapTypeName();
	const char* mtc = mt.c_str();
	memcpy(buff+32, mtc, mt.size()+1);
	unsigned int *buffi = reinterpret_cast<unsigned int*>(buff + 64);
	*buffi = NB_ORBITS;
	fs.write(reinterpret_cast<const char*>(buff), 256);
	delete buff;

	// save all attribs
	for (unsigned int i = 0; i < NB_ORBITS; ++i)
		m_attribs[i].saveBin(fs, i);

	m_mrattribs.saveBin(fs, 00);

	fs.write(reinterpret_cast<const char*>(&m_mrCurrentLevel), sizeof(unsigned int));

	unsigned int nb = m_mrNbDarts.size();
	fs.write(reinterpret_cast<const char*>(&nb), sizeof(unsigned int));
	fs.write(reinterpret_cast<const char*>(&(m_mrNbDarts[0])), nb *sizeof(unsigned int));

	return true;
}

bool MapMulti::loadMapBin(const std::string& filename)
{
	CGoGNistream fs(filename.c_str(), std::ios::in|std::ios::binary);
	if (!fs)
	{
		CGoGNerr << "Unable to open file for loading" << CGoGNendl;
		return false;
	}

	GenericMap::clear(true);

	// read info
    char buff[256] ;
	fs.read(reinterpret_cast<char*>(buff), 256);

	std::string buff_str(buff);
	// Check file type
	if (buff_str == "CGoGN_Map")
	{
		CGoGNerr << "Wrong binary file format, file is not a MR-Map" << CGoGNendl;
		return false;
	}
	if (buff_str != "CGoGN_MRMap")
	{
		CGoGNerr << "Wrong binary file format" << CGoGNendl;
		return false;
	}

	// Check map type
	buff_str = std::string(buff + 32);

	std::string localType = this->mapTypeName();

	std::string fileType = buff_str;

	if (fileType != localType)
	{
		CGoGNerr << "Not possible to load "<< fileType << " into " << localType << " object" << CGoGNendl;
		return false;
	}

	// Check max nb orbit
	unsigned int *ptr_nbo = reinterpret_cast<unsigned int*>(buff + 64);
	unsigned int nbo = *ptr_nbo;
	if (nbo != NB_ORBITS)
	{
		CGoGNerr << "Wrong max orbit number in file" << CGoGNendl;
		return  false;
	}

	// load attrib container
	for (unsigned int i = 0; i < NB_ORBITS; ++i)
	{
		unsigned int id = AttributeContainer::loadBinId(fs);
		m_attribs[id].loadBin(fs);
	}

	AttributeContainer::loadBinId(fs); // not used but need to read to skip
	m_mrattribs.loadBin(fs);

	fs.read(reinterpret_cast<char*>(&m_mrCurrentLevel), sizeof(unsigned int));
	unsigned int nb;
	fs.read(reinterpret_cast<char*>(&nb), sizeof(unsigned int));
	m_mrNbDarts.resize(nb);
	fs.read(reinterpret_cast<char*>(&(m_mrNbDarts[0])), nb *sizeof(unsigned int));

	// restore shortcuts
	GenericMap::restore_shortcuts();
	restore_topo_shortcuts();

	return true;
}

bool MapMulti::copyFrom(const GenericMap& map)
{
	const MapMulti& mapMR = reinterpret_cast<const MapMulti&>(map);

	if (mapTypeName() != map.mapTypeName())
	{
		CGoGNerr << "try to copy from incompatible type map" << CGoGNendl;
		return false;
	}

	// clear the map but do not insert boundary markers dart attribute
	GenericMap::init(false);


	// load attrib container
	for (unsigned int i = 0; i < NB_ORBITS; ++i)
		m_attribs[i].copyFrom(mapMR.m_attribs[i]);

	m_mrattribs.copyFrom(mapMR.m_mrattribs);
	m_mrCurrentLevel = mapMR.m_mrCurrentLevel;

	unsigned int nb = mapMR.m_mrNbDarts.size();
	m_mrNbDarts.resize(nb);
	for (unsigned int i = 0; i < nb; ++i)
		m_mrNbDarts[i] = mapMR.m_mrNbDarts[i];

	// restore shortcuts
	GenericMap::restore_shortcuts();
	restore_topo_shortcuts();

	return true;
}

void MapMulti::restore_topo_shortcuts()
{
	m_involution.clear();
	m_permutation.clear();
	m_permutation_inv.clear();

	m_involution.resize(getNbInvolutions());
	m_permutation.resize(getNbPermutations());
	m_permutation_inv.resize(getNbPermutations());

	std::vector<std::string> listeNames;
	m_attribs[DART].getAttributesNames(listeNames);

	for (unsigned int i = 0;  i < listeNames.size(); ++i)
	{
		std::string sub = listeNames[i].substr(0, listeNames[i].size() - 1);
		if (sub == "involution_")
		{
			unsigned int relNum = listeNames[i][11] - '0';
			AttributeMultiVector<Dart>* rel = getRelation(listeNames[i]);
			m_involution[relNum] = rel;
		}
		else if (sub == "permutation_")
		{
			unsigned int relNum = listeNames[i][12] - '0';
			AttributeMultiVector<Dart>* rel = getRelation(listeNames[i]);
			m_permutation[relNum] = rel;
		}
		else if (sub == "permutation_inv_")
		{
			unsigned int relNum = listeNames[i][16] - '0';
			AttributeMultiVector<Dart>* rel = getRelation(listeNames[i]);
			m_permutation_inv[relNum] = rel;
		}
	}

	std::vector<std::string> names;
	m_mrattribs.getAttributesNames(names);
	m_mrDarts.resize(names.size() - 1);
	for (unsigned int i = 0; i < m_mrDarts.size(); ++i)
		m_mrDarts[i] = NULL;

	for (unsigned int i = 0;  i < names.size(); ++i)
	{
		std::string sub = names[i].substr(0, 7);

		if (sub == "MRLevel")
			m_mrLevels = m_mrattribs.getDataVector<unsigned int>(i);

		if (sub == "MRdart_")
		{
			sub = names[i].substr(7);	// compute number following MRDart_
			unsigned int idx = 0;
			for (unsigned int j = 0; j < sub.length(); j++)
				idx = 10 * idx + (sub[j] - '0');
			if (idx < names.size() - 1)
				m_mrDarts[idx] = m_mrattribs.getDataVector<unsigned int>(i);
			else
				CGoGNerr << "Warning problem updating MR_DARTS" << CGoGNendl;
		}
	}

	// check if all pointers are != NULL
	for (unsigned int i = 0; i < m_mrDarts.size(); ++i)
	{
		if (m_mrDarts[i] == NULL)
			CGoGNerr << "Warning problem MR_DARTS = NULL" << CGoGNendl;
	}
}

} //namespace CGoGN
