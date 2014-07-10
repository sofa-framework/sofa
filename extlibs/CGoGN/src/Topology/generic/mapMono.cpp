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

#include "Topology/generic/mapImpl/mapMono.h"

namespace CGoGN
{

/****************************************
 *             SAVE & LOAD              *
 ****************************************/

bool MapMono::saveMapBin(const std::string& filename) const
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

    memcpy(buff, "CGoGN_Map", 10);

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

    return true;
}

bool MapMono::loadMapBin(const std::string& filename)
{
    CGoGNistream fs(filename.c_str(), std::ios::in|std::ios::binary);
    if (!fs)
    {
        CGoGNerr << "Unable to open file for loading" << CGoGNendl;
        return false;
    }

    GenericMap::clear(true);

    // read info
    char* buff = new char[256];
    fs.read(reinterpret_cast<char*>(buff), 256);

    std::string buff_str(buff);
    // Check file type
    if (buff_str == "CGoGN_MRMap")
    {
        CGoGNerr<< "Wrong binary file format, file is a MR-Map"<< CGoGNendl;
        return false;
    }
    if (buff_str != "CGoGN_Map")
    {
        CGoGNerr<< "Wrong binary file format"<< CGoGNendl;
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

    // restore shortcuts
    GenericMap::restore_shortcuts();
    restore_topo_shortcuts();

    return true;
}

bool MapMono::copyFrom(const GenericMap& map)
{

	if (mapTypeName() != map.mapTypeName())
	{
		CGoGNerr << "try to copy from incompatible type map" << CGoGNendl;
		return false;
	}

	const MapMono& mapM = reinterpret_cast<const MapMono&>(map);

	// clear the map but do not insert boundary markers dart attribute
	GenericMap::init(false);

	// copy attrib containers
	for (unsigned int i = 0; i < NB_ORBITS; ++i)
		m_attribs[i].copyFrom(mapM.m_attribs[i]);

	GenericMap::garbageMarkVectors();

	// restore shortcuts
	GenericMap::restore_shortcuts();
	restore_topo_shortcuts();

	return true;
}

void MapMono::restore_topo_shortcuts()
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
}

} //namespace CGoGN
