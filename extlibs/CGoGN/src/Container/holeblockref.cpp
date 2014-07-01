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

#include "Container/holeblockref.h"

#include <map>
#include <string>
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#include <libxml/parser.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

namespace CGoGN
{

HoleBlockRef::HoleBlockRef() : m_nbfree(0), m_nbref(0), m_nb(0)
{
	m_tableFree = new unsigned int[_BLOCKSIZE_ + 10];
	m_refCount = new unsigned int[_BLOCKSIZE_];
}

HoleBlockRef::HoleBlockRef(const HoleBlockRef& hb)
{
	m_nbfree = hb.m_nbfree;
	m_nbref = hb.m_nbref;
	m_nb = hb.m_nb;

	m_tableFree = new unsigned int[_BLOCKSIZE_ + 10];
	memcpy(m_tableFree, hb.m_tableFree, (_BLOCKSIZE_ + 10) * sizeof(unsigned int));

	m_refCount = new unsigned int[_BLOCKSIZE_];
	memcpy(m_refCount, hb.m_refCount, _BLOCKSIZE_ * sizeof(unsigned int));
}

HoleBlockRef::~HoleBlockRef()
{
	delete[] m_tableFree;
	delete[] m_refCount;
}

void HoleBlockRef::swap(HoleBlockRef& hb)
{
	unsigned int temp = m_nbfree;
	m_nbfree = hb.m_nbfree;
	hb.m_nbfree = temp;

	temp = m_nbref;
	m_nbref = hb.m_nbref;
	hb.m_nbref = temp;

	temp = m_nb;
	m_nb = hb.m_nb;
	hb.m_nb = temp;

	unsigned int* ptr = m_tableFree;
	m_tableFree = hb.m_tableFree;
	hb.m_tableFree = ptr;

	unsigned int* ptr2 = m_refCount;
	m_refCount = hb.m_refCount;
	hb.m_refCount = ptr2;
}

unsigned int HoleBlockRef::newRefElt(unsigned int& nbEltsMax)
{
	// no hole then add a line at the end of block
	if (m_nbfree == 0)
	{
		unsigned int nbElts = m_nbref;

 		m_refCount[m_nbref++] = 1;

		m_nb++;
		nbEltsMax++;
		return nbElts;
	}

	unsigned int index = m_tableFree[--m_nbfree];

	m_refCount[index] = 1;

	m_nb++;
	return index;
}

bool  HoleBlockRef::compressFree()
{
	if (m_nb)
	{
		m_nbfree = 0;
		m_nbref = m_nb;
		return false;
	}
	return true;
}

void HoleBlockRef::overwrite(unsigned int i, HoleBlockRef *bf, unsigned int j)
{
	m_refCount[i] = bf->m_refCount[j];
	bf->m_refCount[j] = 0;

	incNb();
	bf->decNb();
}

void HoleBlockRef::clear()
{
	m_nb = 0;
	m_nbfree = 0;
	m_nbref = 0;
}

bool HoleBlockRef::updateHoles(unsigned int nb)
{
	m_nbfree = 0;
	m_nbref = nb;
	bool notfull = false;
	for (unsigned int i = 0; i < nb; ++i)
	{
		if (m_refCount[i] == 0)
		{
			m_tableFree[m_nbfree++] = i;
			notfull = true;
		}
	}
	return notfull;
}


bool  HoleBlockRef::removeFromFreeElts(unsigned int nb) {
    unsigned int* end = m_tableFree + m_nbfree ;
    unsigned int * elt = NULL;
    elt = std::find(m_tableFree, end, nb);
    if (elt == end) {
        std::cerr << "WARNING : " << __FILE__ << ":" << __LINE__ << std::endl;
        return false;
    }
    else {
        ++m_nb ;
        std::swap(*elt, m_tableFree[--m_nbfree]);
        return true;
    }
}

void HoleBlockRef::printTableFree()
{
    if (m_nbfree > 0u) {
        std::cerr << "m_tableFree : " ;
        for (int i = 0 ; i < m_nbfree ; ++i)
            std::cerr << m_tableFree[i] << " " ;
        std::cerr << std::endl;
    }
}

bool HoleBlockRef::updateHole(unsigned int indx) {
    if (!used(indx)) {
//        std::cerr << "updateHole working : index " << indx << std::endl;
        unsigned int* end = m_tableFree + m_nbfree ;
        // first check if the index is already present
//        printTableFree();
        if (std::find(m_tableFree, end, indx) == end) {
            m_tableFree[m_nbfree++] = indx ;
            --m_nb ;
        }
        return true;
    }
    return false;
}


void HoleBlockRef::saveBin(CGoGNostream& fs)
{
//	CGoGNout << "save bf "<< m_nb<< " / "<< m_nbref<< " / "<< m_nbfree << CGoGNendl;

	// on sauve les trois nombres;
	unsigned int numbers[3];
	numbers[0] = m_nb;
	numbers[1] = m_nbref;
	numbers[2] = m_nbfree;
	fs.write(reinterpret_cast<const char*>(numbers), 3*sizeof(unsigned int) );

	// sauve les ref count
	fs.write(reinterpret_cast<const char*>(m_refCount), _BLOCKSIZE_*sizeof(unsigned int));

	// sauve les free lines
	fs.write(reinterpret_cast<const char*>(m_tableFree), m_nbfree*sizeof(unsigned int));
}

bool HoleBlockRef::loadBin(CGoGNistream& fs)
{
	unsigned int numbers[3];

	fs.read(reinterpret_cast<char*>(numbers), 3*sizeof(unsigned int));
	m_nb = numbers[0];
	m_nbref = numbers[1];
	m_nbfree = numbers[2];

	fs.read(reinterpret_cast<char*>(m_refCount), _BLOCKSIZE_*sizeof(unsigned int));
	fs.read(reinterpret_cast<char*>(m_tableFree), m_nbfree*sizeof(unsigned int));

	return true;
}

} // namespace CGoGN
