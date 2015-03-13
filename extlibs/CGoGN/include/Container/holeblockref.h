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

#ifndef __HOLE_BLOCK_REF__
#define __HOLE_BLOCK_REF__

#include <fstream>
#include <iostream>
#include <assert.h>

#include "Container/sizeblock.h"

namespace CGoGN
{

class HoleBlockRef
{
protected:
	/**
	* Table of free index
	*/
	unsigned int* m_tableFree;
	unsigned int m_nbfree;

	/**
	* Reference counter (if 0 it is a hole)
	*/
	unsigned int* m_refCount;
	unsigned int m_nbref;

	/**
	* nb elements in block
	*/
	unsigned int m_nb;

public:
	/**
	* constructor
	*/
	HoleBlockRef();

	/**
	 * copy constructor
	 */
	HoleBlockRef(const HoleBlockRef& hb);

	/**
	* destructor
	*/
	~HoleBlockRef();

	/**
	 * swapping
	 */
	void swap(HoleBlockRef& hb);

	/**
	* add a element and return its index (refCount = 1)
	* @param nbEltsMax (IN/OUT) max number of element stored
	* @return index on new element
	*/
	unsigned int newRefElt(unsigned int& nbEltsMax);

	/**
	* remove an element
	*/
	inline void removeElt(unsigned int idx)
	{
		m_nb--;
		m_tableFree[m_nbfree++] = idx;
		m_refCount[idx] = 0;
	}

	/**
	* is the block full
	*/
	inline bool full() const { return m_nb == _BLOCKSIZE_;  }
//    inline bool hasFreeIndices() const { return m_nbfree > 0u ;}
	/**
	*  is the block empty
	*/
	inline bool empty() const { return m_nb == 0; }

	/**
	* is this index used or not
	*/
	inline bool used(unsigned int i) const { return m_refCount[i] != 0; }

	/**
	* use with caution: from compress only !
	*/
	inline void incNb() { m_nb++; }

	/**
	* use with caution: from compress only !
	*/
	inline void decNb() { m_nb--; }

	/**
	* return the size of table
	*/
	inline unsigned int sizeTable() { return m_nbref; }

	/**
	* compress the free value (use only in compress )
	* @return true if it is empty
	*/
	bool compressFree();


	inline void compressFull(unsigned int nb)
	{
		m_nbfree = 0;
		m_nbref = nb;
		m_nb = nb;
	}

	/**
	* clear the container of free block
	*/
	void clear();

	/**
	* overwrite a line with another (called from compact)
	* @param i index of line in the block
	* @param bf ptr on the block of other line
	* @param j index of the other line in bf
	*/
	void overwrite(unsigned int i, HoleBlockRef *bf, unsigned int j);

	/**
	* increment ref counter of element i
	*/
	inline void ref(unsigned int i)
	{
		m_refCount[i]++;
	}

	/**
	* decrement ref counter of element i
	* @return true if ref=0 and element has been destroyed
	*/
	inline bool unref(unsigned int i)
	{
//		assert(m_refCount[i] > 1);
		m_refCount[i]--;
		if (m_refCount[i] == 1)
		{
			removeElt(i);
			return true;
		}
		return false;
	}

	/**
	* set ref counter of element i with j
	*/
	inline void setNbRefs(unsigned int i, unsigned int nb) { m_refCount[i] = nb; }

	/**
	* number of references of element i
	* @return the number of references +1 (stored as n+1, 0 = not used, 1 used but not refs, ...)
	*/
	inline unsigned int nbRefs(unsigned int i) { return m_refCount[i]; }

	bool updateHoles(unsigned int nb);

//    bool updateHole(unsigned int indx);

	void saveBin(CGoGNostream& fs);

	bool loadBin(CGoGNistream& fs);
    void printTableFree();
    bool removeFromFreeElts(unsigned int nb);
	unsigned int* getTableFree(unsigned int & nb) {nb =m_nbfree; return m_tableFree;}
};

} // namespace CGoGN

#endif
