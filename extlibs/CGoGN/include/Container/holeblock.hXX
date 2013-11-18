#ifndef __HOLE_BLOCK_REF__
#define __HOLE_BLOCK_REF__

#include <fstream>

#include "sizeblock.h"


class HoleBlock
{
protected:
	typedef unsigned int uint;

	/**
	* Table of free index
	*/
	uint* m_tableFree;
	uint m_nbfree;

	/**
	* Reference counter (if 0 it is a hole)
	*/

	char* m_used;
	uint m_nbused;

	/**
	* nb element in block
	*/
	uint m_nb;

public:
	/**
	* constructor
	*/
	HoleBlock();

	/**
	* destructor
	*/
	~HoleBlock();

	/**
	 * swapping
	 */
	void swap(HoleBlock& hb);


	/**
	* add a element and return its index
	* @param nbEltsMax (IN/OUT) max number of element stored
	* @return index on new element
	*/
	unsigned int newElt(unsigned int& nbEltsMax);

	/**
	* remove an element
	*/
	void removeElt(unsigned int idx);

	/**
	* is the block full
	*/
	inline bool full() const { return m_nb == _BLOCKSIZE_; /*(m_refCount.size() - m_tableFree.size())>=_BLOCKSIZE_;*/ }

	/**
	*  is the block empty
	*/
	inline bool empty() const { return m_nb == 0;}

	/**
	* is this index used or not
	*/
	inline bool used(unsigned int i) const { /*std::cout << "Used["<<i<<"] ? ->"<<m_refCount[i]<<std::endl; */ return m_refCount[i]>0;}

	/**
	* use with caution: from compress only !
	*/
	inline void incNb() {m_nb++;}

	/**
	* use with caution: from compress only !
	*/
	inline void decNb() {m_nb--;}

	/**
	* return the size of table
	*/
	inline unsigned int sizeTable() { return m_nbref;}

	/**
	* compress the free value (use only in compress )
	* @return true if it is empty
	*/
	bool compressFree();

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
	void overwrite(int i, HoleBlockRef *bf, int j);


	bool updateHoles(unsigned int nb);

	void saveBin(std::ofstream& fs);

	bool loadBin(std::ifstream& fs);
};


#endif
