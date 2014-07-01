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

#ifndef __ATTRIBUTE_MULTI_VECTOR__
#define __ATTRIBUTE_MULTI_VECTOR__

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

#include <typeinfo>

#include "Container/sizeblock.h"

namespace CGoGN
{

class AttributeMultiVectorGen
{
protected:
	/**
	 * Name of the attribute
	 */
	std::string m_attrName;

	/**
	 * Name of the type of the attribute
	 */
	std::string m_typeName;

	/**
	 * orbit of the attribute
	 */
	unsigned int m_orbit;

	/**
	 * index of the attribute in its container
	 */
	unsigned int m_index;

public:
	AttributeMultiVectorGen(const std::string& strName, const std::string& strType);

	AttributeMultiVectorGen();

 	virtual ~AttributeMultiVectorGen();

 	virtual AttributeMultiVectorGen* new_obj() = 0;

	/**************************************
	 *             ACCESSORS              *
	 **************************************/

	/**
	* get / set orbit of the attribute
	*/
 	unsigned int getOrbit() const ;

 	void setOrbit(unsigned int id) ;

	/**
	* get / set index of the attribute
	*/
 	unsigned int getIndex() const ;

 	void setIndex(unsigned int id) ;

	/**
	* get / set name of the attribute
	*/
	const std::string& getName() const;

	void setName(const std::string& n);

	/**
	* get / set name of the type of the attribute
	*/
	const std::string& getTypeName() const;

	void setTypeName(const std::string& n);

	/**
	 * get block size
	 */
	unsigned int getBlockSize() const;

	/**************************************
	 *       MULTI VECTOR MANAGEMENT      *
	 **************************************/

	/**
	* add a block of data in the multi vector
	*/
	virtual void addBlock() = 0;

	/**
	* set the number of blocks
	*/
	virtual void setNbBlocks(unsigned int nbb) = 0;

	virtual unsigned int getNbBlocks() const = 0;

//	virtual void addBlocksBefore(unsigned int nbb) = 0;

	virtual bool copy(const AttributeMultiVectorGen* atmvg) = 0;

	virtual bool swap(AttributeMultiVectorGen* atmvg) = 0;

	virtual bool merge(const AttributeMultiVectorGen& att) = 0;

	/**
	* free the used memory
	*/
	virtual void clear() = 0;

	/**
	 * get size of type
	 */
	virtual int getSizeOfType() const = 0;

	/**************************************
	 *             DATA ACCESS            *
	 **************************************/

	virtual unsigned int getBlocksPointers(std::vector<void*>& addr, unsigned int& byteBlockSize) const = 0;

	/**************************************
	 *          LINES MANAGEMENT          *
	 **************************************/

	virtual void initElt(unsigned int id) = 0;

	virtual void copyElt(unsigned int dst, unsigned int src) = 0;

	virtual void swapElt(unsigned int id1, unsigned int id2) = 0;

	virtual void overwrite(unsigned int src_b, unsigned int src_id, unsigned int dst_b, unsigned int dst_id) = 0;


	/**************************************
	 *            SAVE & LOAD             *
	 **************************************/

	virtual void saveBin(CGoGNostream& fs, unsigned int id) = 0;

	static unsigned int loadBinInfos(CGoGNistream& fs, std::string& name, std::string& type);

	virtual bool loadBin(CGoGNistream& fs) = 0;

	static bool skipLoadBin(CGoGNistream& fs);

	/**
	 * lecture binaire
	 * @param fs filestream
	 */
	virtual void dump(unsigned int i) const = 0;
};


/***************************************************************************************************/
/***************************************************************************************************/


template <typename T>
class AttributeMultiVector : public AttributeMultiVectorGen
{
	/**
	* table of blocks of data pointers: vectors!
	*/
	std::vector<T*> m_tableData;

public:
	AttributeMultiVector(const std::string& strName, const std::string& strType);

	AttributeMultiVector();

	~AttributeMultiVector();

	AttributeMultiVectorGen* new_obj();

	/**************************************
	 *       MULTI VECTOR MANAGEMENT      *
	 **************************************/

	void addBlock();

	void setNbBlocks(unsigned int nbb);

	unsigned int getNbBlocks() const;

	void addBlocksBefore(unsigned int nbb);

	bool copy(const AttributeMultiVectorGen* atmvg);

	bool swap(AttributeMultiVectorGen* atmvg);

	bool merge(const AttributeMultiVectorGen& att);

	void clear();

	int getSizeOfType() const;

	/**************************************
	 *             DATA ACCESS            *
	 **************************************/

	/**
	 * get a reference on a elt
	 * @param i index of element
	 */
	T& operator[](unsigned int i);

	/**
	 * get a const reference on a elt
	 * @param i index of element
	 */
	const T& operator[](unsigned int i) const;

	/**
	 * Get the addresses of each block of data
	 */
	unsigned int getBlocksPointers(std::vector<void*>& addr, unsigned int& byteBlockSize) const;

	/**************************************
	 *          LINES MANAGEMENT          *
	 **************************************/

	void initElt(unsigned int id);

	void copyElt(unsigned int dst, unsigned int src);

	void swapElt(unsigned int id1, unsigned int id2);

	/**
	* swap two elements in container (useful for compact function)
	* @param src_b  block index of source element
	* @param src_id index in block of source element
	* @param dst_b  block index of destination element
	* @param dst_id index in block of destination element
	*/
	void overwrite(unsigned int src_b, unsigned int src_id, unsigned int dst_b, unsigned int dst_id);

	/**************************************
	 *       ARITHMETIC OPERATIONS        *
	 **************************************/

//	void affect(unsigned int i, unsigned int j);

//	void add(unsigned int i, unsigned int j);

//	void sub(unsigned int i, unsigned int j);

//	void mult(unsigned int i, double alpha);

//	void div(unsigned int i, double alpha);

//	void lerp(unsigned res, unsigned int i, unsigned int j, double alpha);

	/**************************************
	 *            SAVE & LOAD             *
	 **************************************/

	/**
	 * Sauvegarde binaire
	 * @param fs filestream
	 * @param id id of mv
	 */
	void saveBin(CGoGNostream& fs, unsigned int id);

	/**
	 * lecture binaire
	 * @param fs filestream
	 */
	bool loadBin(CGoGNistream& fs);

	/**
	 * lecture binaire
	 * @param fs filestream
	 */
	virtual void dump(unsigned int i) const;

};

} // namespace CGoGN

#include "attributeMultiVectorBool.hpp"
#include "attributeMultiVector.hpp"

#endif
