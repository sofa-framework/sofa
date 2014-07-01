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

#ifndef __ATTRIBUTE_CONTAINER__
#define __ATTRIBUTE_CONTAINER__

#include "Container/sizeblock.h"
#include "Container/holeblockref.h"
#include "Container/attributeMultiVector.h"

#include <vector>
#include <map>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#include <libxml/parser.h>

namespace CGoGN
{

class RegisteredBaseAttribute;
class AttributeContainer;


class ContainerBrowser
{
public:
	virtual unsigned int begin() const = 0;
	virtual unsigned int end() const = 0;
	virtual void next(unsigned int &it) const = 0;
	virtual void enable() = 0;
	virtual void disable() = 0;
};

/**
 * Container for AttributeMultiVectors
 * All the attributes always have the same size and
 * the management of holes is shared by all attributes
 */
class AttributeContainer
{
public:
	/**
	* constante d'attribut inconnu
	*/
	static const unsigned int UNKNOWN = 0xffffffff;

	/**
	* Taille du bloc
	*/
//	static const unsigned int BlockSize = _BLOCKSIZE_;

protected:
	/**
	* vector of pointers to AttributeMultiVectors
	*/
	std::vector<AttributeMultiVectorGen*> m_tableAttribs;

	/**
	 * vector of free indices in the vector of AttributeMultiVectors
	 */
	std::vector<unsigned int> m_freeIndices;

	/**
	* vector of pointers to HoleBlockRef -> structure that manages holes and refs
	*/
	std::vector<HoleBlockRef*> m_holesBlocks;

	/**
	* vector of indices of blocks that have free space
	*/
	std::vector<unsigned int> m_tableBlocksWithFree;

	/**
	* vector of indices of blocks that are empty
	*/
	std::vector<unsigned int> m_tableBlocksEmpty;

	ContainerBrowser* m_currentBrowser;

	/**
	 * orbit of the container
	 */
	unsigned int m_orbit;

	/**
	* number of attributes
	*/ 
	unsigned int m_nbAttributes;

	/**
	 * counter for attributes without name
	 */
	unsigned int m_nbUnknown;

	/**
	* size (number of elts) of the container
	*/
	unsigned int m_size;

	/**
	* size of the container with holes
	*/
	unsigned int m_maxSize;

	/**
	* memory cost of each line
	*/
	unsigned int m_lineCost;

	/**
	 * map pointer (shared for all container of the same map) for attribute registration
	 */
	std::map<std::string, RegisteredBaseAttribute*>* m_attributes_registry_map;

public:
	AttributeContainer();

	~AttributeContainer();

	unsigned int getOrbit() const;

	void setOrbit(unsigned int orbit);

	void setRegistry(std::map<std::string, RegisteredBaseAttribute*>* re);

	void setContainerBrowser(ContainerBrowser* bro) { m_currentBrowser = bro;}

	/**************************************
	 *          BASIC FEATURES            *
	 **************************************/

	/**
	 * add a new attribute to the container
	 * @param T (template) type of the new attribute
	 * @param attribName name of the new attribute
	 * @return pointer to the new AttributeMultiVector
	 */
	template <typename T>
	AttributeMultiVector<T>* addAttribute(const std::string& attribName);

protected:
	/**
	 * add a new attribute with a given index (for load only)
	 * @param T (template) type of the new attribute
	 * @param attribName name of the new attribute
	 * @param typeName name of the new attribute's type
	 * @param index index of the new attribute
	 */
	template <typename T>
	void addAttribute(const std::string& attribName, const std::string& typeName, unsigned int index);

public:
	/**
	* Remove an attribute (destroys data)
	* @param attribName name of the attribute to remove
	* @return removed or not
	*/
	template <typename T>
	bool removeAttribute(const std::string& attribName);

	/**
	* Remove an attribute (destroys data)
	* @param index index of the attribute to remove
	* @return removed or not
	*/
	template <typename T>
	bool removeAttribute(unsigned int index);

	/**************************************
	 *      INFO ABOUT THE CONTAINER      *
	 **************************************/

	/**
	 * Number of attributes of the container
	 */
	unsigned int getNbAttributes() const;

	/**
	* Size of the container (number of lines)
	*/
	unsigned int size() const;

	/**
	* Capacity of the container (number of lines including holes)
	*/
	unsigned int capacity() const;

	/**
	* Total memory cost of container
	*/
	unsigned int memoryTotalSize() const;

	/**
	* Memory cost of every used line
	*/
	unsigned int memorySize() const;

	/**
	* is the line used in the container
	*/
	inline bool used(unsigned int index) const;

	/**************************************
	 *         CONTAINER TRAVERSAL        *
	 **************************************/

	/**
	 * return the index of the first line of the container
	 */
	unsigned int begin() const;

	/**
	 * return the index of the last line of the container
	 */
	unsigned int end() const;

	/**
	 * get the index of the line after it in the container
	 * MUST BE USED INSTEAD OF ++ !
	 */
	void next(unsigned int &it) const;



	/**
	 * return the index of the first line of the container
	 */
    inline unsigned int realBegin() const;

	/**
	 * return the index of the last line of the container
	 */
    inline unsigned int realEnd() const;

	/**
	 * get the index of the line after it in the container
	 * MUST BE USED INSTEAD OF ++ !
	 */
    inline void realNext(unsigned int &it) const;


	/**************************************
	 *       INFO ABOUT ATTRIBUTES        *
	 **************************************/

	/**
	* recuperation du code d'un attribut
	* @param attribName nom de l'attribut
	* @return l'indice de l'attribut
	*/
	unsigned int getAttributeIndex(const std::string& attribName);

	/**
	 * get the name of an attribute, given its index in the container
	 */
	const std::string& getAttributeName(unsigned int attrIndex) const;

	/**
	 * fill a vector with pointers to the blocks of the given attribute
	 * @param attrIndex index of the attribute
	 * @param vect_addr (OUT) vector of pointers
	 * @param byteBlockSize (OUT) size in bytes of each block
	 * @return number of blocks
	 */
	template<typename T>
	unsigned int getAttributeBlocksPointers(unsigned int attrIndex, std::vector<T*>& vect_ptr, unsigned int& byteBlockSize);

	/**
	 * fill a vector with attributes names
	 * @param names vector of names
	 * @return number of attributes
	 */
	unsigned int getAttributesNames(std::vector<std::string>& names) const;

	/**
	 * fill a vector with attribute type names
	 * @param types vector of type names
	 * @return number of attributes
	 */
	unsigned int getAttributesTypes(std::vector<std::string>& types);

	/**************************************
	 *        CONTAINER MANAGEMENT        *
	 **************************************/

	/**
	 * swap two containers
	 */
	void swap(AttributeContainer& cont);

	/**
	 * clear the container
	 * @param removeAttrib remove the attributes (not only their data)
	 */
	void clear(bool clearAttrib = false);

	/**
	 * container compacting
	 * @param mapOldNew table that contains a map from old indices to new indices (holes -> 0xffffffff)
	 */
	void compact(std::vector<unsigned int>& mapOldNew);

	/**************************************
	 *          LINES MANAGEMENT          *
	 **************************************/

    void printFreeIndices();
    void updateHole(unsigned int index);
	/**
	* insert a line in the container
	* @return index of the line
	*/
	unsigned int insertLine();

	/**
	* remove a line in the container
	* @param index index of the line to remove
	*/
	void removeLine(unsigned int index);

	/**
	 * initialize a line of the container (an element of each attribute)
	 */
	void initLine(unsigned int index);

	/**
	 * copy the content of line src in line dst
	 */
	void copyLine(unsigned int dstIndex, unsigned int srcIndex);

	/**
	* increment the ref counter of the given line
	* @param index index of the line
	*/
	void refLine(unsigned int index);

	/**
	* decrement the ref counter of the given line
	* @param index index of the line
	* @return true if the line was removed
	*/
	bool unrefLine(unsigned int eltIdx);

	/**
	* get the number of refs of the given line
	* @param index index of the line
	* @return number of refs of the line
	*/
    unsigned int getNbRefs(unsigned int index) const;

	/**
	* set the number of refs of the given line
	* @param index index of the line
	* @param nb number of refs
	*/
	void setNbRefs(unsigned int eltIdx, unsigned int nb);

	/**************************************
	 *       ATTRIBUTES MANAGEMENT        *
	 **************************************/

	/**
	 * copy the data of attribute src in attribute dst (type has to be the same)
	 */
	bool copyAttribute(unsigned int dstIndex, unsigned int srcIndex);

	/**
	 * swap the data of attribute 1 with attribute 2 (type has to be the same)
	 */
	bool swapAttributes(unsigned int index1, unsigned int index2);

	/**************************************
	 *       ATTRIBUTES DATA ACCESS       *
	 **************************************/

	/**
	* get an AttributeMultiVector
	* @param attrIndex index of the attribute
	*/
	template<typename T>
	AttributeMultiVector<T>* getDataVector(unsigned int attrIndex);

	AttributeMultiVectorGen* getVirtualDataVector(unsigned int attrIndex);

	/**
	* get an AttributeMultiVector
	* @param attribName name of the attribute
	*/
	template<typename T>
	AttributeMultiVector<T>* getDataVector(const std::string& attribName);

	AttributeMultiVectorGen* getVirtualDataVector(const std::string& attribName);

	/**
	* get a given element of a given attribute
	* @param T type of the attribute
	* @param attrIndex index of the attribute
	* @param eltIndex index of the element
	* @return a reference on the element
	*/
	template <typename T>
	T& getData(unsigned int attrIndex, unsigned int eltIndex);

	/**
	* get a given const element of a given attribute
	* @param T type of the attribute
	* @param attrIndex index of the attribute
	* @param eltIndex index of the element
	* @return a const reference on the element
	*/
	template <typename T>
	const T& getData(unsigned int attrIndex, unsigned int eltIndex) const;

	/**
	* set a given element of a given attribute
	* @param T type of the attribute
	* @param attrIndex index of the attribute
	* @param eltIndex index of the element
	* @param data data to insert
	*/
	template <typename T>
	void setData(unsigned int attrIndex, unsigned int eltIndex, const T& data);



	/**************************************
	 *            SAVE & LOAD             *
	 **************************************/

public:
	/**
	* save binary file
	* @param fs a file stream
	* @param id the id to save
	*/
	void saveBin(CGoGNostream& fs, unsigned int id) const;

	/**
	* get id from file binary stream
	* @param fs file stream
	* @return the id of attribute container
	*/
	static unsigned int loadBinId(CGoGNistream& fs);

	/**
	* load from binary file
	* @param fs a file stream
	* @param id  ??
	*/
	bool loadBin(CGoGNistream& fs);

	/**
	 * copy container
	 * TODO a version that compact on the fly ?
	 */
	void copyFrom(const AttributeContainer& cont);
	/**
	 * dump the container in CSV format (; separated columns)
	 */
	void dumpCSV() const;

	void dumpByLines() const;

    void removeFromFreeIndices(unsigned int index);
};

} // namespace CGoGN

#include "attributeContainer.hpp"

#endif
