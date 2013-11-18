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

#ifndef __ATTRIB_MAP__
#define __ATTRIB_MAP__

#include "Topology/generic/genericmap.h"
#include "Topology/generic/attributeHandler.h"

namespace CGoGN
{

class AttribMap : public GenericMap
{
private:
	void init() ;

public:
	static const unsigned int UNKNOWN_ATTRIB = AttributeContainer::UNKNOWN ;

	AttribMap() ;

	virtual void clear(bool removeAttrib) ;

	/****************************************
	 *        ATTRIBUTES MANAGEMENT         *
	 ****************************************/

	/**
	* Create an attribute for a given orbit
	* @param nameAttr attribute name
	* @return an AttributeHandler
	*/
	template <typename T, unsigned int ORBIT>
	AttributeHandler<T, ORBIT> addAttribute(const std::string& nameAttr) ;

	/**
	 * remove an attribute
	 * @param attr a handler to the attribute to remove
	 * @return true if remove succeed else false
	 */
	template <typename T, unsigned int ORBIT>
	bool removeAttribute(AttributeHandler<T, ORBIT>& attr) ;

	/**
	* search an attribute for a given orbit
	* @param nameAttr attribute name
	* @return an AttributeHandler
	*/
	template <typename T, unsigned int ORBIT>
	AttributeHandler<T, ORBIT> getAttribute(const std::string& nameAttr) ;

	/**
	 * @brief get a generic pointer to an existing attribute multi vector
	 * @param orbit the concerned orbit
	 * @param nameAttr attribute name
	 * @return a pointer to an AttributeMultiVectorGen
	 */
	AttributeMultiVectorGen* getAttributeVectorGen(unsigned int orbit, const std::string& nameAttr) ;

	/**
	* check if an attribute exist ( get, test if valid and add if necessary)
	* @param nameAttr attribute name
	* @return an AttributeHandler
	*/
	template <typename T, unsigned int ORBIT>
	AttributeHandler<T, ORBIT> checkAttribute(const std::string& nameAttr) ;

	/**
	 * swap the content of two attributes (efficient, only swap pointers)
	 */
	template <typename T, unsigned int ORBIT>
	bool swapAttributes(AttributeHandler<T, ORBIT>& attr1, AttributeHandler<T, ORBIT>& attr2) ;

	/**
	 * copy the content of src attribute to dst attribute
	 */
	template <typename T, unsigned int ORBIT>
	bool copyAttribute(AttributeHandler<T, ORBIT>& dst, AttributeHandler<T, ORBIT>& src) ;


	/****************************************
	 *               UTILITIES              *
	 ****************************************/

	/**
	 * use the given attribute to store the indices of the cells of the corresponding orbit
	 * @return the number of cells of the orbit
	 */
	template <unsigned int ORBIT>
	unsigned int computeIndexCells(AttributeHandler<unsigned int, ORBIT>& idx) ;

	/**
	 * ensure that each orbit as one embedding and that each embedding is handle by only one orbit
	 */
	template <unsigned int ORBIT>
	void bijectiveOrbitEmbedding();

} ;

} // namespace CGoGN

#include "Topology/generic/attribmap.hpp"

#endif
