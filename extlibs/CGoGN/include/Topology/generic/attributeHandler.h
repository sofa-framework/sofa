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

#ifndef __ATTRIBUTE_HANDLER_H__
#define __ATTRIBUTE_HANDLER_H__

#include <vector>
#include <map>

#include "Topology/generic/genericmap.h"
#include "Container/fakeAttribute.h"

namespace CGoGN
{

class AttributeHandlerGen
{
protected:
	friend class GenericMap ;
	friend class AttribMap ;

	// the map that contains the linked attribute
	GenericMap* m_map ;
	// boolean that states the validity of the handler
	bool valid ;

public:
	AttributeHandlerGen(GenericMap* m, bool v) :
		m_map(m),
		valid(v)
	{}

	GenericMap* map() const
	{
		return m_map ;
	}

	bool isValid() const
	{
		return valid ;
	}

	virtual int getSizeOfType() const = 0;

	virtual unsigned int getOrbit() const = 0;

	virtual const std::string& name() const = 0;
	virtual const std::string& typeName() const = 0;

	virtual AttributeMultiVectorGen* getDataVectorGen() const = 0;

protected:
	void setInvalid()
	{
		valid = false ;
	}
} ;

/**
 * Class that create an access-table to an existing attribute
 * Main available operations are:
 * - [ index ]
 * - [ dart ]
 * - begin / end / next to manage indexing
 */
template <typename T, unsigned int ORBIT>
class AttributeHandler : public AttributeHandlerGen
{
protected:
	// the multi-vector that contains attribute data
	AttributeMultiVector<T>* m_attrib;

	void registerInMap() ;
	void unregisterFromMap() ;

public:
	typedef T DATA_TYPE ;

	/**
	 * Default constructor
	 * Constructs a non-valid AttributeHandler (i.e. not linked to any attribute)
	 */
	AttributeHandler() ;

	/**
	 * Constructor
	 * @param m the map which belong attribute
	 * @param amv a pointer to the AttributeMultiVector
	 */
	AttributeHandler(GenericMap* m, AttributeMultiVector<T>* amv) ;

	/**
	 * Copy constructor
	 * @param ta the table attribute
	 */
	AttributeHandler(const AttributeHandler<T, ORBIT>& ta) ;

	/**
	 * Transmute Constructor
	 * Construct an attribute of Orbit from Orbit2
	 * @param h the table attribute
	 */
	template <unsigned int ORBIT2>
	AttributeHandler(const AttributeHandler<T, ORBIT2>& h) ;

	/**
	 * affectation operator
	 * @param ta the table attribute to affect to this
	 */
	AttributeHandler<T, ORBIT>& operator=(const AttributeHandler<T, ORBIT>& ta) ;

	/**
	 * transmuted affectation operator
	 * @param ta the table attribute to affect to this
	 */
	template <unsigned int ORBIT2>
	AttributeHandler<T, ORBIT>& operator=(const AttributeHandler<T, ORBIT2>& ta) ;

	/**
	 * Destructor (empty & virtual)
	 */
	virtual ~AttributeHandler() ;

	/**
	 * get attribute data vector
	 */
	AttributeMultiVector<T>* getDataVector() const ;

	/**
	 * get attribute data vector (generic MultiVector)
	 */
	virtual AttributeMultiVectorGen* getDataVectorGen() const ;

	/**
	 * get size of attribute type
	 */
	virtual int getSizeOfType() const ;

	/**
	 * get attribute orbit
	 */
	virtual unsigned int getOrbit() const ;

	/**
	 * get attribute index
	 */
	unsigned int getIndex() const ;

	/**
	 * get attribute name
	 */
	virtual const std::string& name() const ;

	/**
	 * get attribute type name
	 */
	virtual const std::string& typeName() const ;

	/**
	 * give the number of elements of the attribute container
	 */
	unsigned int nbElements() const;

	/**
	 * [] operator with dart parameter
	 */
	T& operator[](Dart d) ;

	/**
	 * const [] operator with dart parameter
	 */
	const T& operator[](Dart d) const ;

	/**
	 * at operator (same as [] but with index parameter)
	 */
	T& operator[](unsigned int a) ;

	/**
	 * const at operator (same as [] but with index parameter)
	 */
	const T& operator[](unsigned int a) const ;

	/**
	 * insert an element (warning we add here a complete line in container)
	 */
	unsigned int insert(const T& elt) ;

	/**
	 * insert an element with default value (warning we add here a complete line in container)
	 */
	unsigned int newElt() ;

	/**
	 * initialize all the lines of the attribute with the given value
	 */
	void setAllValues(const T& v) ;

	/**
	 * begin of table
	 * @return the iterator of the begin of container
	 */
	unsigned int begin() const;

	/**
	 * end of table
	 * @return the iterator of the end of container
	 */
	unsigned int end() const;

	/**
	 * Next on iterator (equivalent to stl ++)
	 * @param iter iterator to
	 */
	void next(unsigned int& iter) const;
} ;

/**
 *  shortcut class for Vertex Attribute (Handler)
 */
template <typename T>
class DartAttribute : public AttributeHandler<T, DART>
{
public:
	DartAttribute() : AttributeHandler<T, DART>() {}
	DartAttribute(const AttributeHandler<T, DART>& ah) : AttributeHandler<T, DART>(ah) {}
	DartAttribute(GenericMap* m, AttributeMultiVector<T>* amv) : AttributeHandler<T, DART>(m,amv) {}
	DartAttribute<T>& operator=(const AttributeHandler<T, DART>& ah) { this->AttributeHandler<T, DART>::operator=(ah); return *this; }
};


/**
 *  shortcut class for Vertex Attribute (Handler)
 */
template <typename T>
class VertexAttribute : public AttributeHandler<T, VERTEX>
{
public:
	VertexAttribute() : AttributeHandler<T, VERTEX>() {}
	VertexAttribute(const AttributeHandler<T, VERTEX>& ah) : AttributeHandler<T, VERTEX>(ah) {}
	VertexAttribute(GenericMap* m, AttributeMultiVector<T>* amv) : AttributeHandler<T, VERTEX>(m,amv) {}
	VertexAttribute<T>& operator=(const AttributeHandler<T, VERTEX>& ah) { this->AttributeHandler<T, VERTEX>::operator=(ah); return *this; }
	VertexAttribute<T>& operator=(const AttributeHandler<T, EDGE>& ah) { this->AttributeHandler<T,VERTEX>::operator=(ah); return *this; }
	VertexAttribute<T>& operator=(const AttributeHandler<T, FACE>& ah) { this->AttributeHandler<T,VERTEX>::operator=(ah); return *this; }
	VertexAttribute<T>& operator=(const AttributeHandler<T, VOLUME>& ah) { this->AttributeHandler<T,VERTEX>::operator=(ah); return *this; }
};


/**
 *  shortcut class for Edge Attribute (Handler)
 */
template <typename T>
class EdgeAttribute : public AttributeHandler<T, EDGE>
{
public:
	EdgeAttribute() : AttributeHandler<T, EDGE>() {}
	EdgeAttribute(const AttributeHandler<T, EDGE>& ah) : AttributeHandler<T, EDGE>(ah) {}
	EdgeAttribute(GenericMap* m, AttributeMultiVector<T>* amv) : AttributeHandler<T, EDGE>(m,amv) {}
	EdgeAttribute<T>& operator=(const AttributeHandler<T, EDGE>& ah) { this->AttributeHandler<T, EDGE>::operator=(ah); return *this; }
};


/**
 *  shortcut class for Face Attribute (Handler)
 */
template <typename T>
class FaceAttribute : public AttributeHandler<T, FACE>
{
public:
	FaceAttribute() : AttributeHandler<T, FACE>() {}
	FaceAttribute(const AttributeHandler<T, FACE>& ah) : AttributeHandler<T, FACE>(ah) {}
	FaceAttribute(GenericMap* m, AttributeMultiVector<T>* amv) : AttributeHandler<T, FACE>(m,amv) {}
	FaceAttribute<T>& operator=(const AttributeHandler<T, FACE>& ah) { this->AttributeHandler<T, FACE>::operator=(ah); return *this; }
	FaceAttribute<T>& operator=(const AttributeHandler<T, VERTEX>& ah) { this->AttributeHandler<T,FACE>::operator=(ah); return *this; }
};


/**
 *  shortcut class for Volume Attribute (Handler)
 */
template <typename T>
class VolumeAttribute : public AttributeHandler<T, VOLUME>
{
public:
	VolumeAttribute() : AttributeHandler<T, VOLUME>() {}
	VolumeAttribute(const AttributeHandler<T, VOLUME>& ah) : AttributeHandler<T, VOLUME>(ah) {}
	VolumeAttribute(GenericMap* m, AttributeMultiVector<T>* amv) : AttributeHandler<T, VOLUME>(m,amv) {}
	VolumeAttribute<T>& operator=(const AttributeHandler<T, VOLUME>& ah) { this->AttributeHandler<T, VOLUME>::operator=(ah); return *this; }
	VolumeAttribute<T>& operator=(const AttributeHandler<T, VERTEX>& ah) { this->AttributeHandler<T, VOLUME>::operator=(ah); return *this; }
};


// turn_to<b>(A*</b> obj) changes class of the object
// that means it just replaces VTBL of the object by VTBL of another class.
// NOTE: these two classes has to be ABI compatible!
template <typename TO_T, typename FROM_T>
inline void turn_to(FROM_T* p)
{
assert(sizeof(FROM_T) == sizeof(TO_T));
::new(p) TO_T(); // use of placement new
}

} // namespace CGoGN

#include "Topology/generic/attributeHandler.hpp"

#endif
