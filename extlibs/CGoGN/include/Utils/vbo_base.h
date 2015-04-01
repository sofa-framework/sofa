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

#ifndef __CGoGN_GLSL_VBO__
#define __CGoGN_GLSL_VBO__


#include <vector>
#include "Utils/gl_def.h"
#include "Container/convert.h"
#include "Topology/generic/attributeHandler.h"
#include <GL/glew.h>

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class GLSLShader;

/**
 * Encapsulation of OpenGL Vertex Buffer Object
 * Manage
 * - alloc / release of GL buffer
 * - ref by Shaders
 * - size of data (invidual cells)
 */
class CGoGN_UTILS_API VBO
{
protected:
	// VBO id
	CGoGNGLuint m_id;

	// size of data (in floats)
	unsigned int m_data_size;

	// shaders that ref this vbo
	std::vector<GLSLShader*> m_refs;

	unsigned int m_nbElts;
	mutable bool m_lock;

	/// name of the last attribute used to fill the VBO
	std::string m_name;

	/// type name of the last attribute used to fill the VBO
	std::string m_typeName;

public:
	/**
	 * constructor: allocate the OGL VBO
	 */
	VBO(const std::string& name = "");

	/**
	 * copy constructor, new VBO copy content
	 */
	VBO(const VBO& vbo);

	/**
	 * destructor: release the OGL VBO and clean references between VBO/Shaders
	 */
	~VBO();

	/**
	 * get id of vbo
	 */
	inline GLuint id() const { return *m_id; }

	/**
	 * get dataSize
	 */
	inline unsigned int dataSize() const { return m_data_size; }

	/**
	 * get name
	 */
	inline const std::string& name() const { return m_name; }

	/**
	 * get type name
	 */
	inline const std::string& typeName() const { return m_typeName; }

	/**
	 * set the data size (in number of float)
	 */
	inline void setDataSize(unsigned int ds) { m_data_size = ds; }

	/**
	 * get nb element in vbo (vertices, colors ...)
	 */
	inline unsigned int nbElts() { return m_nbElts; }

	/**
	 * bind array vbo
	 */
	inline void bind() const  { glBindBuffer(GL_ARRAY_BUFFER, *m_id); }

	/**
	 * alloc buffer of same size than parameter
	 */
	void sameAllocSameBufferSize(const VBO& vbo);

	/**
	 * update data from attribute multivector to the vbo (automatic conversion if necessary and possible)
	 */
	void updateData(const AttributeMultiVectorGen* attrib);

	/**
	 * update data from attribute handler to the vbo
	 */
	inline void updateData(const AttributeHandlerGen& attrib)
	{
		updateData(attrib.getDataVectorGen()) ;
	}

	/**
	 * update data from given data vector
	 * @warning use only with include vbo.h (not vbo_base.h)
	 */
	template <typename T>
	void updateData(std::vector<T>& data);

	void* lockPtr();

	const void* lockPtr() const;

	void releasePtr() const;

	void copyData(void *ptr) const;

	void allocate(unsigned int nbElts);

	/**
	 * update the VBO from Attribute Handler of vectors with on the fly conversion
	 * template paramters:
	 * T_IN input type  attribute handler
	 * NB_COMPONENTS 3 for vbo of pos/normal, 2 for texture etc..
	 * @param attribHG the attribute handler source
	 * @param conv lambda or function/fonctor that take a const T_IN& and return a Vector<NB_COMPONENTS,float>
	 */
	template <typename T_IN, unsigned int NB_COMPONENTS, typename CONVFUNC>
	inline void updateDataConversion(const AttributeHandlerGen& attribHG, CONVFUNC conv)
	{
		const AttributeMultiVectorGen* attrib = attribHG.getDataVectorGen();
        updateDataConversion<T_IN,typename Geom::Vector<NB_COMPONENTS,float>::type,NB_COMPONENTS,CONVFUNC>(attrib,conv);
	}

	/**
	 * update the VBO from Attribute Handler of vectors with on the fly conversion
	 * template paramters:
	 * T_IN input type  attribute handler
	 * @param attribHG the attribute handler source
	 * @param conv lambda or function/fonctor that take a const T_IN& and return a Vector<NB_COMPONENTS,float>
	 */
	template <typename T_IN, typename CONVFUNC>
	inline void updateDataConversion(const AttributeHandlerGen& attribHG, CONVFUNC conv)
	{
		const AttributeMultiVectorGen* attrib = attribHG.getDataVectorGen();
		updateDataConversion<T_IN,float,1,CONVFUNC>(attrib,conv);
	}


protected:
	/**
	 * update the VBO from Attribute Handler with on the fly conversion
	 * template paramters:
	 * T_IN input type  attribute handler
	 * NB_COMPONENTS 3 for vbo of pos/normal, 2 for texture etc..
	 * @param attrib the attribute multivector source
	 * @param conv lambda function that take a const T_IN& and return a Vector<NB_COMPONENTS,float>
	 */
	template <typename T_IN, typename T_OUT, unsigned int NB_COMPONENTS, typename CONVFUNC>
	void updateDataConversion(const AttributeMultiVectorGen* attrib, CONVFUNC conv)
	{
		unsigned int old_nbb =  sizeof(float) * m_data_size * m_nbElts;
		m_name = attrib->getName();
		m_typeName = attrib->getTypeName();
		m_data_size = NB_COMPONENTS;

		// alloue la memoire pour le buffer et initialise le conv
		T_OUT* typedBuffer = new T_OUT[_BLOCKSIZE_];

		std::vector<void*> addr;
		unsigned int byteTableSize;
		unsigned int nbb = attrib->getBlocksPointers(addr, byteTableSize);

		m_nbElts = nbb * _BLOCKSIZE_/(sizeof(T_OUT));

		unsigned int offset = 0;
		unsigned int szb = _BLOCKSIZE_*sizeof(T_OUT);

		// bind buffer to update
		glBindBuffer(GL_ARRAY_BUFFER, *m_id);
		if (nbb!=old_nbb)
			glBufferData(GL_ARRAY_BUFFER, nbb * szb, 0, GL_STREAM_DRAW);

		for (unsigned int i = 0; i < nbb; ++i)
		{
			// convertit les donnees dans le buffer de conv
			const T_IN* typedIn = reinterpret_cast<const T_IN*>(addr[i]);
			T_OUT* typedOut = typedBuffer;
			// compute conversion
			for (unsigned int j = 0; j < _BLOCKSIZE_; ++j)
				*typedOut++ = conv(*typedIn++);

			// update sub-vbo
			glBufferSubData(GL_ARRAY_BUFFER, offset, szb, reinterpret_cast<void*>(typedBuffer));
			// block suivant
			offset += szb;
		}

		// libere la memoire de la conversion
		delete[] typedBuffer;

	}

};



} // namespace Utils

} // namespace CGoGN

#endif
