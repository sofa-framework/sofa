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

#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL3
{

// inline functions:
inline void MapRender::enableVertexAttrib(unsigned int index)
{
	m_usedAttributes[index] = true ;
}

inline void MapRender::disableVertexAttrib(unsigned int index)
{
	m_usedAttributes[index] = false ;
}

/**
 * enable a vertex attribute for rendering (updateDate automatically enable attrib)
 */
inline void MapRender::enableVertexAttrib(const std::string& name)
{
	std::map<std::string,unsigned int>::iterator it = m_attributebyName.find(name);
	if (it != m_attributebyName.end())
		enableVertexAttrib(it->second);
	else
		CGoGNerr <<"enableVertexAttrib: unknown attribute "<< name << CGoGNendl;
}

inline void MapRender::disableVertexAttrib(const std::string& name)
{
	std::map<std::string,unsigned int>::iterator it = m_attributebyName.find(name);
	if (it != m_attributebyName.end())
		disableVertexAttrib(it->second);
	else
		CGoGNerr <<"disableVertexAttrib: unknown attribute "<< name << CGoGNendl;
}

inline unsigned int MapRender::useVertexAttributeName(const std::string& name, const Utils::GLSLShader& sh)
{
	unsigned int vertex_attrib = 0;

	std::map<std::string,unsigned int>::iterator it = m_attributebyName.find(name);
	if (it == m_attributebyName.end())
	{
		vertex_attrib = m_nbVertexAttrib++;
		m_attributebyName.insert(std::pair<std::string,unsigned int>(name, vertex_attrib));
	}
	else
		vertex_attrib = it->second;

	sh.bindAttrib(vertex_attrib, name.c_str());

	return vertex_attrib;
}

template <typename ATTR_HANDLER>
void MapRender::updateData(unsigned int vertex_attrib, const ATTR_HANDLER& attrib, ConvertAttrib* conv)
{
	// choisit le bon buffer en fonction du paramètre upType
	unsigned int indexVBO = vertex_attrib + FIRST_ATTRIBUTE_BUFFER;

	if (! m_allocatedAttributes[vertex_attrib] )
	{
		glGenBuffersARB(1, &(m_VBOBuffers[indexVBO])) ;
		m_allocatedAttributes[vertex_attrib] = true ;
	}

	m_usedAttributes[vertex_attrib] = true ;
	m_AttributesDataSize[vertex_attrib] = sizeof(typename ATTR_HANDLER::DATA_TYPE) / sizeof(float);

	if (conv)
		fillBufferConvert(indexVBO, attrib, conv) ;
	else
		fillBufferDirect(indexVBO, attrib) ;
}

template <typename ATTR_HANDLER>
void MapRender::updateData(const std::string& name, const ATTR_HANDLER& attrib, ConvertAttrib* conv)
{
	unsigned int vertex_attrib = 0;

	std::map<std::string,unsigned int>::iterator it = m_attributebyName.find(name);
	if (it == m_attributebyName.end())
	{
		vertex_attrib = m_nbVertexAttrib++;
		m_attributebyName.insert(std::pair<std::string,unsigned int>(name,vertex_attrib));
		CGoGNerr << "warning update data with unknown name, adding vertex attribute"<< CGoGNendl;
	}
	else
		vertex_attrib = it->second;

	updateData<ATTR_HANDLER>(vertex_attrib, attrib, conv);
}

template <typename ATTR_HANDLER>
void MapRender::fillBufferDirect(unsigned int indexVBO, const ATTR_HANDLER& attrib)
{
	AttributeMultiVector<typename ATTR_HANDLER::DATA_TYPE>* mv = attrib.getDataVector() ;

	std::vector<void*> addr;
	unsigned int byteTableSize;
	unsigned int nbb = mv->getBlocksPointers(addr, byteTableSize);

	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[indexVBO]);
	glBufferDataARB(GL_ARRAY_BUFFER, nbb * byteTableSize, 0, GL_STREAM_DRAW);

	unsigned int offset = 0;

	for (unsigned int i = 0; i < nbb; ++i)
	{
		glBufferSubDataARB(GL_ARRAY_BUFFER, offset, byteTableSize, addr[i]);
		offset += byteTableSize;
	}
}

template <typename ATTR_HANDLER>
void MapRender::fillBufferConvert(unsigned int indexVBO, const ATTR_HANDLER& attrib, ConvertAttrib* conv)
{
	AttributeMultiVector<typename ATTR_HANDLER::DATA_TYPE>* mv = attrib.getDataVector() ;

	std::vector<void*> addr;
	unsigned int byteTableSize;
	unsigned int nbb = mv->getBlocksPointers(addr, byteTableSize);

	// alloue la memoire pour le buffer et initialise le conv
	conv->reserve(mv->getBlockSize());

	// bind buffer to update
	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[indexVBO]);
	glBufferDataARB(GL_ARRAY_BUFFER, nbb * conv->sizeBuffer(), 0, GL_STREAM_DRAW);

	unsigned int offset = 0;

	for (unsigned int i = 0; i < nbb; ++i)
	{
		// convertit les donnees dans le buffer de conv
		conv->convert(addr[i]);
		// update sub-vbo
		glBufferSubDataARB(GL_ARRAY_BUFFER, offset, conv->sizeBuffer(), conv->buffer());
		// block suivant
		offset += conv->sizeBuffer();
	}

	// libere la memoire de la conversion
	conv->release();
}

template<typename PFP>
inline void MapRender::addTri(typename PFP::MAP& map, Dart d, std::vector<GLuint>& tableIndices)
{
	Dart a = d;
	Dart b = map.phi1(a);
	Dart c = map.phi1(b);

	// loop to cut a polygon in triangle on the fly (works only with convex faces)
	do
	{
		tableIndices.push_back(map.getEmbedding(d, VERTEX));
		tableIndices.push_back(map.getEmbedding(b, VERTEX));
		tableIndices.push_back(map.getEmbedding(c, VERTEX));
		b = c;
		c = map.phi1(b);
	} while (c != d);
}


template<typename PFP>
void MapRender::initTriangles(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
	DartMarker m(map,thread);
	tableIndices.reserve(4*map.getNbDarts()/3);

	for(Dart dd = map.begin(); dd != map.end(); map.next(dd))
	{
		if(!m.isMarked(dd))
		{
			addTri<PFP>(map, dd, tableIndices);
			m.markOrbit<FACE>(dd);
		}
	}
}

template<typename PFP>
void MapRender::initTrianglesOptimized(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
#define LIST_SIZE 20
	DartMarker m(map,thread);
	// reserve memory for triangles ( nb indices == nb darts )
	// and a little bit more
	// if lots of polygonal faces, realloc is done by vector
	tableIndices.reserve(4*map.getNbDarts()/3);

	for (Dart dd = map.begin(); dd != map.end(); map.next(dd))
	{
		if (!m.isMarked(dd))
		{
			std::list<Dart> bound;

			addTri<PFP>(map,dd,tableIndices);
			m.markOrbit<FACE>(dd);
			bound.push_back(dd);
			int nb = 1;
			do
			{
				Dart e = bound.back();
				Dart ee = e;
				do
				{
					Dart f = ee;
					do
					{
						if (!m.isMarked(f))
						{
							addTri<PFP>(map, f, tableIndices);
							m.markOrbit<FACE>(f);
							bound.push_back(map.phi1(f));
							++nb;
							if (nb > LIST_SIZE)
							{
								bound.pop_front();
								--nb;
							}
						}
						f = map.phi1(map.phi2(f));
					} while (f != ee);
					ee = map.phi1(ee);
				} while (ee != e);

				bound.pop_back();
				--nb;
			} while (!bound.empty());
		}
	}
#undef LIST_SIZE
}

template<typename PFP>
void MapRender::initLines(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
	DartMarker m(map,thread);
	tableIndices.reserve(map.getNbDarts());

	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!m.isMarked(d))
		{
			tableIndices.push_back(map.getEmbedding(d, VERTEX));
			tableIndices.push_back(map.getEmbedding(map.phi2(d), VERTEX));
			m.markOrbit<EDGE>(d);
		}
	}
}

template<typename PFP>
void MapRender::initLinesOptimized(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
#define LIST_SIZE 20

	DartMarker m(map,thread);

	// reserve memory for edges indices ( nb indices == nb darts)
	tableIndices.reserve(map.getNbDarts());

	for (Dart dd = map.begin(); dd != map.end(); map.next(dd))
	{
		if (!m.isMarked(dd))
		{
			std::list<Dart> bound;
			bound.push_back(dd);
			int nb = 1;
			do
			{
				Dart e = bound.back();
				Dart ee = e;
				do
				{
					Dart f = map.phi2(ee);
					if (!m.isMarked(ee))
					{
						tableIndices.push_back(map.getEmbedding(ee, VERTEX));
						tableIndices.push_back(map.getEmbedding(map.phi1(ee), VERTEX));
						m.markOrbit<EDGE>(f);

						bound.push_back(f);
						++nb;
						if (nb > LIST_SIZE)
						{
							bound.pop_front();
							--nb;
						}
					}
					ee = map.phi1(f);
				} while (ee != e);
				bound.pop_back();
				--nb;
			} while (!bound.empty());
		}
	}
#undef LIST_SIZE
}

template<typename PFP>
void MapRender::initPoints(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
	CellMarker m(map, VERTEX,thread) ;
	tableIndices.reserve(map.getNbDarts()/5);

	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!m.isMarked(d))
		{
			tableIndices.push_back(map.getEmbedding(d, VERTEX));
			m.mark(d) ;
		}
	}
}

template<typename PFP>
void MapRender::initPrimitives(typename PFP::MAP& map, int prim, bool optimized, unsigned int thread)
{
	std::vector<GLuint> tableIndices;

	// indice du VBO a utiliser
	int vbo_ind = 0;


	switch(prim)
	{
		case FLAT_TRIANGLES:
			break;
		case TRIANGLES:
			if(optimized)
				initTrianglesOptimized<PFP>(map,tableIndices,thread);
			else
				initTriangles<PFP>(map,tableIndices,thread) ;
			m_nbIndicesTri = tableIndices.size();
			vbo_ind = m_VBOBuffers[TRIANGLE_INDICES];
			break;
		case LINES:
			if(optimized)
				initLinesOptimized<PFP>(map,tableIndices,thread);
			else
				initLines<PFP>(map,tableIndices,thread) ;
			m_nbIndicesLines = tableIndices.size();
			vbo_ind = m_VBOBuffers[LINE_INDICES];
			break;
		case POINTS:
			initPoints<PFP>(map,tableIndices,thread);
			m_nbIndicesPoints = tableIndices.size();
			vbo_ind = m_VBOBuffers[POINT_INDICES];
			break;
		default:
			CGoGNerr << "problem initializing VBO indices" << CGoGNendl;
			break;
	}
	unsigned int size = tableIndices.size();

	// setup du buffer d'indices
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, vbo_ind);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, size*sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);
}




template<typename PFP>
void MapRender::initFlatTriangles( typename PFP::MAP& map, unsigned int vertex_attrib_position, unsigned int thread)
{
	std::vector<Geom::Vec3f> tableFlat;
	tableFlat.reserve(3*map.getNbDarts()); // 3 in case of polygonal faces (less chance of realloc, but bigger allocation)

	// map VBO of points for vertices positions
	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[vertex_attrib_position+FIRST_ATTRIBUTE_BUFFER]);
	Geom::Vec3f* tablePos = reinterpret_cast<Geom::Vec3f*>(glMapBuffer(GL_ARRAY_BUFFER,GL_READ_ONLY));

	m_nbFlatElts=0;
	// traversal of map for creating buffers
	DartMarker m(map,thread);
	for(Dart dd = map.begin(); dd != map.end(); map.next(dd))
	{
		if(!m.isMarked(dd))
		{
			Dart a = dd;
			Dart b = map.phi1(a);
			Dart c = map.phi1(b);

			Geom::Vec3f& P = tablePos[map.getEmbedding(a, VERTEX)];
			Geom::Vec3f& Q = tablePos[map.getEmbedding(b, VERTEX)];
			Geom::Vec3f& R = tablePos[map.getEmbedding(c, VERTEX)];

			Geom::Vec3f U = Q-P;
			Geom::Vec3f V = R-P;
			Geom::Vec3f N = U^V;
			N.normalize();

			// loop to cut a polygon in triangle on the fly (works only with convex faces)
			do
			{
				tableFlat.push_back(tablePos[map.getEmbedding(a, VERTEX)]);
				tableFlat.push_back(N);
				tableFlat.push_back(tablePos[map.getEmbedding(b, VERTEX)]);
				tableFlat.push_back(N);
				tableFlat.push_back(tablePos[map.getEmbedding(c, VERTEX)]);
				tableFlat.push_back(N);
				b = c;
				c = map.phi1(b);
			} while (c != dd);
			m.markOrbit<FACE>(dd);
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_nbFlatElts = tableFlat.size()/2;

	//creating VBO for flat
	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[FLAT_BUFFER]);
	glBufferDataARB(GL_ARRAY_BUFFER, tableFlat.size() * sizeof(Geom::Vec3f), (char*)(&(tableFlat[0])), GL_STREAM_DRAW);

}





} // namespace VBO

} // namespace Render

} // namespace Algo

} // namespace CGoGN
