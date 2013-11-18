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
#include <cmath>

#include "Geometry/vector_gen.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"
#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{



inline DataPerFaceRender::DataPerFaceRender():
m_nbTris(0)
{
}

template<typename PFP, unsigned int ORBIT, typename T>
void
DataPerFaceRender::updateVBO(Utils::VBO& vboPosition, Utils::VBO& vboData, typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& positions, const AttributeHandler<T,ORBIT>& dataPerXXX)
{
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	std::vector<VEC3> buffer;
	buffer.reserve(16384);

	std::vector<T> bufferData;
	bufferData.reserve(16384);

	TraversorCell<typename PFP::MAP, FACE> traFace(map);

	for (Dart d=traFace.begin(); d!=traFace.end(); d=traFace.next())
	{
		Dart a = d;
		Dart b = map.phi1(a);
		Dart c = map.phi1(b);
		// loop to cut a polygon in triangle on the fly (works only with convex faces)
		do
		{
			buffer.push_back(positions[d]);
			bufferData.push_back(dataPerXXX[d]);
			buffer.push_back(positions[b]);
			bufferData.push_back(dataPerXXX[b]);
			buffer.push_back(positions[c]);
			bufferData.push_back(dataPerXXX[c]);
			b = c;
			c = map.phi1(b);
		} while (c != d);
	}

	m_nbTris = buffer.size()/3;

	vboPosition.setDataSize(3);
	vboPosition.allocate(buffer.size());
	VEC3* ptrPos = reinterpret_cast<VEC3*>(vboPosition.lockPtr());
	memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3));
	vboPosition.releasePtr();

	assert(sizeof(T) % sizeof(float) == 0 || !"DataPerFaceRender::updateVBO: VBO's only work with data types which are a multiple of 4 bytes") ;
	vboData.setDataSize(sizeof(T) / sizeof(float)); // Warning : works only for data types of 4 bytes (as do the VBO's)
	vboData.allocate(bufferData.size());
	T* ptrData = reinterpret_cast<T*>(vboData.lockPtr());
	memcpy(ptrData,&bufferData[0],bufferData.size()*sizeof(T));
	vboData.releasePtr();
}

inline void DataPerFaceRender::draw(Utils::GLSLShader* sh)
{
	sh->enableVertexAttribs();
	glDrawArrays(GL_TRIANGLES , 0 , m_nbTris*3 );
	sh->disableVertexAttribs();
}

}//end namespace VBO

}//end namespace Algo

}//end namespace Render

}//end namespace CGoGN
