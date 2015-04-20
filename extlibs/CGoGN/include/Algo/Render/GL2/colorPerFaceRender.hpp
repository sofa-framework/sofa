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



inline ColorPerFaceRender::ColorPerFaceRender():
m_nbTris(0)
{
}



template<typename PFP, unsigned int ORBIT>
void ColorPerFaceRender::updateVBO(Utils::VBO& vboPosition, Utils::VBO& vboColor, typename PFP::MAP& map,
            const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const AttributeHandler<typename PFP::VEC3,ORBIT, typename PFP::MAP>& colorPerXXX)
{
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;
	typedef Geom::Vec3f VEC3F;

	std::vector<VEC3F> buffer;
	buffer.reserve(16384);

	std::vector<VEC3F> bufferColors;
	bufferColors.reserve(16384);

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
			bufferColors.push_back(colorPerXXX[d]);
			buffer.push_back(positions[b]);
			bufferColors.push_back(colorPerXXX[b]);
			buffer.push_back(positions[c]);
			bufferColors.push_back(colorPerXXX[c]);
			b = c;
			c = map.phi1(b);
		} while (c != d);
	}

	m_nbTris = buffer.size()/3;

	vboPosition.setDataSize(3);
	vboPosition.allocate(buffer.size());
	VEC3F* ptrPos = reinterpret_cast<VEC3F*>(vboPosition.lockPtr());
	memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));
	vboPosition.releasePtr();

	vboColor.setDataSize(3);
	vboColor.allocate(bufferColors.size());
	VEC3F* ptrCol = reinterpret_cast<VEC3F*>(vboColor.lockPtr());
	memcpy(ptrCol,&bufferColors[0],bufferColors.size()*sizeof(VEC3F));
	vboColor.releasePtr();
}

template<typename PFP, unsigned int ORBIT>
void ColorPerFaceRender::updateVBO(Utils::VBO& vboPosition, Utils::VBO& vboNormal, Utils::VBO& vboColor, typename PFP::MAP& map,
            const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normals,
            const AttributeHandler<typename PFP::VEC3,ORBIT, typename PFP::MAP>& colorPerXXX)
{
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;
	typedef Geom::Vec3f VEC3F;

	std::vector<VEC3F> buffer;
	buffer.reserve(16384);

	std::vector<VEC3F> bufferNormals;
	bufferNormals.reserve(16384);

	std::vector<VEC3F> bufferColors;
	bufferColors.reserve(16384);

	TraversorCell<typename PFP::MAP, FACE> traFace(map);

	for (Dart d=traFace.begin(); d!=traFace.end(); d=traFace.next())
	{
		Dart a = d;
		Dart b = map.phi1(a);
		Dart c = map.phi1(b);
		// loop to cut a polygon in triangle on the fly (works only with convex faces)
		do
		{
			buffer.push_back(PFP::toVec3f(positions[d]));
			bufferNormals.push_back(PFP::toVec3f(normals[d]));
			bufferColors.push_back(PFP::toVec3f(colorPerXXX[d]));
			buffer.push_back(PFP::toVec3f(positions[b]));
			bufferNormals.push_back(PFP::toVec3f(normals[b]));
			bufferColors.push_back(PFP::toVec3f(colorPerXXX[b]));
			buffer.push_back(PFP::toVec3f(positions[c]));
			bufferNormals.push_back(PFP::toVec3f(normals[c]));
			bufferColors.push_back(PFP::toVec3f(colorPerXXX[c]));
			b = c;
			c = map.phi1(b);
		} while (c != d);
	}

	m_nbTris = buffer.size()/3;

	vboPosition.setDataSize(3);
	vboPosition.allocate(buffer.size());
	VEC3F* ptrPos = reinterpret_cast<VEC3F*>(vboPosition.lockPtr());
	memcpy(ptrPos, &buffer[0], buffer.size()*sizeof(VEC3F));
	vboPosition.releasePtr();

	vboNormal.setDataSize(3);
	vboNormal.allocate(bufferColors.size());
	VEC3F* ptrNorm = reinterpret_cast<VEC3F*>(vboNormal.lockPtr());
	memcpy(ptrNorm, &bufferColors[0], bufferColors.size()*sizeof(VEC3F));
	vboNormal.releasePtr();

	vboColor.setDataSize(3);
	vboColor.allocate(bufferColors.size());
	VEC3F* ptrCol = reinterpret_cast<VEC3F*>(vboColor.lockPtr());
	memcpy(ptrCol, &bufferColors[0], bufferColors.size()*sizeof(VEC3F));
	vboColor.releasePtr();
}



/*

template<typename PFP, unsigned int ORBIT>
void ColorPerFaceRender::updateVBO(Utils::VBO& vboPosition, Utils::VBO& vboColor, Utils::VBO& vboTexCoord,
								   typename PFP::MAP& map,
								   const VertexAttribute<typename PFP::VEC3>& positions,
								   const AttributeHandler<typename PFP::VEC3,ORBIT>& colorPerXXX,
								   CellMarker<VERTEX>& specialVertices,
								   const VertexAttribute<typename PFP::VEC3>& texCoordPerVertex,
								   const AttributeHandler<typename PFP::VEC3,VERTEX1>& texCoordPerDart)

{
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	std::vector<VEC3> buffer;
	buffer.reserve(16384);

	std::vector<VEC3> bufferColors;
	bufferColors.reserve(16384);


	std::vector<VEC3> bufferTC;
	bufferColors.reserve(16384);

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
			bufferColors.push_back(colorPerXXX[d]);
			if (specialVertices.isMarked(d))
				bufferTC.push_back(texCoordPerDart[d]);
			else
				bufferTC.push_back(texCoordPerVertex[d]);

			buffer.push_back(positions[b]);
			bufferColors.push_back(colorPerXXX[b]);
			if (specialVertices.isMarked(c))
				bufferTC.push_back(texCoordPerDart[b]);
			else
				bufferTC.push_back(texCoordPerVertex[b]);

			buffer.push_back(positions[c]);
			bufferColors.push_back(colorPerXXX[c]);
			if (specialVertices.isMarked(c))
				bufferTC.push_back(texCoordPerDart[c]);
			else
				bufferTC.push_back(texCoordPerVertex[c]);

			b = c;
			c = map.phi1(b);
		} while (c != d);
	}

	m_nbTris = buffer.size()/3;

	vboPosition.setDataSize(3);
	vboPosition.allocate(buffer.size());
	VEC3* ptrPos = reinterpret_cast<VEC3*>(vboPosition.lockPtr());
	memcpy(ptrPos, &buffer[0], buffer.size()*sizeof(VEC3));
	vboPosition.releasePtr();

	vboTexCoord.setDataSize(2);
	vboTexCoord.allocate(bufferTC.size());
	VEC3* ptrTC = reinterpret_cast<VEC3*>(vboTexCoord.lockPtr());
	memcpy(ptrTC, &bufferTC[0], bufferTC.size()*sizeof(VEC2));
	vboTexCoord.releasePtr();

	vboColor.setDataSize(3);
	vboColor.allocate(bufferColors.size());
	VEC3* ptrCol = reinterpret_cast<VEC3*>(vboColor.lockPtr());
	memcpy(ptrCol, &bufferColors[0], bufferColors.size()*sizeof(VEC3));
	vboColor.releasePtr();
}

*/


inline void ColorPerFaceRender::draw(Utils::GLSLShader* sh)
{
	sh->enableVertexAttribs();
	glDrawArrays(GL_TRIANGLES , 0 , m_nbTris*3 );
	sh->disableVertexAttribs();
}




}//end namespace VBO

}//end namespace Algo

}//end namespace Render

}//end namespace CGoGN
