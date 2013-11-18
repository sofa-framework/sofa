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



inline ColorPerEdgeRender::ColorPerEdgeRender():
m_nbEdges(0)
{
}



template<typename PFP, typename ATTRIB>
void ColorPerEdgeRender::updateVBO(Utils::VBO& vboPosition, Utils::VBO& vboColor, typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& positions, const ATTRIB& colorPerXXX)
{
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	std::vector<VEC3> buffer;
	buffer.reserve(16384);

	std::vector<VEC3> bufferColors;
	bufferColors.reserve(16384);

	TraversorCell<typename PFP::MAP, EDGE> traEdge(map);

	for (Dart d=traEdge.begin(); d!=traEdge.end(); d=traEdge.next())
	{
		buffer.push_back(positions[d]);
		bufferColors.push_back(colorPerXXX[d]);
		Dart e = map.phi2(d);
		buffer.push_back(positions[e]);
		bufferColors.push_back(colorPerXXX[e]);
	}

	m_nbEdges = buffer.size()/2;

	vboPosition.setDataSize(3);
	vboPosition.allocate(buffer.size());
	VEC3* ptrPos = reinterpret_cast<VEC3*>(vboPosition.lockPtr());
	memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3));
	vboPosition.releasePtr();

	vboColor.setDataSize(3);
	vboColor.allocate(bufferColors.size());
	VEC3* ptrCol = reinterpret_cast<VEC3*>(vboColor.lockPtr());
	memcpy(ptrCol,&bufferColors[0],bufferColors.size()*sizeof(VEC3));
	vboColor.releasePtr();
}




inline void ColorPerEdgeRender::draw(Utils::GLSLShader* sh)
{
	sh->enableVertexAttribs();
	glDrawArrays(GL_LINES , 0 , m_nbEdges*2 );
	sh->disableVertexAttribs();
}




}//end namespace VBO

}//end namespace Algo

}//end namespace Render

}//end namespace CGoGN
