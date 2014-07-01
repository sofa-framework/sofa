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

#include "Geometry/vector_gen.h"
#include "Topology/generic/autoAttributeHandler.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"

#include "Topology/map/map3.h"
#include "Topology/gmap/gmap3.h"

#include "Topology/generic/traversor/traversorCell.h"
#include "Algo/Geometry/centroid.h"

#include "Geometry/distances.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

template <typename PFP>
Topo3PrimalRender<PFP>::Topo3PrimalRender():
	m_nbDarts(0),
	m_nbRel2(0),
	m_topo_dart_width(2.0f),
	m_topo_relation_width(3.0f),
	m_color_save(NULL),
	m_dartsColor(1.0f,1.0f,1.0f),
	m_boundaryDartsColor(0.5f,0.5f,0.5f),
	m_bufferDartPosition(NULL)
{
	m_vbo0 = new Utils::VBO();
	m_vbo1 = new Utils::VBO();
	m_vbo2 = new Utils::VBO();

	m_vbo0->setDataSize(3);
	m_vbo1->setDataSize(3);
	m_vbo2->setDataSize(3);


	m_shader1 = new Utils::ShaderSimpleColor();
	m_shader2 = new Utils::ShaderColorPerVertex();

	// binding VBO - VA
	m_vaId = m_shader1->setAttributePosition(m_vbo1);

	m_shader2->setAttributePosition(m_vbo0);
	m_shader2->setAttributeColor(m_vbo2);

	// registering for auto matrices update
	Utils::GLSLShader::registerShader(NULL, m_shader1);
	Utils::GLSLShader::registerShader(NULL, m_shader2);
}

template <typename PFP>
Topo3PrimalRender<PFP>::~Topo3PrimalRender()
{
	Utils::GLSLShader::unregisterShader(NULL, m_shader2);
	Utils::GLSLShader::unregisterShader(NULL, m_shader1);

	delete m_shader2;
	delete m_shader1;
	delete m_vbo2;
	delete m_vbo1;
	delete m_vbo0;

	if (m_attIndex.isValid())
		m_attIndex.map()->removeAttribute(m_attIndex);

	if (m_color_save != NULL)
		delete[] m_color_save;

	if (m_bufferDartPosition!=NULL)
		delete[] m_bufferDartPosition;
}

template <typename PFP>
void Topo3PrimalRender<PFP>::setDartWidth(float dw)
{
	m_topo_dart_width = dw;
}

template <typename PFP>
void Topo3PrimalRender<PFP>::setRelationWidth(float pw)
{
	m_topo_relation_width = pw;
}

template <typename PFP>
void Topo3PrimalRender<PFP>::setDartColor(Dart d, float r, float g, float b)
{
	float RGB[6];
	RGB[0]=r; RGB[1]=g; RGB[2]=b;
	RGB[3]=r; RGB[4]=g; RGB[5]=b;
	m_vbo2->bind();
	glBufferSubData(GL_ARRAY_BUFFER, m_attIndex[d]*3*sizeof(float), 6*sizeof(float),RGB);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::setAllDartsColor(float r, float g, float b)
{
	m_vbo2->bind();
	GLvoid* ColorDartsBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);
	float* colorDartBuf = reinterpret_cast<float*>(ColorDartsBuffer);
	for (unsigned int i=0; i < 2*m_nbDarts; ++i)
	{
		*colorDartBuf++ = r;
		*colorDartBuf++ = g;
		*colorDartBuf++ = b;
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::setInitialDartsColor(float r, float g, float b)
{
	m_dartsColor = Geom::Vec3f(r,g,b);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::setInitialBoundaryDartsColor(float r, float g, float b)
{
	m_boundaryDartsColor = Geom::Vec3f(r,g,b);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::drawDarts()
{
	if (m_nbDarts==0)
		return;

	m_shader2->enableVertexAttribs();

	glLineWidth(m_topo_dart_width);
	glDrawArrays(GL_LINES, 0, m_nbDarts*2);

	// change the stride to take 1/2 vertices
	m_shader2->enableVertexAttribs(6*sizeof(GL_FLOAT));

	glPointSize(2.0f*m_topo_dart_width);
	glDrawArrays(GL_POINTS, 0, m_nbDarts);

	m_shader2->disableVertexAttribs();
}

template <typename PFP>
void Topo3PrimalRender<PFP>::drawRelation2()
{
	if (m_nbRel2==0)
		return;

	m_shader1->changeVA_VBO(m_vaId, m_vbo1);
	m_shader1->setColor(Geom::Vec4f(1.0f,0.0f,0.0f,0.0f));
	m_shader1->enableVertexAttribs();

	glLineWidth(m_topo_relation_width);
	glDrawArrays(GL_LINES, 0, m_nbRel2*2);

	m_shader1->disableVertexAttribs();
}

template <typename PFP>
void Topo3PrimalRender<PFP>::drawTopo()
{
	drawDarts();
	drawRelation2();
}

template <typename PFP>
void Topo3PrimalRender<PFP>::overdrawDart(Dart d, float width, float r, float g, float b)
{
	unsigned int indexDart =  m_attIndex[d];

	m_shader1->changeVA_VBO(m_vaId, m_vbo0);
	m_shader1->setColor(Geom::Vec4f(r,g,b,0.0f));
	m_shader1->enableVertexAttribs();

	glLineWidth(width);
	glDrawArrays(GL_LINES, indexDart, 2);

	glPointSize(2.0f*width);
	glDrawArrays(GL_POINTS, indexDart, 1);

	m_shader2->disableVertexAttribs();
}

template <typename PFP>
void Topo3PrimalRender<PFP>::pushColors()
{
	m_color_save = new float[6*m_nbDarts];
	m_vbo2->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(m_color_save, colorBuffer, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::popColors()
{
	m_vbo2->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(colorBuffer, m_color_save, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);

	delete[] m_color_save;
	m_color_save=0;
}

template <typename PFP>
Dart Topo3PrimalRender<PFP>::colToDart(float* color)
{
	unsigned int r = (unsigned int)(color[0]*255.0f);
	unsigned int g = (unsigned int)(color[1]*255.0f);
	unsigned int b = (unsigned int)(color[2]*255.0f);

	unsigned int id = r + 255*g +255*255*b;

	if (id == 0)
		return NIL;
	return Dart(id-1);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::dartToCol(Dart d, float& r, float& g, float& b)
{
	// here use dart.index beacause it is what we want (and not map.dartIndex(d) !!)
	unsigned int lab = d.index + 1; // add one to avoid picking the black of screen

	r = float(lab%255) / 255.0f; lab = lab/255;
	g = float(lab%255) / 255.0f; lab = lab/255;
	b = float(lab%255) / 255.0f; lab = lab/255;
	if (lab!=0)
		CGoGNerr << "Error picking color, too many darts"<< CGoGNendl;
}

template <typename PFP>
Dart Topo3PrimalRender<PFP>::pickColor(unsigned int x, unsigned int y)
{
	//more easy picking for
	unsigned int dw = m_topo_dart_width;
	m_topo_dart_width+=2;

	// save clear color and set to zero
	float cc[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE,cc);

	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_LIGHTING);
	// draw in back buffer (not shown on screen)

	drawDarts();

	// restore dart with
	m_topo_dart_width = dw;

	// read the pixel under the mouse in back buffer
	glReadBuffer(GL_BACK);
	float color[3];
	glReadPixels(x,y,1,1,GL_RGB,GL_FLOAT,color);

	glClearColor(cc[0], cc[1], cc[2], cc[3]);


	std::cout << color[0] << ", "<<color[1] << ", "<<color[2] <<std::endl;
	return colToDart(color);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::svgout2D(const std::string& filename, const glm::mat4& model, const glm::mat4& proj)
{
	Utils::SVG::SVGOut svg(filename,model,proj);
	toSVG(svg);
	svg.write();
}

template <typename PFP>
void Topo3PrimalRender<PFP>::toSVG(Utils::SVG::SVGOut& svg)
{
	// alpha2
	Utils::SVG::SvgGroup* svg2 = new Utils::SVG::SvgGroup("alpha2", svg.m_model, svg.m_proj);
	Geom::Vec3f* ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo1->lockPtr());
	svg2->setWidth(m_topo_relation_width);
	svg2->beginLines();
	for (unsigned int i=0; i<m_nbRel2; ++i)
	{
		Geom::Vec3f P = ptr[2*i];
		Geom::Vec3f Q = ptr[2*i+1];
		svg2->addLine(P, Q, Geom::Vec3f(0.8f,0.0f,0.0f));
	}
	svg2->endLines();
	m_vbo1->releasePtr();
	svg.addGroup(svg2);

	const Geom::Vec3f* colorsPtr = reinterpret_cast<const Geom::Vec3f*>(m_vbo2->lockPtr());
	ptr= reinterpret_cast<Geom::Vec3f*>(m_vbo0->lockPtr());

	Utils::SVG::SvgGroup* svg4 = new Utils::SVG::SvgGroup("darts", svg.m_model, svg.m_proj);
	svg4->setWidth(m_topo_dart_width);

	svg4->beginLines();
	for (unsigned int i=0; i<m_nbDarts; ++i)
	{
		Geom::Vec3f col = colorsPtr[2*i];
		if (col.norm2()>2.9f)
			col = Geom::Vec3f(1.0f,1.0f,1.0f) - col;
		svg4->addLine(ptr[2*i], ptr[2*i+1], col);
	}
	svg4->endLines();

	svg.addGroup(svg4);

	Utils::SVG::SvgGroup* svg5 = new Utils::SVG::SvgGroup("dartEmb", svg.m_model, svg.m_proj);
	svg5->setWidth(m_topo_dart_width);
	svg5->beginPoints();
	for (unsigned int i=0; i<m_nbDarts; ++i)
	{
		Geom::Vec3f col = colorsPtr[2*i];
		if (col.norm2()>2.9f)
			col = Geom::Vec3f(1.0f,1.0f,1.0f) - col;
		svg5->addPoint(ptr[2*i], col);
	}
	svg5->endPoints();
	svg.addGroup(svg5);

	m_vbo0->releasePtr();
	m_vbo2->releasePtr();

}

template <typename PFP>
void Topo3PrimalRender<PFP>::setDartsIdColor(typename PFP::MAP& map)
{
	m_vbo2->bind();
	float* colorBuffer = reinterpret_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb = 0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (nb < m_nbDarts)
		{
			float r,g,b;
			dartToCol(d, r,g,b);

			float* local = colorBuffer+3*m_attIndex[d]; // get the right position in VBO
			*local++ = r;
			*local++ = g;
			*local++ = b;
			*local++ = r;
			*local++ = g;
			*local++ = b;
			nb++;
		}
		else
		{
			CGoGNerr << "Error buffer too small for color picking (change the selector parameter ?)" << CGoGNendl;
			break;
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::updateColors(MAP& map, const VertexAttribute<VEC3, MAP>& colors)
{
	m_vbo2->bind();
	Geom::Vec3f* colorBuffer =  reinterpret_cast<Geom::Vec3f*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb=0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (nb < m_nbDarts)
		{
			colorBuffer[m_attIndex[d]] = colors[d];
			nb++;
		}
		else
		{
			CGoGNerr << "Error buffer too small for color picking (change the selector parameter ?)" << CGoGNendl;
			break;
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template <typename PFP>
Dart Topo3PrimalRender<PFP>::picking(MAP& map, int x, int y)
{
	pushColors();
	setDartsIdColor(map);
	Dart d = pickColor(x,y);
	popColors();
	return d;
}

//template<typename PFP>
//void Topo3PrimalRender<PFP>::updateData(typename PFP::MAP& mapx, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf)
//{
//    updateDataGen<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(mapx,positions,ke,kf);
//}

template <typename PFP>
void Topo3PrimalRender<PFP>::updateData(MAP& mapx, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf)
{
	if (m_attIndex.map() != &mapx)
		m_attIndex  = mapx.template getAttribute<unsigned int, DART, MAP>("dart_index");
	if (!m_attIndex.isValid())
		m_attIndex  = mapx.template addAttribute<unsigned int, DART, MAP>("dart_index");

//	m_nbDarts = 0;
//	for (Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
//	{
//		m_nbDarts++;
//	}

	m_nbDarts = mapx.getNbDarts();

	// beta2/3
	DartAutoAttribute<VEC3, MAP> fv2(mapx);

	m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	Geom::Vec3f* colorDartBuf = reinterpret_cast<Geom::Vec3f*>(ColorDartsBuffer);

	if (m_bufferDartPosition!=NULL)
		delete m_bufferDartPosition;
	m_bufferDartPosition = new Geom::Vec3f[2*m_nbDarts];
	Geom::Vec3f* positionDartBuf = reinterpret_cast<Geom::Vec3f*>(m_bufferDartPosition);

	unsigned int posDBI = 0;

	int nbf = 0;
	//traverse each face of each volume
	TraversorF<MAP> traFace(mapx);
	for (Dart d = traFace.begin(); d != traFace.end(); d = traFace.next())
	{
		std::vector<VEC3> vecPos;
		vecPos.reserve(16);

		VEC3 centerFace = Algo::Surface::Geometry::faceCentroidELW<PFP>(mapx, d, positions);

		//shrink the face
		float okf = 1.0f - kf;
		Dart dd = d;
		do
		{
			VEC3 P = centerFace*okf + positions[dd]*kf;
			vecPos.push_back(P);
			dd = mapx.phi1(dd);
		} while (dd != d);
		
		unsigned int nb = vecPos.size();
		
		vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

		// compute position of points to use for drawing topo
		float oke = 1.0f - ke;
		for (unsigned int i = 0; i < nb; ++i)
		{
			VEC3 P = vecPos[i]*ke + vecPos[i+1]*oke;
			VEC3 Q = vecPos[i+1]*ke + vecPos[i]*oke;

//			VEC3 PP = 0.52f*P + 0.48f*Q;
//			VEC3 QQ = 0.52f*Q + 0.48f*P;

            VEC3 PP = 0.56f*P + 0.44f*Q;
            VEC3 QQ = 0.56f*Q + 0.44f*P;

			*positionDartBuf++ = PFP::toVec3f(P);
			*positionDartBuf++ = PFP::toVec3f(PP);
			if (mapx.template isBoundaryMarked<3>(d))
			{
				*colorDartBuf++ = m_boundaryDartsColor;
				*colorDartBuf++ = m_boundaryDartsColor;
			}
			else
			{
				*colorDartBuf++ = m_dartsColor;
				*colorDartBuf++ = m_dartsColor;
			}

			m_attIndex[d] = posDBI;
			posDBI+=2;
			fv2[d] = (P+PP)*0.5f;

			*positionDartBuf++ = PFP::toVec3f(Q);
			*positionDartBuf++ = PFP::toVec3f(QQ);

			Dart dx = mapx.phi3(d);
			if (mapx.template isBoundaryMarked<3>(dx))
			{
				*colorDartBuf++ = m_boundaryDartsColor;
				*colorDartBuf++ = m_boundaryDartsColor;
			}
			else
			{
				*colorDartBuf++ = m_dartsColor;
				*colorDartBuf++ = m_dartsColor;
			}

			m_attIndex[dx] = posDBI;
			posDBI+=2;
			fv2[dx] = (Q+QQ)*0.5f;

			d = mapx.phi1(d);
		}
		nbf++;
	}
	m_vbo2->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(Geom::Vec3f), m_bufferDartPosition, GL_STREAM_DRAW);

	// alpha2
	m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	Geom::Vec3f* positionF2 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer2);

	m_nbRel2 = 0;

	for (Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
	{
		Dart e = mapx.phi2(mapx.phi3(d));
		//if (d < e)
		{
			*positionF2++ = PFP::toVec3f(fv2[d]);
			*positionF2++ = PFP::toVec3f(fv2[e]);
			m_nbRel2++;
		}
	}

	m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template <typename PFP>
void Topo3PrimalRender<PFP>::computeDartMiddlePositions(MAP& map, DartAttribute<VEC3, MAP>& posExpl)
{
	m_vbo0->bind();
	Geom::Vec3f* positionsPtr = reinterpret_cast<Geom::Vec3f*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		const Geom::Vec3f& v =(positionsPtr[m_attIndex[d]] + positionsPtr[m_attIndex[d]+1])*0.5f;
		posExpl[d] = PFP::toVec3f(v);
	}

	m_vbo0->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template <typename PFP>
Dart Topo3PrimalRender<PFP>::coneSelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle)
{
	float AB2 = rayAB*rayAB;
	Dart dFinal;
	double sin2 = sin(M_PI/180.0 * angle);
	sin2 = sin2*sin2;
	double dist2 = std::numeric_limits<double>::max();

	for(Dart d = map.begin(); d!=map.end(); map.next(d))
	{
		// get back position of segment PQ
		const Geom::Vec3f& P = m_bufferDartPosition[m_attIndex[d]];
		const Geom::Vec3f& Q =m_bufferDartPosition[m_attIndex[d]+1];
		float ld2 = Geom::squaredDistanceLine2Seg(rayA, rayAB, AB2, P, Q);
		Geom::Vec3f V = (P+Q)/2.0f - rayA;
		double d2 = double(V*V);
		double s2 = double(ld2) / d2;
		if (s2 < sin2)
		{
			if (d2<dist2)
			{
				dist2 = d2;
				dFinal = d;
			}
		}
	}
	return dFinal;
}

template <typename PFP>
Dart Topo3PrimalRender<PFP>::raySelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float dmax)
{
	float AB2 = rayAB*rayAB;
	Dart dFinal;
	float dm2 = dmax*dmax;
	double dist2 = std::numeric_limits<double>::max();

	for(Dart d = map.begin(); d!=map.end(); map.next(d))
	{
		// get back position of segment PQ
		const Geom::Vec3f& P = m_bufferDartPosition[m_attIndex[d]];
		const Geom::Vec3f& Q =m_bufferDartPosition[m_attIndex[d]+1];
		float ld2 = Geom::squaredDistanceLine2Seg(rayA, rayAB, AB2, P, Q);
		if (ld2 < dm2)
		{
			Geom::Vec3f V = (P+Q)/2.0f - rayA;
			double d2 = double(V*V);
			if (d2 < dist2)
			{
				dist2 = d2;
				dFinal = d;
			}
		}
	}
	return dFinal;
}

} //end namespace GL2

} //end namespace Render

} //end namespace Algo

} //end namespace CGoGN
