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
#include "Utils/svg.h"
#include "Utils/cgognStream.h"
#include <algorithm>
#include <typeinfo>
#include <GL/glew.h>
#include <Geometry/vector_gen.h>

namespace CGoGN
{

namespace Utils
{

namespace SVG
{


const std::vector<Geom::Vec3f>& SvgObj::vertices() const
{
	return  m_vertices;
}

void SvgObj::addVertex(const Geom::Vec3f& v)
{
	m_vertices.push_back(v);
	m_colors.push_back(m_color);
}

void SvgObj::addVertex3D(const Geom::Vec3f& v)
{
	m_vertices3D.push_back(v);
	m_colors.push_back(m_color);
}

void SvgObj::addVertex(const Geom::Vec3f& v, const Geom::Vec3f& c)
{
	m_vertices.push_back(v);
	if (m_colors.size() < m_vertices.size())
		m_colors.push_back(c);
}

void SvgObj::addVertex3D(const Geom::Vec3f& v, const Geom::Vec3f& c)
{
	m_vertices3D.push_back(v);
	if (m_colors.size() < m_vertices.size())
		m_colors.push_back(c);
}


void SvgObj::addString(const Geom::Vec3f& v, const std::string& str)
{
	m_vertices.push_back(v);
	m_strings.push_back(str);
	m_colors.push_back(m_color);
}

void SvgObj::addString(const Geom::Vec3f& v, const std::string& str, const Geom::Vec3f& c)
{
	m_vertices.push_back(v);
	m_strings.push_back(str);
	m_colors.push_back(c);
}


void SvgObj::setWidth(float w)
{
	m_width=w;
}


void SvgObj::setColor(const Geom::Vec3f& c)
{
	m_color=c;
}


void SvgObj::close()
{
	m_vertices.push_back(m_vertices.front());
}

unsigned int SvgObj::nbv() const
{
	return m_vertices.size();
}

const Geom::Vec3f& SvgObj::P(unsigned int i) const
{
	return m_vertices[i];
}


Geom::Vec3f SvgObj::normal()
{
	if (m_vertices.size()<3)
	{
		CGoGNerr << "Error SVG normal computing (not enough points)"<<CGoGNendl;
		return Geom::Vec3f(0.0f,0.0f,0.0f);
	}

	Geom::Vec3f U = m_vertices3D[2] - m_vertices3D[1];
	Geom::Vec3f V = m_vertices3D[0] - m_vertices3D[1];

    Geom::Vec3f N = U.cross(V);

	N.normalize(); // TO DO verify that is necessary

	return N;
}


void SvgPoints::save(std::ofstream& out) const
{
//	std::stringstream ss;

	unsigned int nb = m_vertices.size();
	for (unsigned int i=0; i<nb; ++i)
	{
		saveOne(out,i);
//		out << "<circle cx=\""<< m_vertices[i][0];
//		out << "\" cy=\""<< m_vertices[i][1];
//		out << "\" r=\""<< m_width;
//		out << "\" style=\"stroke: none; fill: #";
//
//		out << std::hex;
//		unsigned int wp = out.width(2);
//		char prev = out.fill('0');
//		out << int(m_colors[i][0]*255);
//		out.width(2); out.fill('0');
//		out<< int(m_colors[i][1]*255);
//		out.width(2); out.fill('0');
//		out << int(m_colors[i][2]*255)<<std::dec;
//		out.fill(prev);
//		out.width(wp);
//
//		out <<"\"/>"<< std::endl;
	}
}


void SvgPoints::saveOne(std::ofstream& out, unsigned int i, unsigned int /*bbl*/) const
{
	std::stringstream ss;

	out << "<circle cx=\""<< m_vertices[i][0];
	out << "\" cy=\""<< m_vertices[i][1];
	out << "\" r=\""<< m_width;
	out << "\" style=\"stroke: none; fill: #";

	out << std::hex;
	unsigned int wp = out.width(2);
	char prev = out.fill('0');
	out << int(m_colors[i][0]*255);
	out.width(2); out.fill('0');
	out<< int(m_colors[i][1]*255);
	out.width(2); out.fill('0');
	out << int(m_colors[i][2]*255)<<std::dec;
	out.fill(prev);
	out.width(wp);

	out <<"\"/>"<< std::endl;
}

void SvgPoints::saveOneDepthAttenuation(std::ofstream& out, float da, unsigned int i, unsigned int /*bbl*/) const
{
	std::stringstream ss;

	out << "<circle cx=\""<< m_vertices[i][0];
	out << "\" cy=\""<< m_vertices[i][1];
	out << "\" r=\""<< m_width;
	out << "\" style=\"stroke: none; fill: #";

	Geom::Vec3f color = m_colors[i] * da;

	out << std::hex;
	unsigned int wp = out.width(2);
	char prev = out.fill('0');
	out << int(color[0]*255);
	out.width(2); out.fill('0');
	out<< int(color[1]*255);
	out.width(2); out.fill('0');
	out << int(color[2]*255)<<std::dec;
	out.fill(prev);
	out.width(wp);

	out <<"\"/>"<< std::endl;
}

unsigned int SvgPoints::nbPrimtives() const
{
	return m_vertices.size();
}


void SvgPoints::fillDS(std::vector<DepthSort>& vds, unsigned int idObj) const
{
	for (unsigned int i = 0; i< m_vertices.size(); ++i)
	{
		vds.push_back(DepthSort(idObj,i,m_vertices[i][2]));
	}
}

void SvgLines::save(std::ofstream& out) const
{
	std::stringstream ss;

	unsigned int nb = m_vertices.size()/2;
	for (unsigned int i=0; i<nb; ++i)
		saveOne(out,i);
}

void SvgLines::saveOne(std::ofstream& out, unsigned int i, unsigned int /*bbl*/) const
{
	std::stringstream ss;

	out << "<polyline fill=\"none\" stroke=\"#";
	out << std::hex;
	unsigned int wp = out.width(2);
	char prev = out.fill('0');
	out << int(m_colors[2*i][0]*255);
	out.width(2); out.fill('0');
	out<< int(m_colors[2*i][1]*255);
	out.width(2); out.fill('0');
	out << int(m_colors[2*i][2]*255)<<std::dec;
	out <<"\" stroke-width=\""<<m_width<<"\" points=\"";
	out.fill(prev);
	out.width(wp);
	out << m_vertices[2*i][0] << ","<< m_vertices[2*i][1]<< " ";
	out << m_vertices[2*i+1][0] << ","<< m_vertices[2*i+1][1];
	out <<"\"/>"<< std::endl;
}

void SvgLines::saveOneDepthAttenuation(std::ofstream& out, float da, unsigned int i, unsigned int /*bbl*/) const
{
	std::stringstream ss;

	Geom::Vec3f color = m_colors[i] * da + Geom::Vec3f(1.0f,1.0f,1.0f)*(1.0f-da);

	out << "<polyline fill=\"none\" stroke=\"#";
	out << std::hex;
	unsigned int wp = out.width(2);
	char prev = out.fill('0');
	out << int(color[0]*255);
	out.width(2); out.fill('0');
	out<< int(color[1]*255);
	out.width(2); out.fill('0');
	out << int(color[2]*255)<<std::dec;
	out <<"\" stroke-width=\""<<m_width<<"\" points=\"";
	out.fill(prev);
	out.width(wp);
	out << m_vertices[2*i][0] << ","<< m_vertices[2*i][1]<< " ";
	out << m_vertices[2*i+1][0] << ","<< m_vertices[2*i+1][1];
	out <<"\"/>"<< std::endl;
}



unsigned int SvgLines::nbPrimtives() const
{
	return m_vertices.size()/2;
}

void SvgLines::fillDS(std::vector<DepthSort>& vds, unsigned int idObj) const
{
	unsigned int nb = m_vertices.size()/2;
	for (unsigned int i = 0; i<nb; ++i)
	{
		float depth = (m_vertices[2*i][2] + m_vertices[2*i+1][2])/2.0f;
		vds.push_back(DepthSort(idObj,i,depth));
	}
}




void SvgStrings::save(std::ofstream& out) const
{
	unsigned int nb = m_vertices.size();
	for (unsigned int i=0; i<nb; ++i)
	{
		saveOne(out,i);
	}
}


void SvgStrings::saveOne(std::ofstream& out, unsigned int i, unsigned int bbl) const
{

	float scale = m_sf * sqrt(1.0f - m_vertices[i][2])*0.007f*float(bbl);

	out << "<text style=\"font-size:24px;";
	out << "font-family:Bitstream Vera Sans\" ";
	 out << "transform=\"scale(" << scale <<"," << scale <<")\""<< std::endl;
	out << "x=\""<< (m_vertices[i][0])/scale << "\" y=\""<< (m_vertices[i][1])/scale<< "\">";
	out << "<tspan style=\"fill:#";

	out << std::hex;
	unsigned int wp = out.width(2);
	char prev = out.fill('0');
	out << int(m_colors[i][0]*255);
	out.width(2); out.fill('0');
	out<< int(m_colors[i][1]*255);
	out.width(2); out.fill('0');
	out << int(m_colors[i][2]*255)<<std::dec;
	out.fill(prev);
	out.width(wp);

	out <<"\">"<< m_strings[i]<< "</tspan></text>"<< std::endl;
}

void SvgStrings::saveOneDepthAttenuation(std::ofstream& out, float da, unsigned int i, unsigned int bbl) const
{

	float scale = m_sf * sqrt(1.0f - m_vertices[i][2])*0.007f*float(bbl);

	Geom::Vec3f color = m_colors[i] * da;

	out << "<text style=\"font-size:24px;";
	out << "font-family:Bitstream Vera Sans\" ";
	 out << "transform=\"scale(" << scale <<"," << scale <<")\""<< std::endl;
	out << "x=\""<< (m_vertices[i][0])/scale << "\" y=\""<< (m_vertices[i][1])/scale<< "\">";
	out << "<tspan style=\"fill:#";

	out << std::hex;
	unsigned int wp = out.width(2);
	char prev = out.fill('0');
	out << int(color[0]*255);
	out.width(2); out.fill('0');
	out<< int(color[1]*255);
	out.width(2); out.fill('0');
	out << int(color[2]*255)<<std::dec;
	out.fill(prev);
	out.width(wp);

	out <<"\">"<< m_strings[i]<< "</tspan></text>"<< std::endl;
}



unsigned int SvgStrings::nbPrimtives() const
{
	return m_vertices.size();
}

void SvgStrings::fillDS(std::vector<DepthSort>& vds, unsigned int idObj) const
{
	for (unsigned int i = 0; i< m_vertices.size(); ++i)
	{
		vds.push_back(DepthSort(idObj,i,m_vertices[i][2]));
	}
}



//void SvgPolyline::save(std::ofstream& out)
//{
//	std::stringstream ss;
//
//	out << "<polyline fill=\"none\" stroke=\"#";
//	out << std::hex;
//	unsigned int wp = out.width(2);
//	char prev = out.fill('0');
//	out << int(m_color[0]*255);
//	out.width(2); out.fill('0');
//	out<< int(m_color[1]*255);
//	out.width(2); out.fill('0');
//	out << int(m_color[2]*255)<<std::dec;
//	out <<"\" stroke-width=\""<<m_width<<"\" points=\"";
//	out.fill(prev);
//	out.width(wp);
//	for (std::vector<Geom::Vec3f>::iterator it =m_vertices.begin(); it != m_vertices.end(); ++it)
//	{
//		out << (*it)[0] << ","<< (*it)[1]<< " ";
//	}
//	out <<"\"/>"<< std::endl;
//}
//
//void SvgPolygon::setColorFill(const Geom::Vec3f& c)
//{
//	m_colorFill=c;
//}
//
//
//void SvgPolygon::save(std::ofstream& out)
//{
//	std::stringstream ss;
//
//	out << "<polyline fill=\"";
//	out << std::hex;
//	unsigned int wp = out.width(2);
//	char prev = out.fill('0');
//	out << int(m_colorFill[0]*255);
//	out.width(2); out.fill('0');
//	out<< int(m_colorFill[1]*255);
//	out.width(2); out.fill('0');
//	out << int(m_colorFill[2]*255);
//
//	out << "none\" stroke=\"#";
//	wp = out.width(2);
//	prev = out.fill('0');
//	out << int(m_color[0]*255);
//	out.width(2); out.fill('0');
//	out<< int(m_color[1]*255);
//	out.width(2); out.fill('0');
//	out << int(m_color[2]*255)<<std::dec;
//
//	out <<"\" stroke-width=\""<<m_width<<"\" points=\"";
//	out.fill(prev);
//	out.width(wp);
//	for (std::vector<Geom::Vec3f>::iterator it =m_vertices.begin(); it != m_vertices.end(); ++it)
//	{
//		out << (*it)[0] << ","<< (*it)[1]<< " ";
//	}
//	out <<"\"/>"<< std::endl;
//
//}

//void SvgLayers::save(std::ofstream& out) const
//{
//	saveOne(out,0,0);
//}
//
//void SvgLayers::saveOne(std::ofstream& out, unsigned int i, unsigned int bbl) const
//{
//	if(m_end)
//	{
//		out << "</g>" << std::endl;
//	}
//	else
//	{
//		out << "<g inkscape:groupemode=\"layer\"" << std::endl;
//		out << "   id=\"" << m_layer << "\"" << std::endl;
//		out << "   inkscape:label=\"" << m_layer <<"\">" << std::endl;
//	}
//}



SvgGroup::SvgGroup(const std::string& name, const glm::mat4& model, const glm::mat4& proj):
		m_name(name), m_model(model),m_proj(proj),global_color(Geom::Vec3f(0.0f,0.0f,0.0f)), global_width(2.0f), m_isLayer(false)
{
	m_objs.reserve(1000);
	glGetIntegerv(GL_VIEWPORT, &(m_viewport[0]));
}


//SvgGroup::SvgGroup(const glm::mat4& model, const glm::mat4& proj):
//		m_groupName(""), m_model(model),m_proj(proj),global_color(Geom::Vec3f(0.0f,0.0f,0.0f)), global_width(2.0f), m_isLayer(false)
//{
//	m_objs.reserve(1000);
//	glGetIntegerv(GL_VIEWPORT, &(m_viewport[0]));
//}


SvgGroup::~SvgGroup()
{
	for (std::vector<SvgObj*>::iterator it = m_objs.begin(); it != m_objs.end(); ++it)
		delete (*it);
}

void SvgGroup::setColor(const Geom::Vec3f& col)
{
	global_color = col;
}

void SvgGroup::setWidth(float w)
{
	global_width = w;
}

void SvgGroup::beginPoints()
{
	m_current = new SvgPoints();
	m_current->setColor(global_color);
	m_current->setWidth(global_width);
}

void SvgGroup::endPoints()
{
	m_objs.push_back(m_current);
	m_current = NULL;
}

void SvgGroup::addPoint(const Geom::Vec3f& P)
{
	glm::vec3 Q = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,m_proj,m_viewport);
//	glm::vec3 R = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,glm::mat4(1.0),m_viewport);
	m_current->addVertex(Geom::Vec3f(float(Q[0]),float(m_viewport[3])-float(Q[1]),float(Q[2])));
}

void SvgGroup::addPoint(const Geom::Vec3f& P, const Geom::Vec3f& C)
{
	glm::vec3 Q = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,m_proj,m_viewport);
//	glm::vec3 R = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,glm::mat4(1.0),m_viewport);
	m_current->addVertex(Geom::Vec3f(float(Q[0]),float(m_viewport[3])-float(Q[1]),float(Q[2])),C);
}


void SvgGroup::beginLines()
{
	m_current = new SvgLines();
	m_current->setColor(global_color);
	m_current->setWidth(global_width);
}


void SvgGroup::endLines()
{
	m_objs.push_back(m_current);
	m_current = NULL;
}

void SvgGroup::addLine(const Geom::Vec3f& P, const Geom::Vec3f& P2)
{
	glm::vec3 Q = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,m_proj,m_viewport);
//	glm::vec3 R = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,glm::mat4(1.0),m_viewport);

	glm::vec3 Q2 = glm::project(glm::vec3(P2[0],P2[1],P2[2]),m_model,m_proj,m_viewport);
//	glm::vec3 R2 = glm::project(glm::vec3(P2[0],P2[1],P2[2]),m_model,glm::mat4(1.0),m_viewport);

	m_current->addVertex(Geom::Vec3f(float(Q[0]),float(m_viewport[3])-float(Q[1]),float(Q[2])));
	m_current->addVertex(Geom::Vec3f(float(Q2[0]),float(m_viewport[3])-float(Q2[1]),float(Q2[2])));
}

void SvgGroup::addLine(const Geom::Vec3f& P, const Geom::Vec3f& P2, const Geom::Vec3f& C)
{
	glm::vec3 Q = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,m_proj,m_viewport);
//	glm::vec3 R = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,glm::mat4(1.0),m_viewport);

	glm::vec3 Q2 = glm::project(glm::vec3(P2[0],P2[1],P2[2]),m_model,m_proj,m_viewport);
//	glm::vec3 R2 = glm::project(glm::vec3(P2[0],P2[1],P2[2]),m_model,glm::mat4(1.0),m_viewport);

	m_current->addVertex(Geom::Vec3f(float(Q[0]),float(m_viewport[3])-float(Q[1]),float(Q[2])),C);
	m_current->addVertex(Geom::Vec3f(float(Q2[0]),float(m_viewport[3])-float(Q2[1]),float(Q2[2])),C);
}

void SvgGroup::beginStrings(float scalefactor)
{
	m_current = new SvgStrings(scalefactor);
	m_current->setColor(global_color);
	m_current->setWidth(global_width);
}

void SvgGroup::endStrings()
{
	m_objs.push_back(m_current);
	m_current = NULL;
}

void SvgGroup::addString(const Geom::Vec3f& P, const std::string& str)
{
	glm::vec3 Q = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,m_proj,m_viewport);
	m_current->addString(Geom::Vec3f(float(Q[0]),float(m_viewport[3])-float(Q[1]),float(Q[2])), str);
}


void SvgGroup::addString(const Geom::Vec3f& P, const std::string& str, const Geom::Vec3f& C)
{
	glm::vec3 Q = glm::project(glm::vec3(P[0],P[1],P[2]),m_model,m_proj,m_viewport);
	m_current->addString(Geom::Vec3f(float(Q[0]),float(m_viewport[3])-float(Q[1]),float(Q[2])), str, C);
}

void SvgGroup::sortSimpleDepth(std::vector<DepthSort>& vds)
{
	unsigned int nb=0;
	for (std::vector<SvgObj*>::iterator it = m_objs.begin(); it != m_objs.end(); ++it)
		nb += (*it)->nbPrimtives();

	vds.reserve(nb);

	for (unsigned int i=0; i< m_objs.size(); ++i)
	{
		m_objs[i]->fillDS(vds,i);
	}
	std::sort(vds.begin(),vds.end());
}


SVGOut::SVGOut(const std::string& filename, const glm::mat4& model, const glm::mat4& proj):
	m_model(model),m_proj(proj),m_attFact(0.0f)
{
	m_groups.reserve(1000);

	m_out = new std::ofstream(filename.c_str()) ;
	if (!m_out->good())
	{
		CGoGNerr << "Unable to open file " << CGoGNendl ;
	}

	glGetIntegerv(GL_VIEWPORT, &(m_viewport[0]));
}

SVGOut::SVGOut(const glm::mat4& model, const glm::mat4& proj):
	m_model(model),m_proj(proj),m_attFact(0.0f)
{
	m_groups.reserve(1000);

	m_out = NULL;

	glGetIntegerv(GL_VIEWPORT, &(m_viewport[0]));
}

SVGOut::~SVGOut()
{
	delete m_out;

	for (std::vector<SvgGroup*>::iterator it = m_groups.begin(); it != m_groups.end(); ++it)
		delete (*it);
}

void SVGOut::computeBB(unsigned int& a, unsigned int& b, unsigned int& c, unsigned& d)
{
	for (std::vector<SvgGroup*>::iterator it = m_groups.begin(); it != m_groups.end(); ++it)
	{
		const std::vector<SvgObj*>& objs = (*it)->m_objs;

		for (std::vector<SvgObj*>::const_iterator ito = objs.begin(); ito != objs.end(); ++ito)
		{
			const std::vector<Geom::Vec3f>& vert = (*ito)->vertices();
			for (std::vector<Geom::Vec3f>::const_iterator j = vert.begin(); j != vert.end(); ++j)
			{
				const Geom::Vec3f& P = *j;
				if (P[0]<a)
					a = (unsigned int)(P[0]);
				if (P[1]<b)
					b = (unsigned int)(P[1]);
				if (P[0]>c)
					c = (unsigned int)(P[0]);
				if (P[1]>d)
					d = (unsigned int)(P[1]);
			}

			if (a>10)
				a-=10;
			if (b>10)
				b-=10;
			c+=10;
			d+=10;
		}
	}
}


void SVGOut::write()
{
	m_bbX0 = 100000;
	m_bbY0 = 100000;
	m_bbX1 = 0;
	m_bbY1 = 0;
	computeBB(m_bbX0, m_bbY0, m_bbX1, m_bbY1);

	*m_out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"<< std::endl;
	*m_out << "<svg xmlns=\"http://www.w3.org/2000/svg\""<< std::endl;
	*m_out << " xmlns:xlink=\"http://www.w3.org/1999/xlink\""<< std::endl;
	*m_out << "viewBox=\""<< m_bbX0 <<" "<< m_bbY0 <<" "<< m_bbX1-m_bbX0 << " " << m_bbY1-m_bbY0 <<"\">"<< std::endl;
	*m_out << "<title>test</title>"<< std::endl;
	*m_out << "<desc>"<< std::endl;
	*m_out << "Rendered from CGoGN"<< std::endl;

	*m_out << "</desc>"<< std::endl;
	*m_out << "<defs>"<< std::endl;
	*m_out << "</defs>"<< std::endl;
	//*m_out << "<g shape-rendering=\"crispEdges\">" << std::endl;

	for(std::vector<SvgGroup*>::iterator it = m_groups.begin() ; it != m_groups.end() ; ++it)
	{
		if((*it)->m_isLayer)
		{
			*m_out << "<g inkscape:groupmode=\"layer\"" << std::endl;
			*m_out << "id =\"" << (*it)->m_name << "\"" << std::endl;
			*m_out << "inkscape:label=\"" << (*it)->m_name << "\">" << std::endl;

		}
		else
		{
			*m_out << "<g " << std::endl;
			*m_out << "id =\"" << (*it)->m_name << "\">" << std::endl;
		}

		std::vector<DepthSort> vds;
		(*it)->sortSimpleDepth(vds);

		for (std::vector<DepthSort>::iterator itd = vds.begin(); itd != vds.end(); ++itd)
		{
			if (m_attFact > 0.0f)
			{
				float da = float( pow(double(itd-vds.begin()),double(m_attFact)) / pow(double(vds.size()),double(m_attFact)));
				(*it)->m_objs[itd->obj]->saveOneDepthAttenuation(*m_out,da,itd->id, m_bbX1-m_bbX0);
			}
			else
				(*it)->m_objs[itd->obj]->saveOne(*m_out,itd->id, m_bbX1-m_bbX0);
		}

		*m_out << "</g>" << std::endl;
	}

	//*m_out << "</g>" << std::endl;
	*m_out << "</svg>" << std::endl;
	m_out->close();
}



void AnimatedSVGOut::add(SVGOut* svg)
{
	m_svgs.push_back(svg);
}

//void AnimatedSVGOut::write(const std::string& filename, float timeStep)
//{
//    std::ofstream outfile(filename.c_str()) ;

//    unsigned int bbX0=1000000;
//    unsigned int bbY0=1000000;
//    unsigned int bbX1=0;
//    unsigned int bbY1=0;

//    for (unsigned int i=0; i< m_svgs.size(); ++i)
//        m_svgs[i]->computeBB(bbX0, bbY0, bbX1, bbY1);

//    outfile << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"<< std::endl;
//    outfile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<< std::endl;
//    outfile << " xmlns:xlink=\"http://www.w3.org/1999/xlink\""<< std::endl;
//    outfile << "viewBox=\""<< bbX0 <<" "<< bbY0 <<" "<< bbX1-bbX0 << " " << bbY1-bbY0 <<"\">"<< std::endl;
//    outfile << "<title>Animated SVG</title>"<< std::endl;
//    outfile << "<desc>"<< std::endl;
//    outfile << "Rendered from CGoGN"<< std::endl;
//    outfile << "</desc>"<< std::endl;

//    outfile << "<defs>"<< std::endl;


//    for (unsigned int i=0; i< m_svgs.size(); ++i)
//    {
//        std::vector<DepthSort> vds;
//        m_svgs[i]->sortSimpleDepth(vds);

//        outfile << "<g id=\"slice"<<i<< "\">" << std::endl;
//        for (std::vector<DepthSort>::iterator it = vds.begin(); it != vds.end(); ++it)
//            m_svgs[i]->m_groups[it->obj]->saveOne(outfile,it->id, bbX1-bbX0);
//        outfile << "</g>" << std::endl;
//    }

//    outfile << "</defs>"<< std::endl;


//    for (unsigned int i=0; i< m_svgs.size(); ++i)
//    {
//        unsigned int nbo = m_svgs[i]->m_opacities_animations.size();

//        outfile << "<use xlink:href=\"#slice"<<i<<"\" >" << std::endl;
//        outfile << "<animate attributeName=\"opacity\" dur=\""<<timeStep*nbo<<"s\" fill=\"freeze\" values=\"";


//        for (unsigned int j = 0; j < nbo; ++j)
//        {
//            outfile << m_svgs[i]->m_opacities_animations[j];
//            if ( j != (nbo-1))
//                outfile<< ";";
//        }

//        outfile << "\" calcMode=\"discrete\" repeatCount=\"indefinite\" />" << std::endl;
//        outfile << "</use>" << std::endl;
//    }

//    outfile << "</svg>" << std::endl;
//    outfile.close();
//}


} // namespace SVG
} // namespace Utils
} // namespace CGoGN


