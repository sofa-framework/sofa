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

#include "Utils/pickables.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/norm.hpp"
#include "Geometry/distances.h"
#include "Geometry/intersection.h"
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

namespace CGoGN
{

namespace Utils
{

LineDrawable::LineDrawable()
{
	m_vboPos = new VBO();
	m_vboPos->setDataSize(3);

	m_shader = new ShaderSimpleColor();

	m_shader->setAttributePosition(m_vboPos);
	m_shader->setColor(Geom::Vec4f(1.,1.,0.,0.));
	GLSLShader::registerShader(NULL, m_shader);

	glGenBuffers(1, &(*m_ind));
}

LineDrawable::~LineDrawable()
{
	delete m_vboPos;
	GLSLShader::unregisterShader(NULL, m_shader);
	delete m_shader;

	glDeleteBuffers(1, &(*m_ind));
}


void LineDrawable::setColor(const Geom::Vec4f& col)
{
	m_color=col;
	m_shader->setColor(col);
}

const Geom::Vec4f&  LineDrawable::getColor()
{
	return m_color;
}

void LineDrawable::draw()
{
	m_shader->enableVertexAttribs();
	glDrawArrays(GL_LINES, 0, m_nb);
	m_shader->disableVertexAttribs();
}

void LineDrawable::getPrecisionDrawing(unsigned int& sub, unsigned int& sub2)
{
	sub = m_sub1;
	sub2 = m_sub2;
}


Pickable::Pickable(LineDrawable* ld, unsigned int id):
	m_drawable(ld),m_transfo(1.0f), m_id(id), m_allocated(false)
{
}

Pickable::Pickable(int object, unsigned int id):
	m_transfo(1.0f), m_id(id),  m_allocated(true)
{
	switch (object)
	{
		case GRID:
			m_drawable = new Grid();
			break;
		case SPHERE:
			m_drawable = new Sphere();
			break;
		case CONE:
			m_drawable = new Sphere();
			break;
		case CYLINDER:
			m_drawable = new Cylinder();
			break;
		case CUBE:
			m_drawable = new Cube();
			break;
		case ICOSPHERE:
			m_drawable = new IcoSphere();
			break;
		default:
			break;
	}
}

Pickable::~Pickable()
{
	if (m_allocated)
		delete m_drawable;
}


void Pickable::invertPV(const Geom::Vec3f& P, const Geom::Vec3f& V, const glm::mat4& transfo, Geom::Vec3f& PP, Geom::Vec3f& VV)
{
	glm::mat4 invtr = glm::inverse(transfo);
	glm::vec4 xP(P[0],P[1],P[2],1.0f);
	glm::vec4 xQ(P[0]+V[0],P[1]+V[1],P[2]+V[2],1.0f);

	glm::vec4 tP = invtr*xP;
	glm::vec4 tQ = invtr*xQ;

	PP = Geom::Vec3f(tP[0]/tP[3], tP[1]/tP[3], tP[2]/tP[3]);
	VV = Geom::Vec3f(tQ[0]/tQ[3] - PP[0], tQ[1]/tQ[3] - PP[1], tQ[2]/tQ[3]- PP[2]);
}




bool Pickable::pick(const Geom::Vec3f& P, const Geom::Vec3f& V,  Geom::Vec3f& I, float epsilon)
{
	Geom::Vec3f PP;
	Geom::Vec3f VV;
	invertPV(P,V,m_transfo,PP,VV);

	return m_drawable->pick(PP,VV,I,epsilon) != 0;

}


void Pickable::draw()
{
	glm::mat4 store = Utils::GLSLShader::currentTransfo();
	Utils::GLSLShader::currentTransfo() *= m_transfo;
	Utils::GLSLShader::updateCurrentMatrices();
	m_drawable->draw();
	Utils::GLSLShader::currentTransfo() = store;
	Utils::GLSLShader::updateCurrentMatrices();
}

void Pickable::drawNoTransfo()
{
	m_drawable->draw();
}



glm::mat4&  Pickable::transfo()
{
	return m_transfo;
}

void Pickable::rotate(float angle, const Geom::Vec3f& Axis)
{
	m_transfo = glm::rotate(m_transfo, angle, glm::vec3(Axis[0],Axis[1],Axis[2]));
}
//void Pickable::rotate(float angle, const Geom::Vec3f& Axis)
//{
//	glm::mat4 tr = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(Axis[0],Axis[1],Axis[2]));
//	m_transfo = tr*m_transfo;
//}

void Pickable::translate(const Geom::Vec3f& P)
{
	m_transfo = glm::translate(m_transfo, glm::vec3(P[0],P[1],P[2]));
}
//void Pickable::translate(const Geom::Vec3f& P)
//{
//	glm::mat4 tr = glm::translate(glm::mat4(1.0f), glm::vec3(P[0],P[1],P[2]));
//	m_transfo = tr*m_transfo;
//}

void Pickable::scale(const Geom::Vec3f& S)
{
	m_transfo = glm::scale(m_transfo, glm::vec3(S[0],S[1],S[2]));
}
//void Pickable::scale(const Geom::Vec3f& S)
//{
//	glm::mat4 tr = glm::scale(glm::mat4(1.0f), glm::vec3(S[0],S[1],S[2]));
//	m_transfo = tr*m_transfo;
//}


// TODO check why BUG
void Pickable::randomOrientation()
{
	Geom::Vec3f V1(float(rand() - RAND_MAX/2), float(rand() - RAND_MAX/2), float(rand() - RAND_MAX/2));
	V1.normalize();
	float angle = float(rand()%360);
	rotate(angle,V1);
}

void Pickable::randomScale(float min, float max)
{
	const unsigned int MAX_NB=10000;
	float amp = (max - min)/MAX_NB;
	float sx = float((rand()%MAX_NB))*amp + min;
	float sy = float((rand()%MAX_NB))*amp + min;
	float sz = float((rand()%MAX_NB))*amp + min;
	scale(Geom::Vec3f(sx,sy,sz));
}

void Pickable::randomUniformScale(float min, float max)
{
	const unsigned int MAX_NB=10000;
	float amp = (max - min)/MAX_NB;
	float sc = float((rand()%MAX_NB))*amp + min;

	scale(Geom::Vec3f(sc,sc,sc));
}



float Pickable::distancefrom(const Geom::Vec3f& P)
{
	Geom::Vec3f origin(m_transfo[3][0],m_transfo[3][1],m_transfo[3][2]);
	origin -= P;
	return float(origin.norm());
}


Pickable* Pickable::pick(const std::vector<Pickable*>& picks,const Geom::Vec3f& P, const Geom::Vec3f& V)
{
	float mdist = std::numeric_limits<float>::max();
	Pickable* res=NULL;
	Geom::Vec3f I;

	for (std::vector<Utils::Pickable*>::const_iterator it=picks.begin(); it != picks.end(); ++it)
	{
		if ((*it)->pick(P,V,I))
		{
			std::cout << "I="<<I <<std::endl;

//			float dist = (*it)->distancefrom(P);
			glm::mat4 tr = (*it)->transfo();
			glm::vec4 ii(I[0],I[1],I[2],1.0f);
			glm::vec4 IWglm = tr*ii;
			Geom::Vec3f IW(IWglm[0]/IWglm[3],IWglm[1]/IWglm[3],IWglm[2]/IWglm[3]);
			IW -= P;
			float dist = float(IW.norm());
			if (dist < mdist)
			{
				res = *it;
				mdist = dist;
			}
		}
	}
	return res;
}


bool Pickable::distOrder(const std::pair<float, Pickable*>& e1, const std::pair<float, Pickable*>& e2)
{
	return (e1.first < e2.first);
}

std::vector<Pickable*> Pickable::sortedPick(std::vector<Pickable*>& picks, const Geom::Vec3f& P, const Geom::Vec3f& V)
{
	Geom::Vec3f I;
	std::vector< std::pair<float, Pickable*> > sorted;
	sorted.reserve(picks.size());

	for (std::vector<Utils::Pickable*>::const_iterator it=picks.begin(); it != picks.end(); ++it)
	{
		if ((*it)->pick(P,V,I))
		{
//			float dist = (*it)->distancefrom(P);
			glm::mat4 tr = (*it)->transfo();
			glm::vec4 ii(I[0],I[1],I[2],1.0f);
			glm::vec4 IWglm = tr*ii;
			Geom::Vec3f IW(IWglm[0]/IWglm[3],IWglm[1]/IWglm[3],IWglm[2]/IWglm[3]);
			IW -= P;
			float dist = IW.norm();
			sorted.push_back(std::pair<float, Pickable*>(dist,*it));
		}
	}

	std::sort(sorted.begin(), sorted.end(),distOrder);

	std::vector<Pickable*> res;
	res.reserve(sorted.size());
	for (unsigned int i=0; i<sorted.size(); ++i)
	{
		res.push_back(sorted[i].second);
	}

	return res;
}


Geom::Vec3f Pickable::getPosition()
{
	return Geom::Vec3f(m_transfo[3][0],m_transfo[3][1],m_transfo[3][2]);
}

Geom::Vec3f Pickable::getAxisScale(unsigned int ax, float& scale)
{
	Geom::Vec3f tempo(m_transfo[ax][0],m_transfo[ax][1],m_transfo[ax][2]);
    scale = tempo.norm();
    tempo/= scale ;
	return tempo;
}




Grid::Grid(unsigned int sub)
{
	changeTopo(sub);
}

void Grid::changeTopo(unsigned int sub)
{
	std::vector<Geom::Vec3f> points;
	points.resize((sub+1)*2*2);

	m_nb=0;

	for (unsigned int i=0; i<=sub; ++i)
	{
		float a = -1.0f + (2.0f/sub)*i;

		points[4*i] = Geom::Vec3f(a,-1.0f,0.0f);
		points[4*i+1] = Geom::Vec3f(a,1.0f,0.0f);
		points[4*i+2] = Geom::Vec3f(-1.0f,a,0.0f);
		points[4*i+3] = Geom::Vec3f(1.0f,a,0.0f);
		m_nb+=4;
	}

	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, m_nb * sizeof(Geom::Vec3f), &(points[0]), GL_STREAM_DRAW);
}

void Grid::updatePrecisionDrawing(unsigned int sub, unsigned int sub2)
{
	m_sub1 = sub;
	m_sub2 = sub2;
	changeTopo(sub);
}


unsigned int Grid::pick(const Geom::Vec3f& P, const Geom::Vec3f& V, Geom::Vec3f& I, float /*epsilon*/)
{
	if (fabs(V[2])>=0.0000001f)
	{
		float a = -1.0f*P[2]/V[2];
		I = Geom::Vec3f(P+a*V);	// intersection with plane z=0

		if ( (fabs(I[0])<=1.0f) && (fabs(I[1])<=1.0f) )
			return 1;
	}

	return 0;
}






Sphere::Sphere(unsigned int par, unsigned int mer)
{
	changeTopo(par,mer);
}


void Sphere::changeTopo(unsigned int parp, unsigned int mer)
{
	if (parp<2)
		parp=2;
	if (mer<2)
		mer=2;
	// to obtain right number of slice
	unsigned int par = parp-1;

	unsigned int merfactor=1;
	unsigned int parfactor=1;

	if (mer<8)
		merfactor = 8;
	else if (mer<16)
		merfactor = 4;
	else if (mer<32)
		merfactor = 2;

	if (par<8)
		parfactor = 8;
	else if (par<16)
		parfactor = 4;
	else if (par<32)
		parfactor = 2;

	unsigned int merAll = merfactor * mer;
	unsigned int parAll = parfactor* (par+1);

	std::vector<Geom::Vec3f> points;
	points.reserve(parAll*merAll+2);

	for (unsigned int i=0; i<parAll; ++i)
	{
		float beta = float(i+1)*M_PI/float(parAll+1);
		float z = -cos(beta);

		float radius = sin(beta);

		for (unsigned int j=0; j<merAll; ++j)
		{
			float alpha = 2.0f*float(j)*M_PI/float(merAll);
			float x = radius*cos(alpha);
			float y = radius*sin(alpha);
			points.push_back(Geom::Vec3f(x,y,z));
		}
	}
	//poles
	unsigned int north = points.size();
	points.push_back(Geom::Vec3f(0.0f,0.0f,-1.0f));
	unsigned int south = points.size();
	points.push_back(Geom::Vec3f(0.0f,0.0f,1.0f));

	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Geom::Vec3f), &(points[0]), GL_STREAM_DRAW);


	// indices
	std::vector<unsigned int> tableIndices;
	tableIndices.reserve(2*(mer*parAll + par*merAll + 8));


	for (unsigned int i=0; i<par; ++i)
	{
		unsigned int k = i*parfactor*merAll + parfactor*merAll;
		for (unsigned int j=0; j<merAll; ++j)
		{
			tableIndices.push_back(k+j);
			tableIndices.push_back(k+(j+1)%merAll);
		}
	}


	for (unsigned int i=0; i<mer; ++i)
	{
		tableIndices.push_back(north);
		unsigned int k=i*merfactor;
		tableIndices.push_back(k);
		for (unsigned int j=1; j<parAll; ++j)
		{
			tableIndices.push_back(k);
			k += merAll;
			tableIndices.push_back(k);
		}
		tableIndices.push_back(k);
		tableIndices.push_back(south);
	}

	m_nb=tableIndices.size();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_ind);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nb*sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);
}

void Sphere::updatePrecisionDrawing(unsigned int sub, unsigned int sub2)
{
	m_sub1 = sub;
	m_sub2 = sub2;

	if (sub2)
		changeTopo(sub,sub2);
	else
		changeTopo(sub,sub);
}



void Sphere::draw()
{
	m_shader->enableVertexAttribs();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_ind);
	glDrawElements(GL_LINES, m_nb, GL_UNSIGNED_INT, 0);
	m_shader->disableVertexAttribs();
}


unsigned int Sphere::pick(const Geom::Vec3f& P, const Geom::Vec3f& V, Geom::Vec3f& I, float /*epsilon*/)
{
	float dist = Geom::squaredDistanceLine2Point<Geom::Vec3f>(P,V,V*V, Geom::Vec3f(0.0f,0.0f,0.0f));

	if (dist > 1.0f)
		return 0;


	I=P;
	I.normalize();			// grossiere approximation TODO ameliorer approxim ?

	return 1;
}





Cone::Cone(unsigned int par, unsigned int mer)
{
	changeTopo(par,mer);
}


void Cone::changeTopo(unsigned int par, unsigned int mer)
{
	if (par<2)
		par=2;
	if (mer<2)
		mer=2;

	unsigned int merfactor=1;

	if (mer<8)
		merfactor = 8;
	else if (mer<16)
		merfactor = 4;
	else if (mer<32)
		merfactor = 2;

	unsigned int merAll = merfactor * mer;

	std::vector<Geom::Vec3f> points;
	points.reserve(par*merAll+2);

	for (unsigned int i=0; i<par; ++i)
	{
		float radius = 1.0f / float(par) * float(i+1);
		float z = 1.0f - 2.0f*radius;

		for (unsigned int j=0; j<merAll; ++j)
		{
			float alpha = 2.0f*float(j)*M_PI/float(merAll);
			float x = radius*cos(alpha);
			float y = radius*sin(alpha);
			points.push_back(Geom::Vec3f(x,y,z));
		}
	}
	//poles
	points.push_back(Geom::Vec3f(0.0f,0.0f,1.0f));
	points.push_back(Geom::Vec3f(0.0f,0.0f,-1.0f));

	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Geom::Vec3f), &(points[0]), GL_STREAM_DRAW);

	// indices
	std::vector<unsigned int> tableIndices;
	tableIndices.reserve(4*par*mer+4*mer);


	for (unsigned int i=0; i<par; ++i)
	{
		unsigned int k = i*merAll;
		for (unsigned int j=0; j<merAll; ++j)
		{
			tableIndices.push_back(k+j);
			tableIndices.push_back(k+(j+1)%merAll);
		}
	}

	for (unsigned int j=0; j<mer; ++j)
	{
		tableIndices.push_back(par*merAll);
		tableIndices.push_back(j*merfactor + (par-1)*merAll);
		tableIndices.push_back(j*merfactor + (par-1)*merAll);
		tableIndices.push_back(par*merAll+1);

	}


	m_nb=tableIndices.size();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_ind);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nb*sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);
}

void Cone::updatePrecisionDrawing(unsigned int sub, unsigned int sub2)
{
	m_sub1 = sub;
	m_sub2 = sub2;

	if (sub2)
		changeTopo(sub,sub2);
	else
		changeTopo(sub,sub);
}







unsigned int Cone::pick(const Geom::Vec3f& P, const Geom::Vec3f& V, Geom::Vec3f& I, float /*epsilon*/)
{
	Geom::Vec3f Z,Q;
	if (Geom::lineLineClosestPoints<Geom::Vec3f>(P, V, Geom::Vec3f(0.0f,0.0f,0.0f), Geom::Vec3f(0.0f,0.0f,1.0f), Q, Z))
	{
		if ((Q[2]>=-1.0f)&&(Q[2]<=1.0f))
		{
			float dist = Q[0]*Q[0] + Q[1]*Q[1];
			float cdist = (1.0f - Q[2])/2.0f;
			if (dist <= cdist*cdist) // squared !!
			{
				I = Q; 					// WARNING VERY BAD APPROXIMATON : TODO better approxim.
				return 1;
			}

		}
		// else check inter with base
		// Z=-1
		float a = (-1.0f - P[2]) / V[2];
		I = Geom::Vec3f(P+a*V);
		float dist = I[0]*I[0] + I[1]*I[1];
		if (dist > 1.0f)
			return 0;
		return 1;
	}

	// ray in Z direction
	float dist = P[0]*P[0] + P[1]*P[1];
	if (dist > 1.0f)
		return 0;
	I=P;
	I[2]=-1.0f;
	return 1;
}




Cylinder::Cylinder(unsigned int par, unsigned int mer)
{
	changeTopo(par,mer);
}


void Cylinder::changeTopo(unsigned int parp, unsigned int mer)
{
	if (parp<2)
		parp=2;
	if (mer<2)
		mer=2;

	// to obtain right number of slice (with almost same code as sphere)
	unsigned int par = parp+1;

	unsigned int merfactor=1;

	if (mer<8)
		merfactor = 8;
	else if (mer<16)
		merfactor = 4;
	else if (mer<32)
		merfactor = 2;

	unsigned int merAll = merfactor * mer;

	std::vector<Geom::Vec3f> points;
	points.reserve(par*merAll+2);

	for (unsigned int i=0; i<par; ++i)
	{
		float z = -1.0f + 2.0f/float(par-1) * float(i);

		for (unsigned int j=0; j<merAll; ++j)
		{
			float alpha = 2.0f*float(j)*M_PI/float(merAll);
			float x = cos(alpha);
			float y = sin(alpha);
			points.push_back(Geom::Vec3f(x,y,z));
		}
	}
	//poles
	points.push_back(Geom::Vec3f(0.0f,0.0f,1.0f));
	points.push_back(Geom::Vec3f(0.0f,0.0f,-1.0f));

	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Geom::Vec3f), &(points[0]), GL_STREAM_DRAW);

	// indices
	std::vector<unsigned int> tableIndices;
	tableIndices.reserve(4*par*mer+4*mer);


	for (unsigned int i=0; i<par; ++i)
	{
		unsigned int k = i*merAll;
		for (unsigned int j=0; j<merAll; ++j)
		{
			tableIndices.push_back(k+j);
			tableIndices.push_back(k+(j+1)%merAll);
		}
	}

	for (unsigned int j=0; j<mer; ++j)
	{
		tableIndices.push_back(par*merAll);
		tableIndices.push_back(j*merfactor + (par-1)*merAll);
		tableIndices.push_back(j*merfactor + (par-1)*merAll);
		tableIndices.push_back(j*merfactor );
		tableIndices.push_back(j*merfactor );
		tableIndices.push_back(par*merAll+1);

	}


	m_nb=tableIndices.size();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_ind);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nb*sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);

}

void Cylinder::updatePrecisionDrawing(unsigned int sub, unsigned int sub2)
{
	m_sub1 = sub;
	m_sub2 = sub2;

	if (sub2)
		changeTopo(sub,sub2);
	else
		changeTopo(sub,sub);
}



unsigned int Cylinder::pick(const Geom::Vec3f& P, const Geom::Vec3f& V,  Geom::Vec3f& I, float /*epsilon*/)
{
	Geom::Vec3f Z,Q;
	if (Geom::lineLineClosestPoints<Geom::Vec3f>(P, V, Geom::Vec3f(0.0f,0.0f,0.0f), Geom::Vec3f(0.0f,0.0f,1.0f), Q, Z))
	{
		if ((Q[2]>=-1.0f)&&(Q[2]<=1.0f))
		{
			float dist = Q[0]*Q[0] + Q[1]*Q[1];
			if (dist < 1.0f)
			{
				I = Q; 					// WARNING VERY BAD APPROXIMATON : TODO better approxim.
				return 1;
			}

		}
		// else check inter with bases
		// Z=1
		float a = (1.0f - P[2]) / V[2];
		I = Geom::Vec3f(P+a*V);
		float dist = I[0]*I[0] + I[1]*I[1];
		if (dist < 1.0f)
			return 1;
		// Z=-1
		a = (-1.0f - P[2]) / V[2];
		I = Geom::Vec3f(P+a*V);
		dist = I[0]*I[0] + I[1]*I[1];
		if (dist < 1.0f)
			return 1;
		//else no inter
		return 0;
	}

	// ray in Z direction
	float dist = P[0]*P[0] + P[1]*P[1];
	if (dist > 1.0f)
		return 0;
	I=P;
	if (V[2]<0.0f)
		I[2]=-1.0f;
	else
		I[2]=1.0f;
	return 1;
}







Cube::Cube(unsigned int sub)
{
	changeTopo(sub);
}

void Cube::changeTopo(unsigned int sub)
{
	// subdiv = number of internal points on each edge
	unsigned int subdiv = sub-1;
	std::vector<Geom::Vec3f> points;
	points.reserve(8+12*subdiv);

	points.push_back(Geom::Vec3f(-1.0f,-1.0f,-1.0f));
	points.push_back(Geom::Vec3f( 1.0f,-1.0f,-1.0f));
	points.push_back(Geom::Vec3f( 1.0f, 1.0f,-1.0f));
	points.push_back(Geom::Vec3f(-1.0f, 1.0f,-1.0f));
	points.push_back(Geom::Vec3f(-1.0f,-1.0f, 1.0f));
	points.push_back(Geom::Vec3f( 1.0f,-1.0f, 1.0f));
	points.push_back(Geom::Vec3f( 1.0f, 1.0f, 1.0f));
	points.push_back(Geom::Vec3f(-1.0f, 1.0f, 1.0f));

	for (unsigned int i=0; i< subdiv; ++i)
	{
		float v = -1.0f + float(2*i+2)/float(subdiv+1);

		points.push_back(Geom::Vec3f(-1.0f,-1.0f, v));
		points.push_back(Geom::Vec3f(-1.0f, 1.0f, v));
		points.push_back(Geom::Vec3f( 1.0f, 1.0f, v));
		points.push_back(Geom::Vec3f( 1.0f,-1.0f, v));

		points.push_back(Geom::Vec3f(-1.0f, v,-1.0f));
		points.push_back(Geom::Vec3f(-1.0f, v, 1.0f));
		points.push_back(Geom::Vec3f( 1.0f, v, 1.0f));
		points.push_back(Geom::Vec3f( 1.0f, v,-1.0f));

		points.push_back(Geom::Vec3f(v,-1.0f,-1.0f));
		points.push_back(Geom::Vec3f(v,-1.0f, 1.0f));
		points.push_back(Geom::Vec3f(v, 1.0f, 1.0f));
		points.push_back(Geom::Vec3f(v, 1.0f,-1.0f));
	}

	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Geom::Vec3f), &(points[0]), GL_STREAM_DRAW);

	// indices
	std::vector<unsigned int> tableIndices;
	tableIndices.reserve(24+24*subdiv);

	for (unsigned int i=0; i<4; ++i)
	{
		tableIndices.push_back(i);
		tableIndices.push_back((i+1)%4);
		tableIndices.push_back(4 + i);
		tableIndices.push_back(4 + (i+1)%4);
		tableIndices.push_back(i);
		tableIndices.push_back(4 + i);
	}
	for (unsigned int i=0; i< subdiv; ++i)
	{
		for (unsigned int j=0; j< 3; ++j) // direction X Y Z (or edge)
		{
			for (unsigned int k=0; k< 4; ++k)	// turn around cube
			{
				tableIndices.push_back(8 + i*12 + (j*4) + k);
				tableIndices.push_back(8 + i*12 + (j*4) + (k+1)%4);
			}
		}
	}

	m_nb=tableIndices.size();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_ind);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nb*sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);
}


void Cube::updatePrecisionDrawing(unsigned int sub, unsigned int sub2)
{
	m_sub1 = sub;
	m_sub2 = sub2;

	changeTopo(sub);
}



void Cube::draw()
{
	m_shader->enableVertexAttribs();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_ind);
	glDrawElements(GL_LINES, m_nb, GL_UNSIGNED_INT, 0);
	m_shader->disableVertexAttribs();
}


unsigned int Cube::pick(const Geom::Vec3f& P, const Geom::Vec3f& V, Geom::Vec3f& I, float /*epsilon*/)
{

//	// firs quick picking with bounding sphere
	float dist2 = Geom::squaredDistanceLine2Point<Geom::Vec3f>(P,V,V*V, Geom::Vec3f(0.0f,0.0f,0.0f));
	if (dist2 > 3.0f)
		return 0;

	I=P;
	I.normalize();			// grossiere approximation TODO amelioerer approxim ?

	std::vector<Geom::Vec3f> intersections;

	for (unsigned int i=0; i<3; ++i)
	{
		if (fabs(V[i])>=0.0000001f)
		{
			float a = (-1.0f-P[i])/V[i];
			Geom::Vec3f Q = Geom::Vec3f(P+a*V);	// intersection with plane z=0
			if ( (fabs(Q[(i+1)%3])<=1.0f) && (fabs(Q[(i+2)%3])<=1.0f) )
				return 1;
			a = (1.0f-P[i])/V[i];
			Q = Geom::Vec3f(P+a*V);	// intersection with plane z=0
			if ( (fabs(Q[(i+1)%3])<=1.0f) && (fabs(Q[(i+2)%3])<=1.0f) )
				return 1;
		}
	}

	return 0;
}




IcoSphere::IcoSphere(unsigned int sub):
		m_sub(0xffffffff)
{
	changeTopo(sub);
}

unsigned int IcoSphere::insertPoint(std::vector<Geom::Vec3f>& points, const Geom::Vec3f& P)
{
	for (unsigned int i=0; i< points.size();++i)
		if (((P-points[i]).norm2())< 0.00000001f)
			return i;
	points.push_back(P);
	return points.size()-1;
}

void IcoSphere::subdivide(std::vector<unsigned int>& triangles, std::vector<Geom::Vec3f>& points)
{
	std::vector<unsigned int> newTriangles;
	newTriangles.reserve(triangles.size()*4);

	unsigned int nbtris = triangles.size()/3;

	for(unsigned int t=0; t<nbtris;++t)
	{
		const Geom::Vec3f& Pa =points[triangles[3*t]];
		const Geom::Vec3f& Pb =points[triangles[3*t+1]];
		const Geom::Vec3f& Pc =points[triangles[3*t+2]];

		Geom::Vec3f Pab =(Pa+Pb)/2.0f;
		Pab.normalize();
		Geom::Vec3f Pac =(Pa+Pc)/2.0f;
		Pac.normalize();
		Geom::Vec3f Pbc =(Pb+Pc)/2.0f;
		Pbc.normalize();

		unsigned int iAB = insertPoint(points,Pab);
		unsigned int iAC = insertPoint(points,Pac);
		unsigned int iBC = insertPoint(points,Pbc);

		newTriangles.push_back(triangles[3*t]); //Pa
		newTriangles.push_back(iAB); //Pab
		newTriangles.push_back(iAC); //Pac

		newTriangles.push_back(triangles[3*t+1]); //Pb
		newTriangles.push_back(iBC); //Pbc
		newTriangles.push_back(iAB); //Pab

		newTriangles.push_back(triangles[3*t+2]); //Pc
		newTriangles.push_back(iAC); //Pac
		newTriangles.push_back(iBC); //Pbc

		newTriangles.push_back(iAB); //Pab
		newTriangles.push_back(iBC); //Pbc
		newTriangles.push_back(iAC); //Pac
	}

	triangles.swap(newTriangles);
}

void IcoSphere::changeTopo(unsigned int sub)
{
	if (sub<2)
		sub=2;
	int subd = int(log(0.5*sub)/log(2.0))-1;
	if (subd<0)
		subd=0;

	changeTopoSubdivision(subd);
}

void IcoSphere::changeTopoSubdivision(unsigned int sub)
{
	if (m_sub == sub)
		return;

	m_sub = sub;

	unsigned int subEdge = (unsigned int)(powf(2.0f,4.0f-sub));

	std::vector<Geom::Vec3f> points;
	points.reserve(10000);

	unsigned int uitriangles[60]={0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
			1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
			3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
			4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1 };

	std::vector<unsigned int> triangles;
	triangles.reserve(60);
	for (unsigned int i=0; i<60; ++i)
		triangles.push_back(uitriangles[i]);


	// create 12 vertices of an icosahedron
	float t = (1.0f + sqrtf(5.0f)) / 2.0f;

	points.push_back(Geom::Vec3f(-1,  t,  0));
	points.push_back(Geom::Vec3f( 1,  t,  0));
	points.push_back(Geom::Vec3f(-1, -t,  0));
	points.push_back(Geom::Vec3f( 1, -t,  0));
	points.push_back(Geom::Vec3f( 0, -1,  t));
	points.push_back(Geom::Vec3f( 0,  1,  t));
	points.push_back(Geom::Vec3f( 0, -1, -t));
	points.push_back(Geom::Vec3f( 0,  1, -t));
	points.push_back(Geom::Vec3f( t,  0, -1));
	points.push_back(Geom::Vec3f( t,  0,  1));
	points.push_back(Geom::Vec3f(-t,  0, -1));
	points.push_back(Geom::Vec3f(-t,  0,  1));

	for (std::vector<Geom::Vec3f>::iterator pt=points.begin(); pt!=points.end(); ++pt)
		pt->normalize();

	// subdivide
	for (unsigned int i=0; i<sub; ++i)
		subdivide(triangles, points);


	// add some vertices on edges for nice round edges
	unsigned int idxNewPoints=points.size();

	for (unsigned int i=0; i<triangles.size()/3; ++i)
	{
		unsigned int a = triangles[3*i];
		unsigned int b = triangles[3*i+1];
		unsigned int c = triangles[3*i+2];

		if (a < b)
		{
			for(unsigned int i=0; i < subEdge; ++i)
			{
				float x = float(i+1)/float(subEdge+1);
				Geom::Vec3f P(x*points[b] + (1.0f-x)*points[a]);
				P.normalize();
				points.push_back(P);
			}
		}
		if (b < c)
		{
			for(unsigned int i=0; i < subEdge; ++i)
			{
				float x = float(i+1)/float(subEdge+1);
				Geom::Vec3f P(x*points[c] + (1.0f-x)*points[b]);
				P.normalize();
				points.push_back(P);
			}

		}
		if (c < a)
		{
			for(unsigned int i=0; i < subEdge; ++i)
			{
				float x = float(i+1)/float(subEdge+1);
				Geom::Vec3f P(x*points[a] + (1.0f-x)*points[c]);
				P.normalize();
				points.push_back(P);
			}
		}
	}

	// send buffer
	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Geom::Vec3f), &(points[0]), GL_STREAM_DRAW);

	// indices
	std::vector<unsigned int> tableIndices;
	tableIndices.reserve(triangles.size()/2 + 1000);

	unsigned int k=0;
	for (unsigned int i=0; i<triangles.size()/3; ++i)
	{
		unsigned int a = triangles[3*i];
		unsigned int b = triangles[3*i+1];
		unsigned int c = triangles[3*i+2];

		if (a < b)
		{
			tableIndices.push_back(a);
			for(unsigned int j=0; j < subEdge; ++j)
			{
				tableIndices.push_back(idxNewPoints+subEdge*k+j);
				tableIndices.push_back(idxNewPoints+subEdge*k+j);
			}
			tableIndices.push_back(b);
			k++;
		}
		if (b < c)
		{
			tableIndices.push_back(b);
			for(unsigned int j=0; j < subEdge; ++j)
			{
				tableIndices.push_back(idxNewPoints+subEdge*k+j);
				tableIndices.push_back(idxNewPoints+subEdge*k+j);
			}
			tableIndices.push_back(c);
			k++;
		}
		if (c < a)
		{
			tableIndices.push_back(c);
			for(unsigned int j=0; j < subEdge; ++j)
			{
				tableIndices.push_back(idxNewPoints+subEdge*k+j);
				tableIndices.push_back(idxNewPoints+subEdge*k+j);
			}
			tableIndices.push_back(a);
			k++;
		}
	}

	m_nb=tableIndices.size();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_ind);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nb*sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);
}


void IcoSphere::updatePrecisionDrawing(unsigned int sub, unsigned int sub2)
{
	m_sub1 = sub;
	m_sub2 = sub2;
	changeTopo(sub);
}



}
}








