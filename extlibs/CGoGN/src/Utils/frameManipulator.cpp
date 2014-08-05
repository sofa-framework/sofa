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

#include "Utils/frameManipulator.h"

#include "Geometry/distances.h"
#include "Geometry/intersection.h"

#include <glm/ext.hpp>

#define _USE_MATH_DEFINES
#include <math.h>


namespace CGoGN
{

namespace Utils
{

const float FrameManipulator::ring_half_width = 0.08f;

FrameManipulator::FrameManipulator():
		m_highlighted(NONE),
		m_rotations(1.0f),
		m_scaleRendering(1.0f),
		m_trans(0.0f,0.0f,0.0f),
		m_scale(1.0f,1.0f,1.0f)
{

	for (unsigned int i=0; i<11; ++i)
	{
		m_locked_axis[i]=false;
		m_lockedPicking_axis[i]=false;
	}

	m_vboPos = new VBO();
	m_vboPos->setDataSize(3);

	m_vboCol = new VBO();
	m_vboCol->setDataSize(3);

	m_shader = new ShaderSimpleColor();

	m_shader->setAttributePosition(m_vboPos);

	GLSLShader::registerShader(NULL, m_shader);

	std::vector<Geom::Vec3f> points;
	points.reserve(6*nb_segments+30);
	points.resize(6*nb_segments+6);

	unsigned int second = 2*(nb_segments+1);
	unsigned int third = 4*(nb_segments+1);

	for (unsigned int i=0; i<=nb_segments; ++i)
	{
		float alpha = float(i)*M_PI/float(nb_segments/2);
		float x = (1.0f+ring_half_width) * cos(alpha);
		float y = (1.0f+ring_half_width) * sin(alpha);
		float xx = (1.0f-ring_half_width) * cos(alpha);
		float yy = (1.0f-ring_half_width) * sin(alpha);

		points[2*i] = Geom::Vec3f(0.0f,x,y);
		points[2*i+1] = Geom::Vec3f(0.0f,xx,yy);
		points[second + 2*i] = Geom::Vec3f(x,0.0f,y);
		points[second + 2*i+1] = Geom::Vec3f(xx,0.0f,yy);
		points[third + 2*i] = Geom::Vec3f(x,y,0.0f);
		points[third + 2*i+1] = Geom::Vec3f(xx,yy,0.0f);
	}

	points.push_back(Geom::Vec3f(0.0f,0.0f,0.0f));
	points.push_back(Geom::Vec3f(0.23f,0.0f,0.0f));

	points.push_back(Geom::Vec3f(0.0f,0.0f,0.0f));
	points.push_back(Geom::Vec3f(0.0f,0.23f,0.0f));

	points.push_back(Geom::Vec3f(0.0f,0.0f,0.0f));
	points.push_back(Geom::Vec3f(0.0f,0.0f,0.23f));

	points.push_back(Geom::Vec3f(0.27f,0.0f,0.0f));
	points.push_back(Geom::Vec3f(0.75f,0.0f,0.0f));
	points.push_back(Geom::Vec3f(0.9f,0.0f,0.0f));
	points.push_back(Geom::Vec3f(0.7f,-0.03f,0.0f));
	points.push_back(Geom::Vec3f(0.7f,0.0f,-0.03f));
	points.push_back(Geom::Vec3f(0.7f,0.03f,0.0f));
	points.push_back(Geom::Vec3f(0.7f,0.0f,0.03f));
	points.push_back(Geom::Vec3f(0.7f,-0.03f,0.0f));

	points.push_back(Geom::Vec3f(0.0f,  0.27f,0.0f));
	points.push_back(Geom::Vec3f(0.0f,  0.75f, 0.0f));
	points.push_back(Geom::Vec3f(0.0f,  0.9f, 0.0f));
	points.push_back(Geom::Vec3f(0.0f,  0.7f, 0.03f));
	points.push_back(Geom::Vec3f(0.03f, 0.7f, 0.0f));
	points.push_back(Geom::Vec3f(0.0f,  0.7f,-0.03f));
	points.push_back(Geom::Vec3f(-0.03f,0.7f, 0.0f));
	points.push_back(Geom::Vec3f(0.0f,  0.7f, 0.03f));

	points.push_back(Geom::Vec3f(0.0f,0.0f,  0.27f));
	points.push_back(Geom::Vec3f(0.0f,0.0f,  0.75f));
	points.push_back(Geom::Vec3f(0.0f,0.0f,  0.9f));
	points.push_back(Geom::Vec3f(0.03f,0.0f, 0.7f));
	points.push_back(Geom::Vec3f(0.0f,0.03f, 0.7f));
	points.push_back(Geom::Vec3f(-0.03f,0.0f,0.7f));
	points.push_back(Geom::Vec3f(0.0f,-0.03f,0.7f));
	points.push_back(Geom::Vec3f(0.03f,0.0f, 0.7f));

	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Geom::Vec3f), &(points[0]), GL_STREAM_DRAW);

	setLengthAxes();
}

void FrameManipulator::setSize(float radius)
{
	if (m_scaleRendering >0.0f)
		m_scaleRendering = radius;
}

void FrameManipulator::addSize(float radius)
{
	m_scaleRendering += radius;
	if (m_scaleRendering <= 0.0f)
		m_scaleRendering -= radius;
}

float FrameManipulator::getSize()
{
	return m_scaleRendering;
}

void FrameManipulator::draw()
{
	Utils::GLSLShader::pushTransfo();
	Utils::GLSLShader::applyTransfo(transfoRenderFrame());
	Utils::GLSLShader::updateCurrentMatrices();

 	glPushAttrib(GL_LINE_BIT);
	m_shader->enableVertexAttribs();

	if (!m_locked_axis[Xr])
	{
		if (m_highlighted == Xr)
			m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
		else
			m_shader->setColor(Geom::Vec4f(1.0f,0.0f,0.0f,0.0f));
		m_shader->bind();
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 2*nb_segments+2);
	}

	if (!m_locked_axis[Yr])
	{
		if (m_highlighted == Yr)
			m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
		else
			m_shader->setColor(Geom::Vec4f(0.0f,1.0f,0.0f,0.0f));
		m_shader->bind();
		glDrawArrays(GL_TRIANGLE_STRIP, 2*nb_segments+2, 2*nb_segments+2);
	}

	if (!m_locked_axis[Zr])
	{
		if (m_highlighted == Zr)
			m_shader->setColor(Geom::Vec4f(1.0,1.0,0.0f,0.0f));
		else
			m_shader->setColor(Geom::Vec4f(0.0f,0.0f,1.0f,0.0f));
		m_shader->bind();
		glDrawArrays(GL_TRIANGLE_STRIP, 4*nb_segments+4, 2*nb_segments+2);
	}

	if ((!m_locked_axis[CENTER]) && (m_highlighted == CENTER))
	{
		glLineWidth(6.0f);
		m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
		m_shader->bind();
		glDrawArrays(GL_LINES, 6*nb_segments+6, 6);
	}
	else
	{
		if (!m_locked_axis[Xs])
		{
			if (m_highlighted == Xs)
			{
				glLineWidth(6.0f);
				m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
			}
			else
			{
				glLineWidth(3.0f);
				m_shader->setColor(Geom::Vec4f(1.0f,0.0f,0.0f,0.0f));
			}
			m_shader->bind();
			glDrawArrays(GL_LINES, 6*nb_segments+6, 2);
		}

		if (!m_locked_axis[Ys])
		{
			if (m_highlighted == Ys)
			{
				glLineWidth(6.0f);
				m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
			}
			else
			{
				glLineWidth(3.0f);
				m_shader->setColor(Geom::Vec4f(0.0f,0.7f,0.0f,0.0f));
			}
			m_shader->bind();
			glDrawArrays(GL_LINES, 6*nb_segments+8, 2);
		}

		if (!m_locked_axis[Zs])
		{
			if (m_highlighted == Zs)
			{
				glLineWidth(6.0f);
				m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
			}
			else
			{
				glLineWidth(3.0f);
				m_shader->setColor(Geom::Vec4f(0.0f,0.0f,0.7f,0.0f));
			}
			m_shader->bind();
			glDrawArrays(GL_LINES, 6*nb_segments+10, 2);
		}
	}


	if (!m_locked_axis[Xt])
	{
		if (m_highlighted == Xt)
		{
			m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
			glLineWidth(6.0f);
		}
		else
		{
			glLineWidth(3.0f);
			m_shader->setColor(Geom::Vec4f(1.0f,0.0f,0.0f,0.0f));
		}

		m_shader->bind();
		glDrawArrays(GL_LINES, 6*nb_segments+12, 2);
		glDrawArrays(GL_TRIANGLE_FAN, 6*nb_segments+14, 6);
	}

	if (!m_locked_axis[Yt])
	{
		if (m_highlighted == Yt)
		{
			glLineWidth(6.0f);
			m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
		}
		else
		{
			glLineWidth(3.0f);
			m_shader->setColor(Geom::Vec4f(0.0f,1.0f,0.0f,0.0f));
		}
		m_shader->bind();
		glDrawArrays(GL_LINES, 6*nb_segments+20, 2);
		glDrawArrays(GL_TRIANGLE_FAN, 6*nb_segments+22, 6);
	}

	if (!m_locked_axis[Zt])
	{
		if (m_highlighted == Zt)
		{
			glLineWidth(6.0f);
			m_shader->setColor(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
		}
		else
		{
			glLineWidth(3.0f);
			m_shader->setColor(Geom::Vec4f(0.0f,0.0f,1.0f,0.0f));
		}
		m_shader->bind();
		glDrawArrays(GL_LINES, 6*nb_segments+28, 2);
		glDrawArrays(GL_TRIANGLE_FAN, 6*nb_segments+30, 6);
	}

 	m_shader->disableVertexAttribs();
 	glPopAttrib();

 	Utils::GLSLShader::popTransfo();
	Utils::GLSLShader::updateCurrentMatrices();
	m_shader->unbind();
}

void FrameManipulator::highlight(unsigned int axis)
{
	if (m_highlighted == axis)
		m_highlighted = NONE;
	else
		m_highlighted = axis;
}

unsigned int FrameManipulator::pick(const Geom::Vec3f& PP, const Geom::Vec3f& VV, float /*epsilon*/)
{
	Geom::Vec3f P,V;

	Pickable::invertPV(PP,VV, transfoRenderFrame(),P,V );

	// origin of frame
	Geom::Vec3f origin(0.0f,0.0f,0.0f);

	// intersection possible between line and frame (10% margin)?
	float dist2 = Geom::squaredDistanceLine2Point<Geom::Vec3f>(P,V,V*V,origin);

	float distMax= std::max(m_lengthAxes[0],std::max(m_lengthAxes[1],m_lengthAxes[2]));
	distMax *=3.6f;
	distMax= std::max(distMax,1.0f+ring_half_width);

	if (dist2 > distMax*distMax)
		return NONE;

	// click on center
	if (dist2 < 0.02f*0.02f)
	{
		if (axisPickable(CENTER))
			return CENTER;
		else
			return NONE;
	}

	float dist_target[9];
	float dist_cam[9];

	for (unsigned int i=0; i<9; ++i)
	{
		dist_target[i] = 2.0f;
		dist_cam[i] = std::numeric_limits<float>::max();
	}

	// Circles:
	// plane X=0
	Geom::Vec3f Qx;
	Geom::Intersection inter = Geom::intersectionLinePlane<Geom::Vec3f>(P,V,origin, Geom::Vec3f(1.0f,0.0f,0.0f), Qx);

	if (axisPickable(Xr))
	{
		if (inter == Geom::FACE_INTERSECTION)
			dist_target[3] = Qx.norm() - 1.0f;
		else if (inter == Geom::EDGE_INTERSECTION)
			dist_target[3] = sqrt(dist2) - 1.0f;

		if (fabs(dist_target[3]) < ring_half_width )
			dist_cam[3] = (P-Qx)*(P-Qx);
	}

	// plane Y=0
	Geom::Vec3f Qy;
	inter = Geom::intersectionLinePlane<Geom::Vec3f>(P,V,origin, Geom::Vec3f(0.0f,1.0f,0.0f), Qy);

	if (axisPickable(Yr))
	{
		if (inter == Geom::FACE_INTERSECTION)
			dist_target[4] = Qy.norm() - 1.0f;
		else if (inter == Geom::EDGE_INTERSECTION)
			dist_target[4] = sqrt(dist2) - 1.0f;

		if (fabs(dist_target[4]) < ring_half_width )
			dist_cam[4] = (P-Qy)*(P-Qy);
	}

	// plane Z=0
	Geom::Vec3f Qz;
	inter = Geom::intersectionLinePlane<Geom::Vec3f>(P,V,origin, Geom::Vec3f(0.0f,0.0f,1.0f), Qz);

	if (axisPickable(Zr))
	{
		if (inter == Geom::FACE_INTERSECTION)
			dist_target[5] = Qz.norm() - 1.0f;
		else if (inter == Geom::EDGE_INTERSECTION)
			dist_target[5] = sqrt(dist2) - 1.0f;

		if (fabs(dist_target[5]) <  ring_half_width )
			dist_cam[5] = (P-Qz)*(P-Qz);
	}

	// axes:

	if (axisPickable(Xt) || axisPickable(Xs))
	{
		Geom::Vec3f PX(3.6f*m_lengthAxes[0],0.0f,0.0f);
		dist_target[0] = sqrt(Geom::squaredDistanceLine2Seg(P, V, V*V, origin, PX)) ;
		if (fabs(dist_target[0]) < 0.02f)
		{
			if (axisPickable(Xt) && !axisPickable(Xs))
				dist_cam[0] = (P-PX)*(P-PX);
			else
			{
				if ( Qz.norm() > m_lengthAxes[0])
					dist_cam[0] = (P-PX)*(P-PX);
				else
					dist_cam[6] = P*P;
			}
		}
	}

	if (axisPickable(Yt) || axisPickable(Ys))
	{
		Geom::Vec3f PY(0.0f,3.6f*m_lengthAxes[1],0.0f);
		dist_target[1] = sqrt(Geom::squaredDistanceLine2Seg(P, V, V*V, origin, PY)) ;
		if (fabs(dist_target[1]) < 0.02f)
		{
			if (axisPickable(Yt) && !axisPickable(Ys))
				dist_cam[1] = (P-PY)*(P-PY);
			else
			{
				if (Qz.norm() > m_lengthAxes[1])
					dist_cam[1] = (P-PY)*(P-PY);
				else
					dist_cam[7] = P*P;
			}
		}
	}

	if (axisPickable(Zt) || axisPickable(Zs))
	{
		Geom::Vec3f PZ(0.0f,0.0f,3.6f*m_lengthAxes[2]);
		dist_target[2] = sqrt(Geom::squaredDistanceLine2Seg(P, V, V*V, origin, PZ));
		if (fabs(dist_target[2]) < 0.02f )
		{
			if (axisPickable(Zt) && !axisPickable(Zs))
				dist_cam[2] = (P-PZ)*(P-PZ);
			else
			{
				if (Qx.norm() > m_lengthAxes[2])
					dist_cam[2] = (P-PZ)*(P-PZ);
				else
					dist_cam[8] = P*P;
			}
		}
	}

	// find min dist_cam value;
	unsigned int min_index=0;
	float min_val = dist_cam[0];
	for (unsigned int i=1; i<9; ++i)
	{
		if  (dist_cam[i] < min_val)
		{
			min_val = dist_cam[i];
			min_index = i;
		}
	}

	if (min_val < std::numeric_limits<float>::max())
	{
//		if  (! m_locked_axis[Xt+min_index])
		if (axisPickable(Xt+min_index))
			return Xt+min_index;
	}

	return NONE;
}

void FrameManipulator::rotate(unsigned int axis, float angle)
{
	// create axis
	glm::vec3 ax(0.0f,0.0f,0.0f);
	ax[axis-Xr]=1.0f;

	glm::mat4 tr = glm::rotate(glm::mat4(1.0f),angle,ax);
	m_rotations = m_rotations*tr;
}

void FrameManipulator::translate(unsigned int axis, float x)
{
	m_trans += x*m_scaleRendering * glm::vec3(m_rotations[axis-Xt][0],m_rotations[axis-Xt][1],m_rotations[axis-Xt][2]);
}

void FrameManipulator::setLengthAxes()
{
	float avgScale =(m_scale[0]+m_scale[1]+m_scale[2])/3.0f;

	float* positions = reinterpret_cast<float*>(m_vboPos->lockPtr());
	unsigned int ind=3*(6*nb_segments+6+1);

	float sc0 = m_scale[0]/avgScale;
	float sc1 = m_scale[1]/avgScale;
	float sc2 = m_scale[2]/avgScale;

	positions[ind] = 0.23f*sc0;
	ind+=7;
	positions[ind] = 0.23f*sc1;
	ind+=7;
	positions[ind] = 0.23f*sc2;
	ind++;
	if ((m_locked_axis[Xs])&&(m_highlighted!=CENTER))
		positions[ind] = 0.0f;
	else
		positions[ind] = 0.27f*sc0;
	ind+=3;
	positions[ind] = 0.75f*sc0;
	ind+=3;
	positions[ind] = 0.9f*sc0;
	ind+=3;
	float le = 0.7f*sc0;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=4;

	if ((m_locked_axis[Ys])&&(m_highlighted!=CENTER))
		positions[ind] = 0.0f;
	else
		positions[ind] = 0.27f*sc1;
	ind+=3;
	positions[ind] = 0.75f*sc1;
	ind+=3;
	positions[ind] = 0.9f*sc1;
	ind+=3;
	le = 0.7f*sc1;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=4;

	if ((m_locked_axis[Zs])&&(m_highlighted!=CENTER))
		positions[ind] = 0.0f;
	else
		positions[ind] = 0.27f*sc2;
	ind+=3;
	positions[ind] = 0.75f*sc2;
	ind+=3;
	positions[ind] = 0.9f*sc2;
	ind+=3;
	le = 0.7f*sc2;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;
	ind+=3;
	positions[ind] = le;

	m_vboPos->releasePtr();

	m_lengthAxes = Geom::Vec3f(0.25f*sc0, 0.25f*sc1, 0.25f*sc2);
}

void FrameManipulator::scale(unsigned int axis, float sc)
{
	if (axis==CENTER)
	{
		m_scale[0] *= sc;
		m_scale[1] *= sc;
		m_scale[2] *= sc;
	}
	else
		m_scale[axis-Xs] *= sc;

	setLengthAxes();
}

glm::mat4 FrameManipulator::transfoRenderFrame()
{
	glm::mat4 tr = glm::translate(glm::mat4(1.0f), m_trans);
	tr *= m_rotations;

	float avgScale =(m_scale[0]+m_scale[1]+m_scale[2])/3.0f;
	avgScale *= m_scaleRendering;
	return glm::scale(tr,glm::vec3(avgScale,avgScale,avgScale));
}

glm::mat4 FrameManipulator::transfo()
{
	glm::mat4 tr = glm::translate(glm::mat4(1.0f), m_trans);
	tr *= m_rotations;
	return glm::scale(tr,glm::vec3(m_scale[0],m_scale[1],m_scale[2]));
}

void FrameManipulator::setTranslation(const Geom::Vec3f& P)
{
	m_trans[0] = P[0];
	m_trans[1] = P[1];
	m_trans[2] = P[2];
}

void FrameManipulator::setScale(const Geom::Vec3f& S)
{
	m_scale[0] = S[0];
	m_scale[1] = S[1];
	m_scale[2] = S[2];

	setLengthAxes();
}

bool FrameManipulator::setOrientation(const Geom::Vec3f& X, const Geom::Vec3f& Y)
{
    Geom::Vec3f Z = X.cross(Y);

	if ((X.norm() != 1.0f) || (Y.norm() != 1.0f) || (Z.norm() != 1.0f))
		return false;

	m_rotations[0][0] = X[0];
	m_rotations[0][1] = X[1];
	m_rotations[0][2] = X[2];

	m_rotations[1][0] = Y[0];
	m_rotations[1][1] = Y[1];
	m_rotations[1][2] = Y[2];

	m_rotations[2][0] = Z[0];
	m_rotations[2][1] = Z[1];
	m_rotations[2][2] = Z[2];

	return true;
}

void FrameManipulator::setTransformation( const glm::mat4& transfo)
{
    typedef Geom::Vec3f::value_type Real;
	setTranslation(Geom::Vec3f(transfo[3][0],transfo[3][1],transfo[3][2]));

	Geom::Vec3f Rx(	transfo[0][0], transfo[0][1], transfo[0][2]);
	Geom::Vec3f Ry(	transfo[1][0], transfo[1][1], transfo[1][2]);
	Geom::Vec3f Rz(	transfo[2][0], transfo[2][1], transfo[2][2]);

    const Real& N1 = Rx.norm();
    const Real& N2 = Ry.norm();
    const Real& N3 = Rz.norm();
    setScale(Geom::Vec3f(N1, N2, N3));
    Rx/=N1;
    Ry/=N2;
    Rz/=N3;

	m_rotations[0][0] = Rx[0];
	m_rotations[0][1] = Rx[1];
	m_rotations[0][2] = Rx[2];
	m_rotations[1][0] = Ry[0];
	m_rotations[1][1] = Ry[1];
	m_rotations[1][2] = Ry[2];
	m_rotations[2][0] = Rz[0];
	m_rotations[2][1] = Rz[1];
	m_rotations[2][2] = Rz[2];
}

void FrameManipulator::lock(unsigned int axis)
{
	assert(axis <=Scales);
	switch (axis)
	{
	case Translations:
		m_locked_axis[Xt] = true;
		m_locked_axis[Yt] = true;
		m_locked_axis[Zt] = true;
		break;
	case Rotations:
		m_locked_axis[Xr] = true;
		m_locked_axis[Yr] = true;
		m_locked_axis[Zr] = true;
		break;
	case Scales:
		m_locked_axis[Xs] = true;
		m_locked_axis[Ys] = true;
		m_locked_axis[Zs] = true;
		break;
	default:
		m_locked_axis[axis] = true;
		break;
	}
	setLengthAxes();
}

void FrameManipulator::unlock(unsigned int axis)
{
	assert(axis <=Scales);
	switch (axis)
	{
	case Translations:
		m_locked_axis[Xt] = false;
		m_locked_axis[Yt] = false;
		m_locked_axis[Zt] = false;
		break;
	case Rotations:
		m_locked_axis[Xr] = false;
		m_locked_axis[Yr] = false;
		m_locked_axis[Zr] = false;
		break;
	case Scales:
		m_locked_axis[Xs] = false;
		m_locked_axis[Ys] = false;
		m_locked_axis[Zs] = false;
		break;
	default:
		m_locked_axis[axis] = false;
		break;
	}
	setLengthAxes();
}

bool FrameManipulator::locked(unsigned int axis)
{
	assert(axis <=Zs);
	return m_locked_axis[axis];
}

void FrameManipulator::lockPicking(unsigned int axis)
{
	assert(axis <=Scales);
	switch (axis)
	{
	case Translations:
		m_lockedPicking_axis[Xt] = true;
		m_lockedPicking_axis[Yt] = true;
		m_lockedPicking_axis[Zt] = true;
		break;
	case Rotations:
		m_lockedPicking_axis[Xr] = true;
		m_lockedPicking_axis[Yr] = true;
		m_lockedPicking_axis[Zr] = true;
		break;
	case Scales:
		m_lockedPicking_axis[Xs] = true;
		m_lockedPicking_axis[Ys] = true;
		m_lockedPicking_axis[Zs] = true;
		break;
	default:
		m_lockedPicking_axis[axis] = true;
		break;
	}
	setLengthAxes();
}

void FrameManipulator::unlockPicking(unsigned int axis)
{
	assert(axis <=Scales);
	switch (axis)
	{
	case Translations:
		m_lockedPicking_axis[Xt] = false;
		m_lockedPicking_axis[Yt] = false;
		m_lockedPicking_axis[Zt] = false;
		break;
	case Rotations:
		m_lockedPicking_axis[Xr] = false;
		m_lockedPicking_axis[Yr] = false;
		m_lockedPicking_axis[Zr] = false;
		break;
	case Scales:
		m_lockedPicking_axis[Xs] = false;
		m_lockedPicking_axis[Ys] = false;
		m_lockedPicking_axis[Zs] = false;
		break;
	default:
		m_lockedPicking_axis[axis] = false;
		break;
	}
	setLengthAxes();
}

bool FrameManipulator::lockedPicking(unsigned int axis)
{
	return m_lockedPicking_axis[axis];
}


Geom::Vec3f  FrameManipulator::getAxis(unsigned int ax)
{
	unsigned int i = (ax-Xt)%3;

	return Geom::Vec3f(m_rotations[i][0],m_rotations[i][1],m_rotations[i][2]);
}

void FrameManipulator::storeProjection(unsigned int ax)
{
	Geom::Vec3f O = getPosition();

	glm::i32vec4 viewport;
	glGetIntegerv(GL_VIEWPORT, &(viewport[0]));
	glm::vec3 winO = glm::project(glm::vec3(O[0],O[1],O[2]), GLSLShader::currentModelView(), GLSLShader::currentProjection(), viewport);
	m_projectedOrigin = Geom::Vec3f(winO[0], winO[1], winO[2]);

	if (ax>CENTER)
	{
		Geom::Vec3f A = getAxis(ax);
		A += O;
		glm::vec3 winA = glm::project(glm::vec3(A[0],A[1],A[2]), GLSLShader::currentModelView(), GLSLShader::currentProjection(), viewport);
		m_projectedSelectedAxis = Geom::Vec3f(winA[0]-winO[0], winA[1]-winO[1],winA[2]-winO[2]);
	}
}

float FrameManipulator::angleFromMouse(int x, int y, int dx, int dy)
{
	Geom::Vec3f V(float(x) - m_projectedOrigin[0], float(y) - m_projectedOrigin[1],0.0f);
	Geom::Vec3f dV(float(dx), float(dy), 0.0f);
    Geom::Vec3f W = V.cross(dV);

	float alpha=dV.norm()/4.0f;
	// which direction ?
	if (W*m_projectedSelectedAxis > 0.0f)
		alpha *= -1.0f;
	return alpha;
}

float FrameManipulator::distanceFromMouse(int dx, int dy)
{
	Geom::Vec3f dV(float(dx), float(dy), 0.0f);
	float tr = dV*m_projectedSelectedAxis;
	if (tr>0)
		tr = dV.norm()/100.0f;
	else
		tr = dV.norm()/-100.0f;
	return tr;
}

float FrameManipulator::scaleFromMouse(int dx, int dy)
{
	if (abs(dx) > abs(dy))
	{
		if (dx>0)
			return 1.01f;
		return 0.99f;
	}
	else
	{
		if (dy>0)
			return 1.01f;
		return 0.99f;
	}
}

void FrameManipulator::translateInScreen(int dx, int dy)
{
	glm::i32vec4 viewport;
	glGetIntegerv(GL_VIEWPORT, &(viewport[0]));

	Geom::Vec3f NO = m_projectedOrigin+Geom::Vec3f(float(dx), float(dy), 0.0f);

	glm::vec3 P = glm::unProject(glm::vec3(NO[0],NO[1],NO[2]), GLSLShader::currentModelView(), GLSLShader::currentProjection(), viewport);

	m_trans[0] = P[0];
	m_trans[1] = P[1];
	m_trans[2] = P[2];
	storeProjection(NONE);
}

void FrameManipulator::rotateInScreen(int dx, int dy)
{
	glm::i32vec4 viewport;
	glGetIntegerv(GL_VIEWPORT, &(viewport[0]));

	Geom::Vec3f NO = m_projectedOrigin+Geom::Vec3f(float(-dy), float(dx), 0.0f);

	glm::vec3 P = glm::unProject(glm::vec3(NO[0],NO[1],NO[2]), GLSLShader::currentModelView(), GLSLShader::currentProjection(), viewport);

	Geom::Vec3f axisRotation(P[0]-m_trans[0], P[1]-m_trans[1], P[2]-m_trans[2]);
	axisRotation.normalize();

	glm::mat4 tr = glm::rotate(glm::mat4(1.0f),sqrtf(float(dx*dx+dy*dy))/2.0f,glm::vec3(axisRotation[0],axisRotation[1],axisRotation[2]));
	m_rotations = tr*m_rotations;
}

} // namespace Utils

} // namespace CGoGN
