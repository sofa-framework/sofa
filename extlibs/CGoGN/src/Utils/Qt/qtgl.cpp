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

#include "Utils/GLSLShader.h"

#include <iostream>
#include "Utils/trackball.h"
#include "Utils/Qt/qtgl.h"
#include "Utils/Qt/qtSimple.h"
#include "glm/gtc/type_precision.hpp"



namespace CGoGN
{

namespace Utils
{

namespace QT
{

float GLWidget::FAR_PLANE = 500.0f;

GLWidget::GLWidget(SimpleQT* cbs, QWidget *parent) :
	QGLWidget(parent),
	m_cbs(cbs),
	m_state_modifier(0),
	allow_rotation(true)
{
	makeCurrent();
	glewExperimental = GL_TRUE;
	glewInit();

	newModel = 1;
	m_cbs->trans_x() = 0.;
	m_cbs->trans_y() = 0.;
	float f = FAR_PLANE;
	m_cbs->trans_z() = -f / 5.0f;
	foc = 2.0f;

	// init trackball
	trackball(m_cbs->curquat(), 0.0f, 0.0f, 0.0f, 0.0f);
}


GLWidget::~GLWidget()
{
}

void GLWidget::setParamObject(float width, float* pos)
{
	m_obj_sc = ((FAR_PLANE / 5.0f) / foc) / width;
	m_obj_pos = glm::vec3(-pos[0], -pos[1], -pos[2]);
	m_obj_pos_save = glm::vec3(pos[0], pos[1], pos[2]);
	m_obj_width = width;
}

void GLWidget::resetCenterOfRotation(float width, float* pos)
{
	m_cbs->trans_x() = 0.;
	m_cbs->trans_y() = 0.;
	m_cbs->trans_z() = -FAR_PLANE / 5.0f;
	m_obj_sc = ((FAR_PLANE / 5.0f) / foc) / width;

	m_obj_pos = glm::vec3(-pos[0], -pos[1], -pos[2]);
	newModel=1;

}


void GLWidget::setRotation(bool b)
{
	allow_rotation = b;
}

void  GLWidget::setFocal(float df)
{
	if (df > 5.0f)
		df = 5.0f;
	if (df < 0.2f)
		df = 0.2f;

	m_obj_sc *= foc / df;

	foc = df;
	resizeGL(W, H);
}

QSize GLWidget::minimumSizeHint() const
{
    return QSize(200, 200);
}

QSize GLWidget::sizeHint() const
{
    return QSize(500, 500);
}

void GLWidget::recalcModelView()
{
	m_cbs->modelViewMatrix()= glm::mat4(1.0f);

	// positionne l'objet / mvt souris
	oglTranslate(m_cbs->trans_x(), m_cbs->trans_y(), m_cbs->trans_z());

	// tourne l'objet / mvt souris
	glm::mat4 m;
	build_rotmatrixgl3(m, m_cbs->curquat());
	// update matrice
	m_cbs->modelViewMatrix() *= m;

	// transfo pour que l'objet soit centre et a la bonne taille
	oglScale(m_obj_sc, m_obj_sc, m_obj_sc);
	oglTranslate(m_obj_pos[0], m_obj_pos[1], m_obj_pos[2]);

	// ajout transformation
	// m_cbs->modelViewMatrix() *= m_cbs->transfoMatrix();

	newModel = 0;

	if (m_cbs)
		m_cbs->cb_updateMatrix();
}

void GLWidget::changeCenterOfRotation(const glm::vec3& newCenter)
{
	glm::mat4 storeMVM(m_cbs->modelViewMatrix());

	m_cbs->modelViewMatrix() = glm::mat4(1.0f);

	// positionne l'objet / mvt souris
	oglTranslate(m_cbs->trans_x(), m_cbs->trans_y(), m_cbs->trans_z());

	// tourne l'objet / mvt souris
	glm::mat4 m;
	build_rotmatrixgl3(m, m_cbs->curquat());
	// update matrice
	m_cbs->modelViewMatrix() *= m;

	// ajout transformation in screen
	m_cbs->modelViewMatrix()*= m_cbs->transfoMatrix();

	// transfo pour que l'objet soit centre et a la bonne taille
	oglScale(m_obj_sc, m_obj_sc, m_obj_sc);
	oglTranslate(m_obj_pos[0], m_obj_pos[1], m_obj_pos[2]);

	oglTranslate(newCenter[0], newCenter[1], newCenter[2]);
	oglScale(1.0f / m_obj_sc, 1.0f / m_obj_sc, 1.0f / m_obj_sc);

	m = glm::inverse(m_cbs->transfoMatrix());
	m_cbs->modelViewMatrix() *= m;

	matrix_to_quat( m_cbs->curquat(), m_cbs->modelViewMatrix());

	m_cbs->trans_x() = m_cbs->modelViewMatrix()[3][0];
	m_cbs->trans_y() = m_cbs->modelViewMatrix()[3][1];
	m_cbs->trans_z() = m_cbs->modelViewMatrix()[3][2];

	m_cbs->modelViewMatrix() = storeMVM;

	m_obj_pos = glm::vec3(-newCenter[0], -newCenter[1], -newCenter[2]);
}

glm::vec3& GLWidget::getObjPos()
{
	return m_obj_pos ;
}

void GLWidget::initializeGL()
{
	glEnable(GL_DEPTH_TEST);

	if (m_cbs)
		m_cbs->cb_initGL();
}

void GLWidget::resizeGL(int w, int h)
{
	W = w;
	H = h;

	glViewport(0, 0, W, H);
	float f = FAR_PLANE;
	m_cbs->projectionMatrix() = glm::frustum(-1.0f, 1.0f, -1.0f * H / W, 1.0f * H / W, foc, f);

	recalcModelView();
}

void GLWidget::paintGL()
{
	if (newModel)
	    recalcModelView();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (m_cbs)
	{
		Utils::GLSLShader::s_current_matrices = m_cbs->matricesPtr();
		m_cbs->cb_redraw();
	}
}

void GLWidget::mousePressEvent(QMouseEvent* event)
{
	beginx = event->x();
	beginy = event->y();
	clickPoint = event->pos();
	m_current_button = event->button();

	if (m_cbs)
		m_cbs->cb_mousePress(event->button(), event->x()*pixelRatio(), getHeight() - event->y()*pixelRatio());
	setFocus(Qt::MouseFocusReason);
}

void GLWidget::mouseReleaseEvent(QMouseEvent* event)
{
	if (m_cbs)
		m_cbs->cb_mouseRelease(event->button(), event->x()*pixelRatio(), getHeight() - event->y()*pixelRatio());

	if(event->pos() == clickPoint)
		mouseClickEvent(event) ;
}

void GLWidget::mouseClickEvent(QMouseEvent* event)
{

	if (m_cbs)
		m_cbs->cb_mouseClick(event->button(), event->x()*pixelRatio(), getHeight() - event->y()*pixelRatio());
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent* event)
{
	if (event->button()==1)
	{
		GLint x = event->x()*pixelRatio();
		GLint y = getHeight() - event->y()*pixelRatio();
		GLfloat depth;
		glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
		if (depth < 1.0f)
		{
			glm::i32vec4 viewport;
			glGetIntegerv(GL_VIEWPORT, &(viewport[0]));
			glm::vec3 win(x, y, depth);
			glm::vec3 P = glm::unProject(win, m_cbs->modelViewMatrix(), m_cbs->projectionMatrix(), viewport);
			changeCenterOfRotation(P);
		}
		else
		{
			resetCenterOfRotation(m_obj_width, static_cast<float*>(&m_obj_pos_save.x)) ;
			updateGL();
		}
	}
}

void GLWidget::mouseMoveEvent(QMouseEvent* event)
{
	// move object only if no special keys pressed
	if (!(m_state_modifier & ( Qt::ShiftModifier | Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier)))
	{
		int x = event->x();
		int y = event->y();

		switch (m_current_button)
		{
			case Qt::RightButton:
			{
				float wl;
				if (m_cbs->trans_z() > -20.0f)
					wl = 20.0f / foc;
				else
					wl = -2.0f * m_cbs->trans_z() / foc;
				m_cbs->trans_x() += wl / W * (x - beginx);
				m_cbs->trans_y() += wl / H * (beginy - y);
			}
				break;
			case Qt::MidButton:
			{
				float wl = -0.5f * FAR_PLANE / foc;
				m_cbs->trans_z() -= wl / W * (x - beginx);
				m_cbs->trans_z() -= wl / H * (y - beginy);
			}
				break;
			case Qt::LeftButton:
			{
				if(allow_rotation)
				{
					trackball(
						m_cbs->lastquat(),
						(2.0f * beginx - W) / W,
						(H - 2.0f * beginy) / H,
						(2.0f * x - W) / W,(H - 2.0f * y) / H
					);
					add_quats(m_cbs->lastquat(), m_cbs->curquat(), m_cbs->curquat());
				}
			}
				break;
		}

		beginx = x;
		beginy = y;
		newModel = 1;
		updateGL();
	}

	if (m_cbs)
		m_cbs->cb_mouseMove(event->buttons(), event->x(), getHeight() - event->y());
}

void GLWidget::wheelEvent(QWheelEvent* event)
{
	if (!(m_state_modifier & ( Qt::ShiftModifier | Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier)))
	{
		float wl = -0.05f * FAR_PLANE / foc;

		if (event->delta() > 0)
			m_cbs->trans_z() += wl;
		else
			m_cbs->trans_z() -= wl;

		newModel = 1;
		updateGL();
	}

	if (m_cbs)
		m_cbs->cb_wheelEvent(event->delta(), event->x(), getHeight() - event->y());
}


void GLWidget::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Escape)
	{
		close();
		m_cbs->close();
		return;
	}

	m_state_modifier = event->modifiers();

	int k = event->key();
	if ( (k >= 65) && (k <= 91) && !(event->modifiers() & Qt::ShiftModifier) )
		k += 32;

	if (m_cbs)
		m_cbs->cb_keyPress(k);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
//	QWidget::keyReleaseEvent(event);

	m_state_modifier = event->modifiers();
    int k = event->key();

    // align on axis
	if ((k == 'Z') && (event->modifiers() & Qt::ShiftModifier))
	{
	    float Z[3] = { 0.0f, 0.0f, 1.0f };
		axis_to_quat(Z, 0.0f, m_cbs->curquat());
		newModel = 1;
		updateGL();
	}

	if ((k == 'Y') && (event->modifiers() & Qt::ShiftModifier))
	{
		float X[3] = { 1.0f, 0.0f, 0.0f };
		axis_to_quat(X, float(M_PI / 2.0), m_cbs->curquat());
		newModel = 1;
		updateGL();
	}

	if ((k == 'X') && (event->modifiers() & Qt::ShiftModifier))
	{
		float Y[3] = { 0.0f, 1.0f, 0.0f };
		axis_to_quat(Y, float(-M_PI / 2.0), m_cbs->curquat());
		newModel = 1;
		updateGL();
	}

    if ( (k >= 65) && (k <= 91) && (event->modifiers() != Qt::ShiftModifier) )
    	k += 32;

	if (m_cbs)
		m_cbs->cb_keyRelease(k);
}


void GLWidget::glMousePosition(int& x, int& y)
{
	QPoint xy = mapFromGlobal(QCursor::pos());
	x = xy.x();
	y = getHeight() - xy.y();
}


void GLWidget::oglRotate(float angle, float x, float y, float z)
{
	m_cbs->modelViewMatrix() = glm::rotate(m_cbs->modelViewMatrix(), glm::radians(angle), glm::vec3(x,y,z));
}

void GLWidget::oglTranslate(float tx, float ty, float tz)
{
	m_cbs->modelViewMatrix() = glm::translate(m_cbs->modelViewMatrix(), glm::vec3(tx,ty,tz));
}

void GLWidget::oglScale(float sx, float sy, float sz)
{
	m_cbs->modelViewMatrix() = glm::scale(m_cbs->modelViewMatrix(), glm::vec3(sx,sy,sz));
}


GLfloat GLWidget::getOrthoScreenRay(int x, int y, Geom::Vec3f& rayA, Geom::Vec3f& rayB, int radius)
{
	// get Z from depth buffer
	int yy = y;
	GLfloat depth_t[25];
	glReadPixels(x-2, yy-2, 5, 5, GL_DEPTH_COMPONENT, GL_FLOAT, depth_t);

	GLfloat depth=0.0f;
	unsigned int nb=0;
	for (unsigned int i=0; i< 25; ++i)
	{
		if (depth_t[i] != 1.0f)
		{
			depth += depth_t[i];
			nb++;
		}
	}
	if (nb>0)
		depth /= float(nb);
	else
		depth = 0.5f;

	glm::i32vec4 viewport;
	glGetIntegerv(GL_VIEWPORT, &(viewport[0]));

	glm::vec3 win(x, yy, 0.0f);

	glm::vec3 P = glm::unProject(win, m_cbs->modelViewMatrix(), m_cbs->projectionMatrix(), viewport);

	rayA[0] = P[0];
	rayA[1] = P[1];
	rayA[2] = P[2];

	win[2] = depth;

	P = glm::unProject(win, m_cbs->modelViewMatrix(), m_cbs->projectionMatrix(), viewport);
	rayB[0] = P[0];
	rayB[1] = P[1];
	rayB[2] = P[2];

	if (depth == 1.0f)	// depth vary in [0-1]
		win[2] = 0.5f;

	win[0] += radius;
	P = glm::unProject(win, m_cbs->modelViewMatrix(), m_cbs->projectionMatrix(), viewport);
	Geom::Vec3f Q;
	Q[0] = P[0];
	Q[1] = P[1];
	Q[2] = P[2];

	// compute & return distance
	Q -= rayB;
	return float(Q.norm());
}

float GLWidget::getWidthInWorld(unsigned int pixel_width, const Geom::Vec3f& center)
{

	glm::i32vec4 viewport;
	glGetIntegerv(GL_VIEWPORT, &(viewport[0]));

	glm::vec3 win = glm::project(glm::vec3(center[0],center[1],center[2]), m_cbs->modelViewMatrix(), m_cbs->projectionMatrix(), viewport);

	win[0]-= pixel_width/2;

	glm::vec3 P = glm::unProject(win, m_cbs->modelViewMatrix(), m_cbs->projectionMatrix(), viewport);

	win[0] += pixel_width;

	glm::vec3 Q = glm::unProject(win, m_cbs->modelViewMatrix(), m_cbs->projectionMatrix(), viewport);

	return glm::distance(P,Q);
}

void GLWidget::transfoRotate(float angle, float x, float y, float z)
{
	m_cbs->transfoMatrix() = glm::rotate( m_cbs->transfoMatrix(), glm::radians(angle), glm::vec3(x,y,z));
	recalcModelView() ;
}

void GLWidget::transfoTranslate(float tx, float ty, float tz)
{
	m_cbs->transfoMatrix() = glm::translate( m_cbs->transfoMatrix(), glm::vec3(tx,ty,tz));
}

void GLWidget::transfoScale(float sx, float sy, float sz)
{
	m_cbs->transfoMatrix() = glm::scale( m_cbs->transfoMatrix(), glm::vec3(sx,sy,sz));
}

void GLWidget::pushTransfoMatrix()
{
	m_stack_trf.push( m_cbs->transfoMatrix());
}

bool GLWidget::popTransfoMatrix()
{
	if (m_stack_trf.empty())
		return false;
	 m_cbs->transfoMatrix() = m_stack_trf.top();
	m_stack_trf.pop();
	return true;
}

/**
 * current transfo matrix
 */
const glm::mat4& GLWidget::transfoMatrix() const { return m_cbs->transfoMatrix(); }
glm::mat4& GLWidget::transfoMatrix() { return m_cbs->transfoMatrix(); }

/**
 * current modelview matrix
 */
const glm::mat4& GLWidget::modelViewMatrix() const { return m_cbs->modelViewMatrix(); }
glm::mat4& GLWidget::modelViewMatrix() { return m_cbs->modelViewMatrix(); }

/**
 * current projection matrix
 */
const glm::mat4& GLWidget::projectionMatrix() const { return m_cbs->projectionMatrix(); }
glm::mat4& GLWidget::projectionMatrix() { return m_cbs->projectionMatrix(); }



} // namespace QT

} // namespace Utils

} // namespace CGoGN
