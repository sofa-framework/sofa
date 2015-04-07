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

#ifndef __QT_GL2__
#define __QT_GL2__

#include "Utils/gl_def.h"
#include <QGLWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include <iostream>
#include <stack>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "Geometry/vector_gen.h"


#ifdef WIN32
#if defined CGoGN_QT_DLL_EXPORT
#define CGoGN_UTILS_API __declspec(dllexport)
#else
#define CGoGN_UTILS_API __declspec(dllimport)
#endif
#else
#define CGoGN_UTILS_API
#endif

namespace CGoGN
{

namespace Utils
{

namespace QT
{

class SimpleQT;

class CGoGN_UTILS_API GLWidget : public QGLWidget
{
	Q_OBJECT

public:
	GLWidget(SimpleQT* cbs, QWidget *parent = 0);

	~GLWidget();

    QSize minimumSizeHint() const;

    QSize sizeHint() const;

protected:

	static float FAR_PLANE;

	SimpleQT* m_cbs;

	int m_current_button;
	QPoint clickPoint;
	int beginx;
	int beginy;
	int newModel;

	int moving;
	int scaling;
	int translating;

	float scalefactor;
	float foc;

	float m_obj_sc;
	glm::vec3 m_obj_pos;
	glm::vec3 m_obj_pos_save;
	float m_obj_width;

	// width and height of windows
	int W;
	int H;

	int m_state_modifier;

	/// stack for transformation matrix
	std::stack<glm::mat4> m_stack_trf;

	bool allow_rotation;

	/**
	 * met a jour la matrice modelview
	 */
	void recalcModelView();

	/**
	 * recalcul le quaternion ainsi que le deplacement courant
	 * pour un nouveau centre de rotation
	 */
	void changeCenterOfRotation(const glm::vec3& newCenter);

public:
	void resetCenterOfRotation(float width, float* pos);

	void setParamObject(float width, float* pos);

	void setRotation(bool b);

	void initializeGL();

	void paintGL();

	void resizeGL(int width, int height);

public:
	void mousePressEvent(QMouseEvent* event);

	void mouseReleaseEvent(QMouseEvent* event);

	void mouseClickEvent(QMouseEvent* event);

	void mouseDoubleClickEvent(QMouseEvent* event);

	void mouseMoveEvent(QMouseEvent* event);

	void keyPressEvent(QKeyEvent* event);

	void keyReleaseEvent(QKeyEvent* event);

	void wheelEvent(QWheelEvent* event);

	bool Shift() { return m_state_modifier & Qt::ShiftModifier; }

	bool Control() { return m_state_modifier & Qt::ControlModifier; }

	bool Alt() { return m_state_modifier & Qt::AltModifier; }

	int getHeight() const { return H; }
	int getWidth() const { return W; }

	/**
	 * set the focale distance (for a screen width of 2), default value is 1
	 */
	void setFocal(float df);

	/**
	 * get the focale distance
	 */
	float getFocal() const { return foc; }

	/**
	 * get current state
	 */
	int getStateModifier() const { return m_state_modifier ; }
	int getCurrentButton() const { return m_current_button ; }

	static float getFarPlane() { return FAR_PLANE ; }

	glm::vec3& getObjPos() ;

	void modelModified() { newModel = 1; }

	void glMousePosition(int& x, int& y);

	/**
	 * get a ray (2 points) from a pick point in GL area
	 * @param x mouse position
	 * @param y mouse position
	 * @param rayA first computed point of ray
	 * @param rayA second computed point of ray
	 * @param radius radius on pixel for clicking precision
	 * @return the distance in modelview world corresponding to radius pixel in screen
	 */
	GLfloat getOrthoScreenRay(int x, int y, Geom::Vec3f& rayA, Geom::Vec3f& rayB, int radius=4);

	/**
	 * transform a pixel distance on screen in distance in world
	 * @param pixel_width width on pixel on screen
	 * @param center reference point on world to use (defaut 0,0,0)
	 */
	float getWidthInWorld(unsigned int pixel_width, const Geom::Vec3f& center=Geom::Vec3f(0.0f,0.0f,0.0f));

	/**
	 * current transfo matrix
	 */
	const glm::mat4& transfoMatrix() const;
	glm::mat4& transfoMatrix();

	/**
	 * current modelview matrix
	 */
	const glm::mat4& modelViewMatrix() const;
	glm::mat4& modelViewMatrix();

	/**
	 * current projection matrix
	 */
	const glm::mat4& projectionMatrix() const;
	glm::mat4& projectionMatrix();


	void transfoRotate(float angle, float x, float y, float z);

	void transfoTranslate(float tx, float ty, float tz);

	void transfoScale(float sx, float sy, float sz);

	void pushTransfoMatrix();

	bool popTransfoMatrix();


protected:
	/**
	 * equivalent to old school glRotate
	 */
	void oglRotate(float angle, float x, float y, float z);

	/**
	 * equivalent to old school glTranslate
	 */
	void oglTranslate(float tx, float ty, float tz);

	/**
	 * equivalent to old school glScale
	 */
	void oglScale(float sx, float sy, float sz);

	/**
	 * get the focale distance
	 */
	float getScale() { return scalefactor / foc; }

	inline int pixelRatio() const
	{
		#if (QT_VERSION>>16) == 5
			return this->devicePixelRatio();
		#else
			return 1;
		#endif
	}
};

} // namespace QT

} // namespace Utils

} // namespace CGoGN

#endif
