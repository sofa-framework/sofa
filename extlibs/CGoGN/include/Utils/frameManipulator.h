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


#ifndef __FRAMEMANIPULATOR_H_
#define __FRAMEMANIPULATOR_H_

#include "Utils/vbo_base.h"
#include "glm/glm.hpp"
#include "Utils/Shaders/shaderSimpleColor.h"
#include "Utils/pickables.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API FrameManipulator//: public Pickable
{
public:
	enum AXIS {NONE=0, CENTER, Xt, Yt, Zt, Xr, Yr, Zr, Xs, Ys, Zs, Translations, Rotations, Scales};

protected:

	/**
	 * number of segment for circle drawing
	 */
	static const unsigned int nb_segments = 64;

	static const float ring_half_width;

	/**
	 * locking table
	 */
	bool m_locked_axis[11];

	/**
	 * pinking only locking table
	 */
	bool m_lockedPicking_axis[11];

	/**
	 * VBO for position
	 */
	Utils::VBO* m_vboPos;

	/**
	 * VBO for color
	 */
	Utils::VBO* m_vboCol;

	/**
	 * Shader
	 */
	Utils::ShaderSimpleColor* m_shader;

	/**
	 * the axis to be highlighted
	 */
	unsigned int m_highlighted;

	/**
	 * Rotation matrices
	 */
	glm::mat4 m_rotations;

	/**
	 * scale rendering factor
	 */
	float m_scaleRendering;

	/**
	 * translation
	 */
	glm::vec3 m_trans;

	/**
	 * scale
	 */
	glm::vec3 m_scale;

	/**
	 * epsilon distance for picking
	 */
	float m_epsilon;

	/**
	 * compute and set length axes / scale factors
	 */
	void setLengthAxes();

	/**
	 * length of axes
	 */
	Geom::Vec3f m_lengthAxes;

	Geom::Vec3f m_projectedSelectedAxis;

	Geom::Vec3f m_projectedOrigin;

	/**
	 * get the matrix transformation with the scale factor for rendering
	 */
	glm::mat4 transfoRenderFrame();

	inline bool axisPickable(unsigned int a) { return (!m_locked_axis[a]) && (!m_lockedPicking_axis[a]);}

public:
	FrameManipulator();

	/**
	 * set size of frame (for rendering)
	 */
	void setSize(float radius);

	/**
	 * add a (signed) val to the size of frame
	 */
	void addSize(float val);

	/**
	 * get the size of frame
	 */
	float getSize();

	/**
	 * draw the frame
	 */
	void draw();

	/**
	 * try picking the frame
	 */
	 unsigned int pick(const Geom::Vec3f& P, const Geom::Vec3f& V, float epsilon=0.0f);


	 /**
	  * lock an axis (drawing & picking are disabled)
	  * @param axis the axis Xt/Yt/Zt/Xs/Yx/Zs/Xr/Yr/Zr or group Translations/Scales/Rotations)
	  */
	 void lock(unsigned int axis);

	 /**
	  * unlock an axis
	  * @param axis the axis Xt/Yt/Zt/Xs/Yx/Zs/Xr/Yr/Zr or group Translations/Scales/Rotations)
	  */
	 void unlock(unsigned int axis);

	 /**
	  * is an axis locked
	  * @param axis the axis to test
	  */
	 bool locked(unsigned int axis);

	 /**
	  * lock an axis only for pinking
	  */
	 void lockPicking(unsigned int axis);

	 /**
	  * unlock an axis (only for pinking)
	  */
	 void unlockPicking(unsigned int axis);

	 /**
	  * is an axis locked (only for pinking)
	  */
	 bool lockedPicking(unsigned int axis);

	/**
	 * higlight an axis (change width rendering).
	 * To unhighlight, just highlight NONE or highlight a already highlighted  axis
	 */
	void highlight(unsigned int axis);

	/**
	 * rotate the frame around one of its axis
	 */
	void rotate(unsigned int axis, float angle);

	/**
	 * translate the frame around one of its axis
	 * @param axis
	 * @param x ratio of frame radius
	 */
	void translate(unsigned int axis, float x);

	/**
	 * scale the frame in direction of one axis
	 * @param axis (Xs/Ys/Zs/CENTER)
	 * @param sc scale factor to apply on
	 */
	void scale(unsigned int axis, float sc);

	/**
	 * get the matrix transformation
	 */
	glm::mat4 transfo();

	/**
	 * set the position of frame
	 * @param P the position of origin
	 */
	void setTranslation(const Geom::Vec3f& P);

	inline Geom::Vec3f getPosition() { return Geom::Vec3f(m_trans[0], m_trans[1], m_trans[2]); }

	Geom::Vec3f getAxis(unsigned int ax);

//	void project(unsigned int ax, Geom::Vec3f& origin, Geom::Vec3f& vect);

	void storeProjection(unsigned int ax);

	/**
	 * set the scale of frame
	 * @param P the vector of scale factors
	 */
	void setScale(const Geom::Vec3f& S);

	/**
	 * set the orientation of frame (Z is deduced)
	 * @param X the vector X of frame
	 * @param Y the vector Y of frame
	 * @return return false if parameters are not unit orthogonal vectors
	 */
	bool setOrientation(const Geom::Vec3f& X, const Geom::Vec3f& Y);

	/**
	 * set transformation matrix
	 */
	void setTransformation( const glm::mat4& transfo);



	float angleFromMouse(int x, int y, int dx, int dy);

	float distanceFromMouse(int dx, int dy);

	float scaleFromMouse(int dx, int dy);

//	float factorCenterFromMouse(int dx, int dy);

	inline static bool rotationAxis(unsigned int axis) { return (axis >= Xr) && (axis <= Zr); }
	inline static bool translationAxis(unsigned int axis) { return (axis >= Xt) && (axis <= Zt); }
	inline static bool scaleAxis(unsigned int axis) { return ((axis >= Xs) && (axis <= Zs)) || (axis == CENTER); }

	/**
	 * translate from screen mouse move
	 */
	void translateInScreen(int dx, int dy);

	void rotateInScreen(int dx, int dy);
};


}
}
#endif
