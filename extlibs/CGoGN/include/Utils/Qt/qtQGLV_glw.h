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

#ifndef __QTQGLVIEWER_GLW_H
#define __QTQGLVIEWER_GLW_H

#include <QApplication>
#include <QDesktopWidget>
#include <QMainWindow>
#include <QWidget>
#include <QtGui>
#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>

#include <set>
#include <string>
#include "Geometry/vector_gen.h"
#include "Utils/gl_matrices.h"

namespace CGoGN { namespace Utils { class GLSLShader; } }

namespace CGoGN
{

namespace Utils
{

namespace QT
{
// forward definition
class SimpleQGLV;


class QGLView : public QGLViewer
{
	Q_OBJECT
protected:
	SimpleQGLV* m_sqgl;
	int W;
	int H;
	int m_state_modifier;

public:
	QGLView(SimpleQGLV* ptr, QWidget *parent = 0);
	~QGLView();

	void setObjectBB(float* bbmin, float* bbmax);
	void setParamObject(float width, float* pos);
	void glMousePosition(int& x, int& y);

	virtual void init();
	virtual void preDraw();
	virtual void draw();
	virtual void postDraw() {}
	virtual void resizeGL(int width, int height);

	void mousePressEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void mouseClickEvent(QMouseEvent* event);
//	void mouseDoubleClickEvent(QMouseEvent* event);
//	void mouseMoveEvent(QMouseEvent* event);
	void keyPressEvent(QKeyEvent* event);
	void keyReleaseEvent(QKeyEvent* event);
//	void wheelEvent(QWheelEvent* event);
	bool Shift() { return m_state_modifier & Qt::ShiftModifier; }
	bool Control() { return m_state_modifier & Qt::ControlModifier; }
	bool Alt() { return m_state_modifier & Qt::AltModifier; }
	int getHeight() const { return H; }
	int getWidth() const { return W; }

	QSize minimumSizeHint() const;
	QSize sizeHint() const;

};


} // namespace QT

} // namespace Utils

} // namespace CGoGN

#endif
