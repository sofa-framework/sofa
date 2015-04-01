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

#ifndef __QT_COLORS_CHOOSER_
#define __QT_COLORS_CHOOSER_


#include "Utils/Qt/qtSimple.h"

#include <QColorDialog>
#include <QListWidget>
#include "Utils/Qt/qtpopup.h"


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

//forward definition
class SimpleQT;

/**
 * Class that allow to interactively & easily modify application color:
 *  Example of use
 * 	cc = new Utils::QT::ColorsChooser(&sqt)
 * 	cc->addColor(backgroundColor);
 *  cc->addColor(lineColor);
 *  cc->show();
 *  Colors are automatically updated and
 *
 *  Closing the window make a hide() (no destruction)
 *  To destroy: call delete in exit_cb
 */
class CGoGN_UTILS_API  ColorsChooser : public QtPopUp
{
	Q_OBJECT
protected:
	QListWidget *m_list;
	QColorDialog* m_diag;
	SimpleQT *m_interf;
	std::vector<Geom::Vec3f*> m_colors;
	std::vector<std::string> m_names;
	int m_current;

	/**
	 *  Update callback, on color has changed (optional)
	 *  overload with what you want (glClearColor, setColor of shader ...)
	 *  Called only if interface ptr has been given
	 *  Do not forget to cast interf in the type of your interface !!
	 */
	virtual void updateCallBack(SimpleQT* /*interf*/) {}

public:
	/**
	 * constructor
	 * @param interf a ptr to the SimpltQT interf (if given the updateCallback and updateGL() are called
	 */
	ColorsChooser(SimpleQT *interf=NULL);

	/**
	 * add a color ptr the color chooser
	 * @param ptr the ptr to the color
	 * @param name display name in interface
	 */
	unsigned int addColor(Geom::Vec3f* ptr, const std::string& name);

protected slots:
	void select_color(int x);
	void change_color(const QColor& col);
};

}
}
}

#endif
