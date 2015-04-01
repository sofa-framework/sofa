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

#ifndef __QT_POP_UP__
#define __QT_POP_UP__

#include <QGridLayout>
#include <QDialog>
#include "Utils/Qt/qtSimple.h"

#ifdef WIN32
#ifndef CGoGN_UTILS_API
#if defined CGoGN_QT_DLL_EXPORT
#define CGoGN_UTILS_API __declspec(dllexport)
#else
#define CGoGN_UTILS_API __declspec(dllimport)
#endif
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

/**
*	Easy popup window creation
*   Can contain one a more widgets in a grid
*	Use show/hide to show/hide !
*/
class CGoGN_UTILS_API QtPopUp : public QDialog
{
	Q_OBJECT

	Utils::QT::SimpleQT * m_cbs;

	QGridLayout* m_layout;

public:

	/**
	* create an empty popup
	* @param withButtons  add OK/CANCEL to the popup (exec launch blocking popup & return 1/0)
	*/
	QtPopUp(Utils::QT::SimpleQT* sqt=NULL, bool withButtons=false);

	/**
	 *
	 */
	virtual ~QtPopUp();

	/**
	* add a widget in the grid layout
	* @param wid the widget to use
	* @param row the row in which to insert
	* @param col the column in which to insert
	*/	
	void addWidget(QWidget* wid, int row, int col);
	

protected:
	/// overload keypress event to avoid ESC out
	virtual void keyPressEvent ( QKeyEvent * e );
};

}
}
}

#endif

