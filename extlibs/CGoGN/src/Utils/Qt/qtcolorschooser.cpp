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
#define CGoGN_UTILS_DLL_EXPORT 1
#include "Utils/Qt/qtcolorschooser.h"
#include "Utils/Qt/qtSimple.h"

namespace CGoGN
{

namespace Utils
{

namespace QT
{


ColorsChooser::ColorsChooser(SimpleQT *interf):
		QtPopUp(NULL,false),m_interf(interf),m_current(0)
{
	m_list = new QListWidget();
	m_diag = new QColorDialog();
	m_diag->setOption(QColorDialog::NoButtons);
	addWidget(m_list,0,0);
	addWidget(m_diag,0,1);
	connect(m_list,  SIGNAL(currentRowChanged(int)), this, SLOT(select_color(int)));
	connect(m_diag, SIGNAL(	currentColorChanged(const QColor&)), this, SLOT(change_color(const QColor&)));

}

unsigned int ColorsChooser::addColor(Geom::Vec3f* ptr, const std::string& name)
{
	m_colors.push_back(ptr);
	m_list->addItem(QString(name.c_str()));
	return (unsigned int)(m_colors.size()-1);
}


void ColorsChooser::select_color(int x)
{
	m_current = x;
	const Geom::Vec3f& col = *m_colors[x];
	m_diag->show();
	m_diag->setCurrentColor(QColor(int(255.0f*col[0]), int(255.0f*col[1]), int(255.0f*col[2])) );
}

void ColorsChooser::change_color(const QColor& col)
{
	Geom::Vec3f& out = *m_colors[m_current];
	out[0] = float(col.redF());
	out[1] = float(col.greenF());
	out[2] = float(col.blueF());


	if (m_interf)
	{
		updateCallBack(m_interf);
		m_interf->updateGL();
	}
}

}
}
}
