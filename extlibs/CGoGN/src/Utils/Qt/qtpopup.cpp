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
#include "Utils/Qt/qtpopup.h"
 #include <QVBoxLayout>
 #include <QHBoxLayout>
 #include <QPushButton>
 #include <QKeyEvent>
namespace CGoGN
{
namespace Utils
{
namespace QT
{


QtPopUp::QtPopUp(SimpleQT* sqt, bool withButtons):
m_cbs(sqt)
{
	if (withButtons)
	{
		QVBoxLayout *vlayout = new QVBoxLayout(this);
		m_layout = new QGridLayout(NULL);
		vlayout->addLayout(m_layout);

		QHBoxLayout *layButtons = new QHBoxLayout();
		QPushButton* okbutton = new QPushButton( "OK", this );
		QObject::connect( okbutton, SIGNAL( clicked() ), this, SLOT( accept() ) );
		QPushButton* cancelbutton = new QPushButton( "Cancel", this );
		QObject::connect( cancelbutton, SIGNAL( clicked() ), 	this, SLOT( reject() ) );
		// TO LAYOUT
		layButtons->addWidget(okbutton);
		layButtons->addWidget(cancelbutton);
		vlayout->addLayout(layButtons);

		setLayout(vlayout);

	}
	else
	{
		m_layout = new QGridLayout(this);
		setLayout(m_layout);
	}
}


QtPopUp::~QtPopUp()
{
}



void QtPopUp::addWidget(QWidget* wid, int col, int row)
{
	m_layout->addWidget(wid,col,row);
}

void QtPopUp::keyPressEvent ( QKeyEvent * event )
{
	int k = event->key();
	if ( (k >= 65) && (k <= 91) && !(event->modifiers() & Qt::ShiftModifier) )
		k += 32;

	if (m_cbs)
		m_cbs->cb_keyPress(k);
}

}
}
}
	
