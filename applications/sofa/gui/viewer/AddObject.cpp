/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#include "AddObject.h"
#include <iostream>

#ifdef QT_MODULE_QT3SUPPORT
#include <Q3FileDialog>
#include <QLineEdit>
#else
#include <qfiledialog.h>
#include <qlineedit.h>
#endif

namespace sofa
{

namespace gui
{

namespace guiviewer
{


#ifndef QT_MODULE_QT3SUPPORT
typedef QFileDialog Q3FileDialog;
#endif


AddObject::AddObject( QWidget* parent , const char*, bool, Qt::WFlags )
{
    positionX->setText("0");
    positionY->setText("0");
    positionZ->setText("0");
    scaleValue->setText("1");

    openFilePath->setText(NULL);

    connect( (QObject *) buttonOk, SIGNAL( clicked() ), parent, SLOT( loadObject()));
}

//Set the default file
void AddObject::setPath(const std::string path)
{
    fileName = path;
    openFilePath->setText(QString(fileName.c_str()));
}


//Open a file Dialog and set the path of the selected path in the text field.
void AddObject::fileOpen()
{
    QString s  = Q3FileDialog::getOpenFileName(fileName.empty()?NULL:fileName.c_str(), "Sofa Element (*.xml *.scn)",  this, "open file dialog",  "Choose a file to open" );

    if (s.isNull() ) return;
#ifdef QT_MODULE_QT3SUPPORT
    std::string object_fileName(s.toStdString());
#else
    std::string object_fileName(s.latin1());
#endif

    openFilePath->setText(QString(object_fileName.c_str()));
}


} // namespace qt

} // namespace gui

} // namespace sofa

