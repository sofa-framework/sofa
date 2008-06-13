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
#ifndef SOFA_GUI_FILEMANAGEMENT_H
#define SOFA_GUI_FILEMANAGEMENT_H


#ifdef SOFA_QT4
#include <Q3FileDialog>
#include <QFileDialog>
#else
#include <qfiledialog.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{


#ifndef SOFA_QT4
typedef QFileDialog Q3FileDialog;
#endif

static QString getExistingDirectory ( QWidget* parent, const QString & dir = QString(), const char * name = 0, const QString & caption = QString() )
{
#ifdef SOFA_QT4
    QFileDialog::Options options = QFileDialog::ShowDirsOnly;
    //	options |= QFileDialog::DontUseNativeDialog;
    options |= QFileDialog::DontUseSheet;
    return QFileDialog::getExistingDirectory ( parent, name?QString(name):caption, dir, options );
#else
    return Q3FileDialog::getExistingDirectory( dir, parent, name, caption );
#endif
};

static QString getOpenFileName ( QWidget* parent, const QString & startWith = QString(), const QString & filter = QString(), const char * name = 0, const QString & caption = QString(), QString * selectedFilter = 0 )
{
#ifdef SOFA_QT4
    QFileDialog::Options options = 0;
    //	options |= QFileDialog::DontUseNativeDialog;
    options |= QFileDialog::DontUseSheet;
    return QFileDialog::getOpenFileName ( parent, name?QString(name):caption, startWith, filter, selectedFilter, options );
#else
    return Q3FileDialog::getOpenFileName ( startWith, filter, parent, name, caption, selectedFilter );
#endif
};

static QString getSaveFileName ( QWidget* parent, const QString & startWith = QString(), const QString & filter = QString(), const char * name = 0, const QString & caption = QString(), QString * selectedFilter = 0 )
{
#ifdef SOFA_QT4
    QFileDialog::Options options = 0;
    //	options |= QFileDialog::DontUseNativeDialog;
    options |= QFileDialog::DontUseSheet;
    return QFileDialog::getSaveFileName ( parent, name?QString(name):caption, startWith, filter, selectedFilter, options );
#else
    return Q3FileDialog::getSaveFileName ( startWith, filter, parent, name, caption, selectedFilter );
#endif
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_VIEWER_REALGUI_H
