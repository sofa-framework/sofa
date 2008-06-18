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

#include "FileManagement.h"
namespace sofa
{
namespace gui
{
namespace qt
{


#ifndef SOFA_QT4
typedef QFileDialog Q3FileDialog;
#endif

QString getExistingDirectory ( QWidget* parent, const QString & dir, const char * name, const QString & caption)
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

QString getOpenFileName ( QWidget* parent, const QString & startWith, const QString & filter, const char * name, const QString & caption, QString * selectedFilter )
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

QString getSaveFileName ( QWidget* parent, const QString & startWith, const QString & filter, const char * name, const QString & caption, QString * selectedFilter )
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

