/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/gui/qt/config.h>
#include <vector>

#include <QFileDialog>


namespace sofa::gui::qt
{


QString SOFA_GUI_QT_API getExistingDirectory ( QWidget* parent, const QString & dir = QString(), const char * name = nullptr, const QString & caption = QString() );

QString SOFA_GUI_QT_API getOpenFileName ( QWidget* parent, const QString & startWith = QString(), const QString & filter = QString(), const char * name = nullptr, const QString & caption = QString(), QString * selectedFilter = nullptr );

QString SOFA_GUI_QT_API getSaveFileName ( QWidget* parent, const QString & startWith = QString(), const QString & filter = QString(), const char * name = nullptr, const QString & caption = QString(), QString * selectedFilter = nullptr );


void SOFA_GUI_QT_API getFilesInDirectory( const QString &path, std::vector< QString > &files, bool recursive=true, const std::vector< QString > &filter=std::vector< QString >() );

} //namespace sofa::gui::qt
