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
#include <sofa/core/objectmodel/Base.h>
#include <sofa/gui/qt/config.h>

#include <QWidget>
#include <QTextEdit>
#include <QGroupBox>
#include <QGridLayout>



namespace sofa::gui::qt
{

struct ModifyObjectFlags;
class SOFA_GUI_QT_API QDataDescriptionWidget : public QWidget
{
    Q_OBJECT
public:
    QDataDescriptionWidget(QWidget* parent, core::objectmodel::Base* object);

    void addRow(QGridLayout* grid, const std::string& title,
                const std::string& value, unsigned int row, unsigned int minimumWidth =0);

    void addRowHyperLink(QGridLayout* grid, const std::string& title,
                const std::string& value, unsigned int row, unsigned int minimumWidth =0);
};


} //namespace sofa::gui::qt
