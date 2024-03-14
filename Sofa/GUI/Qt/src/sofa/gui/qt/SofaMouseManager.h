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
#include <sofa/gui/common/PickHandler.h>
#include <QDialog>
#include <memory>

class Ui_MouseManager;

namespace sofa::gui::qt
{


class SofaMouseManager : public QDialog
{
    Q_OBJECT
    std::unique_ptr<Ui_MouseManager> gui;
public:

    SofaMouseManager(QWidget *parent);
    ~SofaMouseManager() override;

    void updateContent();

    void setPickHandler(common::PickHandler *);

    std::map< int, std::string >& getMapIndexOperation()
    {
        return mapIndexOperation;
    }

    void updateOperation( sofa::component::setting::MouseButtonSetting* setting);
    void updateOperation(common::MOUSE_BUTTON button, const std::string &id);


public slots:
    void selectOperation(int);

protected:
    void updateOperation(common::Operation* op);
    common::PickHandler *pickHandler;
    std::map< int, std::string > mapIndexOperation;

    type::fixed_array< std::string, common::NONE > usedOperations;
};


} // namespace sofa::gui::qt
