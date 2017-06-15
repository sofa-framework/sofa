/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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


#ifndef SOFA_GUI_QT_MOUSEMANAGER_H
#define SOFA_GUI_QT_MOUSEMANAGER_H

#include <sofa/gui/PickHandler.h>
#include <QDialog>
#include <memory>

class Ui_MouseManager;

namespace sofa
{
namespace gui
{
namespace qt
{


class SofaMouseManager : public QDialog
{
    Q_OBJECT
    std::unique_ptr<Ui_MouseManager> gui;
public:

    SofaMouseManager();
    ~SofaMouseManager();

    static SofaMouseManager* getInstance()
    {
        static SofaMouseManager instance;
        return &instance;
    }

    void updateContent();

    void setPickHandler(PickHandler *);

    std::map< int, std::string >& getMapIndexOperation()
    {
        return mapIndexOperation;
    }

    void updateOperation( sofa::component::configurationsetting::MouseButtonSetting* setting);
    void updateOperation( MOUSE_BUTTON button, const std::string &id);


public slots:
    void selectOperation(int);

protected:
    void updateOperation(Operation* op);
    PickHandler *pickHandler;
    std::map< int, std::string > mapIndexOperation;

    helper::fixed_array< std::string,NONE > usedOperations;
};


}
}
}

#endif
