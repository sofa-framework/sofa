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
#ifndef SOFA_GUI_VIEWER_QT_QMENUFILESRECENTLYOPENED_H
#define SOFA_GUI_VIEWER_QT_QMENUFILESRECENTLYOPENED_H

#include <sofa/gui/FilesRecentlyOpenedManager.h>
#include "SofaGUIQt.h"

#include <QMenu>

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_SOFAGUIQT_API QMenuFilesRecentlyOpened: public FilesRecentlyOpenedManager
{
public:
    QMenuFilesRecentlyOpened(const std::string &configFile):FilesRecentlyOpenedManager(configFile),menuRecentlyOpenedFiles(0) {};
    virtual ~QMenuFilesRecentlyOpened() {if (menuRecentlyOpenedFiles) delete menuRecentlyOpenedFiles;};
    void openFile(const std::string &file);

    QMenu *createWidget(QWidget *parent, const std::string& =std::string("Recently Opened Files ..."));
    QMenu *getMenu() {return menuRecentlyOpenedFiles;};

protected:
    void updateWidget();
    QMenu *menuRecentlyOpenedFiles;


};


}
}
}

#endif
