/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include "QMenuFilesRecentlyOpened.h"

#include <sofa/helper/system/FileRepository.h>
#include <fstream>
#include <algorithm>


namespace sofa
{
namespace gui
{
namespace qt
{

void QMenuFilesRecentlyOpened::updateWidget()
{
    //Clear the current widget
    menuRecentlyOpenedFiles->clear();
    //while (menuRecentlyOpenedFiles->actions().count())
    //    menuRecentlyOpenedFiles->removeAction(menuRecentlyOpenedFiles->actions()[0]);
    //Add the content of files
    for (unsigned int i=0; i<files.size(); ++i)
        menuRecentlyOpenedFiles->addAction(QString(files[i].c_str()));
}

QMenu *QMenuFilesRecentlyOpened::createWidget(QWidget *parent, const std::string &name)
{
    menuRecentlyOpenedFiles = new QMenu(QString(name.c_str()), parent);
    menuRecentlyOpenedFiles->setTearOffEnabled(true);

    updateWidget();
    return menuRecentlyOpenedFiles;
}

void QMenuFilesRecentlyOpened::openFile(const std::string &file)
{
    FilesRecentlyOpenedManager::openFile(file);
    updateWidget();
    writeFiles();
}


}
}
}

