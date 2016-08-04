/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v16.08                  *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
//Cut/Copy Paste management
void SofaModeler::cut()
{
    if (graph)
    {
        isPasteReady=graph->cut(presetPath+"copyBuffer.scn");
        pasteAction->setEnabled(isPasteReady);
    }
}
void SofaModeler::copy()
{
    if (graph)
    {
        isPasteReady=graph->copy(presetPath+"copyBuffer.scn");
        pasteAction->setEnabled(isPasteReady);
    }
}
void SofaModeler::paste()
{
    if (graph)
    {
        graph->paste(presetPath+"copyBuffer.scn");
    }
}

void SofaModeler::showPluginManager()
{
    plugin_dialog->show();
}

void SofaModeler::displayMessage(const std::string &m)
{
    QString messageLaunch(m.c_str());
    statusBar()->showMessage(messageLaunch,5000);
}

void SofaModeler::displayHelpModeler()
{
    static std::string pathModelerHTML=sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/applications/projects/Modeler/Modeler.html" );
#ifdef WIN32
    infoItem->setSource(QUrl(QString("file:///")+QString(pathModelerHTML.c_str())));
#else
    infoItem->setSource(QUrl(QString(pathModelerHTML.c_str())));
#endif
}


}
}
}
