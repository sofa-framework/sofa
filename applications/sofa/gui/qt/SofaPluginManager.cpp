/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaPluginManager.h"
#include "FileManagement.h"
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/DynamicLibrary.h>

#include <QMessageBox>
#include <QTextEdit>
#include <QPushButton>

#include <iostream>
#include <sstream>


namespace sofa
{
namespace gui
{
namespace qt
{

#define LOCATION_COLUMN 3

SofaPluginManager::SofaPluginManager()
{
    setupUi(this);
    // SIGNAL / SLOTS CONNECTIONS
    this->connect(buttonAdd, SIGNAL(clicked() ),  this, SLOT( addLibrary() ));
    this->connect(buttonRemove, SIGNAL(clicked() ),  this, SLOT( removeLibrary() ));

    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*) ), this, SLOT(updateComponentList(Q3ListViewItem*) ));
    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*) ), this, SLOT(updateDescription(Q3ListViewItem*) ));

    sofa::helper::system::PluginManager::getInstance().initRecentlyOpened();
    initPluginListView();
}



void SofaPluginManager::initPluginListView()
{
    typedef sofa::helper::system::PluginManager::PluginMap PluginMap;
    PluginMap& map = sofa::helper::system::PluginManager::getInstance().getPluginMap();
    typedef PluginMap::iterator PluginIterator;
    listPlugins->clear();
    for( PluginIterator iter = map.begin(); iter != map.end(); ++iter )
    {
        sofa::helper::system::Plugin& plugin = iter->second;
        QString slicense = plugin.getModuleLicense();
        QString sname    = plugin.getModuleName();
        QString sversion = plugin.getModuleVersion();
        QString sfile    = (iter->first).c_str();
        Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, slicense, sversion, sfile);
        item->setSelectable(true);
    }
}

void SofaPluginManager::addLibrary()
{
    // compute the plugin directory path
    QDir dir = QCoreApplication::applicationDirPath();
#if defined (WIN32)
	dir.cd("../bin");
#else
    dir.cd("../lib");
#endif
    QString pluginPath = dir.canonicalPath();
    //get the lib to load
#if defined (__APPLE__)
    QString sfile = getOpenFileName ( this, pluginPath, "dynamic library (*.dylib*)", "load library dialog",  "Choose the component library to load" );
#elif defined (WIN32)
    QString sfile = getOpenFileName ( this, pluginPath, "dynamic library (*.dll)", "load library dialog",  "Choose the component library to load" );
#else
    QString sfile = getOpenFileName ( this, pluginPath, "dynamic library (*.so)", "load library dialog",  "Choose the component library to load" );
#endif
    if(sfile=="")
        return;
#ifndef _DEBUG
    if(sfile.contains(QString("d.")) == true)
        if(QMessageBox::question(this, "library loading warning","This plugin lib seems to be in debug mode whereas you are currently in release mode.\n Are you sure you want to load this lib?",QMessageBox::Yes,QMessageBox::No) != QMessageBox::Yes)
            return;
#else
    if(sfile.contains(QString("d.")) == false)
        if(QMessageBox::question(this, "library loading warning","This plugin lib seems to be in release mode whereas you are currently in debug mode.\n Are you sure you want to load this lib?",QMessageBox::Yes,QMessageBox::No) != QMessageBox::Yes)
            return;
#endif
    std::stringstream sstream;

    std::string pluginFile = std::string(sfile.ascii());
    if(sofa::helper::system::PluginManager::getInstance().loadPlugin(pluginFile,&sstream))
    {
        typedef sofa::helper::system::PluginManager::PluginMap PluginMap;
        typedef sofa::helper::system::Plugin    Plugin;
        if( ! sstream.str().empty())
        {
            QMessageBox * mbox = new QMessageBox(this,"library loading warning");
            mbox->setIcon(QMessageBox::Warning);
            mbox->setText(sstream.str().c_str());
            mbox->show();
        }
        PluginMap& map = sofa::helper::system::PluginManager::getInstance().getPluginMap();
        Plugin& plugin = map[pluginFile];
        QString slicense = plugin.getModuleLicense();
        QString sname    = plugin.getModuleName();
        QString sversion = plugin.getModuleVersion();

        Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, slicense, sversion, pluginFile.c_str());
        item->setSelectable(true);
        sofa::helper::system::PluginManager::getInstance().writeToIniFile();
        emit( libraryAdded() );
    }
    else
    {
        QMessageBox * mbox = new QMessageBox(this,"library loading error");
        mbox->setIcon(QMessageBox::Critical);
        mbox->setText(sstream.str().c_str());
        mbox->show();
    }


}



void SofaPluginManager::removeLibrary()
{
    //get the selected item
    Q3ListViewItem * curItem = listPlugins->selectedItem();
    std::stringstream sstream;
    if (!curItem) return;

    std::string location( curItem->text(LOCATION_COLUMN).toAscii() ); //get the location value

    if( sofa::helper::system::PluginManager::getInstance().unloadPlugin(location,&sstream) )
    {
        listPlugins->removeItem(curItem);
        sofa::helper::system::PluginManager::getInstance().writeToIniFile();
        emit( libraryRemoved() );
        description->clear();
        listComponents->clear();
    }
    else
    {
        std::string errlog;
        sstream >> errlog;
        QMessageBox * mbox = new QMessageBox(this,"library unloading error");
        mbox->setIcon(QMessageBox::Critical);
        mbox->setText(errlog.c_str());
        mbox->show();
    }

}

void SofaPluginManager::updateComponentList(Q3ListViewItem* curItem)
{
    if(curItem == NULL ) return;
    //update the component list when an item is selected
    listComponents->clear();

    std::string location( curItem->text(LOCATION_COLUMN).toAscii() ); //get the location value

    typedef sofa::helper::system::PluginManager::PluginMap PluginMap;
    typedef sofa::helper::system::Plugin    Plugin;
    PluginMap& map = sofa::helper::system::PluginManager::getInstance().getPluginMap();
    Plugin& plugin = map[location];

    QString cpts( plugin.getModuleComponentList() );
    cpts.replace(", ","\n");
    cpts.replace(",","\n");
    std::istringstream in(cpts.ascii());

    while (!in.eof())
    {
        std::string componentText;
        in >> componentText;
        Q3ListViewItem *item=new Q3ListViewItem(listComponents,curItem);
        item->setText(0,componentText.c_str());
    }

}


void SofaPluginManager::updateDescription(Q3ListViewItem* curItem)
{
    if(curItem == NULL ) return;
    //update the component list when an item is selected
    description->clear();
    std::string location( curItem->text(LOCATION_COLUMN).toAscii() ); //get the location value
    typedef sofa::helper::system::PluginManager::PluginMap PluginMap;
    typedef sofa::helper::system::Plugin    Plugin;
    PluginMap& map = sofa::helper::system::PluginManager::getInstance().getPluginMap();
    Plugin plugin = map[location];
    description->setText(QString(plugin.getModuleDescription()));
}



}
}
}

