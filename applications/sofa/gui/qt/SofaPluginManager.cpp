/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "SofaPluginManager.h"
#include "../BaseGUI.h"
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

    this->connect(listPlugins, SIGNAL(itemSelectionChanged() ), this, SLOT(updateComponentList() ));
    this->connect(listPlugins, SIGNAL(itemSelectionChanged() ), this, SLOT(updateDescription() ));

    listPlugins->setHeaderLabels(QStringList() << "Name" << "License" << "Version" << "Location");
    listComponents->setHeaderLabels(QStringList() << "Component list");

    loadPluginsFromIniFile();
    updatePluginsListView();
}



void SofaPluginManager::updatePluginsListView()
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
        //QTreeWidgetItem * item = new QTreeWidgetItem(listPlugins, sname, slicense, sversion, sfile);
        QTreeWidgetItem * item = new QTreeWidgetItem(listPlugins);
        item->setText(0, sname);
        item->setText(1, slicense);
        item->setText(2, sversion);
        item->setText(3, sfile);
        //item->setSelected(true);
        //listPlugins->addTopLevelItem(item);
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

    std::string pluginFile = std::string(sfile.toStdString());
    if(sofa::helper::system::PluginManager::getInstance().loadPluginByPath(pluginFile,&sstream))
    {
        typedef sofa::helper::system::Plugin    Plugin;
        if( ! sstream.str().empty())
        {
            QMessageBox * mbox = new QMessageBox(this);
            mbox->setWindowTitle("library loading warning");
            mbox->setIcon(QMessageBox::Warning);
            mbox->setText(sstream.str().c_str());
            mbox->show();
        }
        Plugin* plugin = sofa::helper::system::PluginManager::getInstance().getPlugin(pluginFile);
        if(!plugin)
        {
            // This should not happen as we are protected by if(loadPluginByPath(...))
            msg_error("SofaPluginManager") << "plugin should be loaded: " << pluginFile << msgendl;
            return;
        }
        QString slicense = plugin->getModuleLicense();
        QString sname    = plugin->getModuleName();
        QString sversion = plugin->getModuleVersion();

        //QTreeWidgetItem * item = new QTreeWidgetItem(listPlugins, sname, slicense, sversion, pluginFile.c_str());
        QTreeWidgetItem * item = new QTreeWidgetItem(listPlugins);
        item->setText(0, sname);
        item->setText(1, slicense);
        item->setText(2, sversion);
        item->setText(3, pluginFile.c_str());
        listPlugins->addTopLevelItem(item);

        //item->setSelectable(true);
        savePluginsToIniFile();
        emit( libraryAdded() );
    }
    else
    {
        QMessageBox * mbox = new QMessageBox(this);
        mbox->setWindowTitle("library loading error");
        mbox->setIcon(QMessageBox::Critical);
        mbox->setText(sstream.str().c_str());
        mbox->show();
    }


}



void SofaPluginManager::removeLibrary()
{
    //get the selected item
    if(listPlugins->selectedItems().count() < 1)
        return;

    QTreeWidgetItem * curItem = listPlugins->selectedItems()[0];
    std::stringstream sstream;
    if (!curItem) return;

    std::string location( curItem->text(LOCATION_COLUMN).toStdString() ); //get the location value

    if( sofa::helper::system::PluginManager::getInstance().unloadPlugin(location,&sstream) )
    {
        //listPlugins->removeItem(curItem);
        delete curItem;

        savePluginsToIniFile();
        emit( libraryRemoved() );

        if(this->listPlugins->selectedItems().count() < 1)
        {
            description->clear();
            listComponents->clear();
        }
    }
    else
    {
        std::string errlog;
        sstream >> errlog;
        QMessageBox * mbox = new QMessageBox(this);
        mbox->setWindowTitle("library unloading error");
        mbox->setIcon(QMessageBox::Critical);
        mbox->setText(errlog.c_str());
        mbox->show();
    }

}

void SofaPluginManager::updateComponentList()
{
    if(this->listPlugins->selectedItems().count() < 1)
        return;

    QTreeWidgetItem* curItem = this->listPlugins->selectedItems()[0];

    if(curItem == NULL ) return;
    //update the component list when an item is selected
    listComponents->clear();

    std::string location( curItem->text(LOCATION_COLUMN).toStdString() ); //get the location value

    typedef sofa::helper::system::Plugin    Plugin;
    Plugin* plugin = sofa::helper::system::PluginManager::getInstance().getPlugin(location);
    if(!plugin)
    {
        msg_warning("SofaPluginManager") << "plugin is not loaded: " << location << msgendl;
        return;
    }

    QString cpts( plugin->getModuleComponentList() );
    cpts.replace(", ","\n");
    cpts.replace(",","\n");
    std::istringstream in(cpts.toStdString());

    while (!in.eof())
    {
        std::string componentText;
        in >> componentText;
        //QTreeWidgetItem *item=new QTreeWidgetItem(listComponents,curItem);

        QTreeWidgetItem * item = new QTreeWidgetItem(listComponents);
        item->setText(0, componentText.c_str());
    }
}


void SofaPluginManager::updateDescription()
{
    if(this->listPlugins->selectedItems().count() < 1)
        return;

    QTreeWidgetItem* curItem = this->listPlugins->selectedItems()[0];

    if(curItem == NULL ) return;
    //update the component list when an item is selected
    description->clear();

    std::string location( curItem->text(LOCATION_COLUMN).toStdString() ); //get the location value

    typedef sofa::helper::system::Plugin    Plugin;
    Plugin* plugin = sofa::helper::system::PluginManager::getInstance().getPlugin(location);
    if(!plugin)
    {
        msg_warning("SofaPluginManager") << "plugin is not loaded: " << location << msgendl;
        return;
    }
    description->setText(QString(plugin->getModuleDescription()));
}

void SofaPluginManager::savePluginsToIniFile()
{
    const std::string pluginsIniFile = sofa::gui::BaseGUI::getConfigDirectoryPath() + "/loadedPlugins.ini";
    sofa::helper::system::PluginManager::getInstance().writeToIniFile(pluginsIniFile);
}

void SofaPluginManager::loadPluginsFromIniFile()
{
    const std::string pluginsIniFile = sofa::gui::BaseGUI::getConfigDirectoryPath() + "/loadedPlugins.ini";
    sofa::helper::system::PluginManager::getInstance().readFromIniFile(pluginsIniFile);
}


}
}
}

