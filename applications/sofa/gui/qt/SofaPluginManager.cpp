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

#include <sofa/core/Plugin.h>

#include <QMessageBox>
#include <QTextEdit>
#include <QPushButton>

#include <iostream>
#include <sstream>

using sofa::core::Plugin;
using sofa::core::PluginManager;
using sofa::helper::system::DynamicLibrary;


namespace sofa
{
namespace gui
{
namespace qt
{

SofaPluginManager::SofaPluginManager(PluginManager& pluginManager):
    m_pluginManager(pluginManager)
{
    setupUi(this);

    // SIGNAL / SLOTS CONNECTIONS
    this->connect(buttonAdd, SIGNAL(clicked()), this, SLOT( addLibrary()));
    this->connect(buttonRemove, SIGNAL(clicked()), this, SLOT( removeLibrary()));
    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*)),
                  this, SLOT(updateComponentList(Q3ListViewItem*)));
    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*)),
                  this, SLOT(updateDescription(Q3ListViewItem*)));
    // m_pluginManager.initRecentlyOpened();
    initPluginListView();
}


std::string SofaPluginManager::getSelectedPluginName()
{
    Q3ListViewItem * selectedItem = listPlugins->selectedItem();
    // the name of the plugin is in the column 0
    return std::string(selectedItem->text(0).toAscii());
}

void SofaPluginManager::initPluginListView()
{
    const PluginManager::LoadedPluginMap& map = m_pluginManager.getLoadedPlugins();
    listPlugins->clear();
    for(PluginManager::LoadedPluginMap::const_iterator iter = map.begin(); iter != map.end(); ++iter)
    {
        sofa::core::Plugin& plugin = *iter->second.plugin;
        QString slicense(plugin.getLicense().c_str());
        QString sname(plugin.getName().c_str());
        QString sversion(plugin.getVersion().c_str());
        QString sfile(iter->first.c_str());
        Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, slicense, sversion, sfile);
        item->setSelectable(true);
    }
}

void SofaPluginManager::addLibrary()
{
    // compute the plugin directory path
    QDir dir = QCoreApplication::applicationDirPath();
	dir.cd("plugins");
    QString pluginPath = dir.canonicalPath();
    //get the lib to load
    std::string fileType = "dynamic library (*." + DynamicLibrary::extension + "*)";
    QString sfile = getOpenFileName(this, pluginPath, fileType.c_str(),
                                    "load library dialog",
                                    "Choose the component library to load");
    if(sfile=="")
        return;
// #ifdef NDEBUG
//     if(sfile.contains(QString("d.")) == true)
//         if(QMessageBox::question(this, "library loading warning","This plugin lib seems to be in debug mode whereas you are currently in release mode.\n Are you sure you want to load this lib?",QMessageBox::Yes,QMessageBox::No) != QMessageBox::Yes)
//             return;
// #else
//     if(sfile.contains(QString("d.")) == false)
//         if(QMessageBox::question(this, "library loading warning","This plugin lib seems to be in release mode whereas you are currently in debug mode.\n Are you sure you want to load this lib?",QMessageBox::Yes,QMessageBox::No) != QMessageBox::Yes)
//             return;
// #endif
    std::stringstream sstream;

    std::string pluginFile = std::string(sfile.ascii());

    try {
        Plugin& plugin = m_pluginManager.loadPlugin(pluginFile);
        if (!plugin.isLegacy())
            m_pluginManager.addComponentsToFactory(*sofa::core::ObjectFactory::getInstance(), plugin);

        QString slicense(plugin.getLicense().c_str());
        QString sname(plugin.getName().c_str());
        QString sversion(plugin.getVersion().c_str());

        Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, slicense, sversion, pluginFile.c_str());
        item->setSelectable(true);
        // m_pluginManager.writeToIniFile();
        emit( libraryAdded() );
    }
    catch (std::exception& e) {
        QMessageBox * mbox = new QMessageBox(this,"Error while loading plugin");
        mbox->setIcon(QMessageBox::Critical);
        mbox->setText(e.what());
        mbox->show();
    }
}



void SofaPluginManager::removeLibrary()
{
    //get the selected item
    Q3ListViewItem * curItem = listPlugins->selectedItem();
    std::stringstream sstream;
    if (!curItem) return;
    const std::string pluginName = getSelectedPluginName();

    try {
        const Plugin* plugin = m_pluginManager.getLoadedPlugins().find(pluginName)->second.plugin;
        if (!plugin->isLegacy()) {
            m_pluginManager.removeComponentsFromFactory(*sofa::core::ObjectFactory::getInstance(),
                                                        *plugin);
            m_pluginManager.unloadPlugin(pluginName);
        }
    }
    catch (std::exception& e) {
        QMessageBox * mbox = new QMessageBox(this,"Error while unloading plugin");
        mbox->setIcon(QMessageBox::Critical);
        mbox->setText(e.what());
        mbox->show();
    }

    listPlugins->removeItem(curItem);
    // m_pluginManager.writeToIniFile();
    emit( libraryRemoved() );
    description->clear();
    listComponents->clear();
}

void SofaPluginManager::updateComponentList(Q3ListViewItem* curItem)
{
    if (curItem == NULL)
        return;

    listComponents->clear();

    std::string pluginName = getSelectedPluginName();
    const sofa::core::Plugin& plugin = m_pluginManager.getLoadedPlugin(pluginName);

    for (std::map<std::string, Plugin::ComponentEntry>::const_iterator
             i = plugin.getComponentEntries().begin();
         i != plugin.getComponentEntries().end();
         i++) {
        Q3ListViewItem *item=new Q3ListViewItem(listComponents,curItem);
        item->setText(0,i->first.c_str());
    }
}


void SofaPluginManager::updateDescription(Q3ListViewItem* curItem)
{
    if (curItem == NULL)
        return;

    std::string pluginName = getSelectedPluginName();
    const sofa::core::Plugin& plugin = m_pluginManager.getLoadedPlugin(pluginName);
    description->setText(QString(plugin.getDescription().c_str()));
}


}
}
}
