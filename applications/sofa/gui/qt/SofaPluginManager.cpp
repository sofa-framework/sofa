/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaPluginManager.h"
#include "FileManagement.h"
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>

#ifdef SOFA_QT4
//#include <Q3Header>
//#include <Q3PopupMenu>
#include <QMessageBox>
#include <QLibrary>
#include <QSettings>
#include <QTextEdit>
#include <QPushButton>
#else
//#include <qheader.h>
//#include <qpopupmenu.h>
#include <qmessagebox.h>
#include <qlibrary.h>
#include <qsettings.h>
#include <qtextedit.h>
#include <qpushbutton.h>
#endif

#include <iostream>
#include <sstream>



#ifndef SOFA_QT4
typedef QListViewItem Q3ListViewItem;
#endif

namespace sofa
{
namespace gui
{
namespace qt
{

#define LOCATION_COLUMN 3

SofaPluginManager::SofaPluginManager()
{
    // SIGNAL / SLOTS CONNECTIONS
    this->connect(buttonAdd, SIGNAL(clicked() ),  this, SLOT( addLibrary() ));
    this->connect(buttonRemove, SIGNAL(clicked() ),  this, SLOT( removeLibrary() ));
#ifdef SOFA_QT4
    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*) ), this, SLOT(updateComponentList(Q3ListViewItem*) ));
    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*) ), this, SLOT(updateDescription(Q3ListViewItem*) ));
#else
    this->connect(listPlugins, SIGNAL(selectionChanged(QListViewItem*) ), this, SLOT(updateComponentList(QListViewItem*) ));
    this->connect(listPlugins, SIGNAL(selectionChanged(QListViewItem*) ), this, SLOT(updateDescription(QListViewItem*) ));
#endif


}

void SofaPluginManager::initPluginList()
{
    //for compatibility with previous version : transfer plugin list from SOFA to sofa path (to be removed one day...)
    transferPluginsToNewPath();

    //read the plugin list in the settings
    QSettings settings;
    std::string binPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str());
    settings.setPath( "SOFA-FRAMEWORK", QString(binPath.c_str()), QSettings::User);

    settings.beginGroup("/plugins");
    int size = settings.readNumEntry("/size");

    listPlugins->clear();
    typedef void (*componentLoader)();
    typedef const char* (*componentStr)();

    for (int i=1 ; i<=size; i++)
    {
        QString config;
        config = config.setNum(i);

        settings.beginGroup(config);
        QString sfile = settings.readEntry("/location");
        settings.endGroup();

        //load the plugin libs -> automatically look at the relase/debug version depending on the current mode we are
#ifndef NDEBUG
        //add the "d" in the name if we are currently in debug mode
        sfile.replace(QString("."), QString("d."));
#endif
        QLibrary lib(sfile);

        componentLoader componentLoaderFunc = (componentLoader) lib.resolve("initExternalModule");
        componentStr componentNameFunc = (componentStr) lib.resolve("getModuleName");

        //fill the list view
        if (componentLoaderFunc && componentNameFunc)
        {
            componentLoaderFunc();
            QString sname(componentNameFunc());

            componentStr componentLicenseFunc = (componentStr) lib.resolve("getModuleLicense");
            QString slicense;
            if (componentLicenseFunc)
                slicense=componentLicenseFunc();

            componentStr componentVersionFunc = (componentStr) lib.resolve("getModuleVersion");
            QString sversion;
            if(componentVersionFunc)
                sversion=componentVersionFunc();

            Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, slicense, sversion, sfile);
            item->setSelectable(true);
            pluginList.insert(std::string(sfile.ascii()) );
            emit( libraryAdded() );
        }
    }
    // 			settings.endArray();
    settings.endGroup();


}



void SofaPluginManager::addLibrary()
{
    //get the lib to load
    std::string pluginPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/lib/sofa-plugins/" );
#if defined (__APPLE__)
    QString sfile = getOpenFileName ( this, QString(pluginPath.c_str()), "dynamic library (*.dylib*)", "load library dialog",  "Choose the component library to load" );
#elif defined (WIN32)
    QString sfile = getOpenFileName ( this, QString(pluginPath.c_str()), "dynamic library (*.dll)", "load library dialog",  "Choose the component library to load" );
#else
    QString sfile = getOpenFileName ( this, QString(pluginPath.c_str()), "dynamic library (*.so)", "load library dialog",  "Choose the component library to load" );
#endif
    if(sfile=="")
        return;
#ifdef NDEBUG
    if(sfile.contains(QString("d.")) == true)
        if(QMessageBox::question(this, "library loading warning","This plugin lib seems to be in debug mode whereas you are currently in release mode.\n Are you sure you want to load this lib?",QMessageBox::Yes,QMessageBox::No) != QMessageBox::Yes)
            return;
#else
    if(sfile.contains(QString("d.")) == false)
        if(QMessageBox::question(this, "library loading warning","This plugin lib seems to be in release mode whereas you are currently in debug mode.\n Are you sure you want to load this lib?",QMessageBox::Yes,QMessageBox::No) != QMessageBox::Yes)
            return;
#endif

    //try to load the lib
    QLibrary lib(sfile);
    if (!lib.load())
        std::cout<<"Error loading plugin " << sfile.latin1() <<std::endl;

    //get the functions
    typedef void (*componentLoader)();
    typedef const char* (*componentStr)();
    componentLoader componentLoaderFunc = (componentLoader) lib.resolve("initExternalModule");
    componentStr componentNameFunc = (componentStr) lib.resolve("getModuleName");

    if (componentLoaderFunc && componentNameFunc)
    {
        //fill the list view
        componentLoaderFunc();
        QString sname(componentNameFunc());

        componentStr componentLicenseFunc = (componentStr) lib.resolve("getModuleLicense");
        QString slicense;
        if (componentLicenseFunc)
            slicense=componentLicenseFunc();


        componentStr componentVersionFunc = (componentStr) lib.resolve("getModuleVersion");
        QString sversion;
        if(componentVersionFunc)
            sversion=componentVersionFunc();

        Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, slicense, sversion, sfile);
        item->setSelectable(true);

        //add to the settings (to record it)
        QSettings settings;
        std::string binPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str());
        settings.setPath( "SOFA-FRAMEWORK", QString(binPath.c_str()), QSettings::User);
        settings.beginGroup("/plugins");
        int size = settings.readNumEntry("/size");
        QString titi;
        titi = titi.setNum(size+1);
        settings.beginGroup(titi);
#ifndef NDEBUG
        //remove the "d" in the name if we are currently in debug mode
        sfile.replace(QString("d."), QString("."));
#endif
        settings.writeEntry("/location", sfile);
        settings.endGroup();
        settings.writeEntry("/size", size+1);
        settings.endGroup();
        pluginList.insert(std::string(sfile.ascii()) );
        emit( libraryAdded() );
    }
    else
    {
        QMessageBox * mbox = new QMessageBox(this,"library loading error");
        mbox->setText("Unable to load this library");
        mbox->show();
    }
}



void SofaPluginManager::removeLibrary()
{
    //get the selected item
    Q3ListViewItem * curItem = listPlugins->selectedItem();
    if (!curItem) return;
    QString location = curItem->text(LOCATION_COLUMN); //get the location value
    //remove it from the list view
    listPlugins->removeItem(curItem);

    //remove it from the settings
    QSettings settings;
    std::string binPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str());
    settings.setPath( "SOFA-FRAMEWORK", QString(binPath.c_str()), QSettings::User);
    settings.beginGroup("/plugins");
    int size = settings.readNumEntry("/size");

    for (int i=1 ; i<=size; i++)
    {
        QString config;
        config = config.setNum(i);
        settings.beginGroup(config);
        QString sfile = settings.readEntry("/location");
        if (sfile == location)
        {
            settings.removeEntry("/location");
            pluginList.erase(pluginList.find(std::string(sfile.ascii()) ) );
        }
        settings.endGroup();
    }

    settings.endGroup();
    description->clear();
    listComponents->clear();

    emit( libraryRemoved() );
}



void SofaPluginManager::updateComponentList(Q3ListViewItem* curItem)
{

    //update the component list when an item is selected
    listComponents->clear();
    QString location = curItem->text(LOCATION_COLUMN); //get the location value
    QLibrary lib(location);
    typedef const char* (*componentStr)();
    componentStr componentListFunc = (componentStr) lib.resolve("getModuleComponentList");
    if(componentListFunc)
    {
        QString cpts( componentListFunc() );
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
}


void SofaPluginManager::updateDescription(Q3ListViewItem* curItem)
{
    //update the component list when an item is selected
    description->clear();
    QString location = curItem->text(LOCATION_COLUMN); //get the location value
    QLibrary lib(location);
    typedef const char* (*componentStr)();
    componentStr componentDescFunc = (componentStr) lib.resolve("getModuleDescription");
    if(componentDescFunc)
    {
        description->setText(QString(componentDescFunc()));
    }
}


void SofaPluginManager::transferPluginsToNewPath()
{
    //read the plugin list in the settings
    QSettings oldSettings;
    oldSettings.setPath( "SOFA-FRAMEWORK", "SOFA", QSettings::User);
    oldSettings.beginGroup("/plugins");
    int size = oldSettings.readNumEntry("/size");

    QSettings newSettings;
    std::string binPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str());
    newSettings.setPath( "SOFA-FRAMEWORK", QString(binPath.c_str()), QSettings::User);
    newSettings.beginGroup("/plugins");

    listPlugins->clear();

    for (int i=1 ; i<=size; i++)
    {
        QString plugId;
        plugId = plugId.setNum(i);

        //get the plugin location in the old list
        oldSettings.beginGroup(plugId);
        QString sfile = oldSettings.readEntry("/location");
        oldSettings.endGroup();

        //put it in the new one
        int sizeNewList = newSettings.readNumEntry("/size");
        QString titi;
        titi = titi.setNum(sizeNewList+1);
        newSettings.beginGroup(titi);
        newSettings.writeEntry("/location", sfile);
        newSettings.endGroup();
        newSettings.writeEntry("/size", sizeNewList+1);

        //remove it from the old list
        oldSettings.beginGroup(plugId);
        oldSettings.removeEntry("/location");
        oldSettings.endGroup();
    }
    oldSettings.writeEntry("/size", 0);
    oldSettings.endGroup();
    newSettings.endGroup();
}

}
}
}

