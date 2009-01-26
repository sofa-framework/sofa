/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
 *                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifdef SOFA_QT4
//#include <Q3Header>
//#include <Q3PopupMenu>
#include <QMessageBox>
#include <QLibrary>
#include <QSettings>
#else
//#include <qheader.h>
//#include <qpopupmenu.h>
#include <qmessagebox.h>
#include <qlibrary.h>
#include <qsettings.h>
#endif

#include <iostream>


#ifndef SOFA_QT4
typedef QListViewItem Q3ListViewItem;
#endif

namespace sofa
{
namespace gui
{
namespace qt
{

SofaPluginManager::SofaPluginManager()
{
#ifdef SOFA_QT4
    // SIGNAL / SLOTS CONNECTIONS
    this->connect(buttonAdd, SIGNAL(clicked() ),  this, SLOT( addLibrary() ));
    this->connect(buttonRemove, SIGNAL(clicked() ),  this, SLOT( removeLibrary() ));
    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*) ), this, SLOT(updateComponentList(Q3ListViewItem*) ));
    this->connect(listPlugins, SIGNAL(selectionChanged(Q3ListViewItem*) ), this, SLOT(updateDescription(Q3ListViewItem*) ));

    //read the plugin list in the settings
    QSettings settings("SOFA-FRAMEWORK", "SOFA");
    int size = settings.beginReadArray("plugins");
    listPlugins->clear();
    typedef void (*componentLoader)();
    typedef char* (*componentStr)();

    for (int i=0 ; i<size; ++i)
    {
        settings.setArrayIndex(i);
        QString sfile = settings.value("location").toString();

        //load the plugin libs
        QLibrary lib(sfile);
        componentLoader componentLoaderFunc = (componentLoader) lib.resolve("initExternalModule");
        componentStr componentNameFunc = (componentStr) lib.resolve("getModuleName");
        //componentStr componentDescFunc = (componentStr) lib.resolve("getModuleDescription");
        //fill the list view
        if (componentLoaderFunc && componentNameFunc/* && componentDescFunc*/)
        {
            componentLoaderFunc();
            QString sname(componentNameFunc());
            //QString sdesc(componentDescFunc());
            Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, sfile/*, sdesc*/);
            item->setSelectable(true);
        }
    }
    settings.endArray();
#endif
}



void SofaPluginManager::addLibrary()
{
#ifdef SOFA_QT4
    //get the lib to load
    QString sfile = getOpenFileName ( this, NULL, "dynamic library (*.dll *.so *.dylib)", "load library dialog",  "Choose the component library to load" );

    //try to load the lib
    QLibrary lib(sfile);
    if (!lib.load())
        std::cout<<lib.errorString().latin1()<<std::endl;

    //get the functions
    typedef void (*componentLoader)();
    typedef char* (*componentStr)();
    componentLoader componentLoaderFunc = (componentLoader) lib.resolve("initExternalModule");
    componentStr componentNameFunc = (componentStr) lib.resolve("getModuleName");
    //componentStr componentDescFunc = (componentStr) lib.resolve("getModuleDescription");

    if (componentLoaderFunc && componentNameFunc/* && componentDescFunc*/)
    {
        //fill the list view
        componentLoaderFunc();
        QString sname(componentNameFunc());
        //QString sdesc(componentDescFunc());
        Q3ListViewItem * item = new Q3ListViewItem(listPlugins, sname, sfile/*, sdesc*/);
        item->setSelectable(true);

        //add to the settings (to record it)
        QSettings settings("SOFA-FRAMEWORK", "SOFA");
        int size = settings.beginReadArray("plugins");
        settings.endArray();
        settings.beginWriteArray("plugins");
        settings.setArrayIndex(size);
        settings.setValue("location", sfile);
        settings.endArray();
    }
    else
    {
        QMessageBox * mbox = new QMessageBox(this,"library loading error");
        mbox->setText("Unable to load this library");
        mbox->show();
    }
#endif
}



void SofaPluginManager::removeLibrary()
{
#ifdef SOFA_QT4
    //get the selected item
    Q3ListViewItem * curItem = listPlugins->selectedItem();
    QString location = curItem->text(1); //get the location value
    //remove it from the list view
    listPlugins->removeItem(curItem);

    //remove it from the settings
    QSettings settings("SOFA-FRAMEWORK", "SOFA");
    int size = settings.beginReadArray("plugins");

    for (int i=0 ; i<size; ++i)
    {
        settings.setArrayIndex(i);
        QString sfile = settings.value("location").toString();
        if (sfile == location)
        {
            settings.endArray();
            settings.beginWriteArray("plugins");
            settings.setArrayIndex(i);
            settings.remove("location");
        }
    }

    settings.endArray();
#endif
}



void SofaPluginManager::updateComponentList(Q3ListViewItem* curItem)
{
#ifdef SOFA_QT4
    //update the component list when an item is selected
    listComponents->clear();
    QString location = curItem->text(1); //get the location value
    QLibrary lib(location);
    typedef char* (*componentStr)();
    componentStr componentListFunc = (componentStr) lib.resolve("getModuleComponentList");
    if(componentListFunc)
    {
        QString cpts( componentListFunc() );
        cpts.replace(", ","\n");
        cpts.replace(",","\n");
        listComponents->setText(cpts);
    }
#endif
}

void SofaPluginManager::updateDescription(Q3ListViewItem* curItem)
{
#ifdef SOFA_QT4
    //update the component list when an item is selected
    description->clear();
    QString location = curItem->text(1); //get the location value
    QLibrary lib(location);
    typedef char* (*componentStr)();
    componentStr componentDescFunc = (componentStr) lib.resolve("getModuleDescription");
    if(componentDescFunc)
    {
        description->setText(QString(componentDescFunc()));
    }
}
#endif
}
}
}

