/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_DISPLAYFLAGSDATAWIDGET_H
#define SOFA_GUI_QT_DISPLAYFLAGSDATAWIDGET_H

#include <sofa/gui/qt/DataWidget.h>
#include <sofa/core/visual/DisplayFlags.h>

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3CheckListItem>
#include <Q3ListViewItem>
#include <Q3Header>
#include <QMouseEvent>
#include <Q3Frame>
#include <Q3GroupBox>
#else
#include <qlistview.h>
#include <qheader.h>
#include <qgroupbox.h>
typedef QListView Q3ListView;
typedef QCheckListItem Q3CheckListItem;
typedef QListViewItem Q3ListViewItem;
typedef QGroupBox Q3GroupBox;
#endif

namespace sofa
{
namespace gui
{
namespace qt
{


class SOFA_SOFAGUIQT_API DisplayFlagWidget : public Q3ListView
{
    Q_OBJECT;
public:

    enum VISUAL_FLAG
    {
        VISUALMODELS,
        BEHAVIORMODELS,
        COLLISIONMODELS,
        BOUNDINGCOLLISIONMODELS,
        MAPPINGS,MECHANICALMAPPINGS,
        FORCEFIELDS,
        INTERACTIONFORCEFIELDS,
        RENDERING,
        WIREFRAME,
        NORMALS,
#ifdef SOFA_SMP
        PROCESSORCOLOR,
#endif
        ALLFLAGS
    };


    DisplayFlagWidget(QWidget* parent, const char* name= 0, Qt::WFlags f= 0 );

    bool getFlag(int idx) {return itemShowFlag[idx]->isOn();}
    void setFlag(int idx, bool value) {itemShowFlag[idx]->setOn(value);}

Q_SIGNALS:
    void change(int,bool);
    void clicked();


protected:
    virtual void contentsMousePressEvent ( QMouseEvent * e );

    void findChildren(Q3CheckListItem *, std::vector<Q3CheckListItem* > &children);


    Q3CheckListItem* itemShowFlag[ALLFLAGS];
    std::map<  Q3CheckListItem*, int > mapFlag;
};


class SOFA_SOFAGUIQT_API DisplayFlagsDataWidget : public TDataWidget< sofa::core::visual::DisplayFlags >
{
    Q_OBJECT;
public:
    typedef sofa::core::visual::DisplayFlags DisplayFlags;
    DisplayFlagsDataWidget(QWidget* parent, const char* name, core::objectmodel::Data<DisplayFlags>* data, bool root = false)
        :TDataWidget<DisplayFlags>(parent,name,data), isRoot(root)
    {
    }

    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);

protected:

    virtual void readFromData();
    virtual void writeToData();
    virtual unsigned int sizeWidget() {return 8;}
    virtual unsigned int numColumnWidget() {return 1;}

    DisplayFlagWidget* flags;
    bool isRoot;


};

}

}

}


#endif // SOFA_GUI_QT_DISPLAYFLAGSDATAWIDGET_H
