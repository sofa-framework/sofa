/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gui/qt/DataWidget.h>
#include <sofa/core/visual/DisplayFlags.h>

#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QHeaderView>
#include <QMouseEvent>
#include <QFrame>
#include <QGroupBox>

namespace sofa::gui::qt
{


class SOFA_GUI_QT_API DisplayFlagWidget : public QTreeWidget
{
    Q_OBJECT;
public:

    enum VISUAL_FLAG
    {
        VISUALMODELS,
        BEHAVIORMODELS,
        COLLISIONMODELS,
        BOUNDINGCOLLISIONMODELS,
        DETECTIONOUTPUTS,
        MAPPINGS,MECHANICALMAPPINGS,
        FORCEFIELDS,
        INTERACTIONFORCEFIELDS,
        RENDERING,
        WIREFRAME,
        NORMALS,
        ALLFLAGS
    };


    DisplayFlagWidget(QWidget* parent, const char* name= nullptr, Qt::WindowFlags f = Qt::WindowType::Widget );

    bool getFlag(int idx) {return itemShowFlag[idx]->checkState(0) == Qt::Checked;}
    void setFlag(int idx, bool value)
    {
        itemShowFlag[idx]->setCheckState(0, (value) ? Qt::Checked : Qt::Unchecked)   ;
    }

Q_SIGNALS:
    void change(int,bool);
    void clicked();


protected:
    void setTreeWidgetNodeCheckable(QTreeWidgetItem* w, const char* name);
    void setTreeWidgetCheckable(QTreeWidgetItem* w, const char* name);

    void mouseReleaseEvent ( QMouseEvent * e ) override;

    void findChildren(QTreeWidgetItem *, std::vector<QTreeWidgetItem* > &children);


    QTreeWidgetItem* itemShowFlag[ALLFLAGS];
    std::map<  QTreeWidgetItem*, int > mapFlag;
};


class SOFA_GUI_QT_API DisplayFlagsDataWidget : public TDataWidget< sofa::core::visual::DisplayFlags >
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

} // namespace sofa::gui::qt
