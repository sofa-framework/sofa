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
#ifndef SOFA_GUI_QT_QTABULATIONMODIFYOBJECT_H
#define SOFA_GUI_QT_QTABULATIONMODIFYOBJECT_H

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/simulation/Node.h>
#include <SofaGui/config.h>

#include <QWidget>
#include <QTextEdit>
#include <QGroupBox>
#include <QTreeWidgetItem>
#include <QTreeWidget>


namespace sofa
{
namespace gui
{
namespace qt
{

struct ModifyObjectFlags;
class DataWidget;

class QTabulationModifyObject : public QWidget
{
    Q_OBJECT
public:
    QTabulationModifyObject(QWidget* parent,
            core::objectmodel::Base *object, QTreeWidgetItem* item,
            unsigned int idx=1);

    void externalWidgetAddition(int num) {size+=num;}
    void addData(sofa::core::objectmodel::BaseData *data, const ModifyObjectFlags& flags);
    void addLink(sofa::core::objectmodel::BaseLink *link, const ModifyObjectFlags& flags);
    void addStretch();

    unsigned int getIndex() const {return index;}
    bool isFull() const;
    void setFull() {pixelSize=pixelMaxSize;}
    bool isEmpty() const;
    bool isDirty() const;

    QString getDataModifiedString() const;

public slots:
    void setTabDirty(bool=true);
    void updateListViewItem();
    void updateDataValue();
    void updateWidgetValue();
    void dataValueChanged(QString dataValue);

signals:
    void UpdateDatas();
    void UpdateDataWidgets();
    void TabDirty(bool);
    void nodeNameModification(simulation::Node *);



protected:
    core::objectmodel::Base *object;
    QTreeWidgetItem* item;


    const unsigned int index;
    unsigned int size;

    bool dirty;
    std::map< QObject*, QString> m_dataValueModified;

    unsigned int pixelSize;
    unsigned int pixelMaxSize;

};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QTABULATIONMODIFYOBJECT_H

