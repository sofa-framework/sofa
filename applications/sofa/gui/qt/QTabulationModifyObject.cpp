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
#include "QTabulationModifyObject.h"

#include "QDisplayDataWidget.h"
#include "QDisplayLinkWidget.h"

#include "ModifyObject.h"
#include <QDebug>
#include <QApplication>
#include <QDesktopWidget>
// uncomment to show traces of GUI operations in this file
//#define DEBUG_GUI

namespace sofa
{

namespace gui
{

namespace qt
{

QTabulationModifyObject::QTabulationModifyObject(QWidget* parent,
        core::objectmodel::Base *o, Q3ListViewItem* i,
        unsigned int idx):
    QWidget(parent), object(o), item(i), index(idx), size(0), dirty(false), pixelSize(0), pixelMaxSize(600)
{
    new QVBoxLayout( this, 0, 1, "tabVisualizationLayout");

    //find correct maxPixelSize according to the current screen resolution
    const int screenHeight = QApplication::desktop()->height();
    pixelMaxSize = screenHeight - 300;
}

void QTabulationModifyObject::addData(sofa::core::objectmodel::BaseData *data, const ModifyObjectFlags& flags)
{

    if (  (!data->isDisplayed()) && flags.HIDE_FLAG )
    {
#ifdef DEBUG_GUI
        std::cout << "GUI: data " << data->getName() << " is hidden." << std::endl;
#endif
        return;
    }

#ifdef DEBUG_GUI
    std::cout << "GUI> addData " << data->getName() << std::endl;
#endif

    data->setDisplayed(true);

    const std::string name=data->getName();
    QDisplayDataWidget* displaydatawidget = new QDisplayDataWidget(this,data,flags);
    this->layout()->add(displaydatawidget);

    size += displaydatawidget->getNumWidgets();
    pixelSize += displaydatawidget->sizeHint().height();


    connect(displaydatawidget, SIGNAL( WidgetDirty(bool) ), this, SLOT( setTabDirty(bool) ) );
    connect(displaydatawidget, SIGNAL( DataOwnerDirty(bool)),  this, SLOT( updateListViewItem() ) );
    connect(this, SIGNAL(UpdateDatas()), displaydatawidget, SLOT( UpdateData()));
    connect(this, SIGNAL(UpdateDataWidgets()), displaydatawidget, SLOT( UpdateWidgets()));
    connect(displaydatawidget, SIGNAL( dataValueChanged(QString) ), SLOT(dataValueChanged(QString) ) );
#ifdef DEBUG_GUI
    std::cout << "GUI< addData " << data->getName() << std::endl;
#endif
}


void QTabulationModifyObject::addLink(sofa::core::objectmodel::BaseLink *link, const ModifyObjectFlags& flags)
{
    //if (  (!link->isDisplayed()) && flags.HIDE_FLAG ) return;

    //link->setDisplayed(true);

    const std::string name=link->getName();
    QDisplayLinkWidget* displaylinkwidget = new QDisplayLinkWidget(this,link,flags);
    this->layout()->add(displaylinkwidget);

    size += displaylinkwidget->getNumWidgets();
    pixelSize += displaylinkwidget->sizeHint().height();

    connect(displaylinkwidget, SIGNAL( WidgetDirty(bool) ), this, SLOT( setTabDirty(bool) ) );
    connect(displaylinkwidget, SIGNAL( LinkOwnerDirty(bool)),  this, SLOT( updateListViewItem() ) );
    connect(this, SIGNAL(UpdateDatas()), displaylinkwidget, SLOT( UpdateLink()));
    connect(this, SIGNAL(UpdateDataWidgets()), displaylinkwidget, SLOT( UpdateWidgets()));
}

void QTabulationModifyObject::dataValueChanged(QString dataValue)
{
    m_dataValueModified[sender()] = dataValue;
}

void QTabulationModifyObject::updateListViewItem()
{
    if (simulation::Node *node=dynamic_cast< simulation::Node *>(object))
    {
        item->setText(0,object->getName().c_str());
        emit nodeNameModification(node);
    }
    else
    {
        QString currentName = item->text(0);

        std::string name=item->text(0).ascii();
        std::string::size_type pos = name.find(' ');
        if (pos != std::string::npos)
            name = name.substr(0,pos);
        name += "  ";
        name += object->getName();
        QString newName(name.c_str());
        if (newName != currentName) item->setText(0,newName);
    }
}

QString QTabulationModifyObject::getDataModifiedString() const
{
    if (m_dataValueModified.empty())
    {
       return QString();
    }

    QString dataModifiedString;
    std::map< QObject*, QString>::const_iterator it_map;
    std::map< QObject*, QString>::const_iterator it_last = m_dataValueModified.end();
    --it_last;

    for (it_map = m_dataValueModified.begin(); it_map != m_dataValueModified.end(); ++it_map)
    {
        const QString& lastDataValue = it_map->second;
        dataModifiedString += lastDataValue;
        if (it_map != it_last)
        {
            dataModifiedString += "\n";
        }
    }

    return dataModifiedString;
}

void QTabulationModifyObject::setTabDirty(bool b)
{
    dirty=b;
    emit TabDirty(b);
}

bool QTabulationModifyObject::isDirty() const
{
    return dirty;
}

bool QTabulationModifyObject::isFull() const
{
    return pixelSize >= pixelMaxSize;
    //return size >= maxSize;
}

bool QTabulationModifyObject::isEmpty() const
{
    return size==0;
}

void QTabulationModifyObject::updateDataValue()
{
    emit UpdateDatas();
}

void QTabulationModifyObject::updateWidgetValue()
{
    emit UpdateDataWidgets();
}

void QTabulationModifyObject::addStretch()
{
    dynamic_cast<QVBoxLayout*>(this->layout())->addStretch();
}

} // namespace qt

} // namespace gui

} // namespace sofa
