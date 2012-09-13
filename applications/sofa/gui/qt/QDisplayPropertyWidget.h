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
#ifndef SOFA_GUI_QT_QDISPLAYPROPERTYWIDGET_H
#define SOFA_GUI_QT_QDISPLAYPROPERTYWIDGET_H

#include <sofa/gui/qt/SofaGUIQt.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/misc/Monitor.h>
#include <sofa/gui/qt/QTransformationWidget.h>
#include <sofa/gui/qt/QEnergyStatWidget.h>
#include <sofa/gui/qt/WDoubleLineEdit.h>

#include <QTreeWidget>
#ifdef SOFA_QT4
#include <QDialog>
#include <QWidget>
#include <Q3ListViewItem>
#include <Q3ListView>
#include <Q3Table>
#include <Q3GroupBox>
#include <Q3Grid>
#include <Q3TextEdit>
#include <QPushButton>
#include <QTabWidget>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <Q3CheckListItem>
#include <QVBoxLayout>
#else
#include <qdialog.h>
#include <qwidget.h>
#include <qlistview.h>
#include <qtable.h>
#include <qgroupbox.h>
#include <qgrid.h>
#include <qtextedit.h>
#include <qtabwidget.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qcheckbox.h>
#include <qspinbox.h>
#include <qlayout.h>
#endif

#ifndef SOFA_QT4
typedef QGroupBox Q3GroupBox;
typedef QTextEdit   Q3TextEdit;
typedef QListView   Q3ListView;
typedef QListViewItem Q3ListViewItem;
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

class SOFA_SOFAGUIQT_API QDisplayTreeItemWidget : public QWidget
{
    Q_OBJECT

public:

    // constructor / destructor
    QDisplayTreeItemWidget(QWidget* parent=0, QTreeWidgetItem* item=0);
    ~QDisplayTreeItemWidget();

protected slots:
    // resize the corresponding TreeItem when the Widget size changes
    void updateDirtyWidget();

private:
    QTreeWidgetItem* treeWidgetItem;
};

class SOFA_SOFAGUIQT_API QDisplayPropertyWidget : public QTreeWidget
{
    friend class GraphHistoryManager;
    friend class LinkComponent;
    Q_OBJECT

public:
    // constructor / destructor
    QDisplayPropertyWidget(QWidget* parent=0);
    ~QDisplayPropertyWidget();

    // add a component in the tree in order to show / change its data and compare them with other component data
    void addComponent(const QString& component, core::objectmodel::Base* base, Q3ListViewItem* listItem, bool clear = true);

    // add a data group
    void addGroup(const QString& component, const QString& group);

    // add a component data to show / change its value
    void addData(const QString& component, const QString& group, sofa::core::objectmodel::BaseData *data);

    // clear unattached components, theirs groups and data
    void clear();

    // clear everything
    void clearAll();

protected slots:
    void updateListViewItem();
    void updateDirtyWidget();

protected:
    // find a component by name
    QTreeWidgetItem* findComponent(const QString& component) const;

    // find a group by name
    QTreeWidgetItem* findGroup(const QString& component, const QString& group) const;

    /*void dragEnterEvent(QDragEnterEvent *event);
    //void dragMoveEvent(QDragMoveEvent *event);
    //void dragLeaveEvent(QDragLeaveEvent *event);
    void dropEvent(QDropEvent *event);
    Qt::DropActions supportedDropActions() const;*/

private:

    // remember the Base Object and its item in the scene graph list view for each component registered in this property view
    std::map<QTreeWidgetItem*, std::pair<core::objectmodel::Base*, Q3ListViewItem*> >		objects;

    static const QString																	defaultGroup;
    QIcon																					pinIcon;

};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_QT_QDISPLAYPROPERTYWIDGET_H
