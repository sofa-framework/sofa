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
#include <sofa/gui/qt/ModifyObject.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/simulation/common/Node.h>
#include <SofaValidation/Monitor.h>
#include <sofa/gui/qt/QTransformationWidget.h>
#ifdef SOFA_HAVE_QWT
#include <sofa/gui/qt/QEnergyStatWidget.h>
#endif
#include <sofa/gui/qt/WDoubleLineEdit.h>

#include <QTreeWidget>
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


#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>

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

// QDisplayPropertyWidget describe a widget where you can view and edit every properties of Sofa components
class SOFA_SOFAGUIQT_API QDisplayPropertyWidget : public QTreeWidget
{
	Q_OBJECT

    friend class GraphHistoryManager;
    friend class LinkComponent;

public:
    // constructor / destructor
    QDisplayPropertyWidget(const ModifyObjectFlags& modifyFlags, QWidget* parent=0);
    ~QDisplayPropertyWidget();

    // add a component in the tree in order to show / change its data and compare them with other component data
    void addComponent(const QString& component, core::objectmodel::Base* base, Q3ListViewItem* listItem, bool clear = true);

    // add a data / link group
    void addGroup(const QString& component, const QString& group);

    // add a component data to show / change its value
    void addData(const QString& component, const QString& group, sofa::core::objectmodel::BaseData *data);

	// add a component link to show / change its value
    void addLink(const QString& component, const QString& group, sofa::core::objectmodel::BaseLink *link);

	// set a component description
    void setDescription(const QString& component, const QString& group, sofa::core::objectmodel::Base *base);

	// set a component console output
    void setConsoleOutput(const QString& component, const QString& group, sofa::core::objectmodel::Base *base);

protected:
	// add a description item
	void addDescriptionItem(QTreeWidgetItem *groupItem, const QString& name, const QString& description);

public:
    // clear non-pinned components, theirs groups, data and links
    void clear();

    // clear everything, even pinned components
    void clearAll();

	// name of the default property group
	static QString DefaultDataGroup()				{return "Property";}
	static QString DefaultLinkGroup()				{return "Link";}
	static QString DefaultInfoGroup()				{return "Info";}
	static QString DefaultLogGroup()				{return "Log";}

protected slots:
	// call this slot when you rename a component of the scene graph to rename its corresponding list view item
    void updateListViewItem();

	// retrieve the component stored as a property in the signal emitter and clear its output messages
	void clearComponentOutput();

	// retrieve the component stored as a property in the signal emitter and clear its warning messages
	void clearComponentWarning();

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
    QIcon																					pinIcon;
	ModifyObjectFlags																		modifyObjectFlags;

};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_QT_QDISPLAYPROPERTYWIDGET_H
