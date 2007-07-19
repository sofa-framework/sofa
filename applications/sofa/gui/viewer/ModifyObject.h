/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef MODIFYOBJECT_H
#define MODIFYOBJECT_H


#include <sofa/core/objectmodel/BaseObject.h>
#include "RealGUI.h"

#ifdef QT_MODULE_QT3SUPPORT
#include <QDialog>
#include <Q3ListViewItem>
#include <Q3ListView>
#else
#include <qdialog.h>
#include <qlistview.h>
#endif


class QPushButton;

namespace sofa
{

namespace gui
{

namespace guiviewer
{

#ifndef QT_MODULE_QT3SUPPORT
typedef QListViewItem Q3ListViewItem;
#endif
class ModifyObject : public QDialog
{
    Q_OBJECT
public:

    ModifyObject( int Id, core::objectmodel::Base* node, Q3ListViewItem* item_clicked, QWidget* parent, const char* name= 0, bool  modal= FALSE, Qt::WFlags f= 0 );
    ~ModifyObject()
    {
        delete buttonUpdate;
        delete list_Object;
        delete list_PointSubset;
    }

    void setNode(core::objectmodel::Base* node, Q3ListViewItem* item_clicked=NULL); //create all the widgets of the dialog window

public slots:
    void updateValues();             //update the node with the values of the field
    void changeValue();              //each time a field is modified
    void changeNumberPoint();        //used to dynamically add points in an object of type pointSubset
    void closeNow () {emit(reject());} //called from outside to close the current widget
    void reject   () { emit(dialogClosed(Id)); QDialog::reject();}                //When closing a window, inform the parent.
    void accept   () { updateValues(); emit(dialogClosed(Id)); QDialog::accept();} //if closing by using Ok button, update the values

signals:
    void objectUpdated();              //update done
    void dialogClosed(int);            //the current window has been closed: we give the Id of the current window
    void transformObject(GNode * current_node, double translationX, double translationY, double translationZ, double scale);


protected:
    virtual void closeEvent ( QCloseEvent * ) {emit(reject());}
    void updateContext( GNode *node );

    QWidget *parent;
    core::objectmodel::Base* node;
    Q3ListViewItem * item;
    QPushButton *buttonUpdate;
    std::list< QObject* >                 *list_Object;
    std::list< std::list< QObject* > * >  *list_PointSubset;

    int Id;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif
