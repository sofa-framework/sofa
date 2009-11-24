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

#ifndef SOFA_GUI_QT_DATAWIDGET_H
#define SOFA_GUI_QT_DATAWIDGET_H


#include "SofaGUIQt.h"
#include "ModifyObject.h"
#include <sofa/helper/Factory.h>
#include <qglobal.h>
#ifdef SOFA_QT4
#include <QDialog>
#include <QLineEdit>
#include <Q3Table>
#else
#include <qdialog.h>
#include <qlineedit.h>
#include <qtable.h>
#endif // SOFA_QT4


namespace sofa
{

namespace core
{
namespace objectmodel
{
class Base;
class BaseData;
}
}
namespace gui
{
namespace qt
{

class ModifyObject;
class DataWidget
{
protected:
    //core::objectmodel::Base* node;
    core::objectmodel::BaseData* baseData;
    QWidget* parent;
    ModifyObject* dialog;
    std::string name;
    bool readOnly;
public:
    typedef core::objectmodel::BaseData MyData;

    DataWidget(MyData* d) : baseData(d), dialog(NULL), readOnly(false) {}
    virtual ~DataWidget() {}
    void setDialog(ModifyObject* d) { dialog = d; }
    void setReadOnly(bool b) { readOnly = b; }
    void setParent(QWidget *p) { parent=p; }
    void setName(std::string n) { name = n;};
    virtual bool createWidgets(QWidget* parent) = 0;
    virtual void readFromData() = 0;
    virtual void writeToData() {}
    virtual bool processChange(const QObject* /*sender*/) { return false; }
    virtual bool isModified() { return false; }
    std::string getName() { return name;};
    virtual void update()
    {
        readFromData();
    }
    virtual void updateVisibility()
    {
        parent->setShown(baseData->isDisplayed());
    };

    virtual unsigned int sizeWidget() {return 1;}
    //
    // Factory related code
    //

    struct CreatorArgument
    {
        std::string name;
        core::objectmodel::BaseData* data;
        ModifyObject* dialog;
        QWidget* parent;
        bool readOnly;
    };

    template<class T>
    static void create(T*& instance, const CreatorArgument& arg)
    {
        typename T::MyData* data = dynamic_cast<typename T::MyData*>(arg.data);
        if (!data) return;
        instance = new T(data);
        instance->setDialog(arg.dialog);
        instance->setReadOnly(arg.readOnly);
        instance->setParent(arg.parent);
        instance->setName(arg.name);
        if (!instance->createWidgets(arg.parent))
        {
            delete instance;
            instance = NULL;
        }
        else instance->updateVisibility();
    }
};

typedef sofa::helper::Factory<std::string, DataWidget, DataWidget::CreatorArgument> DataWidgetFactory;


class DefaultDataWidget : public DataWidget
{
protected:
    typedef QLineEdit Widget;
    MyData* data;
    Widget* w;
    int counter;
    bool modified;
public:
    DefaultDataWidget(MyData* d) : DataWidget(d), data(d), w(NULL), counter(-1), modified(false) {}
    virtual bool createWidgets(QWidget* parent);
    virtual void readFromData()
    {
        std::string s = data->getValueString();
        w->setText(QString(s.c_str()));
        modified = false;
        counter = data->getCounter();
    }
    virtual bool isModified() { return modified; }
    virtual void writeToData()
    {
        if (!modified) return;
        std::string s = w->text().ascii();
        data->read(s);
        counter = data->getCounter();
    }
    virtual bool processChange(const QObject* sender)
    {
        if (sender == w)
        {
            modified = true;
            return true;
        }
        else return false;
    }
    virtual void update()
    {
        if (counter != data->getCounter())
            readFromData();
    }
};
class QTableUpdater : virtual public Q3Table
{
    Q_OBJECT
public:
    QTableUpdater ( int numRows, int numCols, QWidget * parent = 0, const char * name = 0 ):
#ifdef SOFA_QT4
        Q3Table(numRows, numCols, parent, name)
#else
        QTable(numRows, numCols, parent, name)
#endif
    {};

public slots:
    void setDisplayed(bool b) {this->setShown(b);}
public slots:

};

#if defined WIN32 && !defined(SOFA_GUI_QT_DATAWIDGET_CPP)
extern template class SOFA_SOFAGUIQT_API helper::Factory<std::string, DataWidget, DataWidget::CreatorArgument>;
#endif



} // qt
} // gui
} // sofa

#endif // SOFA_GUI_QT_DATAWIDGET_H
