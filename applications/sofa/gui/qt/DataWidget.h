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
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/Factory.h>
#ifdef SOFA_QT4
#include <QDialog>
#include <QLineEdit>
#include <Q3Table>
#include <QPushButton>
#include <QSpinBox>
#else
#include <qspinbox.h>
#include <qdialog.h>
#include <qlineedit.h>
#include <qtable.h>
#include <qpushbutton.h>
#endif // SOFA_QT4

#ifndef SOFA_QT4
typedef QTable    Q3Table;
#endif


namespace sofa
{

namespace core
{
namespace objectmodel
{
class Base;
}
}
namespace gui
{
namespace qt
{

class ModifyObject;
class DataWidget : public QWidget
{
    Q_OBJECT
public slots:



    void updateData()
    {
        if(modified)
        {
            std::string previousName = baseData->getOwner()->getName();
            writeToData();
            updateVisibility();
            if(baseData->getOwner()->getName() != previousName)
            {
                emit dataParentNameChanged();
            }
        }
        modified = false;
        emit requestChange(modified);
    }
    void updateWidget()
    {
        if(!modified)
        {
            readFromData();
        }
    }
    void setDisplayed(bool b)
    {
        if(b)
        {
            readFromData();
        }
    }
    virtual void update()
    {
        readFromData();
    }

signals:
    void requestChange(bool );
    void dataParentNameChanged();

protected:
    core::objectmodel::BaseData* baseData;
    QWidget* parent;
    std::string name;
    bool readOnly;
    bool modified;
protected slots:
    void setModified()
    {
        modified = true; emit requestChange(modified);
    }

public:
    typedef core::objectmodel::BaseData MyData;

    DataWidget(MyData* d) : baseData(d),  readOnly(false), modified(false) {}
    virtual ~DataWidget() {}
    void setReadOnly(bool b) { readOnly = b; }
    void setParent(QWidget *p) { parent=p; }
    void setName(std::string n) { name = n;};
    core::objectmodel::BaseData* getBaseData() const { return baseData; }
    virtual bool createWidgets(QWidget* parent) = 0;
    virtual void readFromData() {};
    virtual void writeToData() = 0;
    virtual bool isModified() { return false; }
    std::string getName() { return name;};


    virtual void updateVisibility()
    {
        parent->setShown(baseData->isDisplayed());
    };

    virtual unsigned int sizeWidget() {return 1;}
    virtual unsigned int numColumnWidget() {return 2;}
    //
    // Factory related code
    //

    struct CreatorArgument
    {
        std::string name;
        core::objectmodel::BaseData* data;
        QWidget* parent;
        bool readOnly;
    };

    template<class T>
    static void create(T*& instance, const CreatorArgument& arg)
    {
        typename T::MyData* data = dynamic_cast<typename T::MyData*>(arg.data);
        if (!data) return;
        instance = new T(data);
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
    void resizeTableV( int number )
    {
        QSpinBox *spinBox = (QSpinBox *) sender();
        QString header;
        if( spinBox == NULL)
        {
            return;
        }
        if (number != numRows())
        {
            setNumRows(number);

        }
    }

    void resizeTableH( int number )
    {
        QSpinBox *spinBox = (QSpinBox *) sender();
        QString header;
        if( spinBox == NULL)
        {
            return;
        }
        if (number != numCols())
        {
            setNumCols(number);

        }
    }

};

//     extern template class SOFA_SOFAGUIQT_API helper::Factory<std::string, DataWidget, DataWidget::CreatorArgument>;

class QPushButtonUpdater: public QPushButton
{
    Q_OBJECT
public:

    QPushButtonUpdater( const QString & text, QWidget * parent = 0 ): QPushButton(text,parent) {};

public slots:
    void setDisplayed(bool b);
};

//Widget used to display the name of a Data and if needed the link to another Data
class QDisplayDataInfoWidget: public QWidget
{
    Q_OBJECT
public:
    QDisplayDataInfoWidget(QWidget* parent, const std::string& helper, core::objectmodel::BaseData* d, bool modifiable);
public slots:
    void linkModification();
    void linkEdited();
    unsigned int getNumLines() const { return numLines_;}
protected:
    void formatHelperString(const std::string& helper, std::string& final_text);
    static unsigned int numLines(const std::string& str);
    core::objectmodel::BaseData* data;
    unsigned int numLines_;
    QLineEdit *linkpath_edit;
};

} // qt
} // gui
} // sofa

#endif // SOFA_GUI_QT_DATAWIDGET_H

