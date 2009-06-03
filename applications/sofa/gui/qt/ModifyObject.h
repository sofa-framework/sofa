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
#ifndef SOFA_GUI_QT_MODIFYOBJECT_H
#define SOFA_GUI_QT_MODIFYOBJECT_H


#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/misc/Monitor.h>
#include <sofa/helper/Factory.h>

#include <qglobal.h>
#ifdef SOFA_QT4
#include <QDialog>
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
#else
#include <qdialog.h>
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
#endif

#include "WFloatLineEdit.h"

#include <qwt_plot.h>

#include <qwt_plot_curve.h>
#include <sofa/gui/qt/DisplayFlagWidget.h>

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

using namespace sofa::defaulttype;
using sofa::simulation::Node;
using sofa::helper::fixed_array;

#ifndef SOFA_QT4
typedef QListView   Q3ListView;
typedef QListViewItem Q3ListViewItem;
typedef QCheckListItem   Q3CheckListItem;
typedef QTable    Q3Table;
typedef QGroupBox Q3GroupBox;
typedef QTextEdit   Q3TextEdit;
typedef QGrid       Q3Grid;
#endif

class DataWidget;

class ModifyObject : public QDialog
{
    Q_OBJECT
public:

    ModifyObject() {};
    ModifyObject( void *Id, core::objectmodel::Base* node, Q3ListViewItem* item_clicked, QWidget* parent, const char* name= 0, bool  modal= FALSE, Qt::WFlags f= 0 );
    ~ModifyObject()
    {
        delete buttonUpdate;
    }

    void setNode(core::objectmodel::Base* node, Q3ListViewItem* item_clicked=NULL); //create all the widgets of the dialog window

    bool hideData(core::objectmodel::BaseData* data) { return (!data->isDisplayed()) && HIDE_FLAG;};
    void readOnlyData(Q3Table *widget, core::objectmodel::BaseData* data);
    void readOnlyData(QWidget *widget, core::objectmodel::BaseData* data);


public slots:
    void updateValues();              //update the node with the values of the field
    void updateTextEdit();            //update the text fields due to unknown data field
    void updateConsole();             //update the console log of warnings and outputs
    void updateTables();              //update the tables of value at each step of the simulation
    void saveTables();                //Save in datafield the content of a QTable
    void saveTextEdit();                //Save in datafield the content of a QTextEdit
    void changeValue();               //each time a field is modified
    void changeVisualValue();               //each time a field of the Visualization tab is modified
    void closeNow () {emit(reject());} //called from outside to close the current widget
    void reject   () {                 emit(dialogClosed(Id)); deleteLater(); QDialog::reject();} //When closing a window, inform the parent.
    void accept   () { updateValues(); emit(dialogClosed(Id)); deleteLater(); QDialog::accept();} //if closing by using Ok button, update the values
    void resizeTable(int);
    void clearWarnings() {node->sendl.clearWarnings(); logWarningEdit->clear();}
    void clearOutputs() {node->sendl.clearOutputs(); logOutputEdit->clear();}
signals:
    void objectUpdated();              //update done
    void dialogClosed(void *);            //the current window has been closed: we give the Id of the current window
    void transformObject(Node * current_node, double translationX, double translationY, double translationZ,
            double rotationX, double rotationY, double rotationZ,
            double scale);

protected:


    const core::objectmodel::BaseData* getData(const QObject *object);
    virtual void closeEvent ( QCloseEvent * ) {emit(reject());}
    void updateContext( Node *node );

    void createGraphMass(QTabWidget *);
    void updateHistory();
    void updateEnergy();

    bool createTable(core::objectmodel::BaseData* field, Q3GroupBox *box=NULL, Q3Table* vectorTable=NULL, Q3Table* vectorTable2=NULL, Q3Table* vectorTable3=NULL );
    void storeTable(std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table);

    //*********************************************************

    //Monitor Special class
    template< class T>
    bool createMonitorQtTable(Data<typename sofa::component::misc::Monitor<T>::MonitorData >* ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2, Q3Table* vectorTable3 );
    template< class T>
    void storeMonitorQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< typename sofa::component::misc::Monitor<T>::MonitorData >* ff );
    //*********************************************************

    Q3Table* addResizableTable(Q3GroupBox *box,int size, int column=1);

    QWidget *parent;
    QTabWidget *dialogTab;
    core::objectmodel::Base* node;
    Q3ListViewItem * item;
    QPushButton *buttonUpdate;

    std::vector<std::pair< core::objectmodel::BaseData*,  QObject*> >  objectGUI;  //vector of all the Qt Object added in the window

    std::set< const core::objectmodel::BaseData* >                     setUpdates; //set of objects that have been modified
    std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >    list_Table;
    std::list< std::pair< Q3TextEdit*, core::objectmodel::BaseData*> > list_TextEdit;
    std::map< core::objectmodel::BaseData*, int >                      dataIndexTab;
    std::map< QSpinBox*, Q3Table* >                                    resizeMap;
    std::set< Q3Table* >                                               setResize;
    WFloatLineEdit* transformation[7]; //Data added to manage transformation of a whole node

    QWidget *warningTab;
    Q3TextEdit *logWarningEdit;
    QWidget *outputTab;
    Q3TextEdit *logOutputEdit;

    typedef std::map<core::objectmodel::BaseData*, DataWidget*> DataWidgetMap;
    DataWidgetMap dataWidgets;

    void *Id;
    bool visualContentModified;

    //Visual Flags
    DisplayFlagWidget *displayFlag;

    std::vector< double > history;
    std::vector< double > energy_history[3];
    QwtPlot *graphEnergy;
    QwtPlotCurve *energy_curve[3];
    unsigned int counterWidget;

    bool HIDE_FLAG; //if we allow to hide Datas
    bool READONLY_FLAG; //if we allow  ReadOnly Datas
    bool EMPTY_FLAG;//if we allow empty datas
    bool RESIZABLE_FLAG;
    bool REINIT_FLAG;
};

class DataWidget
{
protected:
    core::objectmodel::Base* node;
    core::objectmodel::BaseData* baseData;
    QWidget* parent;
    ModifyObject* dialog;
    std::string name;
    bool readOnly;
public:
    typedef core::objectmodel::BaseData MyData;

    DataWidget(MyData* d) : node(NULL), baseData(d), dialog(NULL), readOnly(false) {}
    virtual ~DataWidget() {}
    void setNode(core::objectmodel::Base* n) { node = n; }
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
        core::objectmodel::Base* node;
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
        instance->setNode(arg.node);
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
    virtual bool createWidgets(QWidget* parent)
    {
        w = new QLineEdit(parent);
        if (w == NULL) return false;
        std::string s = data->getValueString();
        w->setText(QString(s.c_str()));
        counter = data->getCounter();
        if (this->readOnly)
            w->setEnabled(false);
        else
            dialog->connect(w, SIGNAL( textChanged(const QString&) ), dialog, SLOT( changeValue() ));
        return true;
    }
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


} // namespace qt

} // namespace gui

} // namespace sofa

#endif

