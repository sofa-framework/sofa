/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef MODIFYOBJECT_H
#define MODIFYOBJECT_H


#include <sofa/core/objectmodel/BaseObject.h>


#include <sofa/component/topology/PointData.h>
#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/component/forcefield/JointSpringForceField.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/misc/Monitor.h>

#include <qglobal.h>
#ifdef SOFA_QT4
#include <QDialog>
#include <Q3ListViewItem>
#include <Q3ListView>
#include <Q3Table>
#include <Q3GroupBox>
#include <Q3TextEdit>
#include <QPushButton>
#include <QTabWidget>
#include <QSpinBox>
#include <Q3CheckListItem>
#else
#include <qdialog.h>
#include <qlistview.h>
#include <qtable.h>
#include <qgroupbox.h>
#include <qtextedit.h>
#include <qtabwidget.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#endif

#include "WFloatLineEdit.h"

#include <qwt_plot.h>

#include <qwt_plot_curve.h>

class QPushButton;

namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::helper::Quater;
using namespace sofa::defaulttype;
using sofa::simulation::Node;

#ifndef SOFA_QT4
typedef QListView   Q3ListView;
typedef QListViewItem Q3ListViewItem;
typedef QCheckListItem   Q3CheckListItem;
typedef QTable    Q3Table;
typedef QGroupBox Q3GroupBox;
typedef QTextEdit   Q3TextEdit;
#endif
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


public slots:
    void updateValues();              //update the node with the values of the field
    void updateTextEdit();            //update the text fields due to unknown data field
    void updateTables();              //update the tables of value at each step of the simulation
    void saveTables();                //Save in datafield the content of a QTalbe
    void saveTextEdit();                //Save in datafield the content of a QTextEdit
    void changeValue();               //each time a field is modified
    void changeVisualValue();               //each time a field of the Visualization tab is modified
    void changeNumberPoint();         //used to dynamically add points in an object of type pointSubset
    void closeNow () {emit(reject());} //called from outside to close the current widget
    void reject   () {                 emit(dialogClosed(Id)); deleteLater(); QDialog::reject();} //When closing a window, inform the parent.
    void accept   () { updateValues(); emit(dialogClosed(Id)); deleteLater(); QDialog::accept();} //if closing by using Ok button, update the values
    void resizeTable(int);
#ifdef SOFA_QT4
    void visualFlagChanged(Q3ListViewItem *item);
#else
    void visualFlagChanged(QListViewItem *item);
#endif
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
// 	void storeTable(Q3Table* table, core::objectmodel::BaseData* field);

    void createVector( core::objectmodel::BaseData* object,const Quater<double> &value, Q3GroupBox *box); //will be created as a Vec<4,double>
    void createVector( core::objectmodel::BaseData* object,const Quater<float>  &value, Q3GroupBox *box); //will be created as a Vec<4,float>

    //*********************************************************
    template< int N, class T>
    void createVector( core::objectmodel::BaseData* object,const Vec<N,T> &value, Q3GroupBox *box);
    template< int N, class T>
    void storeVector( unsigned int &index, Data< Vec<N,T> > *ff);
    template<class T>
    void storeVector( unsigned int &index, Data< Quater<T> > *ff);
    template< int N, class T>
    void storeVector( unsigned int &index, DataPtr< Vec<N,T> > *ff);
    template<class T>
    void storeVector( unsigned int &index, DataPtr< Quater<T> > *ff);
    template< int N, class T>
    void storeVector( unsigned int &index, Vec<N,T> *ff);
    template<class T>
    void storeVector(unsigned int &index, Quater<T> *ff);
    //*********************************************************
    template< class T>
    bool createQtTable(Data< sofa::helper::vector< T > > *ff, Q3GroupBox *box, Q3Table* vectorTable );
    template<class T>
    void storeQtTable(std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< T > >* ff );

    void storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< std::string > >* ff );

    //*********************************************************
    template< class T>
    bool createQtTable(DataPtr< sofa::helper::vector< T > > *ff, Q3GroupBox *box, Q3Table* vectorTable );
    template< class T>
    void storeQtTable(std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< T > >* ff );
    //*********************************************************
    template< int N, class T>
    bool createQtTable(Data< sofa::helper::vector< Vec<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable );
    template< int N, class T>
    void storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< Vec<N,T> > >* ff );
    //*********************************************************
    template< int N, class T>
    bool createQtTable(DataPtr< sofa::helper::vector< Vec<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable );
    template< int N, class T>
    void storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< Vec<N,T> > >* ff );
    //*********************************************************
    template< class T>
    bool createQtTable(Data< sofa::component::topology::PointData< T > > *ff, Q3GroupBox *box, Q3Table* vectorTable );
    template< class T>
    void storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::component::topology::PointData< T > >* ff );
    //*********************************************************

    //Monitor Special class
    template< class T>
    bool createMonitorQtTable(Data<typename sofa::component::misc::Monitor<T>::MonitorData >* ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2, Q3Table* vectorTable3 );
    template< class T>
    void storeMonitorQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< typename sofa::component::misc::Monitor<T>::MonitorData >* ff );
    //*********************************************************


    //Rigid Special Cases
    template< int N, class T>
    bool createQtTable(Data< sofa::helper::vector< RigidCoord<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2  );
    template<class T>
    void storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidCoord<3,T> > >* ff );
    template<class T>
    void storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidCoord<2,T> > >* ff );

    //*********************************************************
    template< int N, class T>
    bool createQtTable(Data< sofa::helper::vector< RigidDeriv<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2  );
    template<class T>
    void storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidDeriv<3,T> > >* ff );
    template<class T>
    void storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidDeriv<2,T> > >* ff );

    //*********************************************************
    template< int N, class T>
    bool createQtTable(DataPtr< sofa::helper::vector< RigidCoord<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2  );
    template<class T>
    void storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidCoord<3,T> > >* ff );
    template<class T>
    void storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidCoord<2,T> > >* ff );

    //*********************************************************
    template< int N, class T>
    bool createQtTable(DataPtr< sofa::helper::vector< RigidDeriv<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2  );
    template<class T>
    void storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidDeriv<3,T> > >* ff );
    template<class T>
    void storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidDeriv<2,T> > >* ff );
    //*********************************************************

    template< int N, class T>
    bool createQtSpringTable(Data<  sofa::helper::vector< typename sofa::component::forcefield::SpringForceField< StdVectorTypes< Vec<N,T>, Vec<N,T>, T> >::Spring > >  *ff, Q3GroupBox *box, Q3Table* vectorTable );
    template< int N, class T>
    void storeQtSpringTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data<  sofa::helper::vector< typename sofa::component::forcefield::SpringForceField< StdVectorTypes< Vec<N,T>, Vec<N,T>, T> >::Spring > >  *ff);

    //*********************************************************
    template< class T>
    bool createQtRigidSpringTable(Data< sofa::helper::vector< typename sofa::component::forcefield::JointSpringForceField< StdRigidTypes< 3,T > >::Spring > >  *ff, Q3GroupBox *box, Q3Table* vectorTable );
    template< class T>
    void storeQtRigidSpringTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< typename sofa::component::forcefield::JointSpringForceField< StdRigidTypes<3,T> >::Spring > >  *ff );



    Q3Table* addResizableTable(Q3GroupBox *box,int size, int column=1);

    QWidget *parent;
    QTabWidget *dialogTab;
    core::objectmodel::Base* node;
    Q3ListViewItem * item;
    QPushButton *buttonUpdate;
    std::vector<std::pair< core::objectmodel::BaseData*,  QObject*> >  objectGUI;  //vector of all the Qt Object added in the window
    std::set< const core::objectmodel::BaseData* >                     setUpdates; //set of objects that have ben modified
    std::list< std::list< QObject* > * >                               list_PointSubset;
    std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >    list_Table;
    std::list< std::pair< Q3TextEdit*, core::objectmodel::BaseData*> > list_TextEdit;
    std::map< core::objectmodel::BaseData*, int >                      dataIndexTab;
    std::map< QSpinBox*, Q3Table* >                                    resizeMap;
    std::set< Q3Table* >                                               setResize;
    WFloatLineEdit* transformation[7]; //Data added to manage transformation of a whole node

    void *Id;
    bool visualContentModified;

    //Visual Flags
    Q3CheckListItem* itemShowFlag[10];

    std::vector< double > history;
    std::vector< double > energy_history[3];
    QwtPlot *graphEnergy;
    QwtPlotCurve *energy_curve[3];
    unsigned int counterWidget;

    bool HIDE_FLAG; //if we allow to hide Datas
    bool EMPTY_FLAG;//if we allow empty datas
    bool RESIZABLE_FLAG;
    bool REINIT_FLAG;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif

