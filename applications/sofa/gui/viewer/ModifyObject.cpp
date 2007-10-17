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

#include "ModifyObject.h"
#include <iostream>

#ifdef QT_MODULE_QT3SUPPORT
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTabWidget>
#include <QGridLayout>
#include <Q3Grid>
#else
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qspinbox.h>
#include <qcheckbox.h>
#include <qlayout.h>
#include <qtabwidget.h>
#include <qgrid.h>
#endif


#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/component/forcefield/JointSpringForceField.h>
#include <sofa/simulation/tree/InitVisitor.h>
#include "WFloatLineEdit.h"

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif

namespace sofa
{

namespace gui
{

namespace guiviewer
{

using namespace  sofa::defaulttype;
using sofa::component::topology::PointSubset;
using sofa::core::objectmodel::FieldBase;

#ifndef QT_MODULE_QT3SUPPORT
typedef QGrid     Q3Grid;
#endif


ModifyObject::ModifyObject(int Id_, core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked,  QWidget* parent_, const char* name, bool, Qt::WFlags f ): parent(parent_), node(NULL), Id(Id_)
{
    //Constructor of the QDialog
    QDialog( parent_, name, f);
    //Initialization of the Widget
    setNode(node_clicked, item_clicked);
    connect ( this, SIGNAL( objectUpdated() ), parent_, SLOT( redraw() ));
    connect ( this, SIGNAL( dialogClosed(int) ) , parent_, SLOT( modifyUnlock(int)));
    connect ( this, SIGNAL( transformObject(GNode *, double, double, double, double)), parent, SLOT(transformObject(GNode *, double, double, double, double)));
}


//Set the default file
void ModifyObject::setNode(core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked)
{
    node = node_clicked;
    item = item_clicked;

    //Layout to organize the whole window
    QVBoxLayout *generalLayout = new QVBoxLayout(this, 0, 1, "generalLayout");

    //Tabulation widget
    QTabWidget *dialogTab = new QTabWidget(this);
    generalLayout->addWidget(dialogTab);

    //Each tab
    QWidget *tab1 = new QWidget();
    dialogTab->addTab(tab1, QString("Properties"));

    bool visualTab = false;
    QWidget *tab2 = NULL; //tab for visualization info: only created if needed ( boolean visualTab gives this answer ).


    QVBoxLayout *tabPropertiesLayout = new QVBoxLayout( tab1, 0, 1, "tabPropertiesLayout");
    QVBoxLayout *tabVisualizationLayout = NULL;

    // displayWidget

    if (node)
    {

        //All the pointers to the QObjects will be kept in memory in list_Object

        const std::vector< std::pair<std::string, FieldBase*> >& fields = node->getFields();

        int i=0;

        for( std::vector< std::pair<std::string, FieldBase*> >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {

            //For each element, we create a layout
            std::ostringstream oss;
            oss << "itemLayout_" << i;
            Q3GroupBox *box = NULL;;
            // The label
            //QLabel *label = new QLabel(QString((*it).first.c_str()), box,0);
            //label->setGeometry( 10, i*25+5, 200, 20 );

            const std::string& fieldname = (*it).second->getValueTypeString();
            if( DataField<bool> * ff = dynamic_cast< DataField<bool> * >( (*it).second )  )
            {
                //Remove from the dialog window everything about showing collision models, visual models...
                //Don't have any effect if the scene is animated: the root will erase the value.
                std::string name((*it).first);
                name.resize(4);
                if (name == "show")
                {
                    if (!visualTab)
                    {
                        visualTab = true;
                        tab2 = new QWidget();
                        dialogTab->addTab(tab2, QString("Visualization"));
                        tabVisualizationLayout = new QVBoxLayout( tab2, 0, 1, "tabVisualizationLayout");
                    }

                    std::string box_name(oss.str());
                    box = new Q3GroupBox(tab2, QString(box_name.c_str()));
                    tabVisualizationLayout->addWidget( box );
                }
                else
                {
                    std::string box_name(oss.str());
                    box = new Q3GroupBox(tab1, QString(box_name.c_str()));
                    tabPropertiesLayout->addWidget( box );
                }

                box->setColumns(4);
                box->setTitle(QString((*it).first.c_str()));

                if( strcmp((*it).second->help,"TODO") )new QLabel((*it).second->help, box);

                // the bool line edit
                QCheckBox* checkBox = new QCheckBox(box);
                list_Object.push_back( (QObject *) checkBox);

                //checkBox->setGeometry( 205, i*25+5, 170, 20 );

                checkBox->setChecked(ff->getValue());
                connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );


                continue;
            }
            else
            {
                std::string box_name(oss.str());
                box = new Q3GroupBox(tab1, QString(box_name.c_str()));
                box->setColumns(4);
                box->setTitle(QString((*it).first.c_str()));

                if( strcmp((*it).second->help,"TODO") )new QLabel((*it).second->help, box);
                //********************************************************************************************************//
                //int
                if( DataField<int> * ff = dynamic_cast< DataField<int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)INT_MIN,(int)INT_MAX,1,box);
                    list_Object.push_back( (QObject *) spinBox);

                    spinBox->setValue(ff->getValue());
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //unsigned int
                else if( DataField<unsigned int> * ff = dynamic_cast< DataField<unsigned int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)0,(int)INT_MAX,1,box);
                    list_Object.push_back( (QObject *) spinBox);

                    spinBox->setValue(ff->getValue());
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //float
                else if( DataField<float> * ff = dynamic_cast< DataField<float> * >( (*it).second )  )
                {
                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    list_Object.push_back( (QObject *) editSFFloat);

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue());
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //double
                else if(DataField<double> * ff = dynamic_cast< DataField<double> * >( (*it).second )  )
                {

                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    list_Object.push_back( (QObject *) editSFFloat);

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue());
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //string
                else if( DataField<std::string> * ff = dynamic_cast< DataField<std::string> * >( (*it).second )  )
                {
// 			if (ff->getValue().empty()) continue;
                    QLineEdit* lineEdit = new QLineEdit(box);
                    list_Object.push_back( (QObject *) lineEdit);

                    lineEdit->setText(QString(ff->getValue().c_str()));
                    connect( lineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                }

                //********************************************************************************************************//
                //Vec6f, Vec6d
                else if( dynamic_cast< DataField<Vec6f> * > ( (*it).second ) ||
                        dynamic_cast< DataField<Vec6d> * > ( (*it).second ) ||
                        dynamic_cast< DataField<Vec<6,int> > *> ( (*it).second ) ||
                        dynamic_cast< DataField<Vec<6,unsigned int> >* > ( (*it).second ) )
                {

                    if( DataField<Vec6f> * ff = dynamic_cast< DataField<Vec6f> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec6d> * ff = dynamic_cast< DataField<Vec6d> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<6,int> > * ff = dynamic_cast< DataField<Vec<6,int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<6,unsigned int> > * ff = dynamic_cast< DataField<Vec<6,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                }
                //********************************************************************************************************//
                //Vec4f,Vec4d
                else if( dynamic_cast< DataField<Vec4f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec4d> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec<4,int> > *>( (*it).second ) ||
                        dynamic_cast< DataField<Vec<4,unsigned int> >* >  ( (*it).second )  )
                {


                    if( DataField<Vec4f> * ff = dynamic_cast< DataField<Vec4f> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec4d> * ff = dynamic_cast< DataField<Vec4d> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<4,int> > * ff = dynamic_cast< DataField<Vec<4,int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<4,unsigned int> > * ff = dynamic_cast< DataField<Vec<4,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                }
                //********************************************************************************************************//
                //Vec3f,Vec3d
                else if( dynamic_cast< DataField<Vec3f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec3d> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec<3,int> > *>( (*it).second ) ||
                        dynamic_cast< DataField<Vec<3,unsigned int> >* >  ( (*it).second ))
                {

                    if( DataField<Vec3f> * ff = dynamic_cast< DataField<Vec3f> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec3d> * ff = dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<3,int> > * ff = dynamic_cast< DataField<Vec<3,int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<3,unsigned int> > * ff = dynamic_cast< DataField<Vec<3,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }

                }
                //********************************************************************************************************//
                //Vec2f,Vec2d
                else if( dynamic_cast< DataField<Vec2f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec2d> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec<2,int> > * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec<2,unsigned int> > * >  ( (*it).second ))
                {


                    if( DataField<Vec2f> * ff = dynamic_cast< DataField<Vec2f> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec2d> * ff = dynamic_cast< DataField<Vec2d> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<2,int> > * ff = dynamic_cast< DataField<Vec<2,int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<2,unsigned int> > * ff = dynamic_cast< DataField<Vec<2,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }

                }
                //********************************************************************************************************//
                //Vec1f,Vec1d
                else if( dynamic_cast< DataField<Vec1f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec1d> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec<1,int> > * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec<1,unsigned int> > *  > ( (*it).second ))
                {

                    if( DataField<Vec1f> * ff = dynamic_cast< DataField<Vec1f> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec1d> * ff = dynamic_cast< DataField<Vec1d> * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<1,int> > * ff = dynamic_cast< DataField<Vec<1,int> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                    else if(DataField<Vec<1,unsigned int> > * ff = dynamic_cast< DataField<Vec<1,unsigned int> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue(), box);
                    }
                }
                //********************************************************************************************************//
                //PointSubset
                else if( DataField<PointSubset> * ff = dynamic_cast< DataField<PointSubset> * >( (*it).second )  )
                {

                    //Get the PointSubset from the DataField
                    PointSubset p= ff->getValue();
                    //Add the structure to the list
                    std::list< QObject *> *current_list = new std::list< QObject *>();
                    list_PointSubset.push_back(current_list);

                    //First line with only the title and the number of points
                    box->setColumns(2);
                    QSpinBox* spinBox = new QSpinBox((int)0,(int)INT_MAX,1,box);

                    current_list->push_back(spinBox);


                    Q3Table *vectorTable = new Q3Table(p.size(),1, box);
                    current_list->push_back(vectorTable);

                    vectorTable->horizontalHeader()->setLabel(0,QString("Index of the Points"));	vectorTable->setColumnStretchable(0,true);
                    connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

                    spinBox->setValue( p.size() );
                    for (unsigned int t=0; t< p.size(); t++)
                    {
                        std::ostringstream oindex;
                        oindex << "Index_" << t;

                        std::ostringstream oss;
                        oss << p[t];
                        vectorTable->setText(t,0, QString(std::string(oss.str()).c_str()));
                    }
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeNumberPoint() ) );
                }
                //********************************************************************************************************//
                //RigidMass<3, double>,RigidMass<3, float>
                else if( dynamic_cast< DataField<RigidMass<3, double> > * >( (*it).second ) ||
                        dynamic_cast< DataField<RigidMass<3, float> > * >( (*it).second ))
                {
                    box->setColumns(2);

                    WFloatLineEdit* editMass = new WFloatLineEdit( box, "editMass" );
                    list_Object.push_back( (QObject *) editMass);
                    editMass->setMinFloatValue( 0.0f );
                    editMass->setMaxFloatValue( (float)INFINITY );

                    new QLabel("Volume", box);
                    WFloatLineEdit* editVolume = new WFloatLineEdit( box, "editMass" );
                    list_Object.push_back( (QObject *) editVolume);
                    editVolume->setMinFloatValue( 0.0f );
                    editVolume->setMaxFloatValue( (float)INFINITY );


                    new QLabel("Inertia Matrix", box);
                    Q3Grid* grid= new Q3Grid(3,box);
                    WFloatLineEdit *matrix[3][3];
                    for (int row=0; row<3; row++)
                    {
                        for (int column=0; column<3; column++)
                        {

                            std::ostringstream oindex;
                            oindex << "InertiaMatrix_" << row<<column;

                            matrix[row][column] = new WFloatLineEdit( grid, oindex.str().c_str() );

                            list_Object.push_back( matrix[row][column] );
                            matrix[row][column]->setMinFloatValue( (float)-INFINITY );
                            matrix[row][column]->setMaxFloatValue( (float)INFINITY );
                        }
                    }

                    new QLabel("Inertia Mass Matrix", box);
                    Q3Grid* massgrid= new Q3Grid(3,box);
                    WFloatLineEdit *massmatrix[3][3];
                    for (int row=0; row<3; row++)
                    {
                        for (int column=0; column<3; column++)
                        {

                            std::ostringstream oindex;
                            oindex << "InertiaMassMassmatrix_" << row<<column;

                            massmatrix[row][column] = new WFloatLineEdit( massgrid, oindex.str().c_str() );

                            list_Object.push_back( massmatrix[row][column] );
                            massmatrix[row][column]->setMinFloatValue( (float)-INFINITY );
                            massmatrix[row][column]->setMaxFloatValue( (float)INFINITY );
                        }
                    }

                    if( DataField<RigidMass<3, double> > * ff = dynamic_cast< DataField<RigidMass<3, double> > * >( (*it).second )  )
                    {
                        RigidMass<3, double> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);
                        editVolume->setFloatValue(current_mass.volume);

                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                matrix[row][column]->setFloatValue(current_mass.inertiaMatrix[row][column]);
                                connect( matrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }

                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                massmatrix[row][column]->setFloatValue(current_mass.inertiaMassMatrix[row][column]);
                                connect( massmatrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }
                    }

                    else if( DataField<RigidMass<3, float> > * ff = dynamic_cast< DataField<RigidMass<3, float> > * >( (*it).second )  )
                    {
                        RigidMass<3, float> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);
                        editVolume->setFloatValue(current_mass.volume);
                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                matrix[row][column]->setFloatValue(current_mass.inertiaMatrix[row][column]);
                                connect( matrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }

                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                massmatrix[row][column]->setFloatValue(current_mass.inertiaMassMatrix[row][column]);
                                connect( massmatrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }
                    }
                    connect( editMass, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editVolume, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

                }
                //********************************************************************************************************//
                //RigidMass<2, double>,RigidMass<2, float>
                else if( dynamic_cast< DataField<RigidMass<2, double> > * >( (*it).second ) ||
                        dynamic_cast< DataField<RigidMass<2, float> > * >( (*it).second ))
                {
                    box->setColumns(2);

                    WFloatLineEdit* editMass = new WFloatLineEdit( box, "editMass" );
                    list_Object.push_back( (QObject *) editMass);
                    editMass->setMinFloatValue( 0.0f );
                    editMass->setMaxFloatValue( (float)INFINITY );

                    new QLabel("Volume", box);
                    WFloatLineEdit* editVolume = new WFloatLineEdit( box, "editMass" );
                    list_Object.push_back( (QObject *) editVolume);
                    editVolume->setMinFloatValue( 0.0f );
                    editVolume->setMaxFloatValue( (float)INFINITY );


                    new QLabel("Inertia Matrix", box);

                    WFloatLineEdit* editInertiaMatrix = new WFloatLineEdit( box, "editInertia" );
                    list_Object.push_back( (QObject *) editInertiaMatrix);
                    editInertiaMatrix->setMinFloatValue( 0.0f );
                    editInertiaMatrix->setMaxFloatValue( (float)INFINITY );

                    new QLabel("Inertia Mass Matrix", box);

                    WFloatLineEdit* editInertiaMassMatrix = new WFloatLineEdit( box, "editInertiaMass" );
                    list_Object.push_back( (QObject *) editInertiaMassMatrix);
                    editInertiaMassMatrix->setMinFloatValue( 0.0f );
                    editInertiaMassMatrix->setMaxFloatValue( (float)INFINITY );



                    if( DataField<RigidMass<2, double> > * ff = dynamic_cast< DataField<RigidMass<2, double> > * >( (*it).second )  )
                    {
                        RigidMass<2, double> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);
                        editVolume->setFloatValue(current_mass.volume);
                        editInertiaMatrix->setFloatValue(current_mass.inertiaMatrix);
                        editInertiaMassMatrix->setFloatValue(current_mass.inertiaMassMatrix);
                    }

                    else if( DataField<RigidMass<2, float> > * ff = dynamic_cast< DataField<RigidMass<2, float> > * >( (*it).second )  )
                    {
                        RigidMass<2, float> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);
                        editVolume->setFloatValue(current_mass.volume);
                        editInertiaMatrix->setFloatValue(current_mass.inertiaMatrix);
                        editInertiaMassMatrix->setFloatValue(current_mass.inertiaMassMatrix);

                    }
                    connect( editMass, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editVolume, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editInertiaMatrix, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editInertiaMassMatrix, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //RigidCoord<3, double>,RigidCoord<3, float>
                else if( dynamic_cast< DataField<RigidCoord<3, double> > * >( (*it).second ) ||
                        dynamic_cast< DataField<RigidCoord<3, float> > * >( (*it).second ))
                {
                    new QLabel("Center", box);

                    if( DataField<RigidCoord<3, double> > * ff = dynamic_cast< DataField<RigidCoord<3, double> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getCenter(), box);
                    }
                    else if(DataField<RigidCoord<3, float> > * ff = dynamic_cast< DataField<RigidCoord<3, float> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getCenter(), box);
                    }

                    new QLabel("Orientation", box);

                    if( DataField<RigidCoord<3, double> > * ff = dynamic_cast< DataField<RigidCoord<3, double> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getOrientation(), box);
                    }
                    else if(DataField<RigidCoord<3, float> > * ff = dynamic_cast< DataField<RigidCoord<3, float> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getOrientation(), box);
                    }
                }
                //********************************************************************************************************//
                //RigidDeriv<3, double>,RigidDeriv<3, float>
                else if( dynamic_cast< DataField<RigidDeriv<3, double> > * >( (*it).second ) ||
                        dynamic_cast< DataField<RigidDeriv<3, float> > * >( (*it).second ))
                {
                    new QLabel("Velocity Center", box);

                    if( DataField<RigidDeriv<3, double> > * ff = dynamic_cast< DataField<RigidDeriv<3, double> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getVCenter(), box);
                    }
                    else if(DataField<RigidDeriv<3, float> > * ff = dynamic_cast< DataField<RigidDeriv<3, float> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getVCenter(), box);
                    }

                    new QLabel("Velocity Orientation", box);

                    if( DataField<RigidDeriv<3, double> > * ff = dynamic_cast< DataField<RigidDeriv<3, double> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getVOrientation(), box);
                    }
                    else if(DataField<RigidDeriv<3, float> > * ff = dynamic_cast< DataField<RigidDeriv<3, float> > * >( (*it).second )  )
                    {
                        createVector(ff->getValue().getVOrientation(), box);
                    }

                }
                else if(DataField<helper::io::Mesh::Material > *ff = dynamic_cast< DataField<helper::io::Mesh::Material > * >( (*it).second ) )
                {
                    helper::io::Mesh::Material material = ff->getValue();
                    new QLabel("Component", box);
                    new QLabel("R", box); new QLabel("G", box); new QLabel("B", box);	new QLabel("A", box);

                    QCheckBox* checkBox;

                    //Diffuse Component
                    checkBox = new QCheckBox(box); list_Object.push_back( (QObject *) checkBox);
                    checkBox->setChecked(material.useDiffuse);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Diffuse", box);
                    createVector(material.diffuse, box);

                    //Ambient Component
                    checkBox = new QCheckBox(box); list_Object.push_back( (QObject *) checkBox);
                    checkBox->setChecked(material.useAmbient);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Ambient", box);
                    createVector(material.ambient, box);


                    //Emissive Component
                    checkBox = new QCheckBox(box); list_Object.push_back( (QObject *) checkBox);
                    checkBox->setChecked(material.useEmissive);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Emissive", box);
                    createVector(material.emissive, box);


                    //Specular Component
                    checkBox = new QCheckBox(box); list_Object.push_back( (QObject *) checkBox);
                    checkBox->setChecked(material.useSpecular);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Specular", box);
                    createVector(material.specular, box);

                    //Shininess Component
                    checkBox = new QCheckBox(box); list_Object.push_back( (QObject *) checkBox);
                    checkBox->setChecked(material.useShininess);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Shininess", box);
                    WFloatLineEdit* editShininess = new WFloatLineEdit( box, "editShininess" );
                    list_Object.push_back( (QObject *) editShininess);
                    editShininess->setMinFloatValue( 0.0f );
                    editShininess->setMaxFloatValue( (float)INFINITY );
                    editShininess->setFloatValue( material.shininess);
                    connect( editShininess, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

                    box->setColumns(6);
                }
                //********************************************************************************************************//
                //Types that needs a QTable: vector of elements
                else if (createTable( (*it).second, box));
                else
                {
                    //Delete the box
                    delete box;
                    box = NULL;
                    core::objectmodel::FieldBase* unknown_datafield = (*it).second;
                    if (unknown_datafield->getValueString() == "") std::cout << "Empty DataField ";
                    std::cerr<<"not added in the dialog : "<<fieldname<<std::endl;
                    std::cout << "Name : " << (*it).first.c_str() << " : " <<  (*it).second->help << "\n";
                }
            }

            ++i;
            if (box != NULL)
            {
                tabPropertiesLayout->addWidget( box );
            }
        }

        //If the current element is a node, we add a box to perform geometric transformation: translation, scaling
        if(dynamic_cast< GNode *>(node_clicked))
        {
            Q3GroupBox *box = new Q3GroupBox(tab1, QString("Transformation"));
            box->setColumns(3);
            box->setTitle(QString("Transformation"));
            new QLabel(QString("Translation"), box);
            for (int i=0; i<2; i++) box->addSpace(0);
            WFloatLineEdit* editTranslationX = new WFloatLineEdit( box, "editTranslationX" );
            list_Object.push_front( (QObject *) editTranslationX);

            editTranslationX->setMinFloatValue( (float)-INFINITY );
            editTranslationX->setMaxFloatValue( (float)INFINITY );

            WFloatLineEdit* editTranslationY = new WFloatLineEdit( box, "editTranslationY" );
            list_Object.push_front( (QObject *) editTranslationY);

            editTranslationY->setMinFloatValue( (float)-INFINITY );
            editTranslationY->setMaxFloatValue( (float)INFINITY );

            WFloatLineEdit* editTranslationZ = new WFloatLineEdit( box, "editTranslationZ" );
            list_Object.push_front( (QObject *) editTranslationZ);

            editTranslationZ->setMinFloatValue( (float)-INFINITY );
            editTranslationZ->setMaxFloatValue( (float)INFINITY );

            QLabel *textScale = new QLabel(QString("Scale"), box);
            WFloatLineEdit* editScale = new WFloatLineEdit( box, "editScale" );
            list_Object.push_front( (QObject *) editScale);

            editScale->setMinFloatValue( (float)-INFINITY );
            editScale->setMaxFloatValue( (float)INFINITY );

            editTranslationX->setFloatValue(0);
            editTranslationY->setFloatValue(0);
            editTranslationZ->setFloatValue(0);
            editScale->setFloatValue(1);

            connect( editTranslationX, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( editTranslationY, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( editTranslationZ, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( editScale, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

            //Option still experimental : disabled !!!!
            textScale->hide();
            editScale->hide();


            tabPropertiesLayout->addWidget( box );
        }


        //Adding buttons at the bottom of the dialog
        QHBoxLayout *lineLayout = new QHBoxLayout( 0, 0, 6, "Button Layout");

        buttonUpdate = new QPushButton( this, "buttonUpdate" );
        lineLayout->addWidget(buttonUpdate);
        buttonUpdate->setText("&Update");
        buttonUpdate->setEnabled(false);

        QSpacerItem *Horizontal_Spacing = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
        lineLayout->addItem( Horizontal_Spacing );

        QPushButton *buttonOk = new QPushButton( this, "buttonOk" );
        lineLayout->addWidget(buttonOk);
        buttonOk->setText( tr( "&OK" ) );

        QPushButton *buttonCancel = new QPushButton( this, "buttonCancel" );
        lineLayout->addWidget(buttonCancel);
        buttonCancel->setText( tr( "&Cancel" ) );

        generalLayout->addLayout( lineLayout );



        //Signals and slots connections
        connect( buttonUpdate,   SIGNAL( clicked() ), this, SLOT( updateValues() ) );
        connect( buttonOk,       SIGNAL( clicked() ), this, SLOT( accept() ) );
        connect( buttonCancel,   SIGNAL( clicked() ), this, SLOT( reject() ) );


        //Title of the Dialog
        setCaption((node->getTypeName()+"::"+node->getName()).data());

        resize( QSize(553, 130).expandedTo(minimumSizeHint()) );
    }


}


//*******************************************************************************************************************
void ModifyObject::changeValue()
{
    if (buttonUpdate == NULL) return;
    buttonUpdate->setEnabled(true);
}

//*******************************************************************************************************************
void ModifyObject::updateValues()
{
    if (buttonUpdate == NULL // || !buttonUpdate->isEnabled()
       ) return;

    //Make the update of all the values
    if (node)
    {

        std::list< QObject *>::iterator list_it=list_Object.begin();
        //If the current element is a node of the graph, we first apply the transformations
        if (GNode* current_node = dynamic_cast< GNode *>(node))
        {
            WFloatLineEdit* editScale        = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
            WFloatLineEdit* editTranslationZ = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
            WFloatLineEdit* editTranslationY = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
            WFloatLineEdit* editTranslationX = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
            if (!(editTranslationX->getFloatValue() == 0 &&
                    editTranslationY->getFloatValue() == 0 &&
                    editTranslationZ->getFloatValue() == 0 &&
                    editScale->getFloatValue() == 1 ))
            {
                emit( transformObject(current_node,
                        editTranslationX->getFloatValue(),editTranslationY->getFloatValue(),editTranslationZ->getFloatValue(),
                        editScale->getFloatValue()));

                editTranslationX->setFloatValue(0);
                editTranslationY->setFloatValue(0);
                editTranslationZ->setFloatValue(0);
                editScale->setFloatValue(1);
                //current_node->execute<InitVisitor>();
            }
        }

        std::list< std::list< QObject*> * >::iterator block_iterator=list_PointSubset.begin();

        const std::vector< std::pair<std::string, FieldBase*> >& fields = node->getFields();
        int i=0;

        for( std::vector< std::pair<std::string, FieldBase*> >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {

            //*******************************************************************************************************************
            if( DataField<int> * ff = dynamic_cast< DataField<int> * >( (*it).second )  )
            {
                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( (*list_it) ); list_it++;
                ff->setValue(spinBox->value());
            }
            //*******************************************************************************************************************
            else if( DataField<unsigned int> * ff = dynamic_cast< DataField<unsigned int> * >( (*it).second )  )
            {

                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( (*list_it) ); list_it++;

                ff->setValue(spinBox->value());
            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataField<float> * >( (*it).second ) ||
                    dynamic_cast< DataField<double> * >( (*it).second ))
            {

                WFloatLineEdit* editSFFloat = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;


                if( DataField<float> * ff = dynamic_cast< DataField<float> * >( (*it).second )  )
                {
                    ff->setValue(editSFFloat->getFloatValue());
                }
                else if(DataField<double> * ff = dynamic_cast< DataField<double> * >( (*it).second )  )
                {
                    ff->setValue((double) editSFFloat->getFloatValue());
                }

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataField<bool> * >( (*it).second ))
            {
                // the bool line edit

                if( DataField<bool> * ff = dynamic_cast< DataField<bool> * >( (*it).second )  )
                {
                    QCheckBox* checkBox = dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;

                    ff->setValue(checkBox->isOn());
                }

            }
            //*******************************************************************************************************************
            else if( DataField<std::string> * ff = dynamic_cast< DataField<std::string> * >( (*it).second )  )
            {

                QLineEdit* lineEdit = dynamic_cast< QLineEdit *> ( (*list_it) ); list_it++;

                ff->setValue(lineEdit->text().ascii());


                if(!strcmp(ff->help,"object name") )
                {
                    std::string name=item->text(0).ascii();
                    std::string::size_type pos = name.rfind(' ');
                    if (pos != std::string::npos)
                        name.resize(pos-1);
                    name += "  ";

                    name+=lineEdit->text().ascii();
                    item->setText(0,name.c_str());
                }
            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataField<Vec6f> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec6d> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec<6,int> > * >( (*it).second )   ||
                    dynamic_cast< DataField<Vec<6,unsigned int> > * >( (*it).second )   )
            {
                if( DataField<Vec6f> * ff = dynamic_cast< DataField<Vec6f> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec6d> * ff = dynamic_cast< DataField<Vec6d> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if( DataField<Vec<6,int> > * ff = dynamic_cast< DataField<Vec<6,int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec<6, unsigned int> > * ff = dynamic_cast< DataField<Vec<6, unsigned int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }


            }
            //*******************************************************************************************************************
            else if(  dynamic_cast< DataField<Vec4f> * >( (*it).second )           ||
                    dynamic_cast< DataField<Vec4d> * >( (*it).second )           ||
                    dynamic_cast< DataField<Vec<4,int> > * >( (*it).second )     ||
                    dynamic_cast< DataField<Vec<4,unsigned int> > * >( (*it).second )  )
            {

                if( DataField<Vec4f> * ff = dynamic_cast< DataField<Vec4f> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec4d> * ff = dynamic_cast< DataField<Vec4d> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if( DataField<Vec<4,int> > * ff = dynamic_cast< DataField<Vec<4,int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec<4, unsigned int> > * ff = dynamic_cast< DataField<Vec<4, unsigned int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataField<Vec3f> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec3d> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec<3,int> > * >( (*it).second )   ||
                    dynamic_cast< DataField<Vec<3,unsigned int> > * >( (*it).second )  )
            {

                if( DataField<Vec3f> * ff = dynamic_cast< DataField<Vec3f> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec3d> * ff = dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if( DataField<Vec<3,int> > * ff = dynamic_cast< DataField<Vec<3,int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec<3, unsigned int> > * ff = dynamic_cast< DataField<Vec<3, unsigned int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }


            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataField<Vec2f> * >( (*it).second )          ||
                    dynamic_cast< DataField<Vec2d> * >( (*it).second )          ||
                    dynamic_cast< DataField<Vec<2,int> > * >( (*it).second )    ||
                    dynamic_cast< DataField<Vec<2,unsigned int> > * >( (*it).second )  )
            {


                if( DataField<Vec2f> * ff = dynamic_cast< DataField<Vec2f> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec2d> * ff = dynamic_cast< DataField<Vec2d> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if( DataField<Vec<2,int> > * ff = dynamic_cast< DataField<Vec<2,int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec<2, unsigned int> > * ff = dynamic_cast< DataField<Vec<2, unsigned int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataField<Vec1f> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec1d> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec<1,int> > * >( (*it).second )   ||
                    dynamic_cast< DataField<Vec<1,unsigned int> > * >( (*it).second )  )
            {


                if( DataField<Vec1f> * ff = dynamic_cast< DataField<Vec1f> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec1d> * ff = dynamic_cast< DataField<Vec1d> * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if( DataField<Vec<1,int> > * ff = dynamic_cast< DataField<Vec<1,int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }
                else if(DataField<Vec<1, unsigned int> > * ff = dynamic_cast< DataField<Vec<1, unsigned int> > * >( (*it).second )  )
                {
                    storeVector(list_it, ff);
                }

            }
            //*******************************************************************************************************************
            else if( DataField<PointSubset> * ff = dynamic_cast< DataField<PointSubset> * >( (*it).second ))
            {
                std::list< QObject *>::iterator element_iterator=(*block_iterator)->begin();
                element_iterator++;
                Q3Table  *table = dynamic_cast< Q3Table *>  ( (*element_iterator) );
                block_iterator++;


                PointSubset p;
                p.resize(table->numRows());
                for ( int index=0; index<table->numRows(); index++)
                {
                    if (table->text(index,0) == "") p[index] = 0;
                    else                            p[index] = atoi(table->text(index,0));
                }
                ff->setValue(p);
            }
            //*******************************************************************************************************************
            else if( DataField<RigidMass<3, double> > * ff = dynamic_cast< DataField<RigidMass<3, double> > * >( (*it).second )  )
            {
                RigidMass<3, double> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.mass = (double) mass->getFloatValue();
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.volume = (double) volume->getFloatValue();
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                        current_mass.inertiaMatrix[row][column] = (double) matrix_element->getFloatValue();
                    }
                }
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                        current_mass.inertiaMassMatrix[row][column] = (double) matrix_element->getFloatValue();
                    }
                }
                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( DataField<RigidMass<3, float> > * ff = dynamic_cast< DataField<RigidMass<3, float> > * >( (*it).second )  )
            {
                RigidMass<3, float> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.mass =  mass->getFloatValue();
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.volume =  volume->getFloatValue();
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                        current_mass.inertiaMatrix[row][column] = (double) matrix_element->getFloatValue();
                    }
                }
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                        current_mass.inertiaMassMatrix[row][column] = (double) matrix_element->getFloatValue();
                    }
                }
                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( DataField<RigidMass<2, double> > * ff = dynamic_cast< DataField<RigidMass<2, double> > * >( (*it).second )  )
            {
                RigidMass<2, double> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.mass = (double) mass->getFloatValue();
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.volume = (double) volume->getFloatValue();
                WFloatLineEdit* inertiaMatrix = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.inertiaMatrix = (double) inertiaMatrix->getFloatValue();

                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( DataField<RigidMass<2, float> > * ff = dynamic_cast< DataField<RigidMass<2, float> > * >( (*it).second )  )
            {
                RigidMass<2, float> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.mass =  mass->getFloatValue();
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.volume =  volume->getFloatValue();
                WFloatLineEdit* inertiaMatrix = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.inertiaMatrix = (double) inertiaMatrix->getFloatValue();
                WFloatLineEdit* inertiaMassMatrix = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                current_mass.inertiaMassMatrix = (double) inertiaMassMatrix->getFloatValue();

                ff->setValue(current_mass);
            }
            else if (DataField<helper::io::Mesh::Material > *ff = dynamic_cast< DataField<helper::io::Mesh::Material > * >( (*it).second ))
            {
                helper::io::Mesh::Material M;  QCheckBox* checkBox;    WFloatLineEdit* value;

                //Diffuse
                checkBox= dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;
                M.useDiffuse = checkBox->isOn();
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;	M.diffuse[i] = value->getFloatValue();}
                //Ambient
                checkBox= dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;
                M.useAmbient = checkBox->isOn();
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;	M.ambient[i] = value->getFloatValue();}
                //Emissive
                checkBox= dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;
                M.useEmissive = checkBox->isOn();
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;  M.emissive[i] = value->getFloatValue();}
                //Specular
                checkBox= dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;
                M.useSpecular = checkBox->isOn();
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;	M.specular[i] = value->getFloatValue();}
                //Shininess
                checkBox= dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;
                M.useShininess = checkBox->isOn();
                value = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                M.shininess = value->getFloatValue();

                ff->setValue(M);
            }
            ++i;
        }
        if (BaseObject *obj = dynamic_cast< BaseObject* >(node))
            obj->reinit();
    }

    saveTables();

    updateContext(dynamic_cast< GNode *>(node));

    emit (objectUpdated());
    buttonUpdate->setEnabled(false);
}


//*******************************************************************************************************************
//Update the Context of a whole node, including its childs
void ModifyObject::updateContext( GNode *node )
{
    if (node == NULL) return;
    //Update the context of the childs

    GNode::ChildIterator it  = node->child.begin();
    while (it != node->child.end())
    {
        core::objectmodel::Context *current_context = dynamic_cast< core::objectmodel::Context *>(node->getContext());
        (*it)->copyContext( (*current_context));
        updateContext( (*it) );
        it++;
    }

}
//*******************************************************************************************************************
//Method called when the number of one of the PointSubset block has been modified : we need to recreate the block modified
void ModifyObject::changeNumberPoint()
{

    //Add or remove fields
    std::list< std::list< QObject*> * >::iterator block_iterator;

    //For each block of the set
    for (block_iterator=list_PointSubset.begin() ; block_iterator != list_PointSubset.end(); block_iterator++)
    {

        //For each block of type PointSubset, we verify the initial number of element and the current
        std::list< QObject *> *current_structure = (*block_iterator);
        if (current_structure == NULL) continue;

        std::list< QObject *>::iterator element_iterator=current_structure->begin();
        QSpinBox *spin  = dynamic_cast< QSpinBox *> ( (*element_iterator) );
        element_iterator++;
        Q3Table  *table = dynamic_cast< Q3Table *>  ( (*element_iterator) );

        if ( spin->value() != table->numRows())
        {
            int initial_number = table->numRows();
            table->setNumRows( spin->value() );
            for (int i=initial_number; i< spin->value(); i++)  table->setText(i,0,QString(std::string("0").c_str()));
        }
    }
    emit( changeValue() );

}














//**************************************************************************************************************************************
//Called each time a new step of the simulation if computed
void ModifyObject::updateTables()
{
    std::list< std::pair< Q3Table*, FieldBase*> >::iterator it_list_Table;
    for (it_list_Table = list_Table.begin(); it_list_Table != list_Table.end(); it_list_Table++)
    {
        if ( dynamic_cast<Field< vector<Rigid3dTypes::Coord> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Field< vector<Rigid3fTypes::Coord> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Field< vector<Rigid3dTypes::Deriv> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Field< vector<Rigid3fTypes::Deriv> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Field< vector<Rigid2dTypes::Coord> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Field< vector<Rigid2fTypes::Coord> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Field< vector<Rigid2dTypes::Deriv> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Field< vector<Rigid2fTypes::Deriv> > *> ( (*it_list_Table).second ) )
        {
            std::list< std::pair< Q3Table*, FieldBase*> >::iterator it_center = it_list_Table;
            it_list_Table++;
            createTable((*it_list_Table).second,NULL,(*it_center).first, (*it_list_Table).first);
        }
        else
            createTable((*it_list_Table).second,NULL,(*it_list_Table).first);
    }
}


//**************************************************************************************************************************************
//create or update an existing table with the contents of a field
bool ModifyObject::createTable( FieldBase* field,Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2)
{
    //********************************************************************************************************//
    //vector< Vec3f >
    if (DataField< vector< Vec3f > >  *ff = dynamic_cast< DataField< vector< Vec3f > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec3f > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];
            for (int j=0; j<3; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector< Vec3d >
    else if ( DataField< vector< Vec3d> >   *ff = dynamic_cast< DataField< vector< Vec3d > >   * >( field ) )
    {
        if (!vectorTable)
        {

            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec3d > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];
            for (int j=0; j<3; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< Vec<3, int> >
    else if( DataField< vector< Vec< 3, int> > >  *ff = dynamic_cast< DataField< vector< Vec< 3, int> > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec< 3, int> > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];
            for (int j=0; j<3; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector< Vec3f >
    else if (Field< vector< Vec3f > >  *ff = dynamic_cast< Field< vector< Vec3f > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec3f > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];
            for (int j=0; j<3; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< Vec3d >
    else if ( Field< vector< Vec3d> >   *ff = dynamic_cast< Field< vector< Vec3d > >   * >( field ) )
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));      vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec3d > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];
            for (int j=0; j<3; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< Vec<3, int> >
    else if( Field< vector< Vec< 3, int> > >  *ff = dynamic_cast< Field< vector< Vec< 3, int> > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec< 3, int> > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];
            for (int j=0; j<3; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector< Vec< 3, int > >
    else if( Field< vector< Vec< 3, int> > >  *ff = dynamic_cast< Field< vector< Vec< 3, int> > >  * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec< 3, int> > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];
            for (int j=0; j<3; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< Vec2f >
    else if (DataField< vector< Vec2f > >  *ff = dynamic_cast< DataField< vector< Vec2f > >  * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);
            vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector< Vec2f > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];
            for (int j=0; j<2; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector< Vec2d >
    else if ( DataField< vector< Vec2d> >   *ff = dynamic_cast< DataField< vector< Vec2d > >   * >( field ) )
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);
            vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec2d > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];
            for (int j=0; j<2; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< Vec< 2, int> >
    else if(  DataField< vector< Vec< 2, int> > >   *ff = dynamic_cast< DataField< vector< Vec< 2, int> > >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);
            vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec< 2, int> > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];
            for (int j=0; j<2; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }

    //********************************************************************************************************//
    //vector< Vec2f >
    else if (Field< vector< Vec2f > >  *ff = dynamic_cast< Field< vector< Vec2f > >  * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);
            vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector< Vec2f > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];
            for (int j=0; j<2; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector< Vec2d >
    else if ( Field< vector< Vec2d> >   *ff = dynamic_cast< Field< vector< Vec2d > >   * >( field ) )
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);
            vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec2d > value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];
            for (int j=0; j<2; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< Vec< 2, int> >
    else if(  Field< vector< Vec< 2, int> > >   *ff = dynamic_cast< Field< vector< Vec< 2, int> > >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);
            vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< Vec< 2, int> > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];
            for (int j=0; j<2; j++)
            {
                oss[j] << value[i][j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //PointData< double >
    else if(  DataField< sofa::component::topology::PointData<double > >   *ff = dynamic_cast< DataField< sofa::component::topology::PointData<double > >   * >( field))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        sofa::component::topology::PointData<double > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //PointData< float >
    else if(  DataField< sofa::component::topology::PointData<float > >   *ff = dynamic_cast< DataField< sofa::component::topology::PointData<float > >   * >( field))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        sofa::component::topology::PointData<float > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //PointData< int >
    else if(  DataField< sofa::component::topology::PointData<int > >   *ff = dynamic_cast< DataField< sofa::component::topology::PointData<int > >   * >( field))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        sofa::component::topology::PointData<int > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< double >
    else if(  DataField< vector< double > >   *ff = dynamic_cast< DataField< vector< double> >   * >( field))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< double > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< float >
    else if(  DataField< vector< float > >   *ff = dynamic_cast< DataField< vector< float> >   * >( field))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< float > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< int >
    else if(  DataField< vector< int > >   *ff = dynamic_cast< DataField< vector< int> >   * >( field))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< int > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector< unsigned int >
    else if(  DataField< vector< unsigned int > >   *ff = dynamic_cast< DataField< vector< unsigned int> >   * >( field))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }
        vector< unsigned int > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<Rigid3dTypes::Coord>
    else if (Field< vector<Rigid3dTypes::Coord> >  *ff = dynamic_cast< Field< vector<Rigid3dTypes::Coord>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	    vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),4, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	    vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);
            vectorTable2->horizontalHeader()->setLabel(3,QString("W"));      vectorTable2->setColumnStretchable(3,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid3dTypes::Coord> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];

            for (int j=0; j<3; j++)
            {
                oss[j] << value[i].getCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2[4];
            for (int j=0; j<4; j++)
            {
                oss2[j] << value[i].getOrientation()[j];
                vectorTable2->setText(i,j,std::string(oss2[j].str()).c_str());
            }
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<Rigid3fTypes::Coord>
    else if (Field< vector<Rigid3fTypes::Coord> >  *ff = dynamic_cast< Field< vector<Rigid3fTypes::Coord>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),4, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	 vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);
            vectorTable2->horizontalHeader()->setLabel(3,QString("W"));      vectorTable2->setColumnStretchable(3,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid3fTypes::Coord> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];

            for (int j=0; j<3; j++)
            {
                oss[j] << value[i].getCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2[4];
            for (int j=0; j<4; j++)
            {
                oss2[j] << value[i].getOrientation()[j];
                vectorTable2->setText(i,j,std::string(oss2[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector<Rigid3dTypes::Deriv>
    else if (Field< vector<Rigid3dTypes::Deriv> >  *ff = dynamic_cast< Field< vector<Rigid3dTypes::Deriv>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	 vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid3dTypes::Deriv> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];

            for (int j=0; j<3; j++)
            {
                oss[j] << value[i].getVCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2[3];
            for (int j=0; j<3; j++)
            {
                oss2[j] << value[i].getVOrientation()[j];
                vectorTable2->setText(i,j,std::string(oss2[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector<Rigid3fTypes::Deriv>
    else if (Field< vector<Rigid3fTypes::Deriv> >  *ff = dynamic_cast< Field< vector<Rigid3fTypes::Deriv>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	 vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid3fTypes::Deriv> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[3];

            for (int j=0; j<3; j++)
            {
                oss[j] << value[i].getVCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2[3];
            for (int j=0; j<3; j++)
            {
                oss2[j] << value[i].getVOrientation()[j];
                vectorTable2->setText(i,j,std::string(oss2[j].str()).c_str());
            }
        }
        return true;

    }
    //********************************************************************************************************//
    //vector<Rigid2dTypes::Coord>
    else if (Field< vector<Rigid2dTypes::Coord> >  *ff = dynamic_cast< Field< vector<Rigid2dTypes::Coord>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	 vectorTable2->setColumnStretchable(0,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid2dTypes::Coord> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];

            for (int j=0; j<2; j++)
            {
                oss[j] << value[i].getCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2;
            oss2 << value[i].getOrientation();
            vectorTable2->setText(i,0,std::string(oss2.str()).c_str());

        }
        return true;
    }
    //********************************************************************************************************//
    //vector<Rigid2fTypes::Coord>
    else if (Field< vector<Rigid2fTypes::Coord> >  *ff = dynamic_cast< Field< vector<Rigid2fTypes::Coord>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	 vectorTable2->setColumnStretchable(0,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid2fTypes::Coord> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];

            for (int j=0; j<2; j++)
            {
                oss[j] << value[i].getCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2;
            oss2 << value[i].getOrientation();
            vectorTable2->setText(i,0,std::string(oss2.str()).c_str());

        }
        return true;
    }
    //********************************************************************************************************//
    //vector<Rigid2dTypes::Deriv>
    else if (Field< vector<Rigid2dTypes::Deriv> >  *ff = dynamic_cast< Field< vector<Rigid2dTypes::Deriv>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	 vectorTable2->setColumnStretchable(0,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid2dTypes::Deriv> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];

            for (int j=0; j<2; j++)
            {
                oss[j] << value[i].getVCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2;
            oss2 << value[i].getVOrientation();
            vectorTable2->setText(i,0,std::string(oss2.str()).c_str());
        }
        return true;

    }
    //********************************************************************************************************//
    //vector<Rigid2fTypes::Deriv>
    else if (Field< vector<Rigid2fTypes::Deriv> >  *ff = dynamic_cast< Field< vector<Rigid2fTypes::Deriv>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            new QLabel("Center", box);

            vectorTable = new Q3Table(ff->getValue().size(),2, box);
            vectorTable->setReadOnly(false);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	 vectorTable2->setColumnStretchable(0,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<Rigid2fTypes::Deriv> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[2];

            for (int j=0; j<2; j++)
            {
                oss[j] << value[i].getVCenter()[j];
                vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
            }
            std::ostringstream oss2;
            oss2 << value[i].getVOrientation();
            vectorTable2->setText(i,0,std::string(oss2.str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec6dTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec6fTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }

    //********************************************************************************************************//
    //vector<SpringForceField< Vec3dTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec3fTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec2dTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec2fTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec1dTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec1fTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),5, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
            vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
            vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
            vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[5];
            oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
            oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
            oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
            oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
            oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
        }
        return true;
    }
    //********************************************************************************************************//
    //vector<JointSpringForceField< Rigid3dTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),16, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1"));
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2"));
            vectorTable->horizontalHeader()->setLabel(2,QString("KsTx : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(3,QString("KsTy : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(4,QString("KsTz : Stiffness"));

            vectorTable->horizontalHeader()->setLabel(5,QString("KsRx : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(6,QString("KsRy : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(7,QString("KsRz : Stiffness"));

            vectorTable->horizontalHeader()->setLabel(8,QString("Kd : damping"));

            vectorTable->horizontalHeader()->setLabel(9,QString("Rest Length Pos X"));
            vectorTable->horizontalHeader()->setLabel(10,QString("Rest Length Pos Y"));
            vectorTable->horizontalHeader()->setLabel(11,QString("Rest Length Pos Z"));

            vectorTable->horizontalHeader()->setLabel(12,QString("Rest Length Quat X"));
            vectorTable->horizontalHeader()->setLabel(13,QString("Rest Length Quat Y"));
            vectorTable->horizontalHeader()->setLabel(14,QString("Rest Length Quat Z"));
            vectorTable->horizontalHeader()->setLabel(15,QString("Rest Length Quat W"));

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[16];
            oss[0]  << value[i].m1;           vectorTable->setText(i ,0 ,std::string(oss[0] .str()).c_str());
            oss[1]  << value[i].m2;           vectorTable->setText(i ,1 ,std::string(oss[1] .str()).c_str());
            oss[2]  << value[i].kst[0];       vectorTable->setText(i ,2 ,std::string(oss[2] .str()).c_str());
            oss[3]  << value[i].kst[1];       vectorTable->setText(i ,3 ,std::string(oss[3] .str()).c_str());
            oss[4]  << value[i].kst[2];       vectorTable->setText(i ,4 ,std::string(oss[4] .str()).c_str());
            oss[5]  << value[i].ksr[0];       vectorTable->setText(i ,5 ,std::string(oss[5] .str()).c_str());
            oss[6]  << value[i].ksr[1];       vectorTable->setText(i ,6 ,std::string(oss[6] .str()).c_str());
            oss[7]  << value[i].ksr[2];       vectorTable->setText(i ,7 ,std::string(oss[7] .str()).c_str());
            oss[8]  << value[i].kd;           vectorTable->setText(i ,8 ,std::string(oss[8] .str()).c_str());
            oss[9]  << value[i].initTrans[0]; vectorTable->setText(i ,9 ,std::string(oss[9] .str()).c_str());
            oss[10] << value[i].initTrans[1]; vectorTable->setText(i,10 ,std::string(oss[10].str()).c_str());
            oss[11] << value[i].initTrans[2]; vectorTable->setText(i,11 ,std::string(oss[11].str()).c_str());
            oss[12] << value[i].initRot[0];   vectorTable->setText(i,12 ,std::string(oss[12].str()).c_str());
            oss[13] << value[i].initRot[1];   vectorTable->setText(i,13 ,std::string(oss[13].str()).c_str());
            oss[14] << value[i].initRot[2];   vectorTable->setText(i,14 ,std::string(oss[14].str()).c_str());
            oss[15] << value[i].initRot[3];   vectorTable->setText(i,15 ,std::string(oss[15].str()).c_str());
        }
        return true;
    }

    //********************************************************************************************************//
    //vector<JointSpringForceField< Rigid3fTypes >::Spring>
    else if (DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),16, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("Index 1"));
            vectorTable->horizontalHeader()->setLabel(1,QString("Index 2"));
            vectorTable->horizontalHeader()->setLabel(2,QString("KsTx : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(3,QString("KsTy : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(4,QString("KsTz : Stiffness"));

            vectorTable->horizontalHeader()->setLabel(5,QString("KsRx : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(6,QString("KsRy : Stiffness"));
            vectorTable->horizontalHeader()->setLabel(7,QString("KsRz : Stiffness"));

            vectorTable->horizontalHeader()->setLabel(8,QString("Kd : damping"));

            vectorTable->horizontalHeader()->setLabel(9,QString("Rest Length Pos X"));
            vectorTable->horizontalHeader()->setLabel(10,QString("Rest Length Pos Y"));
            vectorTable->horizontalHeader()->setLabel(11,QString("Rest Length Pos Z"));

            vectorTable->horizontalHeader()->setLabel(12,QString("Rest Length Quat X"));
            vectorTable->horizontalHeader()->setLabel(13,QString("Rest Length Quat Y"));
            vectorTable->horizontalHeader()->setLabel(14,QString("Rest Length Quat Z"));
            vectorTable->horizontalHeader()->setLabel(15,QString("Rest Length Quat W"));

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
        }

        vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss[16];
            oss[0]  << value[i].m1;           vectorTable->setText(i ,0 ,std::string(oss[0] .str()).c_str());
            oss[1]  << value[i].m2;           vectorTable->setText(i ,1 ,std::string(oss[1] .str()).c_str());
            oss[2]  << value[i].kst[0];       vectorTable->setText(i ,2 ,std::string(oss[2] .str()).c_str());
            oss[3]  << value[i].kst[1];       vectorTable->setText(i ,3 ,std::string(oss[3] .str()).c_str());
            oss[4]  << value[i].kst[2];       vectorTable->setText(i ,4 ,std::string(oss[4] .str()).c_str());
            oss[5]  << value[i].ksr[0];       vectorTable->setText(i ,5 ,std::string(oss[5] .str()).c_str());
            oss[6]  << value[i].ksr[1];       vectorTable->setText(i ,6 ,std::string(oss[6] .str()).c_str());
            oss[7]  << value[i].ksr[2];       vectorTable->setText(i ,7 ,std::string(oss[7] .str()).c_str());
            oss[8]  << value[i].kd;           vectorTable->setText(i ,8 ,std::string(oss[8] .str()).c_str());
            oss[9]  << value[i].initTrans[0]; vectorTable->setText(i ,9 ,std::string(oss[9] .str()).c_str());
            oss[10] << value[i].initTrans[1]; vectorTable->setText(i,10 ,std::string(oss[10].str()).c_str());
            oss[11] << value[i].initTrans[2]; vectorTable->setText(i,11 ,std::string(oss[11].str()).c_str());
            oss[12] << value[i].initRot[0];   vectorTable->setText(i,12 ,std::string(oss[12].str()).c_str());
            oss[13] << value[i].initRot[1];   vectorTable->setText(i,13 ,std::string(oss[13].str()).c_str());
            oss[14] << value[i].initRot[2];   vectorTable->setText(i,14 ,std::string(oss[14].str()).c_str());
            oss[15] << value[i].initRot[3];   vectorTable->setText(i,15 ,std::string(oss[15].str()).c_str());
        }
        return true;
    }

    return false;
}














//**************************************************************************************************************************************
//save in datafield the values of the tables
void ModifyObject::saveTables()
{

    std::list< std::pair< Q3Table*, FieldBase*> >::iterator it_list_Table;
    for (it_list_Table = list_Table.begin(); it_list_Table != list_Table.end(); it_list_Table++)
    {
        storeTable((*it_list_Table).first, (*it_list_Table).second);
    }
}











//**************************************************************************************************************************************
//Read the content of a QTable and store its values in a datafield
void ModifyObject::storeTable(Q3Table* table, FieldBase* field)
{

    QString value;

    //**************************************************************************************************************************************
    if (DataField< vector< Vec3f > >  *ff = dynamic_cast< DataField< vector< Vec3f > >  * >( field ))
    {
        QString vec_value[3];
        vector< Vec3f > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<3; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec3f( atof(vec_value[0]),
                    atof(vec_value[1]),
                    atof(vec_value[2])  ));
        }
        ff->setValue( new_value );

    }

    //**************************************************************************************************************************************
    else if ( DataField< vector< Vec3d> >   *ff = dynamic_cast< DataField< vector< Vec3d > >   * >( field ) )
    {

        QString vec_value[3];
        vector< Vec3d > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<3; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec3d( atof(vec_value[0]),
                    atof(vec_value[1]),
                    atof(vec_value[2])  ));
        }
        ff->setValue( new_value );

    }

    //**************************************************************************************************************************************
    else if( DataField< vector< Vec< 3, int> > >  *ff = dynamic_cast< DataField< vector< Vec< 3, int> > >  * >( field ))
    {
        QString vec_value[3];
        vector< Vec< 3, int> > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<3; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec< 3, int>( atoi(vec_value[0]),
                    atoi(vec_value[1]),
                    atoi(vec_value[2])  ));
        }
        ff->setValue( new_value );
    }

    //**************************************************************************************************************************************
    else if (Field< vector< Vec3f > >  *ff = dynamic_cast< Field< vector< Vec3f > >  * >( field ))
    {
        QString vec_value[3];
        vector< Vec3f > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<3; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec3f( atof(vec_value[0]),
                    atof(vec_value[1]),
                    atof(vec_value[2])  ));
        }
        ff->setValue( new_value );

    }
    //**************************************************************************************************************************************
    else if ( Field< vector< Vec3d> >   *ff = dynamic_cast< Field< vector< Vec3d > >   * >( field ) )
    {

        QString vec_value[3];
        vector< Vec3d > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<3; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec3d( atof(vec_value[0]),
                    atof(vec_value[1]),
                    atof(vec_value[2])  ));
        }
        ff->setValue( new_value );

    }
    //**************************************************************************************************************************************
    else if( Field< vector< Vec< 3, int> > >  *ff = dynamic_cast< Field< vector< Vec< 3, int> > >  * >( field ))
    {
        QString vec_value[3];
        vector< Vec< 3, int> > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<3; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec< 3, int>( atoi(vec_value[0]),
                    atoi(vec_value[1]),
                    atoi(vec_value[2])  ));
        }
        ff->setValue( new_value );

    }

    //**************************************************************************************************************************************
    else if (DataField< vector< Vec2f > >  *ff = dynamic_cast< DataField< vector< Vec2f > >  * >( field ))
    {

        QString vec_value[2];
        vector< Vec2f > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<2; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec2f( atof(vec_value[0]),
                    atof(vec_value[1])  ));
        }
        ff->setValue( new_value );

    }
    //**************************************************************************************************************************************
    else if ( DataField< vector< Vec2d> >   *ff = dynamic_cast< DataField< vector< Vec2d > >   * >( field ) )
    {
        QString vec_value[2];
        vector< Vec2d > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<2; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec2d( atof(vec_value[0]),
                    atof(vec_value[1])  ));
        }
        ff->setValue( new_value );

    }
    //**************************************************************************************************************************************
    else if(  DataField< vector< Vec< 2, int> > >   *ff = dynamic_cast< DataField< vector< Vec< 2, int> > >   * >( field ))
    {

        QString vec_value[2];
        vector< Vec< 2, int> > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<2; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec< 2, int>( atoi(vec_value[0]),
                    atoi(vec_value[1])  ));
        }
        ff->setValue( new_value );
    }

    //**************************************************************************************************************************************
    else if (Field< vector< Vec2f > >  *ff = dynamic_cast< Field< vector< Vec2f > >  * >( field ))
    {

        QString vec_value[2];
        vector< Vec2f > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<2; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec2f( atof(vec_value[0]),
                    atof(vec_value[1])  ));
        }
        ff->setValue( new_value );

    }
    //**************************************************************************************************************************************
    else if ( Field< vector< Vec2d> >   *ff = dynamic_cast< Field< vector< Vec2d > >   * >( field ) )
    {
        QString vec_value[2];
        vector< Vec2d > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<2; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec2d( atof(vec_value[0]),
                    atof(vec_value[1])  ));
        }
        ff->setValue( new_value );

    }
    //**************************************************************************************************************************************
    else if(  Field< vector< Vec< 2, int> > >   *ff = dynamic_cast< Field< vector< Vec< 2, int> > >   * >( field ))
    {

        QString vec_value[2];
        vector< Vec< 2, int> > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            for (int j=0; j<2; j++)
            {
                vec_value[j] = table->text(i,j);
            }
            new_value.push_back( Vec< 2, int>( atoi(vec_value[0]),
                    atoi(vec_value[1])  ));
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if(  DataField< vector< double > >   *ff = dynamic_cast< DataField< vector< double> >   * >( field))
    {
        QString vec_value;
        vector< double  > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            vec_value = table->text(i,0);
            new_value.push_back( atof(vec_value) );
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if(  DataField< vector< float > >   *ff = dynamic_cast< DataField< vector< float> >   * >( field))
    {
        QString vec_value;
        vector< float  > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            vec_value = table->text(i,0);
            new_value.push_back( atof(vec_value) );
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if(  DataField< vector< int > >   *ff = dynamic_cast< DataField< vector< int> >   * >( field))
    {
        QString vec_value;
        vector< int  > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            vec_value = table->text(i,0);
            new_value.push_back( atoi(vec_value) );
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if(  DataField< vector< unsigned int > >   *ff = dynamic_cast< DataField< vector< unsigned int> >   * >( field))
    {
        QString vec_value;
        vector< unsigned int  > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            vec_value = table->text(i,0);
            new_value.push_back( atoi(vec_value) );
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if(  DataField< sofa::component::topology::PointData<double > >   *ff = dynamic_cast< DataField< sofa::component::topology::PointData<double > >   * >( field))
    {

        QString vec_value;
        sofa::component::topology::PointData<double > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            vec_value = table->text(i,0);
            new_value.push_back( atof(vec_value) );
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if(  DataField< sofa::component::topology::PointData<float > >   *ff = dynamic_cast< DataField< sofa::component::topology::PointData<float > >   * >( field))
    {

        QString vec_value;
        sofa::component::topology::PointData<float > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            vec_value = table->text(i,0);
            new_value.push_back( atof(vec_value) );
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if(  DataField< sofa::component::topology::PointData<int > >   *ff = dynamic_cast< DataField< sofa::component::topology::PointData<int > >   * >( field))
    {

        QString vec_value;
        sofa::component::topology::PointData<int > new_value;
        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            vec_value = table->text(i,0);
            new_value.push_back( atoi(vec_value) );
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }

    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[5];
            sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring current_spring;
            for (int j=0; j<5; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int) values[1];

            current_spring.ks       = values[2];
            current_spring.kd       = values[3];
            current_spring.initpos  = values[4];

            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[16];
            sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring current_spring;
            for (int j=0; j<16; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int)values[1];

            current_spring.kst[0]       = values[2];
            current_spring.kst[1]       = values[3];
            current_spring.kst[2]       = values[4];

            current_spring.ksr[0]       = values[5];
            current_spring.ksr[1]       = values[6];
            current_spring.ksr[2]       = values[7];

            current_spring.kd           = values[8];

            current_spring.initTrans[0] = values[9];
            current_spring.initTrans[1] = values[10];
            current_spring.initTrans[2] = values[11];

            current_spring.initRot[0]   = values[12];
            current_spring.initRot[1]   = values[13];
            current_spring.initRot[2]   = values[14];
            current_spring.initRot[3]   = values[15];
            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
    //**************************************************************************************************************************************
    else if (DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  *ff = dynamic_cast< DataField< vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  * >( field ))
    {
        vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > new_value;

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            double values[16];
            sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring current_spring;
            for (int j=0; j<16; j++)
            {
                values[j] = atof(table->text(i,j));
            }
            current_spring.m1 = (int) values[0];
            current_spring.m2 = (int)values[1];

            current_spring.kst[0]       = values[2];
            current_spring.kst[1]       = values[3];
            current_spring.kst[2]       = values[4];

            current_spring.ksr[0]       = values[5];
            current_spring.ksr[1]       = values[6];
            current_spring.ksr[2]       = values[7];

            current_spring.kd           = values[8];

            current_spring.initTrans[0] = values[9];
            current_spring.initTrans[1] = values[10];
            current_spring.initTrans[2] = values[11];

            current_spring.initRot[0]   = values[12];
            current_spring.initRot[1]   = values[13];
            current_spring.initRot[2]   = values[14];
            current_spring.initRot[3]   = values[15];
            new_value.push_back(current_spring);
        }
        ff->setValue( new_value );
    }
}








void ModifyObject::createVector(const Vec<6,double> &value, Q3GroupBox *box)
{

    const int size = 6;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<6,float> &value, Q3GroupBox *box)
{

    const int size = 6;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<6,int> &value, Q3GroupBox *box)
{

    const int size = 6;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<6,unsigned int> &value, Q3GroupBox *box)
{

    const int size = 6;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)0 );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<4,double> &value, Q3GroupBox *box)
{

    const int size = 4;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}
void ModifyObject::createVector(const Vec<4,float> &value, Q3GroupBox *box)
{

    const int size = 4;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}


void ModifyObject::createVector(const Vec<4,int> &value, Q3GroupBox *box)
{

    const int size = 4;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<4,unsigned int> &value, Q3GroupBox *box)
{

    const int size = 4;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)0 );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<3,double> &value, Q3GroupBox *box)
{

    const int size = 3;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<3,float> &value, Q3GroupBox *box)
{
    const int size = 3;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<3,int> &value, Q3GroupBox *box)
{

    const int size = 3;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<3,unsigned int> &value, Q3GroupBox *box)
{

    const int size = 3;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)0 );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<2,double> &value, Q3GroupBox *box)
{

    const int size = 2;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}


void ModifyObject::createVector(const Vec<2,float> &value, Q3GroupBox *box)
{

    const int size = 2;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<2,int> &value, Q3GroupBox *box)
{

    const int size = 2;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<2,unsigned int> &value, Q3GroupBox *box)
{

    const int size = 2;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)0 );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<1,double> &value, Q3GroupBox *box)
{

    const int size = 1;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<1,float> &value, Q3GroupBox *box)
{

    const int size = 1;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setFloatValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<1,int> &value, Q3GroupBox *box)
{

    const int size = 1;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Vec<1,unsigned int> &value, Q3GroupBox *box)
{

    const int size = 1;
    box->setColumns(size+1);

    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        list_Object.push_back( (QObject *) editSFFloat[i]);

        editSFFloat[i]->setMinFloatValue( (float)0 );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<size; i++)
    {
        editSFFloat[i]->setIntValue(value[i]);
    }
    for (int i=0; i<size; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}

void ModifyObject::createVector(const Quater<double> &value, Q3GroupBox *box)
{
    Vec<4,double> new_value(value[0], value[1], value[2], value[3]);
    createVector(new_value, box);
}
void ModifyObject::createVector(const Quater<float> &value, Q3GroupBox *box)
{
    Vec<4,float> new_value(value[0], value[1], value[2], value[3]);
    createVector(new_value, box);
}
//********************************************************************************************************************
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<6,double> > *ff)
{
    const int size=6;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, double> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<6,float> > *ff)
{
    const int size=6;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, float> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<6,int> > *ff)
{
    const int size=6;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, int> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<6,unsigned int> > *ff)
{
    const int size=6;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, unsigned int> value;
    for (int i=0; i<size; i++)  value[i] =  (unsigned int) editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<4,double> > *ff)
{
    const int size=4;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, double> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<4,float> > *ff)
{
    const int size=4;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, float> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<4,int> > *ff)
{
    const int size=4;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, int> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<4,unsigned int> > *ff)
{
    const int size=4;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, unsigned int> value;
    for (int i=0; i<size; i++)  value[i] =  (unsigned int) editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<3,double> > *ff)
{
    const int size=3;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, double> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<3,float> > *ff)
{
    const int size=3;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, float> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<3,int> > *ff)
{
    const int size=3;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, int> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<3,unsigned int> > *ff)
{
    const int size=3;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, unsigned int> value;
    for (int i=0; i<size; i++)  value[i] =  (unsigned int) editSFFloat[i]->getIntValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<2,double> > *ff)
{
    const int size=2;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, double> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<2,float> > *ff)
{
    const int size=2;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, float> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<2,int> > *ff)
{
    const int size=2;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, int> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<2,unsigned int> > *ff)
{
    const int size=2;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, unsigned int> value;
    for (int i=0; i<size; i++)  value[i] =  (unsigned int) editSFFloat[i]->getIntValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<1,double> > *ff)
{
    const int size=1;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, double> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<1,float> > *ff)
{
    const int size=1;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, float> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<1,int> > *ff)
{
    const int size=1;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, int> value;
    for (int i=0; i<size; i++)  value[i] =  editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

void ModifyObject::storeVector(std::list< QObject *>::iterator &list_it, DataField< Vec<1,unsigned int> > *ff)
{
    const int size=1;
    WFloatLineEdit* editSFFloat[size];
    for (int i=0; i<size; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
    }

    Vec<size, unsigned int> value;
    for (int i=0; i<size; i++)  value[i] =  (unsigned int) editSFFloat[i]->getIntValue();
    ff->setValue(value);
}

} // namespace qt

} // namespace gui

} // namespace sofa

