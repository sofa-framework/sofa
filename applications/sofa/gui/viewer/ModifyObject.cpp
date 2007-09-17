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
#include <sofa/component/topology/PointSubset.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/simulation/tree/InitAction.h>
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
        // 	    list_Object= new std::list< QObject *>();
        // 	    list_PointSubset= new std::list< std::list<QObject *> *>();

        const std::map< std::string, core::objectmodel::FieldBase* >& fields = node->getFields();

        int i=0;

        for( std::map< std::string, core::objectmodel::FieldBase* >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
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

                if( DataField<int> * ff = dynamic_cast< DataField<int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)INT_MIN,(int)INT_MAX,1,box);
                    list_Object.push_back( (QObject *) spinBox);

                    spinBox->setValue(ff->getValue());
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                else if( DataField<unsigned int> * ff = dynamic_cast< DataField<unsigned int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)0,(int)INT_MAX,1,box);
                    list_Object.push_back( (QObject *) spinBox);

                    spinBox->setValue(ff->getValue());
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                else if( DataField<float> * ff = dynamic_cast< DataField<float> * >( (*it).second )  )
                {
                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    list_Object.push_back( (QObject *) editSFFloat);

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue());
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }
                else if(DataField<double> * ff = dynamic_cast< DataField<double> * >( (*it).second )  )
                {

                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    list_Object.push_back( (QObject *) editSFFloat);

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue());
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }

                else if( DataField<std::string> * ff = dynamic_cast< DataField<std::string> * >( (*it).second )  )
                {

                    QLineEdit* lineEdit = new QLineEdit(box);
                    list_Object.push_back( (QObject *) lineEdit);

                    lineEdit->setText(QString(ff->getValue().c_str()));
                    connect( lineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                }


                else if( dynamic_cast< DataField<Vec6f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec6d> * >( (*it).second ) )
                {
                    box->setColumns(7);
                    WFloatLineEdit* editSFFloat[6];
                    for (int i=0; i<6; i++)
                    {
                        std::ostringstream oss;
                        oss << "editSFFloat_" << i;
                        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())  );
                        list_Object.push_back( (QObject *) editSFFloat[i]);

                        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
                        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
                    }
                    if( DataField<Vec6f> * ff = dynamic_cast< DataField<Vec6f> * >( (*it).second )  )
                    {
                        for (int i=0; i<6; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<6; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }
                    else if(DataField<Vec6d> * ff = dynamic_cast< DataField<Vec6d> * >( (*it).second )  )
                    {

                        for (int i=0; i<6; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<6; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }
                }
                else if( dynamic_cast< DataField<Vec4f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec4d> * >( (*it).second )   )
                {

                    box->setColumns(5);
                    WFloatLineEdit* editSFFloat[4];
                    for (int i=0; i<4; i++)
                    {
                        std::ostringstream oss;
                        oss << "editSFFloat_" << i;
                        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
                        list_Object.push_back( (QObject *) editSFFloat[i]);

                        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
                        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
                    }
                    if( DataField<Vec4f> * ff = dynamic_cast< DataField<Vec4f> * >( (*it).second )  )
                    {
                        for (int i=0; i<4; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<4; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }
                    else if(DataField<Vec4d> * ff = dynamic_cast< DataField<Vec4d> * >( (*it).second )  )
                    {

                        for (int i=0; i<4; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<4; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }
                }
                else if( dynamic_cast< DataField<Vec3f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec3d> * >( (*it).second ))
                {

                    box->setColumns(4);

                    WFloatLineEdit* editSFFloat[3];
                    for (int i=0; i<3; i++)
                    {
                        std::ostringstream oss;
                        oss << "editSFFloat_" << i;
                        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
                        list_Object.push_back( (QObject *) editSFFloat[i]);

                        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
                        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
                    }
                    if( DataField<Vec3f> * ff = dynamic_cast< DataField<Vec3f> * >( (*it).second )  )
                    {
                        for (int i=0; i<3; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<3; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }
                    else if(DataField<Vec3d> * ff = dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
                    {

                        for (int i=0; i<3; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<3; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }
                }
                else if( dynamic_cast< DataField<Vec2f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec2d> * >( (*it).second ))
                {


                    box->setColumns(3);
                    WFloatLineEdit* editSFFloat[2];
                    for (int i=0; i<2; i++)
                    {
                        std::ostringstream oss;
                        oss << "editSFFloat_" << i;
                        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
                        list_Object.push_back( (QObject *) editSFFloat[i]);

                        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
                        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
                    }
                    if( DataField<Vec2f> * ff = dynamic_cast< DataField<Vec2f> * >( (*it).second )  )
                    {
                        for (int i=0; i<2; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<2; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }
                    else if(DataField<Vec2d> * ff = dynamic_cast< DataField<Vec2d> * >( (*it).second )  )
                    {

                        for (int i=0; i<2; i++)
                        {
                            editSFFloat[i]->setFloatValue(ff->getValue()[i]);
                        }
                        for (int i=0; i<2; i++)
                        {
                            connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        }
                    }

                }
                else if( dynamic_cast< DataField<Vec1f> * >( (*it).second ) ||
                        dynamic_cast< DataField<Vec1d> * >( (*it).second ) )
                {
                    box->setColumns(2);
                    WFloatLineEdit* editSFFloatX = new WFloatLineEdit( box, "editSFFloatX" );
                    list_Object.push_back( (QObject *) editSFFloatX);

                    editSFFloatX->setMinFloatValue( (float)-INFINITY );
                    editSFFloatX->setMaxFloatValue( (float)INFINITY );

                    if( DataField<Vec1f> * ff = dynamic_cast< DataField<Vec1f> * >( (*it).second )  )
                    {
                        editSFFloatX->setFloatValue(ff->getValue()[0]);

                        connect( editSFFloatX, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    }
                    else if(DataField<Vec1d> * ff = dynamic_cast< DataField<Vec1d> * >( (*it).second )  )
                    {
                        editSFFloatX->setFloatValue(ff->getValue()[0]);

                        connect( editSFFloatX, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    }
                }
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

                    //Second line contains the sequence of fields
                    Q3Grid *grid = new Q3Grid(width()/150,box);
                    current_list->push_back(grid); //We save the container of the elements

                    spinBox->setValue(p.size());
                    for (unsigned int t=0; t< p.size(); t++)
                    {
                        std::ostringstream oindex;
                        oindex << "editIndex_" << t;

                        WFloatLineEdit* editIndex = new WFloatLineEdit( grid, oindex.str().c_str() );

                        current_list->push_back(editIndex);

                        editIndex->setMinFloatValue( 0);
                        editIndex->setMaxFloatValue( (float)INFINITY );
                        editIndex->setIntValue(p[t]);

                        connect( editIndex, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

                    }
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeNumberPoint() ) );
                }
                //RigidMass<3, double>
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
                    }
                    connect( editMass, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editVolume, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

                }
                else if (createTable(NULL, (*it).second, box));
                else
                {
                    //Delete the box
                    delete box;
                    box = NULL;
                    std::cerr<<"FIELD not added in the dialog : "<<fieldname<<std::endl;

                    core::objectmodel::FieldBase* unknown_datafield = (*it).second;
                    std::cout << "Value: " << unknown_datafield->getValueString()<< "\n";
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
                //current_node->execute<InitAction>();
            }
        }

        std::list< std::list< QObject*> * >::iterator block_iterator=list_PointSubset.begin();

        const std::map< std::string, core::objectmodel::FieldBase* >& fields = node->getFields();
        int i=0;

        for( std::map< std::string, core::objectmodel::FieldBase* >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {
            if( DataField<int> * ff = dynamic_cast< DataField<int> * >( (*it).second )  )
            {
                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( (*list_it) ); list_it++;
                ff->setValue(spinBox->value());
            }


            else if( DataField<unsigned int> * ff = dynamic_cast< DataField<unsigned int> * >( (*it).second )  )
            {

                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( (*list_it) ); list_it++;

                ff->setValue(spinBox->value());
            }

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
            else if( dynamic_cast< DataField<bool> * >( (*it).second ))
            {
                // the bool line edit

                if( DataField<bool> * ff = dynamic_cast< DataField<bool> * >( (*it).second )  )
                {
                    QCheckBox* checkBox = dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;

                    ff->setValue(checkBox->isOn());
                }

            }
            else if( DataField<std::string> * ff = dynamic_cast< DataField<std::string> * >( (*it).second )  )
            {

                QLineEdit* lineEdit = dynamic_cast< QLineEdit *> ( (*list_it) ); list_it++;

#ifdef QT_MODULE_QT3SUPPORT
                std::string value(lineEdit->text());
                ff->setValue(value.c_str());
#else
                ff->setValue(lineEdit->text());
#endif
                if ((*it).first == std::string("name")) item->setText(0,lineEdit->text());
            }
            else if( dynamic_cast< DataField<Vec6f> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec6d> * >( (*it).second ) )
            {
                WFloatLineEdit* editSFFloat[6];
                for (int i=0; i<6; i++)
                {
                    editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                }

                if( DataField<Vec6f> * ff = dynamic_cast< DataField<Vec6f> * >( (*it).second )  )
                {
                    Vec<6, float> value;
                    for (int i=0; i<6; i++)  value[i] =  editSFFloat[i]->getFloatValue();
                    ff->setValue(value);
                }
                else if(DataField<Vec6d> * ff = dynamic_cast< DataField<Vec6d> * >( (*it).second )  )
                {
                    Vec<6, double> value;
                    for (int i=0; i<6; i++)  value[i] =  ((double)editSFFloat[i]->getFloatValue());

                    ff->setValue(value);
                }

            }
            else if(  dynamic_cast< DataField<Vec4f> * >( (*it).second )          ||
                    dynamic_cast< DataField<Vec4d> * >( (*it).second ) )
            {
                WFloatLineEdit* editSFFloat[4];
                for (int i=0; i<4; i++)
                {
                    editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                }

                if( DataField<Vec4f> * ff = dynamic_cast< DataField<Vec4f> * >( (*it).second )  )
                {
                    Vec<4, float> value;
                    for (int i=0; i<4; i++)  value[i] =  editSFFloat[i]->getFloatValue();
                    ff->setValue(value);
                }
                else if(DataField<Vec4d> * ff = dynamic_cast< DataField<Vec4d> * >( (*it).second )  )
                {
                    Vec<4, double> value;
                    for (int i=0; i<4; i++)  value[i] =  ((double)editSFFloat[i]->getFloatValue());

                    ff->setValue(value);
                }

            }
            else if( dynamic_cast< DataField<Vec3f> * >( (*it).second )        ||
                    dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
            {
                WFloatLineEdit* editSFFloat[3];
                for (int i=0; i<3; i++)
                {
                    editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                }

                if( DataField<Vec3f> * ff = dynamic_cast< DataField<Vec3f> * >( (*it).second )  )
                {
                    Vec<3, float> value;
                    for (int i=0; i<3; i++)  value[i] =  editSFFloat[i]->getFloatValue();
                    ff->setValue(value);
                }
                else if(DataField<Vec3d> * ff = dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
                {
                    Vec<3, double> value;
                    for (int i=0; i<3; i++)  value[i] =  ((double)editSFFloat[i]->getFloatValue());

                    ff->setValue(value);
                }

            }
            else if( dynamic_cast< DataField<Vec2f> * >( (*it).second )          ||
                    dynamic_cast< DataField<Vec2d> * >( (*it).second ))
            {
                WFloatLineEdit* editSFFloat[2];
                for (int i=0; i<2; i++)
                {
                    editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                }

                if( DataField<Vec2f> * ff = dynamic_cast< DataField<Vec2f> * >( (*it).second )  )
                {
                    Vec<2, float> value;
                    for (int i=0; i<2; i++)  value[i] =  editSFFloat[i]->getFloatValue();
                    ff->setValue(value);
                }
                else if(DataField<Vec2d> * ff = dynamic_cast< DataField<Vec2d> * >( (*it).second )  )
                {
                    Vec<2, double> value;
                    for (int i=0; i<2; i++)  value[i] =  ((double)editSFFloat[i]->getFloatValue());

                    ff->setValue(value);
                }

            }
            else if( dynamic_cast< DataField<Vec1f> * >( (*it).second )         ||
                    dynamic_cast< DataField<Vec1d> * >( (*it).second ))
            {
                WFloatLineEdit* editSFFloat;
                editSFFloat = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;

                if( DataField<Vec1f> * ff = dynamic_cast< DataField<Vec1f> * >( (*it).second )  )
                {
                    Vec<1, float> value(editSFFloat->getFloatValue());
                    ff->setValue(value);
                }
                else if(DataField<Vec1d> * ff = dynamic_cast< DataField<Vec1d> * >( (*it).second )  )
                {
                    Vec<1, double> value((double)editSFFloat->getFloatValue());
                    ff->setValue(value);
                }

            }

            else if( DataField<PointSubset> * ff = dynamic_cast< DataField<PointSubset> * >( (*it).second ))
            {


                PointSubset p=ff->getValue();

                //Size of the block, once the spinbox and the grid have been removed
                p.resize((*block_iterator)->size()-2);
                std::list< QObject* >::iterator element_iterator = (*block_iterator)->begin();
                element_iterator++; element_iterator++;
                for (int index=0; element_iterator != (*block_iterator)->end(); element_iterator++,index++)
                {
                    WFloatLineEdit* field = dynamic_cast< WFloatLineEdit *> ( (*element_iterator) );
                    p[index] = field->getIntValue();
                }
                ff->setValue(p);
                block_iterator++;

            }

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
                ff->setValue(current_mass);
            }
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
                ff->setValue(current_mass);
            }

            ++i;
        }

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

        //The number of fields, once the QSpinBox and the grid have been removed
        int initial_size = (current_structure->size()-2);

        //Get the spin box containing the number wanted of fields
        std::list< QObject *>::iterator element_iterator=current_structure->begin();
        QSpinBox *spin = dynamic_cast< QSpinBox *>( (*element_iterator) );
        element_iterator++;
        Q3Grid   *grid = dynamic_cast< Q3Grid *>  ( (*element_iterator) );

        if ( initial_size == spin->value()) {continue; }
        else if ( initial_size < spin->value())
        {
            //We need to add fields
            for (int i=initial_size; i<spin->value(); i++)
            {
                std::ostringstream oindex;
                oindex << "editIndex_" << i;

                WFloatLineEdit* field = new WFloatLineEdit( grid, oindex.str().c_str() );
                current_structure->push_back(field);
                connect( field, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                field->setMinFloatValue( 0 );
                field->setMaxFloatValue( (float) INFINITY );
                field->setIntValue(0);
                field->show();
            }

        }
        else if ( initial_size > spin->value())
        {
            //We need to remove fields
            element_iterator=current_structure->end();
            element_iterator--; //last element
            WFloatLineEdit* field;
            for (int i=initial_size ; i > spin->value(); i--, element_iterator--)
            {
                field = dynamic_cast< WFloatLineEdit *> ( (*element_iterator) );
                delete field;
            }
            current_structure->resize(spin->value()+2);
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
        createTable((*it_list_Table).first, (*it_list_Table).second);
    }
}


//**************************************************************************************************************************************
//create or update an existing table with the contents of a field
bool ModifyObject::createTable( Q3Table* vectorTable, FieldBase* field,Q3GroupBox *box)
{

    if (DataField< vector< Vec3f > >  *ff = dynamic_cast< DataField< vector< Vec3f > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	    vectorTable->setColumnStretchable(0,true);
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
    else if ( DataField< vector< Vec3d> >   *ff = dynamic_cast< DataField< vector< Vec3d > >   * >( field ) )
    {
        if (!vectorTable)
        {

            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	    vectorTable->setColumnStretchable(0,true);
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
    else if( DataField< vector< Vec< 3, int> > >  *ff = dynamic_cast< DataField< vector< Vec< 3, int> > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	    vectorTable->setColumnStretchable(0,true);
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
    else if (Field< vector< Vec3f > >  *ff = dynamic_cast< Field< vector< Vec3f > >  * >( field ))
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	    vectorTable->setColumnStretchable(0,true);
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
    else if ( Field< vector< Vec3d> >   *ff = dynamic_cast< Field< vector< Vec3d > >   * >( field ) )
    {
        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;

            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	    vectorTable->setColumnStretchable(0,true);
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
    else if( Field< vector< Vec< 3, int> > >  *ff = dynamic_cast< Field< vector< Vec< 3, int> > >  * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0)  return false;
            box->setColumns(1);
            vectorTable = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X")); 	    vectorTable->setColumnStretchable(0,true);
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
        sofa::component::topology::PointData<float > value = ff->getValue();

        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;
            oss << value[i];
            vectorTable->setText(i,0,std::string(oss.str()).c_str());
        }
        return true;
    }
    else
    {
        std::cout << "UNKNOWN PointData<float,allocator<float> >\n";
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
}

} // namespace qt

} // namespace gui

} // namespace sofa

