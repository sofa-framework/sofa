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

#include <sofa/gui/qt/ModifyObject.h>
#include <iostream>
#ifdef SOFA_QT4
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTabWidget>
#include <QGridLayout>
#include <Q3Grid>
#include <QTabWidget>
#include <Q3ListView>
#else
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcheckbox.h>
#include <qlayout.h>
#include <qtabwidget.h>
#include <qgrid.h>
#endif

#include <sofa/component/topology/PointSubset.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/TransformationVisitor.h>



#include <qwt_legend.h>

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif


#define WIDGET_BY_TAB 10
#define SIZE_TEXT     75

namespace sofa
{

namespace gui
{

namespace qt
{

using namespace  sofa::defaulttype;
using sofa::component::topology::PointSubset;
using sofa::core::objectmodel::BaseData;

#ifndef SOFA_QT4
typedef QGrid       Q3Grid;
typedef QScrollView Q3ScrollView;
#endif



ModifyObject::ModifyObject(void *Id_, core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked,  QWidget* parent_, const char* name, bool, Qt::WFlags // f
                          ):
    parent(parent_), node(NULL), Id(Id_),visualContentModified(false)
{
    //Title of the Dialog
    setCaption(name);

    HIDE_FLAG = true;
    READONLY_FLAG = true;
    EMPTY_FLAG = false;
    RESIZABLE_FLAG = false;
    REINIT_FLAG = true;

    energy_curve[0]=NULL;	        energy_curve[1]=NULL;	        energy_curve[2]=NULL;
    //Initialization of the Widget
    setNode(node_clicked, item_clicked);
    connect ( this, SIGNAL( objectUpdated() ), parent_, SLOT( redraw() ));
    connect ( this, SIGNAL( dialogClosed(void *) ) , parent_, SLOT( modifyUnlock(void *)));


}


//Set the default file
void ModifyObject::setNode(core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked)
{
    node = node_clicked;
    item = item_clicked;
    //Layout to organize the whole window
    QVBoxLayout *generalLayout = new QVBoxLayout(this, 0, 1, "generalLayout");

    //Tabulation widget
    dialogTab = new QTabWidget(this);
    generalLayout->addWidget(dialogTab);
    connect(dialogTab, SIGNAL( currentChanged( QWidget*)), this, SLOT( updateTables()));

    //Each tab
    counterWidget=0;

    unsigned int counterTab=0;
    QWidget *currentTab=NULL;
    QWidget *currentTab_save=NULL;
    bool emptyTab = false;


    bool visualTab = false;
    bool isNode = (dynamic_cast< Node *>(node_clicked) != NULL);
    QWidget *tabVisualization = NULL; //tab for visualization info: only created if needed ( boolean visualTab gives this answer ).

    QVBoxLayout *currentTabLayout = NULL;
    QVBoxLayout *tabPropertiesLayout=NULL;
    QVBoxLayout *tabVisualizationLayout = NULL;

    // displayWidget
    if (node_clicked)
    {
        //If the current element is a node, we add a box to perform geometric transformation: translation, rotation, scaling
        if(REINIT_FLAG && isNode)
        {
            emptyTab = false;
            currentTab_save  = currentTab= new QWidget();
            currentTabLayout = tabPropertiesLayout = new QVBoxLayout( currentTab, 0, 1, QString("tabPropertiesLayout") + QString::number(counterWidget));
            counterWidget = 2;

            dialogTab->addTab(currentTab, QString("Properties ") + QString::number(counterWidget/WIDGET_BY_TAB));
            ++counterTab;

            Q3GroupBox *box = new Q3GroupBox(currentTab, QString("Transformation"));
            box->setColumns(4);
            box->setTitle(QString("Transformation"));
            //********************************************************************************
            //Translation
            new QLabel(QString("Translation"), box);
            transformation[0] = new WFloatLineEdit( box, "editTranslationX" );
            transformation[0]->setMinFloatValue( (float)-INFINITY );
            transformation[0]->setMaxFloatValue( (float)INFINITY );

            transformation[1] = new WFloatLineEdit( box, "transformation[1]" );
            transformation[1]->setMinFloatValue( (float)-INFINITY );
            transformation[1]->setMaxFloatValue( (float)INFINITY );

            transformation[2] = new WFloatLineEdit( box, "transformation[2]" );
            transformation[2]->setMinFloatValue( (float)-INFINITY );
            transformation[2]->setMaxFloatValue( (float)INFINITY );


            //********************************************************************************
            //Rotation
            new QLabel(QString("Rotation"), box);
            transformation[3] = new WFloatLineEdit( box, "transformation[3]" );
            transformation[3]->setMinFloatValue( (float)-INFINITY );
            transformation[3]->setMaxFloatValue( (float)INFINITY );

            transformation[4] = new WFloatLineEdit( box, "transformation[4]" );
            transformation[4]->setMinFloatValue( (float)-INFINITY );
            transformation[4]->setMaxFloatValue( (float)INFINITY );

            transformation[5] = new WFloatLineEdit( box, "transformation[5]" );
            transformation[5]->setMinFloatValue( (float)-INFINITY );
            transformation[5]->setMaxFloatValue( (float)INFINITY );


            //********************************************************************************
            //Scale
            QLabel *textScale = new QLabel(QString("Scale"), box);
            transformation[6] = new WFloatLineEdit( box, "transformation[6]" );
            transformation[6]->setMinFloatValue( (float)-INFINITY );
            transformation[6]->setMaxFloatValue( (float)INFINITY );


            //********************************************************************************
            //Default values
            transformation[0]->setFloatValue(0);
            transformation[1]->setFloatValue(0);
            transformation[2]->setFloatValue(0);

            transformation[3]->setFloatValue(0);
            transformation[4]->setFloatValue(0);
            transformation[5]->setFloatValue(0);

            transformation[6]->setFloatValue(1);

            connect( transformation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[3], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[4], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[5], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[6], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

            //Option still experimental : disabled !!!!
            textScale->hide();
            transformation[6]->hide();


            tabPropertiesLayout->addWidget( box );
        }

        //All the pointers to the QObjects will be kept in memory in list_Object

        const std::vector< std::pair<std::string, BaseData*> >& fields = node->getFields();

        int i=0;
        for( std::vector< std::pair<std::string, BaseData*> >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {
            currentTab = currentTab_save; //in case we modified currentTab to the visualTab
            currentTabLayout = tabPropertiesLayout;
            if (!emptyTab && counterWidget/WIDGET_BY_TAB==counterTab)
            {
                emptyTab = true;
                if (tabPropertiesLayout!= NULL) tabPropertiesLayout->addStretch();
                currentTab_save  = currentTab= new QWidget();
                currentTabLayout = tabPropertiesLayout = new QVBoxLayout( currentTab, 0, 1, QString("tabPropertiesLayout") + QString::number(counterWidget));
            }
            //For each element, we create a layout
            std::ostringstream oss;
            oss << "itemLayout_" << i;
            Q3GroupBox *box = NULL;;
            // The label
            //QLabel *label = new QLabel(QString((*it).first.c_str()), box,0);
            //label->setGeometry( 10, i*25+5, 200, 20 );

            // 		const std::string& fieldname = (*it).second->getValueTypeString();
            std::string name((*it).first);
            name.resize(4);
            if (name == "show")
            {
                if (!visualTab)
                {
                    visualTab = true;
                    tabVisualization = new QWidget();
                    tabVisualizationLayout = new QVBoxLayout( tabVisualization, 0, 1, "tabVisualizationLayout");
                }

                currentTab = tabVisualization;
                currentTabLayout = tabVisualizationLayout;

                if ( dynamic_cast< Data<int> * >( (*it).second )
                        &&
                        (
                                (*it).first == "showVisualModels" ||
                                (*it).first == "showBehaviorModels" ||
                                (*it).first == "showCollisionModels" ||
                                (*it).first == "showBoundingCollisionModels" ||
                                (*it).first == "showMappings" ||
                                (*it).first == "showMechanicalMappings" ||
                                (*it).first == "showForceFields" ||
                                (*it).first == "showInteractionForceFields" ||
                                (*it).first == "showWireFrame" ||
                                (*it).first == "showNormals"
                        )
                   )
                {

                    box = new Q3GroupBox(tabVisualization, QString("Tri State"));
                    tabVisualizationLayout->addWidget( box );

                    box->setColumns(2);
                    box->setTitle(QString("Visualization Flags"));

                    displayFlag = new DisplayFlagWidget(box);

                    connect( displayFlag, SIGNAL( change(int,bool)), this, SLOT(changeVisualValue() ));

                    Data<int> *ff;

                    for (unsigned int i=0; i<10; ++i)
                    {
                        ff=dynamic_cast< Data<int> * >( (*it).second);
                        objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) displayFlag));
                        if (i!=0) displayFlag->setFlag(i,(ff->getValue()==1));
                        else      displayFlag->setFlag(i,(ff->getValue()!=0));
                        it++;
                    }
                    it--;
                    continue;
                }
            }

            {
                if (hideData(it->second)) continue;
                std::string box_name(oss.str());
                box = new Q3GroupBox(currentTab, QString(box_name.c_str()));
                box->setColumns(4);
                box->setTitle(QString((*it).first.c_str()));

                std::string label_text=(*it).second->help;

                std::string final_text;
                unsigned int number_line=0;
                while (!label_text.empty())
                {
                    std::string::size_type pos = label_text.find('\n');
                    std::string current_sentence;
                    if (pos != std::string::npos)
                        current_sentence  = label_text.substr(0,pos+1);
                    else
                        current_sentence = label_text;


                    if (current_sentence.size() > SIZE_TEXT)
                    {
                        unsigned int index_cut;
                        unsigned int cut = current_sentence.size()/SIZE_TEXT;
                        for (index_cut=1; index_cut<=cut; index_cut++)
                        {
                            std::string::size_type numero_char=current_sentence.rfind(' ',SIZE_TEXT*index_cut);
                            current_sentence = current_sentence.insert(numero_char+1,1,'\n');
                            number_line++;
                        }
                    }
                    if (pos != std::string::npos) label_text = label_text.substr(pos+1);
                    else label_text = "";
                    final_text += current_sentence;
                    number_line++;
                }
                counterWidget += number_line/3; //each 3lines, a new widget is counted
                if (label_text != "TODO") new QLabel(final_text.c_str(), box);

                //********************************************************************************************************//
                //int
                if( Data<int> * ff = dynamic_cast< Data<int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)INT_MIN,(int)INT_MAX,1,box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) spinBox));

                    spinBox->setValue(ff->getValue()); readOnlyData(spinBox,ff);

                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //char
                else if( Data<bool> * ff = dynamic_cast< Data<bool> * >( (*it).second )  )
                {
                    QCheckBox* checkBox = new QCheckBox(box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));

                    checkBox->setChecked(ff->getValue());
                    readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //unsigned int
                else if( Data<unsigned int> * ff = dynamic_cast< Data<unsigned int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)0,(int)INT_MAX,1,box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) spinBox));

                    spinBox->setValue(ff->getValue()); readOnlyData(spinBox,ff);
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //float
                else if( Data<float> * ff = dynamic_cast< Data<float> * >( (*it).second )  )
                {
                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editSFFloat));

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue()); readOnlyData(editSFFloat,ff);
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //double
                else if(Data<double> * ff = dynamic_cast< Data<double> * >( (*it).second )  )
                {
                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editSFFloat));

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue()); readOnlyData(editSFFloat,ff);
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //string
                else if( Data<std::string> * ff = dynamic_cast< Data<std::string> * >( (*it).second )  )
                {
                    // 			if (ff->getValue().empty()) continue;
                    QLineEdit* lineEdit = new QLineEdit(box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) lineEdit));

                    lineEdit->setText(QString(ff->getValue().c_str())); readOnlyData(lineEdit,ff);
                    connect( lineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //int
                else if( DataPtr<int> * ff = dynamic_cast< DataPtr<int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)INT_MIN,(int)INT_MAX,1,box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) spinBox));

                    spinBox->setValue(ff->getValue()); readOnlyData(spinBox,ff);
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //unsigned int
                else if( DataPtr<unsigned int> * ff = dynamic_cast< DataPtr<unsigned int> * >( (*it).second )  )
                {
                    QSpinBox* spinBox = new QSpinBox((int)0,(int)INT_MAX,1,box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) spinBox));

                    spinBox->setValue(ff->getValue()); readOnlyData(spinBox,ff);
                    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //char
                else if( DataPtr<bool> * ff = dynamic_cast< DataPtr<bool> * >( (*it).second )  )
                {
                    QCheckBox* checkBox = new QCheckBox(box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));

                    checkBox->setChecked(ff->getValue()); readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //float
                else if( DataPtr<float> * ff = dynamic_cast< DataPtr<float> * >( (*it).second )  )
                {
                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editSFFloat));

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue()); readOnlyData(editSFFloat,ff);
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //double
                else if(DataPtr<double> * ff = dynamic_cast< DataPtr<double> * >( (*it).second )  )
                {
                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( box, "editSFFloat" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editSFFloat));

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );

                    editSFFloat->setFloatValue(ff->getValue()); readOnlyData(editSFFloat,ff);
                    connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //string
                else if( DataPtr<std::string> * ff = dynamic_cast< DataPtr<std::string> * >( (*it).second )  )
                {
                    // 			if (ff->getValue().empty()) continue;
                    QLineEdit* lineEdit = new QLineEdit(box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) lineEdit));

                    lineEdit->setText(QString(ff->getValue().c_str())); readOnlyData(lineEdit,ff);
                    connect( lineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                }
                else if(DataPtr<bool> *ff = dynamic_cast< DataPtr< bool> * > ( (*it).second) )
                {
                    QCheckBox* checkBox = new QCheckBox(box);
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));

                    checkBox->setChecked(ff->getValue()); readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //Vec6f, Vec6d
                else if( dynamic_cast< Data<Vec6f> * > ( (*it).second ) ||
                        dynamic_cast< Data<Vec6d> * > ( (*it).second ) ||
                        dynamic_cast< Data<Vec<6,int> > *> ( (*it).second ) ||
                        dynamic_cast< Data<Vec<6,unsigned int> >* > ( (*it).second ) )
                {

                    if( Data<Vec6f> * ff = dynamic_cast< Data<Vec6f> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec6d> * ff = dynamic_cast< Data<Vec6d> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<6,int> > * ff = dynamic_cast< Data<Vec<6,int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<6,unsigned int> > * ff = dynamic_cast< Data<Vec<6,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                }
                //********************************************************************************************************//
                //Vec4f,Vec4d
                else if( dynamic_cast< Data<Vec4f> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec4d> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec<4,int> > *>( (*it).second ) ||
                        dynamic_cast< Data<Vec<4,unsigned int> >* >  ( (*it).second )  )
                {


                    if( Data<Vec4f> * ff = dynamic_cast< Data<Vec4f> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec4d> * ff = dynamic_cast< Data<Vec4d> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<4,int> > * ff = dynamic_cast< Data<Vec<4,int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<4,unsigned int> > * ff = dynamic_cast< Data<Vec<4,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                }
                //********************************************************************************************************//
                //Vec3f,Vec3d
                else if( dynamic_cast< Data<Vec3f> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec3d> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec<3,int> > *>( (*it).second ) ||
                        dynamic_cast< Data<Vec<3,unsigned int> >* >  ( (*it).second ))
                {

                    if( Data<Vec3f> * ff = dynamic_cast< Data<Vec3f> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec3d> * ff = dynamic_cast< Data<Vec3d> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<3,int> > * ff = dynamic_cast< Data<Vec<3,int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<3,unsigned int> > * ff = dynamic_cast< Data<Vec<3,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }

                }
                //********************************************************************************************************//
                //Vec2f,Vec2d
                else if( dynamic_cast< Data<Vec2f> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec2d> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec<2,int> > * >( (*it).second ) ||
                        dynamic_cast< Data<Vec<2,unsigned int> > * >  ( (*it).second ))
                {


                    if( Data<Vec2f> * ff = dynamic_cast< Data<Vec2f> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec2d> * ff = dynamic_cast< Data<Vec2d> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<2,int> > * ff = dynamic_cast< Data<Vec<2,int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<2,unsigned int> > * ff = dynamic_cast< Data<Vec<2,unsigned int> >* >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }

                }
                //********************************************************************************************************//
                //Vec1f,Vec1d
                else if( dynamic_cast< Data<Vec1f> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec1d> * >( (*it).second ) ||
                        dynamic_cast< Data<Vec<1,int> > * >( (*it).second ) ||
                        dynamic_cast< Data<Vec<1,unsigned int> > *  > ( (*it).second ))
                {
                    if( Data<Vec1f> * ff = dynamic_cast< Data<Vec1f> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec1d> * ff = dynamic_cast< Data<Vec1d> * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<1,int> > * ff = dynamic_cast< Data<Vec<1,int> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                    else if(Data<Vec<1,unsigned int> > * ff = dynamic_cast< Data<Vec<1,unsigned int> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue(), box);
                    }
                }
                //********************************************************************************************************//
                //PointSubset
                else if( Data<PointSubset> * ff = dynamic_cast< Data<PointSubset> * >( (*it).second )  )
                {
                    Q3Table *vectorTable=NULL;

                    box->setColumns(1);

                    vectorTable = addResizableTable(box,ff->getValue().size(),1);

                    list_Table.push_back(std::make_pair(vectorTable, ff));
                    objectGUI.push_back(std::make_pair(ff,vectorTable));
                    vectorTable->horizontalHeader()->setLabel(0,QString("Value"));
                    vectorTable->setColumnStretchable(0,true);

                    connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );


                    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
                    else	  vectorTable->setNumRows(ff->getValue().size());


                    //Get the PointSubset from the Data
                    PointSubset p= ff->getValue();

                    for (unsigned int i=0; i<ff->getValue().size(); i++)
                    {
                        std::ostringstream oss;
                        oss << p[i];
                        vectorTable->setText(i,0,std::string(oss.str()).c_str());
                    }
                    readOnlyData(vectorTable,ff);
                }
                //********************************************************************************************************//
                //PointSubset
                else if( DataPtr<PointSubset> * ff = dynamic_cast< DataPtr<PointSubset> * >( (*it).second )  )
                {
                    Q3Table *vectorTable=NULL;

                    box->setColumns(1);

                    vectorTable = addResizableTable(box,ff->getValue().size(),1);

                    list_Table.push_back(std::make_pair(vectorTable, ff));
                    objectGUI.push_back(std::make_pair(ff,vectorTable));
                    vectorTable->horizontalHeader()->setLabel(0,QString("Value"));
                    vectorTable->setColumnStretchable(0,true);

                    connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );


                    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
                    else	  vectorTable->setNumRows(ff->getValue().size());


                    //Get the PointSubset from the Data
                    PointSubset p= ff->getValue();

                    for (unsigned int i=0; i<ff->getValue().size(); i++)
                    {
                        std::ostringstream oss;
                        oss << p[i];
                        vectorTable->setText(i,0,std::string(oss.str()).c_str());
                    }
                    readOnlyData(vectorTable,ff);
                }
                //********************************************************************************************************//
                //RigidMass<3, double>,RigidMass<3, float>
                else if( dynamic_cast< Data<RigidMass<3, double> > * >( (*it).second ) ||
                        dynamic_cast< Data<RigidMass<3, float> > * >( (*it).second ))
                {
                    box->setColumns(2);

                    WFloatLineEdit* editMass = new WFloatLineEdit( box, "editMass" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editMass));
                    editMass->setMinFloatValue( 0.0f );
                    editMass->setMaxFloatValue( (float)INFINITY );

                    new QLabel("Volume", box);
                    WFloatLineEdit* editVolume = new WFloatLineEdit( box, "editMass" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editVolume));
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

                            objectGUI.push_back(std::make_pair( (*it).second,  matrix[row][column] ));
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

                            objectGUI.push_back(std::make_pair( (*it).second,  massmatrix[row][column] ));
                            massmatrix[row][column]->setMinFloatValue( (float)-INFINITY );
                            massmatrix[row][column]->setMaxFloatValue( (float)INFINITY );
                        }
                    }

                    if( Data<RigidMass<3, double> > * ff = dynamic_cast< Data<RigidMass<3, double> > * >( (*it).second )  )
                    {
                        RigidMass<3, double> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);     readOnlyData(editMass,ff);
                        editVolume->setFloatValue(current_mass.volume); readOnlyData(editVolume,ff);

                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                matrix[row][column]->setFloatValue(current_mass.inertiaMatrix[row][column]);
                                readOnlyData(matrix[row][column],ff);
                                connect( matrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }

                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                massmatrix[row][column]->setFloatValue(current_mass.inertiaMassMatrix[row][column]);
                                massmatrix[row][column]->setEnabled(false);
                                connect( massmatrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }
                    }

                    else if( Data<RigidMass<3, float> > * ff = dynamic_cast< Data<RigidMass<3, float> > * >( (*it).second )  )
                    {
                        RigidMass<3, float> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);     readOnlyData(editMass,ff);
                        editVolume->setFloatValue(current_mass.volume); readOnlyData(editVolume,ff);
                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                matrix[row][column]->setFloatValue(current_mass.inertiaMatrix[row][column]);
                                readOnlyData(matrix[row][column],ff);
                                connect( matrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }

                        for (int row=0; row<3; row++)
                        {
                            for (int column=0; column<3; column++)
                            {
                                massmatrix[row][column]->setFloatValue(current_mass.inertiaMassMatrix[row][column]);
                                massmatrix[row][column]->setEnabled(false);
                                connect( massmatrix[row][column], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                            }
                        }
                    }
                    connect( editMass, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editVolume, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

                }
                //********************************************************************************************************//
                //RigidMass<2, double>,RigidMass<2, float>
                else if( dynamic_cast< Data<RigidMass<2, double> > * >( (*it).second ) ||
                        dynamic_cast< Data<RigidMass<2, float> > * >( (*it).second ))
                {
                    box->setColumns(2);

                    WFloatLineEdit* editMass = new WFloatLineEdit( box, "editMass" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editMass));
                    editMass->setMinFloatValue( 0.0f );
                    editMass->setMaxFloatValue( (float)INFINITY );

                    new QLabel("Volume", box);
                    WFloatLineEdit* editVolume = new WFloatLineEdit( box, "editMass" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editVolume));
                    editVolume->setMinFloatValue( 0.0f );
                    editVolume->setMaxFloatValue( (float)INFINITY );


                    new QLabel("Inertia Matrix", box);

                    WFloatLineEdit* editInertiaMatrix = new WFloatLineEdit( box, "editInertia" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editInertiaMatrix));
                    editInertiaMatrix->setMinFloatValue( 0.0f );
                    editInertiaMatrix->setMaxFloatValue( (float)INFINITY );

                    new QLabel("Inertia Mass Matrix(Read Only)", box);

                    WFloatLineEdit* editInertiaMassMatrix = new WFloatLineEdit( box, "editInertiaMass" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editInertiaMassMatrix));
                    editInertiaMassMatrix->setMinFloatValue( 0.0f );
                    editInertiaMassMatrix->setMaxFloatValue( (float)INFINITY );



                    if( Data<RigidMass<2, double> > * ff = dynamic_cast< Data<RigidMass<2, double> > * >( (*it).second )  )
                    {
                        RigidMass<2, double> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);                           readOnlyData(editMass,ff);
                        editVolume->setFloatValue(current_mass.volume);		        readOnlyData(editVolume,ff);
                        editInertiaMatrix->setFloatValue(current_mass.inertiaMatrix);         readOnlyData(editInertiaMatrix,ff);
                        editInertiaMassMatrix->setFloatValue(current_mass.inertiaMassMatrix); editInertiaMassMatrix->setEnabled(false);
                    }

                    else if( Data<RigidMass<2, float> > * ff = dynamic_cast< Data<RigidMass<2, float> > * >( (*it).second )  )
                    {
                        RigidMass<2, float> current_mass = ff->getValue();
                        editMass->setFloatValue(current_mass.mass);                           readOnlyData(editMass,ff);
                        editVolume->setFloatValue(current_mass.volume);			readOnlyData(editVolume,ff);
                        editInertiaMatrix->setFloatValue(current_mass.inertiaMatrix);		readOnlyData(editInertiaMatrix,ff);
                        editInertiaMassMatrix->setFloatValue(current_mass.inertiaMassMatrix);	editInertiaMassMatrix->setEnabled(false);

                    }
                    connect( editMass, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editVolume, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editInertiaMatrix, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    connect( editInertiaMassMatrix, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                }
                //********************************************************************************************************//
                //RigidCoord<3, double>,RigidCoord<3, float>
                else if( dynamic_cast< Data<RigidCoord<3, double> > * >( (*it).second ) ||
                        dynamic_cast< Data<RigidCoord<3, float> > * >( (*it).second ))
                {
                    new QLabel("Center", box);

                    if( Data<RigidCoord<3, double> > * ff = dynamic_cast< Data<RigidCoord<3, double> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getCenter(), box);
                    }
                    else if(Data<RigidCoord<3, float> > * ff = dynamic_cast< Data<RigidCoord<3, float> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getCenter(), box);
                    }

                    new QLabel("Orientation", box);

                    if( Data<RigidCoord<3, double> > * ff = dynamic_cast< Data<RigidCoord<3, double> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getOrientation(), box);
                    }
                    else if(Data<RigidCoord<3, float> > * ff = dynamic_cast< Data<RigidCoord<3, float> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getOrientation(), box);
                    }
                }
                //********************************************************************************************************//
                //RigidDeriv<3, double>,RigidDeriv<3, float>
                else if( dynamic_cast< Data<RigidDeriv<3, double> > * >( (*it).second ) ||
                        dynamic_cast< Data<RigidDeriv<3, float> > * >( (*it).second ))
                {
                    new QLabel("Velocity Center", box);

                    if( Data<RigidDeriv<3, double> > * ff = dynamic_cast< Data<RigidDeriv<3, double> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getVCenter(), box);
                    }
                    else if(Data<RigidDeriv<3, float> > * ff = dynamic_cast< Data<RigidDeriv<3, float> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getVCenter(), box);
                    }

                    new QLabel("Velocity Orientation", box);

                    if( Data<RigidDeriv<3, double> > * ff = dynamic_cast< Data<RigidDeriv<3, double> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getVOrientation(), box);
                    }
                    else if(Data<RigidDeriv<3, float> > * ff = dynamic_cast< Data<RigidDeriv<3, float> > * >( (*it).second )  )
                    {
                        createVector(  (*it).second ,ff->getValue().getVOrientation(), box);
                    }

                }
                else if(Data<helper::io::Mesh::Material > *ff = dynamic_cast< Data<helper::io::Mesh::Material > * >( (*it).second ) )
                {
                    helper::io::Mesh::Material material = ff->getValue();
                    new QLabel("Component", box);
                    new QLabel("R", box); new QLabel("G", box); new QLabel("B", box);	new QLabel("A", box);

                    QCheckBox* checkBox;

                    //Diffuse Component
                    checkBox = new QCheckBox(box); objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));
                    checkBox->setChecked(material.useDiffuse); readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Diffuse", box);
                    createVector(  (*it).second ,material.diffuse, box);

                    //Ambient Component
                    checkBox = new QCheckBox(box); objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));
                    checkBox->setChecked(material.useAmbient); readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Ambient", box);
                    createVector(  (*it).second ,material.ambient, box);


                    //Emissive Component
                    checkBox = new QCheckBox(box); objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));
                    checkBox->setChecked(material.useEmissive); readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Emissive", box);
                    createVector(  (*it).second ,material.emissive, box);


                    //Specular Component
                    checkBox = new QCheckBox(box); objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));
                    checkBox->setChecked(material.useSpecular); readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Specular", box);
                    createVector(  (*it).second ,material.specular, box);

                    //Shininess Component
                    checkBox = new QCheckBox(box); objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) checkBox));
                    checkBox->setChecked(material.useShininess); readOnlyData(checkBox,ff);
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                    new QLabel("Shininess", box);
                    WFloatLineEdit* editShininess = new WFloatLineEdit( box, "editShininess" );
                    objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) editShininess));
                    editShininess->setMinFloatValue( 0.0f );
                    editShininess->setMaxFloatValue( (float)128.0f );
                    editShininess->setFloatValue( material.shininess); readOnlyData(editShininess,ff);
                    connect( editShininess, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

                    box->setColumns(6);
                }
                //********************************************************************************************************//
                //Types that needs a QTable: vector of elements
                else if (createTable( (*it).second, box))
                {
                    ++counterWidget; //count for two classic widgets
                }
                else
                {
                    Q3TextEdit* textedit = new Q3TextEdit(box);
                    // 		      objectGUI.push_back(std::make_pair( (*it).second,  (QObject *) textedit);

                    textedit->setText(QString((*it).second->getValueString().c_str()));
                    //if empty field, we don't display it

                    if ((*it).second->getValueString().empty() && !EMPTY_FLAG)
                    {
                        box->hide();
                        std::cerr << (*it).first << " : " << (*it).second->getValueTypeString() << " Not added because empty \n";
                        --counterWidget;
                    }
                    else
                    {
                        list_TextEdit.push_back(std::make_pair(textedit, (*it).second));
                        ++counterWidget; //count for two classic widgets
                        connect( textedit, SIGNAL( textChanged() ), this, SLOT( changeValue() ) );
                    }
                }
            }
            ++i;
            if (box != NULL)
            {
                if (currentTab == currentTab_save && emptyTab && counterWidget/WIDGET_BY_TAB == counterTab)
                {
                    dialogTab->addTab(currentTab, QString("Properties ") + QString::number(counterWidget/WIDGET_BY_TAB));
                    ++counterTab;
                    emptyTab = false;
                }

                dataIndexTab.insert(std::make_pair((*it).second, dialogTab->count()-1));
                ++counterWidget;
                currentTabLayout->addWidget( box );
            }
        }

        if (tabPropertiesLayout!= NULL) tabPropertiesLayout->addStretch();
        if (tabVisualization != NULL )
        {
            dialogTab->addTab(tabVisualization, QString("Visualization"));
            if ( !isNode) tabVisualizationLayout->addStretch();
        }
        for (unsigned int indexTab = 0; indexTab<counterTab; indexTab++)
        {
            if (counterTab == 1)
                dialogTab->setTabLabel(dialogTab->page(indexTab),
                        QString("Properties"));
            else
                dialogTab->setTabLabel(dialogTab->page(indexTab),
                        QString("Properties ") + QString::number(indexTab+1) + QString("/") + QString::number(counterTab));
        }

        if (Node* node = dynamic_cast< Node* >(node_clicked))
        {
            if (REINIT_FLAG && (node->mass!= NULL || node->forceField.size()!=0 ) )
            {
                createGraphMass(dialogTab);
            }
        }

        // Info tab
        {
            emptyTab = false;
            QWidget* tab = new QWidget();
            QVBoxLayout* tabLayout = new QVBoxLayout( tab, 0, 1, QString("tabInfoLayout"));

            dialogTab->addTab(tab, QString("Infos"));
            ++counterTab;
            //Instance
            {
                Q3GroupBox *box = new Q3GroupBox(tab, QString("Instance"));
                box->setColumns(2);
                box->setTitle(QString("Instance"));
                new QLabel(QString("Name"), box);
                new QLabel(QString(node_clicked->getName().c_str()), box);
                new QLabel(QString("Class"), box);
                new QLabel(QString(node_clicked->getClassName().c_str()), box);
                std::string namespacename = node_clicked->decodeNamespaceName(typeid(*node_clicked));
                if (!namespacename.empty())
                {
                    new QLabel(QString("Namespace"), box);
                    new QLabel(QString(namespacename.c_str()), box);
                }
                if (!node_clicked->getTemplateName().empty())
                {
                    new QLabel(QString("Template"), box);
                    new QLabel(QString(node_clicked->getTemplateName().c_str()), box);
                }

                tabLayout->addWidget( box );
            }

            //Class description
            core::ObjectFactory::ClassEntry* entry = core::ObjectFactory::getInstance()->getEntry(node_clicked->getClassName());
            if (entry != NULL && ! entry->creatorList.empty())
            {
                Q3GroupBox *box = new Q3GroupBox(tab, QString("Class"));
                box->setColumns(2);
                box->setTitle(QString("Class"));
                if (!entry->description.empty() && entry->description != std::string("TODO"))
                {
                    new QLabel(QString("Description"), box);
                    new QLabel(QString(entry->description.c_str()), box);
                }
                if (!entry->authors.empty() && entry->authors != std::string("TODO"))
                {
                    new QLabel(QString("Authors"), box);
                    new QLabel(QString(entry->authors.c_str()), box);
                }
                if (!entry->license.empty() && entry->license != std::string("TODO"))
                {
                    new QLabel(QString("License"), box);
                    new QLabel(QString(entry->license.c_str()), box);
                }
                tabLayout->addWidget( box );
            }

            //Log of initialization
            core::objectmodel::Base *obj = dynamic_cast< core::objectmodel::Base* >(node_clicked);
            if (obj && obj->getLogWarning().size() != 0)
            {
                const sofa::helper::vector< std::string > &log = obj->getLogWarning();
                Q3GroupBox *box = new Q3GroupBox(tab, QString("Log"));
                box->setColumns(1);
                box->setTitle(QString("Log"));
                for (unsigned int idLog=0; idLog<log.size(); ++idLog)
                {
                    new QLabel(QString(log[idLog].c_str()), box);
                }
                tabLayout->addWidget( box );
            }
            tabLayout->addStretch();
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



        resize( QSize(553, 130).expandedTo(minimumSizeHint()) );
    }


}


//*******************************************************************************************************************
void ModifyObject::changeValue()
{
    setUpdates.insert(getData(sender()));
    if (buttonUpdate == NULL) return;
    buttonUpdate->setEnabled(true);
}
void ModifyObject::changeVisualValue()
{
    setUpdates.insert(getData(sender()));
    if (buttonUpdate == NULL) return;
    buttonUpdate->setEnabled(true);
    visualContentModified = true;
}
//*******************************************************************************************************************
void ModifyObject::updateValues()
{
    if (buttonUpdate == NULL // || !buttonUpdate->isEnabled()
       ) return;


    saveTextEdit();
    saveTables();

    //Make the update of all the values
    if (node)
    {

        //If the current element is a node of the graph, we first apply the transformations
        if (REINIT_FLAG && dynamic_cast< Node *>(node))
        {
            Node* current_node = dynamic_cast< Node *>(node);
            if (!(transformation[0]->getFloatValue() == 0 &&
                    transformation[1]->getFloatValue() == 0 &&
                    transformation[2]->getFloatValue() == 0 &&
                    transformation[3]->getFloatValue()    == 0 &&
                    transformation[4]->getFloatValue()    == 0 &&
                    transformation[5]->getFloatValue()    == 0 &&
                    transformation[6]->getFloatValue() == 1 ))
            {

                sofa::simulation::TransformationVisitor transform;
                transform.setTranslation(transformation[0]->getFloatValue(),transformation[1]->getFloatValue(),transformation[2]->getFloatValue());
                transform.setRotation(transformation[3]->getFloatValue(),transformation[4]->getFloatValue(),transformation[5]->getFloatValue());
                transform.setScale(transformation[6]->getFloatValue());
                transform.execute(current_node);

                transformation[0]->setFloatValue(0);
                transformation[1]->setFloatValue(0);
                transformation[2]->setFloatValue(0);

                transformation[3]->setFloatValue(0);
                transformation[4]->setFloatValue(0);
                transformation[5]->setFloatValue(0);

                transformation[6]->setFloatValue(1);

            }
        }

        for (unsigned int index_object=0; index_object < objectGUI.size(); ++index_object)
        {
            //Special Treatment for visual flags
            if( visualContentModified && objectGUI[index_object].second == (QObject*) displayFlag)
            {
                for (unsigned int i=0; i<10; ++i)
                {
                    Data<int> * ff = dynamic_cast< Data<int> * >( objectGUI[index_object].first );
                    ff->setValue(displayFlag->getFlag(i));
                    index_object++;
                }
                index_object--;
            }

            if (setUpdates.find(objectGUI[index_object].first ) == setUpdates.end()) continue;
            //*******************************************************************************************************************
            if( Data<int> * ff = dynamic_cast< Data<int> * >( objectGUI[index_object].first )  )
            {

                if (dynamic_cast< QSpinBox *> ( objectGUI[index_object].second ))
                {

                    QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( objectGUI[index_object].second );
                    ff->setValue(spinBox->value());
                }

            }
            //*******************************************************************************************************************
            else if( Data<unsigned int> * ff = dynamic_cast< Data<unsigned int> * >( objectGUI[index_object].first )  )
            {

                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( objectGUI[index_object].second );

                ff->setValue(spinBox->value());
            }
            //*******************************************************************************************************************
            else if( dynamic_cast< Data<float> * >( objectGUI[index_object].first ) ||
                    dynamic_cast< Data<double> * >( objectGUI[index_object].first ))
            {
                WFloatLineEdit* editSFFloat = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );


                if( Data<float> * ff = dynamic_cast< Data<float> * >( objectGUI[index_object].first )  )
                {
                    ff->setValue(editSFFloat->getFloatValue());
                }
                else if(Data<double> * ff = dynamic_cast< Data<double> * >( objectGUI[index_object].first )  )
                {
                    ff->setValue((double) editSFFloat->getFloatValue());
                }

            }
            //*******************************************************************************************************************
            else if( Data<bool> * ff = dynamic_cast< Data<bool> * >( objectGUI[index_object].first ))
            {
                // the bool line edit
                QCheckBox* checkBox = dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                ff->setValue(checkBox->isOn());
            }
            //*******************************************************************************************************************
            else if( Data<std::string> * ff = dynamic_cast< Data<std::string> * >( objectGUI[index_object].first )  )
            {

                QLineEdit* lineEdit = dynamic_cast< QLineEdit *> ( objectGUI[index_object].second );

                ff->setValue(lineEdit->text().ascii());


                if( !dynamic_cast< Node *>(node) && !strcmp(ff->help,"object name") )
                {
                    std::string name=item->text(0).ascii();
                    std::string::size_type pos = name.find(' ');
                    if (pos != std::string::npos)
                        name.resize(pos);
                    name += "  ";

                    name+=lineEdit->text().ascii();
                    item->setText(0,name.c_str());
                }
                else if (dynamic_cast< Node *>(node))
                    item->setText(0,lineEdit->text().ascii());

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< Data<Vec6f> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< Data<Vec6d> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< Data<Vec<6,int> > * >( objectGUI[index_object].first )   ||
                    dynamic_cast< Data<Vec<6,unsigned int> > * >( objectGUI[index_object].first )   )
            {
                if( Data<Vec6f> * ff = dynamic_cast< Data<Vec6f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec6d> * ff = dynamic_cast< Data<Vec6d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( Data<Vec<6,int> > * ff = dynamic_cast< Data<Vec<6,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec<6, unsigned int> > * ff = dynamic_cast< Data<Vec<6, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
            }
            //*******************************************************************************************************************
            else if(  dynamic_cast< Data<Vec4f> * >( objectGUI[index_object].first )           ||
                    dynamic_cast< Data<Vec4d> * >( objectGUI[index_object].first )           ||
                    dynamic_cast< Data<Vec<4,int> > * >( objectGUI[index_object].first )     ||
                    dynamic_cast< Data<Vec<4,unsigned int> > * >( objectGUI[index_object].first )  )
            {

                if( Data<Vec4f> * ff = dynamic_cast< Data<Vec4f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec4d> * ff = dynamic_cast< Data<Vec4d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( Data<Vec<4,int> > * ff = dynamic_cast< Data<Vec<4,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec<4, unsigned int> > * ff = dynamic_cast< Data<Vec<4, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< Data<Vec3f> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< Data<Vec3d> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< Data<Vec<3,int> > * >( objectGUI[index_object].first )   ||
                    dynamic_cast< Data<Vec<3,unsigned int> > * >( objectGUI[index_object].first )  )
            {

                if( Data<Vec3f> * ff = dynamic_cast< Data<Vec3f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec3d> * ff = dynamic_cast< Data<Vec3d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( Data<Vec<3,int> > * ff = dynamic_cast< Data<Vec<3,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec<3, unsigned int> > * ff = dynamic_cast< Data<Vec<3, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }


            }
            //*******************************************************************************************************************
            else if( dynamic_cast< Data<Vec2f> * >( objectGUI[index_object].first )          ||
                    dynamic_cast< Data<Vec2d> * >( objectGUI[index_object].first )          ||
                    dynamic_cast< Data<Vec<2,int> > * >( objectGUI[index_object].first )    ||
                    dynamic_cast< Data<Vec<2,unsigned int> > * >( objectGUI[index_object].first )  )
            {


                if( Data<Vec2f> * ff = dynamic_cast< Data<Vec2f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec2d> * ff = dynamic_cast< Data<Vec2d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( Data<Vec<2,int> > * ff = dynamic_cast< Data<Vec<2,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec<2, unsigned int> > * ff = dynamic_cast< Data<Vec<2, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< Data<Vec1f> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< Data<Vec1d> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< Data<Vec<1,int> > * >( objectGUI[index_object].first )   ||
                    dynamic_cast< Data<Vec<1,unsigned int> > * >( objectGUI[index_object].first )  )
            {


                if( Data<Vec1f> * ff = dynamic_cast< Data<Vec1f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec1d> * ff = dynamic_cast< Data<Vec1d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( Data<Vec<1,int> > * ff = dynamic_cast< Data<Vec<1,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(Data<Vec<1, unsigned int> > * ff = dynamic_cast< Data<Vec<1, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }

            }
            else if( Data<RigidCoord<3,double> > * ff = dynamic_cast< Data<RigidCoord<3,double> > * >( objectGUI[index_object].first )  )
            {
                RigidCoord<3,double> v;
                storeVector(index_object, &v.getCenter());
                storeVector(index_object, &v.getOrientation());
                ff->setValue(v);
            }
            else if( Data<RigidDeriv<3,double> > * ff = dynamic_cast< Data<RigidDeriv<3,double> > * >( objectGUI[index_object].first )  )
            {
                RigidDeriv<3,double> v;
                storeVector(index_object, &v.getVCenter());
                storeVector(index_object, &v.getVOrientation());
                ff->setValue(v);
            }
            //*******************************************************************************************************************
            else if( Data<RigidMass<3, double> > * ff = dynamic_cast< Data<RigidMass<3, double> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<3, double> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass = (double) mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume = (double) volume->getFloatValue();
                index_object++;
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMassMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                index_object--;
                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( Data<RigidMass<3, float> > * ff = dynamic_cast< Data<RigidMass<3, float> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<3, float> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass =  mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume =  volume->getFloatValue();
                index_object++;
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMassMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                index_object--;
                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( Data<RigidMass<2, double> > * ff = dynamic_cast< Data<RigidMass<2, double> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<2, double> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass = (double) mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume = (double) volume->getFloatValue();
                index_object++;
                WFloatLineEdit* inertiaMatrix = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.inertiaMatrix = (double) inertiaMatrix->getFloatValue();


                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( Data<RigidMass<2, float> > * ff = dynamic_cast< Data<RigidMass<2, float> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<2, float> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass =  mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume =  volume->getFloatValue();
                index_object++;
                WFloatLineEdit* inertiaMatrix = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.inertiaMatrix = (double) inertiaMatrix->getFloatValue();
                index_object++;
                WFloatLineEdit* inertiaMassMatrix = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.inertiaMassMatrix = (double) inertiaMassMatrix->getFloatValue();
                index_object++;

                ff->setValue(current_mass);
            }
            else if (Data<helper::io::Mesh::Material > *ff = dynamic_cast< Data<helper::io::Mesh::Material > * >( objectGUI[index_object].first ))
            {
                helper::io::Mesh::Material M;  QCheckBox* checkBox;    WFloatLineEdit* value;

                //Diffuse
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useDiffuse = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second ); 	M.diffuse[i] = value->getFloatValue(); index_object++;}
                //Ambient
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useAmbient = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second ); 	M.ambient[i] = value->getFloatValue(); index_object++;}
                //Emissive
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useEmissive = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );   M.emissive[i] = value->getFloatValue(); index_object++;}
                //Specular
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useSpecular = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second ); 	M.specular[i] = value->getFloatValue(); index_object++;}
                //Shininess
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useShininess = checkBox->isOn();
                index_object++;
                value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                M.shininess = value->getFloatValue();

                ff->setValue(M);
            }
            //**********************************************************************
            //support for Dataptr
            //*******************************************************************************************************************
            else if( DataPtr<int> * ff = dynamic_cast< DataPtr<int> * >( objectGUI[index_object].first )  )
            {
                if (dynamic_cast< QSpinBox *> ( objectGUI[index_object].second ))
                {

                    QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( objectGUI[index_object].second );
                    ff->setValue(spinBox->value());
                }
                else
                {
                    QCheckBox* checkBox = dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                    ff->setValue(checkBox->isOn());
                }

            }
            //*******************************************************************************************************************
            else if( DataPtr<unsigned int> * ff = dynamic_cast< DataPtr<unsigned int> * >( objectGUI[index_object].first )  )
            {

                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( objectGUI[index_object].second );

                ff->setValue(spinBox->value());
            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataPtr<float> * >( objectGUI[index_object].first ) ||
                    dynamic_cast< DataPtr<double> * >( objectGUI[index_object].first ))
            {
                WFloatLineEdit* editSFFloat = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );


                if( DataPtr<float> * ff = dynamic_cast< DataPtr<float> * >( objectGUI[index_object].first )  )
                {
                    ff->setValue(editSFFloat->getFloatValue());
                }
                else if(DataPtr<double> * ff = dynamic_cast< DataPtr<double> * >( objectGUI[index_object].first )  )
                {
                    ff->setValue((double) editSFFloat->getFloatValue());
                }

            }
            //*******************************************************************************************************************
            else if( DataPtr<bool> * ff = dynamic_cast< DataPtr<bool> * >( objectGUI[index_object].first ))
            {
                // the bool line edit
                QCheckBox* checkBox = dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                ff->setValue(checkBox->isOn());
            }
            //*******************************************************************************************************************
            else if( DataPtr<std::string> * ff = dynamic_cast< DataPtr<std::string> * >( objectGUI[index_object].first )  )
            {

                QLineEdit* lineEdit = dynamic_cast< QLineEdit *> ( objectGUI[index_object].second );

                ff->setValue(lineEdit->text().ascii());


                if( !dynamic_cast< Node *>(node) && !strcmp(ff->help,"object name") )
                {
                    std::string name=item->text(0).ascii();
                    std::string::size_type pos = name.find(' ');
                    if (pos != std::string::npos)
                        name.resize(pos);
                    name += "  ";

                    name+=lineEdit->text().ascii();
                    item->setText(0,name.c_str());
                }
                else if (dynamic_cast< Node *>(node))
                    item->setText(0,lineEdit->text().ascii());

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataPtr<Vec6f> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< DataPtr<Vec6d> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< DataPtr<Vec<6,int> > * >( objectGUI[index_object].first )   ||
                    dynamic_cast< DataPtr<Vec<6,unsigned int> > * >( objectGUI[index_object].first )   )
            {
                if( DataPtr<Vec6f> * ff = dynamic_cast< DataPtr<Vec6f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec6d> * ff = dynamic_cast< DataPtr<Vec6d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( DataPtr<Vec<6,int> > * ff = dynamic_cast< DataPtr<Vec<6,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec<6, unsigned int> > * ff = dynamic_cast< DataPtr<Vec<6, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }


            }
            //*******************************************************************************************************************
            else if(  dynamic_cast< DataPtr<Vec4f> * >( objectGUI[index_object].first )           ||
                    dynamic_cast< DataPtr<Vec4d> * >( objectGUI[index_object].first )           ||
                    dynamic_cast< DataPtr<Vec<4,int> > * >( objectGUI[index_object].first )     ||
                    dynamic_cast< DataPtr<Vec<4,unsigned int> > * >( objectGUI[index_object].first )  )
            {

                if( DataPtr<Vec4f> * ff = dynamic_cast< DataPtr<Vec4f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec4d> * ff = dynamic_cast< DataPtr<Vec4d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( DataPtr<Vec<4,int> > * ff = dynamic_cast< DataPtr<Vec<4,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec<4, unsigned int> > * ff = dynamic_cast< DataPtr<Vec<4, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataPtr<Vec3f> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< DataPtr<Vec3d> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< DataPtr<Vec<3,int> > * >( objectGUI[index_object].first )   ||
                    dynamic_cast< DataPtr<Vec<3,unsigned int> > * >( objectGUI[index_object].first )  )
            {

                if( DataPtr<Vec3f> * ff = dynamic_cast< DataPtr<Vec3f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec3d> * ff = dynamic_cast< DataPtr<Vec3d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( DataPtr<Vec<3,int> > * ff = dynamic_cast< DataPtr<Vec<3,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec<3, unsigned int> > * ff = dynamic_cast< DataPtr<Vec<3, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }


            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataPtr<Vec2f> * >( objectGUI[index_object].first )          ||
                    dynamic_cast< DataPtr<Vec2d> * >( objectGUI[index_object].first )          ||
                    dynamic_cast< DataPtr<Vec<2,int> > * >( objectGUI[index_object].first )    ||
                    dynamic_cast< DataPtr<Vec<2,unsigned int> > * >( objectGUI[index_object].first )  )
            {


                if( DataPtr<Vec2f> * ff = dynamic_cast< DataPtr<Vec2f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec2d> * ff = dynamic_cast< DataPtr<Vec2d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( DataPtr<Vec<2,int> > * ff = dynamic_cast< DataPtr<Vec<2,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec<2, unsigned int> > * ff = dynamic_cast< DataPtr<Vec<2, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }

            }
            //*******************************************************************************************************************
            else if( dynamic_cast< DataPtr<Vec1f> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< DataPtr<Vec1d> * >( objectGUI[index_object].first )         ||
                    dynamic_cast< DataPtr<Vec<1,int> > * >( objectGUI[index_object].first )   ||
                    dynamic_cast< DataPtr<Vec<1,unsigned int> > * >( objectGUI[index_object].first )  )
            {


                if( DataPtr<Vec1f> * ff = dynamic_cast< DataPtr<Vec1f> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec1d> * ff = dynamic_cast< DataPtr<Vec1d> * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if( DataPtr<Vec<1,int> > * ff = dynamic_cast< DataPtr<Vec<1,int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }
                else if(DataPtr<Vec<1, unsigned int> > * ff = dynamic_cast< DataPtr<Vec<1, unsigned int> > * >( objectGUI[index_object].first )  )
                {
                    storeVector(index_object, ff);
                }

            }
            else if( DataPtr<RigidCoord<3,double> > * ff = dynamic_cast< DataPtr<RigidCoord<3,double> > * >( objectGUI[index_object].first )  )
            {
                RigidCoord<3,double> v;
                storeVector(index_object, &v.getCenter());
                storeVector(index_object, &v.getOrientation());
                ff->setValue(v);
            }
            else if( DataPtr<RigidDeriv<3,double> > * ff = dynamic_cast< DataPtr<RigidDeriv<3,double> > * >( objectGUI[index_object].first )  )
            {
                RigidDeriv<3,double> v;
                storeVector(index_object, &v.getVCenter());
                storeVector(index_object, &v.getVOrientation());
                ff->setValue(v);
            }
            //*******************************************************************************************************************
            else if( DataPtr<RigidMass<3, double> > * ff = dynamic_cast< DataPtr<RigidMass<3, double> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<3, double> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass = (double) mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume = (double) volume->getFloatValue();
                index_object++;
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMassMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                index_object--;
                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( DataPtr<RigidMass<3, float> > * ff = dynamic_cast< DataPtr<RigidMass<3, float> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<3, float> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass =  mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume =  volume->getFloatValue();
                index_object++;
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                for (int row=0; row<3; row++)
                {
                    for (int column=0; column<3; column++)
                    {
                        WFloatLineEdit* matrix_element = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                        current_mass.inertiaMassMatrix[row][column] = (double) matrix_element->getFloatValue();
                        index_object++;
                    }
                }
                index_object--;
                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( DataPtr<RigidMass<2, double> > * ff = dynamic_cast< DataPtr<RigidMass<2, double> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<2, double> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass = (double) mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume = (double) volume->getFloatValue();
                index_object++;
                WFloatLineEdit* inertiaMatrix = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.inertiaMatrix = (double) inertiaMatrix->getFloatValue();

                ff->setValue(current_mass);
            }
            //*******************************************************************************************************************
            else if( DataPtr<RigidMass<2, float> > * ff = dynamic_cast< DataPtr<RigidMass<2, float> > * >( objectGUI[index_object].first )  )
            {
                RigidMass<2, float> current_mass = ff->getValue();

                WFloatLineEdit* mass = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.mass =  mass->getFloatValue();
                index_object++;
                WFloatLineEdit* volume = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.volume =  volume->getFloatValue();
                index_object++;
                WFloatLineEdit* inertiaMatrix = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.inertiaMatrix = (double) inertiaMatrix->getFloatValue();
                index_object++;
                WFloatLineEdit* inertiaMassMatrix = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                current_mass.inertiaMassMatrix = (double) inertiaMassMatrix->getFloatValue();

                ff->setValue(current_mass);
            }
            else if (DataPtr<helper::io::Mesh::Material > *ff = dynamic_cast< DataPtr<helper::io::Mesh::Material > * >( objectGUI[index_object].first ))
            {
                helper::io::Mesh::Material M;  QCheckBox* checkBox;    WFloatLineEdit* value;

                //Diffuse
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useDiffuse = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second ); 	M.diffuse[i] = value->getFloatValue(); index_object++;}
                //Ambient
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useAmbient = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second ); 	M.ambient[i] = value->getFloatValue(); index_object++;}
                //Emissive
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useEmissive = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );   M.emissive[i] = value->getFloatValue(); index_object++;}
                //Specular
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useSpecular = checkBox->isOn();
                index_object++;
                for (int i=0; i<4; i++) { value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second ); 	M.specular[i] = value->getFloatValue(); index_object++;}
                //Shininess
                checkBox= dynamic_cast< QCheckBox *> ( objectGUI[index_object].second );
                M.useShininess = checkBox->isOn();
                index_object++;
                value = dynamic_cast< WFloatLineEdit *> ( objectGUI[index_object].second );
                M.shininess = value->getFloatValue();

                ff->setValue(M);
            }
        }
        if (REINIT_FLAG)
        {
            if (sofa::core::objectmodel::BaseObject *obj = dynamic_cast< sofa::core::objectmodel::BaseObject* >(node))
                obj->reinit();
            else if (Node *n = dynamic_cast< Node *>(node)) n->reinit();
        }
    }

    if (visualContentModified) updateContext(dynamic_cast< Node *>(node));

    updateTables();
    emit (objectUpdated());
    buttonUpdate->setEnabled(false);
    visualContentModified = false;


    setUpdates.clear();
}


//*******************************************************************************************************************
//Update the Context of a whole node, including its childs
void ModifyObject::updateContext( Node *node )
{
    if (node == NULL) return;
    node->execute< sofa::simulation::UpdateVisualContextVisitor >();
}


void  ModifyObject::createGraphMass(QTabWidget *dialogTab)
{
    QWidget *tabMassStats = new QWidget(); dialogTab->addTab(tabMassStats, QString("Energy Stats"));
    QVBoxLayout *tabMassStatsLayout = new QVBoxLayout( tabMassStats, 0, 1, "tabMassStats");


#ifdef SOFA_QT4
    graphEnergy = new QwtPlot(QwtText("Energy"),tabMassStats);
#else
    graphEnergy = new QwtPlot(tabMassStats,"Energy");
#endif
    history.clear();
    energy_history[0].clear();
    energy_history[1].clear();
    energy_history[2].clear();

    energy_curve[0] = new QwtPlotCurve("Kinetic");	        energy_curve[0]->attach(graphEnergy);
    energy_curve[1] = new QwtPlotCurve("Potential");	energy_curve[1]->attach(graphEnergy);
    energy_curve[2] = new QwtPlotCurve("Mechanical");	energy_curve[2]->attach(graphEnergy);

    energy_curve[0]->setPen(QPen(Qt::red));
    energy_curve[1]->setPen(QPen(Qt::green));
    energy_curve[2]->setPen(QPen(Qt::blue));

    graphEnergy->setAxisTitle(QwtPlot::xBottom, "Time/seconds");
    graphEnergy->setTitle("Energy Graph");
    graphEnergy->insertLegend(new QwtLegend(), QwtPlot::BottomLegend);

    tabMassStatsLayout->addWidget(graphEnergy);
}

void ModifyObject::updateHistory()
{
    if (Node *gnode = dynamic_cast<Node *>(node))
    {
        if ( gnode->mass || gnode->forceField.size() != 0)
        {
            history.push_back(gnode->getTime());
            updateEnergy();
        }
    }
}

void ModifyObject::updateEnergy()
{

    Node *gnode = dynamic_cast<Node *>(node);

    unsigned int index = energy_history[0].size();
    if (gnode->mass)
        energy_history[0].push_back(gnode->mass->getKineticEnergy());
    else
        energy_history[0].push_back(0);

    if (gnode->forceField.size() != 0)
        energy_history[1].push_back(gnode->forceField[0]->getPotentialEnergy());
    else
        energy_history[1].push_back(0);

    energy_history[2].push_back(energy_history[0][index] + energy_history[1][index]);

    if (dialogTab->currentPageIndex() == dialogTab->count()-2)
    {
        energy_curve[0]->setRawData(&history[0],&(energy_history[0][0]), history.size());
        energy_curve[1]->setRawData(&history[0],&(energy_history[1][0]), history.size());
        energy_curve[2]->setRawData(&history[0],&(energy_history[2][0]), history.size());
        graphEnergy->replot();
    }

}




void ModifyObject::updateTextEdit()
{
    std::list< std::pair< Q3TextEdit*, BaseData*> >::iterator it_list_TextEdit;
    for (it_list_TextEdit = list_TextEdit.begin(); it_list_TextEdit != list_TextEdit.end(); it_list_TextEdit++)
    {
        if ((dataIndexTab.find((*it_list_TextEdit).second))->second == dialogTab->currentPageIndex())
        {
            (*it_list_TextEdit).first->setText( QString((*it_list_TextEdit).second->getValueString().c_str()));
        }
    }
}
//**************************************************************************************************************************************
//Called each time a new step of the simulation if computed
void ModifyObject::updateTables()
{
    updateHistory();
    updateTextEdit();
    std::list< std::pair< Q3Table*, BaseData*> >::iterator it_list_Table;
    bool skip;
    for (it_list_Table = list_Table.begin(); it_list_Table != list_Table.end(); it_list_Table++)
    {
        skip = false;

        if ((dataIndexTab.find((*it_list_Table).second))->second != dialogTab->currentPageIndex() ) skip = true;
        if ( dynamic_cast<DataPtr< vector<Rigid3dTypes::Coord> > *> ( (*it_list_Table).second ) || dynamic_cast<DataPtr< vector<Rigid3fTypes::Coord> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<DataPtr< vector<Rigid3dTypes::Deriv> > *> ( (*it_list_Table).second ) || dynamic_cast<DataPtr< vector<Rigid3fTypes::Deriv> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<DataPtr< vector<Rigid2dTypes::Coord> > *> ( (*it_list_Table).second ) || dynamic_cast<DataPtr< vector<Rigid2fTypes::Coord> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<DataPtr< vector<Rigid2dTypes::Deriv> > *> ( (*it_list_Table).second ) || dynamic_cast<DataPtr< vector<Rigid2fTypes::Deriv> > *> ( (*it_list_Table).second ) ||
                dynamic_cast<Data< vector<Rigid3dTypes::Coord> > *> ( (*it_list_Table).second )    || dynamic_cast<Data< vector<Rigid3fTypes::Coord> > *> ( (*it_list_Table).second )    ||
                dynamic_cast<Data< vector<Rigid3dTypes::Deriv> > *> ( (*it_list_Table).second )    || dynamic_cast<Data< vector<Rigid3fTypes::Deriv> > *> ( (*it_list_Table).second )    ||
                dynamic_cast<Data< vector<Rigid2dTypes::Coord> > *> ( (*it_list_Table).second )    || dynamic_cast<Data< vector<Rigid2fTypes::Coord> > *> ( (*it_list_Table).second )    ||
                dynamic_cast<Data< vector<Rigid2dTypes::Deriv> > *> ( (*it_list_Table).second )    || dynamic_cast<Data< vector<Rigid2fTypes::Deriv> > *> ( (*it_list_Table).second )          )
        {
            std::list< std::pair< Q3Table*, BaseData*> >::iterator it_center = it_list_Table;
            it_list_Table++;
            if (!skip) createTable((*it_list_Table).second,NULL,(*it_center).first, (*it_list_Table).first);
        }
        else if ( dynamic_cast < Data<sofa::component::misc::Monitor< Vec3Types >::MonitorData > *> ( (*it_list_Table).second ) )
        {
            std::list< std::pair< Q3Table*, BaseData*> >::iterator it_center = it_list_Table;
            it_list_Table++;
            std::list< std::pair< Q3Table*, BaseData*> >::iterator it_center2 = it_list_Table;
            it_list_Table++; //two times because a monitor is composed of 3 baseData
            if (!skip)
                createTable((*it_list_Table).second,NULL,(*it_center).first,(*it_center2).first, (*it_list_Table).first);
        }
        else if (!skip) createTable((*it_list_Table).second,NULL,(*it_list_Table).first);
    }
}


//**************************************************************************************************************************************
//create or update an existing table with the contents of a field
bool ModifyObject::createTable( BaseData* field,Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2, Q3Table* vectorTable3)
{
    //********************************************************************************************************//
    //vector< Vec3f >
    if (Data< vector< Vec3f > >  *ff = dynamic_cast< Data< vector< Vec3f > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec3d >
    else if ( Data< vector< Vec3d> >   *ff = dynamic_cast< Data< vector< Vec3d > >   * >( field ) )
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec<3, int> >
    else if( Data< vector< Vec< 3, int> > >  *ff = dynamic_cast< Data< vector< Vec< 3, int> > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec<3, int> >
    else if( Data< vector< Vec< 3, unsigned int> > >  *ff = dynamic_cast< Data< vector< Vec< 3, unsigned int> > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec3f >
    else if (DataPtr< vector< Vec3f > >  *ff = dynamic_cast< DataPtr< vector< Vec3f > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec3d >
    else if ( DataPtr< vector< Vec3d> >   *ff = dynamic_cast< DataPtr< vector< Vec3d > >   * >( field ) )
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec<3, int> >
    else if( DataPtr< vector< Vec< 3, int> > >  *ff = dynamic_cast< DataPtr< vector< Vec< 3, int> > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 3, int > >
    else if( DataPtr< vector< Vec< 3, int> > >  *ff = dynamic_cast< DataPtr< vector< Vec< 3, int> > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 3, unsigned int > >
    else if( DataPtr< vector< Vec< 3, unsigned int> > >  *ff = dynamic_cast< DataPtr< vector< Vec< 3, unsigned int> > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec2f >
    else if (Data< vector< Vec2f > >  *ff = dynamic_cast< Data< vector< Vec2f > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec2d >
    else if ( Data< vector< Vec2d> >   *ff = dynamic_cast< Data< vector< Vec2d > >   * >( field ) )
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 2, int> >
    else if(  Data< vector< Vec< 2, int> > >   *ff = dynamic_cast< Data< vector< Vec< 2, int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 2, unsigned int> >
    else if(  Data< vector< Vec< 2, unsigned int> > >   *ff = dynamic_cast< Data< vector< Vec< 2, unsigned  int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec2f >
    else if (DataPtr< vector< Vec2f > >  *ff = dynamic_cast< DataPtr< vector< Vec2f > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec2d >
    else if ( DataPtr< vector< Vec2d> >   *ff = dynamic_cast< DataPtr< vector< Vec2d > >   * >( field ) )
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 2, int> >
    else if(  DataPtr< vector< Vec< 2, int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 2, int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 2, unsigned int> >
    else if(  DataPtr< vector< Vec< 2, unsigned int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 2, unsigned int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec1f >
    else if (Data< vector< Vec1f > >  *ff = dynamic_cast< Data< vector< Vec1f > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec1d >
    else if ( Data< vector< Vec1d> >   *ff = dynamic_cast< Data< vector< Vec1d > >   * >( field ) )
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 1, int> >
    else if(  Data< vector< Vec< 1, int> > >   *ff = dynamic_cast< Data< vector< Vec< 1, int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 1, unsigned int> >
    else if(  Data< vector< Vec< 1, unsigned int> > >   *ff = dynamic_cast< Data< vector< Vec< 1, unsigned  int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec1f >
    else if (DataPtr< vector< Vec1f > >  *ff = dynamic_cast< DataPtr< vector< Vec1f > >  * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec1d >
    else if ( DataPtr< vector< Vec1d> >   *ff = dynamic_cast< DataPtr< vector< Vec1d > >   * >( field ) )
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 1, int> >
    else if(  DataPtr< vector< Vec< 1, int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 1, int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< Vec< 1, unsigned int> >
    else if(  DataPtr< vector< Vec< 1, unsigned int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 1, unsigned int> > >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< double >
    else if(  Data< vector< double > >   *ff = dynamic_cast< Data< vector< double> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< float >
    else if(  Data< vector< float > >   *ff = dynamic_cast< Data< vector< float> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< int >
    else if(  Data< vector< int > >   *ff = dynamic_cast< Data< vector< int> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< unsigned int >
    else if(  Data< vector< unsigned int > >   *ff = dynamic_cast< Data< vector< unsigned int> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< std::string >
    else if(  Data< vector< std::string > >   *ff = dynamic_cast< Data< vector< std::string> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< double >
    else if(  DataPtr< vector< double > >   *ff = dynamic_cast< DataPtr< vector< double> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< float >
    else if(  DataPtr< vector< float > >   *ff = dynamic_cast< DataPtr< vector< float> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< int >
    else if(  DataPtr< vector< int > >   *ff = dynamic_cast< DataPtr< vector< int> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< unsigned int >
    else if(  DataPtr< vector< unsigned int > >   *ff = dynamic_cast< DataPtr< vector< unsigned int> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< std::string >
    else if(  DataPtr< vector< std::string > >   *ff = dynamic_cast< DataPtr< vector< std::string> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }

    //********************************************************************************************************//
    //fixed_array3 double
    else if(  Data<vector<fixed_array<double,3> > > *ff = dynamic_cast< Data<vector<fixed_array<double,3> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array3 float
    else if(  Data<vector<fixed_array<float,3> > > *ff = dynamic_cast< Data<vector<fixed_array<float,3> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array1
    else if(  Data<vector<fixed_array<unsigned int,1> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,1> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array2
    else if(  Data<vector<fixed_array<unsigned int,2> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,2> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array3
    else if(  Data<vector<fixed_array<unsigned int,3> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,3> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array4
    else if(  Data<vector<fixed_array<unsigned int,4> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,4> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array8
    else if(  Data<vector<fixed_array<unsigned int,8> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,8> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //fixed_array3 double
    else if(  DataPtr<vector<fixed_array<double,3> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<double,3> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array3 float
    else if(  DataPtr<vector<fixed_array<float,3> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<float,3> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array1
    else if(  DataPtr<vector<fixed_array<unsigned int,1> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,1> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array2
    else if(  DataPtr<vector<fixed_array<unsigned int,2> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,2> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array3
    else if(  DataPtr<vector<fixed_array<unsigned int,3> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,3> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array4
    else if(  DataPtr<vector<fixed_array<unsigned int,4> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,4> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //fixed_array8
    else if(  DataPtr<vector<fixed_array<unsigned int,8> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,8> > > * >  (field))
    {
        return createQtTable(  ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< double >
    else if(  Data<sofa::component::topology::PointData< double > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< double> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< float >
    else if(  Data<sofa::component::topology::PointData< float > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< float> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< int >
    else if(  Data<sofa::component::topology::PointData< int > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< int> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector< unsigned int >
    else if(  Data<sofa::component::topology::PointData< unsigned int > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< unsigned int> >   * >( field))
    {
        return createQtTable(ff,box,vectorTable);
    }

    //********************************************************************************************************//

    else if(  Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData >   * >( field))
    {
        return createMonitorQtTable < Vec3dTypes >(ff,box,vectorTable, vectorTable2, vectorTable3);
    }

    //********************************************************************************************************//
    else if(  Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData >   * >( field))
    {
        return createMonitorQtTable < Vec3fTypes >(ff,box,vectorTable, vectorTable2, vectorTable3);
    }
    //********************************************************************************************************//
    //vector<Rigid3dTypes::Coord>
    else if (DataPtr< vector<Rigid3fTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid3fTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Deriv> && vector<Rigid3ffTypes::Deriv>
    else if (DataPtr< vector<Rigid3fTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid3fTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Coord>
    else if (Data< vector<Rigid3fTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid3fTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Deriv> && vector<Rigid3ffTypes::Deriv>
    else if (Data< vector<Rigid3fTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid3fTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid2fTypes::Coord>
    else if (DataPtr< vector<Rigid2fTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid2fTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid2dfTypes::Deriv>
    else if (DataPtr< vector<Rigid2fTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid2fTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid2fTypes::Coord>
    else if (Data< vector<Rigid2fTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid2fTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid2dfTypes::Deriv>
    else if (Data< vector<Rigid2fTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid2fTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }

    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Coord>
    else if (DataPtr< vector<Rigid3dTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid3dTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Deriv> && vector<Rigid3fdTypes::Deriv>
    else if (DataPtr< vector<Rigid3dTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid3dTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Coord>
    else if (Data< vector<Rigid3dTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid3dTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Deriv> && vector<Rigid3fdTypes::Deriv>
    else if (Data< vector<Rigid3dTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid3dTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid2dTypes::Coord>
    else if (DataPtr< vector<Rigid2dTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid2dTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid2ddTypes::Deriv>
    else if (DataPtr< vector<Rigid2dTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid2dTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //vector<Rigid2dTypes::Coord>
    else if (Data< vector<Rigid2dTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid2dTypes::Coord>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    //********************************************************************************************************//
    //PointSubset
    else if (Data< vector<Rigid2dTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid2dTypes::Deriv>  >   * >( field ))
    {
        return createQtTable(ff,box,vectorTable, vectorTable2);
    }
    else if (Data< PointSubset > *ff = dynamic_cast< Data< PointSubset > *>(field))
    {
        PointSubset new_value;

        for (int i=0; i<vectorTable->numRows(); ++i)
        {
            QString value = vectorTable->text(i,0);
            if (value.isEmpty())
            {
                new_value.push_back(0);
                vectorTable->setText(i,0,QString("0"));
            }
            else
                new_value.push_back( atoi(value) );
        }
        ff->setValue( new_value );
        changeValue();
    }
    //********************************************************************************************************//
    else if (DataPtr< PointSubset > *ff = dynamic_cast< DataPtr< PointSubset > *>(field))
    {
        PointSubset new_value;

        for (int i=0; i<vectorTable->numRows(); ++i)
        {
            QString value = vectorTable->text(i,0);
            if (value.isEmpty())
            {
                new_value.push_back(0);
                vectorTable->setText(i,0,QString("0"));
            }
            else
                new_value.push_back( atoi(value) );
        }
        ff->setValue( new_value );
        changeValue();
    }
    //********************************************************************************************************//
    //vector<LaparoscopicRigid3Types::Coord>
    else if (DataPtr< vector<LaparoscopicRigid3Types::Coord> >  *ff = dynamic_cast< DataPtr< vector<LaparoscopicRigid3Types::Coord>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0 && !EMPTY_FLAG)  return false;
            box->setColumns(1);
            new QLabel("Translation", box);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            vectorTable->setReadOnly(false);

            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	    vectorTable->setColumnStretchable(0,true);

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
        vector<LaparoscopicRigid3Types::Coord> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;

            {
                oss << value[i].getTranslation();
                vectorTable->setText(i,0,std::string(oss.str()).c_str());
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
    //vector<LaparoscopicRigid3Types::Deriv>
    else if (DataPtr< vector<LaparoscopicRigid3Types::Deriv> >  *ff = dynamic_cast< DataPtr< vector<LaparoscopicRigid3Types::Deriv>  >   * >( field ))
    {

        if (!vectorTable)
        {
            if (ff->getValue().size() == 0 && !EMPTY_FLAG)  return false;
            box->setColumns(1);
            new QLabel("Translation", box);

            vectorTable = new Q3Table(ff->getValue().size(),1, box);
            vectorTable->setReadOnly(false);

            list_Table.push_back(std::make_pair(vectorTable, field));
            vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);

            connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

            new QLabel("Orientation", box);

            vectorTable2 = new Q3Table(ff->getValue().size(),3, box);
            list_Table.push_back(std::make_pair(vectorTable2, field));
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	 vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);

            connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        }
        vector<LaparoscopicRigid3Types::Deriv> value = ff->getValue();


        for (unsigned int i=0; i<ff->getValue().size(); i++)
        {
            std::ostringstream oss;

            {
                oss << value[i].getVTranslation();
                vectorTable->setText(i,0,std::string(oss.str()).c_str());
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
#ifndef WIN32
    //********************************************************************************************************//
    //vector<SpringForceField< Vec6dTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<6,double>(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec6fTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<6,float>(ff,box,vectorTable);
    }

    //********************************************************************************************************//
    //vector<SpringForceField< Vec3dTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<3,double>(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec3fTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<3,float>(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec2dTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<2,double>(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec2fTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<2,float>(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec1dTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<1,double>(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector<SpringForceField< Vec1fTypes >::Spring>
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  * >( field ))
    {
        return createQtSpringTable<1,float>(ff,box,vectorTable);
    }
    //********************************************************************************************************//
    //vector<JointSpringForceField< Rigid3dTypes >::Spring>
    else if (Data< vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  *ff = dynamic_cast< Data< vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  * >( field ))
    {
        return createQtRigidSpringTable<double>(ff,box,vectorTable);
    }

    //********************************************************************************************************//
    //vector<JointSpringForceField< Rigid3fTypes >::Spring>
    else if (Data< vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  *ff = dynamic_cast< Data< vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  * >( field ))
    {
        return createQtRigidSpringTable<float>(ff,box,vectorTable);
    }
#endif

    return false;
}




//**************************************************************************************************************************************
//save in datafield the values of the text edit
void ModifyObject::saveTextEdit()
{

    std::list< std::pair< Q3TextEdit*, core::objectmodel::BaseData*> >::iterator it_list_TextEdit;
    for (it_list_TextEdit = list_TextEdit.begin(); it_list_TextEdit != list_TextEdit.end(); it_list_TextEdit++)
    {
        std::string value = it_list_TextEdit->first->text().ascii();
        it_list_TextEdit->second->read(value);
    }
}


//**************************************************************************************************************************************
//save in datafield the values of the tables
void ModifyObject::saveTables()
{

    std::list< std::pair< Q3Table*, BaseData*> >::iterator it_list_Table;
    for (it_list_Table = list_Table.begin(); it_list_Table != list_Table.end(); it_list_Table++)
    {

        if (setUpdates.find(getData(it_list_Table->first)) == setUpdates.end()) continue;
        storeTable(it_list_Table);
    }
}


//**************************************************************************************************************************************
//Read the content of a QTable and store its values in a datafield
void ModifyObject::storeTable(std::list< std::pair< Q3Table*, BaseData*> >::
        iterator &it_list_table)
{
    ///////////////////////////////////////////
    //it_list_table->first is a Q3Table* table
    //it_list_table->second is a BaseData* field
    ///////////////////////////////////////////

    //**************************************************************************************************************************************
    if (Data< vector< Vec<3,float> > >  *ff = dynamic_cast< Data< vector< Vec<3,float> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( Data< vector< Vec<3,double> > >   *ff = dynamic_cast< Data< vector< Vec<3,double> > >   * >( it_list_table->second ) )
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if( Data< vector< Vec< 3, int> > >  *ff = dynamic_cast< Data< vector< Vec< 3, int> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if( Data< vector< Vec< 3, unsigned int> > >   *ff = dynamic_cast< Data< vector< Vec< 3, unsigned int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( DataPtr< vector< Vec<3,float> > >  *ff = dynamic_cast< DataPtr< vector< Vec<3,float> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( DataPtr< vector< Vec<3,double> > >   *ff = dynamic_cast< DataPtr< vector< Vec<3,double> > >   * >( it_list_table->second ) )
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if( DataPtr< vector< Vec< 3, int> > >  *ff = dynamic_cast< DataPtr< vector< Vec< 3, int> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< Vec< 3, unsigned int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 3, unsigned int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( Data< vector< Vec<2,float> > >  *ff = dynamic_cast< Data< vector< Vec<2,float> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( Data< vector< Vec<2,double> > >   *ff = dynamic_cast< Data< vector< Vec<2,double> > >   * >( it_list_table->second ) )
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if( Data< vector< Vec< 2, int> > >   *ff = dynamic_cast< Data< vector< Vec< 2, int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if( Data< vector< Vec< 2, unsigned int> > >   *ff = dynamic_cast< Data< vector< Vec< 2, unsigned int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (DataPtr< vector< Vec<2,float> > >  *ff = dynamic_cast< DataPtr< vector< Vec<2,float> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( DataPtr< vector< Vec<2,double> > >   *ff = dynamic_cast< DataPtr< vector< Vec<2,double> > >   * >( it_list_table->second ) )
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< Vec< 2, int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 2, int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< Vec< 2, unsigned int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 2, unsigned int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( Data< vector< Vec<1,float> > >  *ff = dynamic_cast< Data< vector< Vec<1,float> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( Data< vector< Vec<1,double> > >   *ff = dynamic_cast< Data< vector< Vec<1,double> > >   * >( it_list_table->second ) )
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if( Data< vector< Vec< 1, int> > >   *ff = dynamic_cast< Data< vector< Vec< 1, int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if( Data< vector< Vec< 1, unsigned int> > >   *ff = dynamic_cast< Data< vector< Vec< 1, unsigned int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (DataPtr< vector< Vec<1,float> > >  *ff = dynamic_cast< DataPtr< vector< Vec<1,float> > >  * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if ( DataPtr< vector< Vec<1,double> > >   *ff = dynamic_cast< DataPtr< vector< Vec<1,double> > >   * >( it_list_table->second ) )
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< Vec< 1, int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 1, int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< Vec< 1, unsigned int> > >   *ff = dynamic_cast< DataPtr< vector< Vec< 1, unsigned int> > >   * >( it_list_table->second ))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data< vector< double > >   *ff = dynamic_cast< Data< vector< double> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data< vector< float > >   *ff = dynamic_cast< Data< vector< float> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data< vector< int > >   *ff = dynamic_cast< Data< vector< int> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data< vector< std::string > >   *ff = dynamic_cast< Data< vector< std::string > >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data< vector< unsigned int > >   *ff = dynamic_cast< Data< vector< unsigned int> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< double > >   *ff = dynamic_cast< DataPtr< vector< double> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< float > >   *ff = dynamic_cast< DataPtr< vector< float> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< int > >   *ff = dynamic_cast< DataPtr< vector< int> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  DataPtr< vector< unsigned int > >   *ff = dynamic_cast< DataPtr< vector< unsigned int> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //********************************************************************************************************//
    //fixed_array3 double
    else if(  Data<vector<fixed_array<double,3> > > *ff = dynamic_cast< Data<vector<fixed_array<double,3> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array3 float
    else if(  Data<vector<fixed_array<float,3> > > *ff = dynamic_cast< Data<vector<fixed_array<float,3> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array1
    else if(  Data<vector<fixed_array<unsigned int,1> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,1> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array2
    else if(  Data<vector<fixed_array<unsigned int,2> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,2> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array3
    else if(  Data<vector<fixed_array<unsigned int,3> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,3> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array4
    else if(  Data<vector<fixed_array<unsigned int,4> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,4> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array8
    else if(  Data<vector<fixed_array<unsigned int,8> > > *ff = dynamic_cast< Data<vector<fixed_array<unsigned int,8> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //********************************************************************************************************//
    //fixed_array3 double
    else if(  DataPtr<vector<fixed_array<double,3> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<double,3> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array3 float
    else if(  DataPtr<vector<fixed_array<float,3> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<float,3> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array1
    else if(  DataPtr<vector<fixed_array<unsigned int,1> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,1> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array2
    else if(  DataPtr<vector<fixed_array<unsigned int,2> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,2> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array3
    else if(  DataPtr<vector<fixed_array<unsigned int,3> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,3> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array4
    else if(  DataPtr<vector<fixed_array<unsigned int,4> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,4> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //fixed_array8
    else if(  DataPtr<vector<fixed_array<unsigned int,8> > > *ff = dynamic_cast< DataPtr<vector<fixed_array<unsigned int,8> > > * >  (it_list_table->second))
    {
        storeQtTable( it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data<sofa::component::topology::PointData< double > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< double> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data<sofa::component::topology::PointData< float > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< float> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data<sofa::component::topology::PointData< int > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< int> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if(  Data<sofa::component::topology::PointData< unsigned int > >   *ff = dynamic_cast< Data<sofa::component::topology::PointData< unsigned int> >   * >( it_list_table->second))
    {
        storeQtTable(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Coord>
    else if (Data< vector<Rigid3fTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid3fTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Coord>
    else if (Data< vector<Rigid2fTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid2fTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Coord>
    else if (DataPtr< vector<Rigid3fTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid3fTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Coord>
    else if (DataPtr< vector<Rigid2fTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid2fTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Deriv>
    else if (Data< vector<Rigid3fTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid3fTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Deriv>
    else if (Data< vector<Rigid2fTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid2fTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Deriv>
    else if (DataPtr< vector<Rigid3fTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid3fTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3dfTypes::Deriv>
    else if (DataPtr< vector<Rigid2fTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid2fTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Coord>
    else if (Data< vector<Rigid3dTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid3dTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Coord>
    else if (Data< vector<Rigid2dTypes::Coord> >  *ff = dynamic_cast< Data< vector<Rigid2dTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Coord>
    else if (DataPtr< vector<Rigid3dTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid3dTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Coord>
    else if (DataPtr< vector<Rigid2dTypes::Coord> >  *ff = dynamic_cast< DataPtr< vector<Rigid2dTypes::Coord>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Deriv>
    else if (Data< vector<Rigid3dTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid3dTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Deriv>
    else if (Data< vector<Rigid2dTypes::Deriv> >  *ff = dynamic_cast< Data< vector<Rigid2dTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Deriv>
    else if (DataPtr< vector<Rigid3dTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid3dTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid3Table(it_list_table, ff);
    }
    //********************************************************************************************************//
    //vector<Rigid3ddTypes::Deriv>
    else if (DataPtr< vector<Rigid2dTypes::Deriv> >  *ff = dynamic_cast< DataPtr< vector<Rigid2dTypes::Deriv>  >   * >( it_list_table->second ))
    {
        storeQtRigid2Table(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (  Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData > * >( it_list_table->second))
    {
        storeMonitorQtTable< Vec3dTypes >(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (  Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData > * >( it_list_table->second))
    {
        storeMonitorQtTable< Vec3fTypes >(it_list_table, ff);
    }
    //**************************************************************************************************************************************
#ifndef WIN32
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6dTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<6,double>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec6fTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<6,float>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3dTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<3,double>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec3fTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<3,float>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2dTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<2,double>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec2fTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<2,float>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1dTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<1,double>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::SpringForceField< Vec1fTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtSpringTable<1,float>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::JointSpringForceField< Rigid3dTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtRigidSpringTable<double>(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (Data< sofa::helper::vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  *ff = dynamic_cast< Data< sofa::helper::vector<sofa::component::forcefield::JointSpringForceField< Rigid3fTypes>::Spring > >  * >( it_list_table->second ))
    {
        storeQtRigidSpringTable<float>(it_list_table, ff);
    }
#endif
}

//********************************************************************************************************************
void ModifyObject::createVector(   core::objectmodel::BaseData* object,const Quater<double> &value, Q3GroupBox *box)
{
    Vec<4,double> new_value(value[0], value[1], value[2], value[3]);
    createVector( object,new_value, box);
}
//********************************************************************************************************************
void ModifyObject::createVector(   core::objectmodel::BaseData* object, const Quater<float> &value, Q3GroupBox *box)
{
    Vec<4,float> new_value(value[0], value[1], value[2], value[3]);
    createVector( object ,new_value, box);
}


//********************************************************************************************************************
//TEMPLATE FUNCTIONS
//********************************************************************************************************************
template< int N, class T>
void ModifyObject::createVector( core::objectmodel::BaseData* object,const Vec<N,T> &value, Q3GroupBox *box)
{
    box->setColumns(N+1);
    WFloatLineEdit* editSFFloat[N];
    for (int i=0; i<N; i++)
    {
        std::ostringstream oss;
        oss << "editSFFloat_" << i;
        editSFFloat[i] = new WFloatLineEdit( box, QString(std::string(oss.str()).c_str())   );
        objectGUI.push_back(std::make_pair( object,  (QObject *) editSFFloat[i]));

        editSFFloat[i]->setMinFloatValue( (float)-INFINITY );
        editSFFloat[i]->setMaxFloatValue( (float)INFINITY );
    }
    for (int i=0; i<N; i++)
    {
        editSFFloat[i]->setFloatValue((float)value[i]); readOnlyData(editSFFloat[i],object);
    }
    for (int i=0; i<N; i++)
    {
        connect( editSFFloat[i], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    }
}
//********************************************************************************************************************
template< int N, class T>
void ModifyObject::storeVector(unsigned int &index, Data< Vec<N,T> > *ff)
{
    WFloatLineEdit* editSFFloat[N];
    for (int i=0; i<N; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( objectGUI[index].second ); index++;
    }
    index--;
    Vec<N, T> value;
    for (int i=0; i<N; i++)  value[i] =  (T) editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}


template<class T>
void ModifyObject::storeVector(unsigned int &index, Data< Quater<T> > *ff)
{
    WFloatLineEdit* editSFFloat[4];
    for (int i=0; i<4; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( objectGUI[index].second ); index++;
    }
    index--;
    Quater<T> value;
    for (int i=0; i<4; i++)  value[i] =  (T) editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}

template< int N, class T>
void ModifyObject::storeVector(unsigned int &index, DataPtr< Vec<N,T> > *ff)
{
    WFloatLineEdit* editSFFloat[N];
    for (int i=0; i<N; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( objectGUI[index].second ); index++;
    }
    index--;

    Vec<N, T> value;
    for (int i=0; i<N; i++)  value[i] =  (T) editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}
template<class T>
void ModifyObject::storeVector(unsigned int &index, DataPtr< Quater<T> > *ff)
{
    WFloatLineEdit* editSFFloat[4];
    for (int i=0; i<4; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( objectGUI[index].second ); index++;
    }
    index--;
    Quater<T> value;
    for (int i=0; i<4; i++)  value[i] =  (T) editSFFloat[i]->getFloatValue();
    ff->setValue(value);
}

template< int N, class T>
void ModifyObject::storeVector(unsigned int &index, Vec<N,T> *ff)
{
    WFloatLineEdit* editSFFloat[N];
    for (int i=0; i<N; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( objectGUI[index].second ); index++;
    }
    index--;
    Vec<N, T> value;
    for (int i=0; i<N; i++)  value[i] =  (T) editSFFloat[i]->getFloatValue();
    *ff = value;
}
template<class T>
void ModifyObject::storeVector(unsigned int &index, Quater<T> *ff)
{
    WFloatLineEdit* editSFFloat[4];
    for (int i=0; i<4; i++)
    {
        editSFFloat[i] = dynamic_cast< WFloatLineEdit *> ( objectGUI[index].second ); index++;
    }
    index--;
    Quater<T> value;
    for (int i=0; i<4; i++)  value[i] =  (T) editSFFloat[i]->getFloatValue();
    *ff = value;
}
//********************************************************************************************************************
//********************************************************************************************************************
template< class T>
bool ModifyObject::createQtTable(Data< sofa::helper::vector< T > > *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {

        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;

        box->setColumns(1);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),1);
        else vectorTable = new Q3Table(ff->getValue().size(),1, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));
        vectorTable->horizontalHeader()->setLabel(0,QString("Value"));
        vectorTable->setColumnStretchable(0,true);

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }

    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());


    vector< T > value = ff->getValue();


    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss;
        oss << value[i];
        vectorTable->setText(i,0,std::string(oss.str()).c_str());
    }
    readOnlyData(vectorTable,ff);
    return true;
}
//********************************************************************************************************************

void ModifyObject::storeQtTable(std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< std::string > >* ff )
{
    Q3Table* table=it_list_table->first;

    QString vec_value;
    vector< std::string > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        vec_value = table->text(i,0);
        new_value.push_back( (std::string)(vec_value.ascii()) );
    }
    ff->setValue( new_value );
}
//********************************************************************************************************************
template<class T>
void ModifyObject::storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< T > >* ff )
{

    Q3Table* table=it_list_table->first;

    QString vec_value;
    vector< T > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        vec_value = table->text(i,0);
        new_value.push_back( (T)atof(vec_value) );
    }
    ff->setValue( new_value );
}
//********************************************************************************************************************
//********************************************************************************************************************
template< class T>
bool ModifyObject::createQtTable(DataPtr< sofa::helper::vector< T > > *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;

        box->setColumns(1);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),1);
        else vectorTable = new Q3Table(ff->getValue().size(),1, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));

        vectorTable->horizontalHeader()->setLabel(0,QString("Value"));
        vectorTable->setColumnStretchable(0,true);

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }

    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    vector< T > value = ff->getValue();

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss;
        oss << value[i];
        vectorTable->setText(i,0,std::string(oss.str()).c_str());
    }
    readOnlyData(vectorTable,ff);
    return true;
}
//********************************************************************************************************************
template<class T>
void ModifyObject::storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< T > >* ff )
{

    Q3Table* table=it_list_table->first;

    QString vec_value;
    vector< T > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        vec_value = table->text(i,0);
        new_value.push_back( (T)atof(vec_value) );
    }
    ff->setValue( new_value );
}
//********************************************************************************************************************
//********************************************************************************************************************
template< int N, class T>
bool ModifyObject::createQtTable(Data< sofa::helper::vector< Vec<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;

        box->setColumns(1);

        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));
        if (N<=2)
        {
            if (N>0) {vectorTable->horizontalHeader()->setLabel(0,QString("X"));   vectorTable->setColumnStretchable(0,true);}
            if (N>1) {vectorTable->horizontalHeader()->setLabel(1,QString("Y"));   vectorTable->setColumnStretchable(1,true);}
            if (N>2) {vectorTable->horizontalHeader()->setLabel(2,QString("Z"));   vectorTable->setColumnStretchable(2,true);}
        }
        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }

    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    sofa::helper::vector< Vec<N,T> > value = ff->getValue();


    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];
        for (int j=0; j<N; j++)
        {
            oss[j] << value[i][j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }
    }
    readOnlyData(vectorTable,ff);
    return true;
}
//*******************************************************************************************************************
template< int N, class T>
void ModifyObject::storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< Vec<N,T> > >* ff )
{

    Q3Table* table=it_list_table->first;

    sofa::helper::vector< Vec<N,T> > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<N,T> vec_value;
        for (int j=0; j<N; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_value.push_back( vec_value );
    }
    ff->setValue( new_value );
}
//********************************************************************************************************************
//********************************************************************************************************************
template< int N, class T>
bool ModifyObject::createQtTable(DataPtr< sofa::helper::vector< Vec<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;

        box->setColumns(1);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));
        if (N<=2)
        {
            if (N>0) {vectorTable->horizontalHeader()->setLabel(0,QString("X"));   vectorTable->setColumnStretchable(0,true);}
            if (N>1) {vectorTable->horizontalHeader()->setLabel(1,QString("Y"));   vectorTable->setColumnStretchable(1,true);}
            if (N>2) {vectorTable->horizontalHeader()->setLabel(2,QString("Z"));   vectorTable->setColumnStretchable(2,true);}
        }
        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }
    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    sofa::helper::vector< Vec<N,T> > value = ff->getValue();

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];
        for (int j=0; j<N; j++)
        {
            oss[j] << value[i][j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }
    }
    readOnlyData(vectorTable,ff);
    return true;
}
//********************************************************************************************************************
template< int N, class T>
void ModifyObject::storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< Vec<N,T> > >* ff )
{

    Q3Table* table=it_list_table->first;

    sofa::helper::vector< Vec<N,T> > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<N,T> vec_value;
        for (int j=0; j<N; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }

        new_value.push_back( vec_value );

    }
    ff->setValue( new_value );
}
//********************************************************************************************************************
//********************************************************************************************************************
template< std::size_t N, class T>
bool ModifyObject::createQtTable(Data< sofa::helper::vector< fixed_array<T,N> > > *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;

        box->setColumns(1);

        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }

    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    sofa::helper::vector< fixed_array<T,N> > value = ff->getValue();


    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];
        for (unsigned int j=0; j<N; j++)
        {
            oss[j] << value[i][j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }
    }
    readOnlyData(vectorTable,ff);
    return true;
}
//*******************************************************************************************************************
template< std::size_t N, class T>
void ModifyObject::storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< fixed_array<T,N> > >* ff )
{

    Q3Table* table=it_list_table->first;

    sofa::helper::vector< fixed_array<T,N> > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        fixed_array<T,N> vec_value;
        for (unsigned int j=0; j<N; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_value.push_back( vec_value );
    }
    ff->setValue( new_value );
}
//********************************************************************************************************************
//********************************************************************************************************************
template< std::size_t N, class T>
bool ModifyObject::createQtTable(DataPtr< sofa::helper::vector< fixed_array<T,N> > > *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;

        box->setColumns(1);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }
    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    sofa::helper::vector< fixed_array<T,N> > value = ff->getValue();

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];
        for (unsigned int j=0; j<N; j++)
        {
            oss[j] << value[i][j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }
    }
    readOnlyData(vectorTable,ff);
    return true;
}
//********************************************************************************************************************
template< std::size_t N, class T>
void ModifyObject::storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< fixed_array<T,N> > >* ff )
{

    Q3Table* table=it_list_table->first;

    sofa::helper::vector< fixed_array<T,N> > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        fixed_array<T,N> vec_value;
        for (unsigned int j=0; j<N; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }

        new_value.push_back( vec_value );

    }
    ff->setValue( new_value );
}
//********************************************************************************************************************
//********************************************************************************************************************
template< class T>
bool ModifyObject::createQtTable(Data< sofa::component::topology::PointData< T > > *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;

        box->setColumns(1);

        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),1);
        else vectorTable = new Q3Table(ff->getValue().size(),1, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));
        vectorTable->setColumnStretchable(0,true);

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }
    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    vector< T > value = ff->getValue();
    /*
      if (vectorTable->numRows() < ff->getValue().size())
      {
      for (unsigned int i=0;
      }
      else if (vectorTable->numRows() > ff->getValue().size())
      {

      }*/

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss;
        oss << value[i];
        vectorTable->setText(i,0,std::string(oss.str()).c_str());
    }
    readOnlyData(vectorTable,ff);
    return true;
}
//********************************************************************************************************************
template<class T>
void ModifyObject::storeQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::component::topology::PointData< T >  >* ff )
{

    Q3Table* table=it_list_table->first;

    QString vec_value;
    sofa::component::topology::PointData< T > new_value;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        vec_value = table->text(i,0);
        new_value.push_back( (T)atof(vec_value) );
    }
    ff->setValue( new_value );
}
//********************************************************************************************************************

template< class T>
bool ModifyObject::createMonitorQtTable(Data<typename sofa::component::misc::Monitor<T>::MonitorData >* ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2, Q3Table* vectorTable3 )
{
    //internal monitorData
    typename sofa::component::misc::Monitor<T>::MonitorData MonitorDataTemp = ff->getValue();
    //number of rows
    unsigned short int nbRowVels = 0, nbRowForces = 0, nbRowPos = 0;

    if (!vectorTable || !vectorTable2 || !vectorTable3)
    {
        if (!MonitorDataTemp.sizeIdxPos() && !MonitorDataTemp.sizeIdxVels()
            && !MonitorDataTemp.sizeIdxForces() && !EMPTY_FLAG )
            return false;

        box->setColumns(2);
        new QLabel("", box);
        new QLabel("Positions", box);

        vectorTable = addResizableTable(box,MonitorDataTemp.sizeIdxPos(),4);
        new QLabel (" ", box);

        vectorTable->setReadOnly(false);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable));

        vectorTable->horizontalHeader()->setLabel(0,QString("particle Indices"));
        vectorTable->setColumnStretchable(0,true);
        vectorTable->horizontalHeader()->setLabel(1,QString("X"));      vectorTable->setColumnStretchable(1,true);
        vectorTable->horizontalHeader()->setLabel(2,QString("Y"));      vectorTable->setColumnStretchable(2,true);
        vectorTable->horizontalHeader()->setLabel(3,QString("Z"));      vectorTable->setColumnStretchable(3,true);

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        new QLabel("Velocities", box);

        vectorTable2 = addResizableTable(box,MonitorDataTemp.sizeIdxVels(),4);
        new QLabel (" ", box);
        vectorTable2->setReadOnly(false);

        list_Table.push_back(std::make_pair(vectorTable2, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable2));

        vectorTable2->horizontalHeader()->setLabel(0,QString("particle Indices"));
        vectorTable2->setColumnStretchable(0,true);
        vectorTable2->horizontalHeader()->setLabel(1,QString("X"));
        vectorTable2->setColumnStretchable(1,true);
        vectorTable2->horizontalHeader()->setLabel(2,QString("Y"));
        vectorTable2->setColumnStretchable(2,true);
        vectorTable2->horizontalHeader()->setLabel(3,QString("Z"));
        vectorTable2->setColumnStretchable(3,true);

        connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        new QLabel("Forces", box);

        vectorTable3 = addResizableTable(box,MonitorDataTemp.sizeIdxForces(),4);
        new QLabel (" ", box);
        vectorTable3->setReadOnly(false);

        list_Table.push_back(std::make_pair(vectorTable3, ff));
        objectGUI.push_back(std::make_pair(ff,vectorTable3));

        vectorTable3->horizontalHeader()->setLabel(0,QString("particle Indices"));
        vectorTable3->setColumnStretchable(0,true);
        vectorTable3->horizontalHeader()->setLabel(1,QString("X"));
        vectorTable3->setColumnStretchable(1,true);
        vectorTable3->horizontalHeader()->setLabel(2,QString("Y"));
        vectorTable3->setColumnStretchable(2,true);
        vectorTable3->horizontalHeader()->setLabel(3,QString("Z"));
        vectorTable3->setColumnStretchable(3,true);

        connect( vectorTable3, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

    } //fin if (!vectorTable)

    //number of rows for positions
    if (MonitorDataTemp.getSizeVecVels())
    {
        if (setResize.find(vectorTable) != setResize.end())
        {
            sofa::helper::vector < int > NewIndPos;
            NewIndPos = MonitorDataTemp.getIndPos();
            NewIndPos.resize(vectorTable->numRows(), 0);
            nbRowPos = NewIndPos.size();
            MonitorDataTemp.setIndPos (NewIndPos);
        }
        else
        {
            nbRowPos = MonitorDataTemp.sizeIdxPos();
            vectorTable->setNumRows(nbRowPos);
        }
    }
    else
    {
        vectorTable->setNumRows(nbRowPos);
    }

    //number of rows for velocities
    if (MonitorDataTemp.getSizeVecVels())
    {
        if (setResize.find(vectorTable2) != setResize.end())
        {
            sofa::helper::vector < int > NewIndVels;
            NewIndVels = MonitorDataTemp.getIndVels();
            NewIndVels.resize(vectorTable2->numRows(), 0);
            nbRowVels = NewIndVels.size();
            MonitorDataTemp.setIndVels (NewIndVels);
        }
        else
        {
            nbRowVels = MonitorDataTemp.sizeIdxVels();
            vectorTable2->setNumRows(nbRowVels);
        }
    }
    else
    {
        vectorTable2->setNumRows(nbRowVels);
    }

    //number of rows for forces
    if (MonitorDataTemp.getSizeVecForces())
    {
        if (setResize.find(vectorTable3) != setResize.end())
        {
            sofa::helper::vector < int > NewIndForces;
            NewIndForces = MonitorDataTemp.getIndForces();
            NewIndForces.resize(vectorTable3->numRows(), 0);
            nbRowForces = NewIndForces.size();
            MonitorDataTemp.setIndForces (NewIndForces);
        }
        else
        {
            nbRowForces = MonitorDataTemp.sizeIdxForces();
            vectorTable3->setNumRows(nbRowForces);
        }
    }
    else
    {
        vectorTable3->setNumRows(nbRowForces);
    }


    for (unsigned int i=0; i<3; i++)
    {
        std::ostringstream *oss = new std::ostringstream[nbRowPos];
        for (unsigned int j=0; j<nbRowPos; j++)
        {
            oss[j] << (MonitorDataTemp.getPos(j))[i];
            vectorTable->setText(j,i+1,std::string(oss[j].str()).c_str());
        }

        std::ostringstream * oss2 = new std::ostringstream[nbRowVels];
        for (unsigned int j=0; j<nbRowVels; j++)
        {
            oss2[j] << (MonitorDataTemp.getVel(j))[i];
            vectorTable2->setText(j,i+1,std::string(oss2[j].str()).c_str());
        }

        std::ostringstream * oss3 = new std::ostringstream[nbRowForces];
        for (unsigned int j=0; j<nbRowForces; j++)
        {
            oss3[j] << (MonitorDataTemp.getForce(j))[i];
            vectorTable3->setText(j,i+1,std::string(oss3[j].str()).c_str());
        }
        delete [] oss;
        delete [] oss2;
        delete [] oss3;
    }
    //vectorTable1
    std::ostringstream  * oss = new std::ostringstream[nbRowPos];
    for (unsigned int j=0; j<nbRowPos; j++)
    {
        oss[j] << MonitorDataTemp.getIndPos()[j];
        vectorTable->setText(j,0,std::string(oss[j].str()).c_str());
    }
    //vectorTable2
    std::ostringstream * oss2 = new std::ostringstream[nbRowVels];
    for (unsigned int j=0; j<nbRowVels; j++)
    {
        oss2[j] << MonitorDataTemp.getIndVels()[j];
        vectorTable2->setText(j,0,std::string(oss2[j].str()).c_str());
    }
    //vectorTable3
    std::ostringstream * oss3 = new std::ostringstream[nbRowForces];
    for (unsigned int j=0; j<nbRowForces; j++)
    {
        oss3[j] << MonitorDataTemp.getIndForces()[j];
        vectorTable3->setText(j,0,std::string(oss3[j].str()).c_str());
    }
    if (vectorTable ) readOnlyData(vectorTable ,ff);
    if (vectorTable2) readOnlyData(vectorTable2,ff);
    if (vectorTable3) readOnlyData(vectorTable3,ff);
    delete [] oss;
    delete [] oss2;
    delete [] oss3;

    counterWidget+=3;
    ff->setValue (MonitorDataTemp);
    return true;
}

//********************************************************************************************************************
template<class T>
void ModifyObject::storeMonitorQtTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data<typename sofa::component::misc::Monitor<T>::MonitorData >* ff )
{
    Q3Table* table = it_list_table->first;
    //internal monitorData
    typename sofa::component::misc::Monitor<T>::MonitorData NewMonitorData = ff->getValue();


    //Qtable positions
    if (NewMonitorData.getSizeVecPos())
    {
        int valueBox;
        sofa::helper::vector < int > values = NewMonitorData.getIndPos();
        for (int i=0; i < table -> numRows(); i++)
        {
            valueBox = (int)atof(table->text(i,0));
            if(valueBox >= 0 && valueBox <= (int)(NewMonitorData.getSizeVecPos() - 1))
                values[i] = valueBox;
        }

        NewMonitorData.setIndPos(values);
    }
    it_list_table++;
    table = it_list_table->first;


    //Qtable velocities

    if (NewMonitorData.getSizeVecVels())
    {
        int valueBox;
        sofa::helper::vector < int > values = NewMonitorData.getIndVels();
        for (int i=0; i < table -> numRows(); i++)
        {
            valueBox = (int)atof(table->text(i,0));
            if(valueBox >= 0 && valueBox <= (int)(NewMonitorData.getSizeVecVels() - 1))
                values[i] = valueBox;
        }

        NewMonitorData.setIndVels(values);
    }
    it_list_table++;
    table=it_list_table->first;


    //Qtable forces

    if (NewMonitorData.getSizeVecForces())
    {
        int valueBox;
        sofa::helper::vector < int > values = NewMonitorData.getIndForces();

        for (int i=0; i < table -> numRows(); i++)
        {
            valueBox = (int)atof(table->text(i,0));
            if(valueBox >= 0 && valueBox <= (int)(NewMonitorData.getSizeVecForces() - 1))
                values[i] = valueBox;
        }

        NewMonitorData.setIndForces(values);
    }

    ff->setValue(NewMonitorData);
}
//********************************************************************************************************************

template< int N, class T>
bool ModifyObject::createQtTable(DataPtr< sofa::helper::vector< RigidCoord<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2 )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0 && !EMPTY_FLAG)  return false;
        box->setColumns(1);
        if (RESIZABLE_FLAG) new QLabel("", box);
        new QLabel("Center", box);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        vectorTable->setReadOnly(false);
        list_Table.push_back(std::make_pair(vectorTable, ff));

        vectorTable->horizontalHeader()->setLabel(0,QString("X"));	    vectorTable->setColumnStretchable(0,true);
        if (N>1) { vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);}
        if (N>2) { vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);}
        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        if (RESIZABLE_FLAG)new QLabel("", box);
        new QLabel("Orientation", box);
        if (RESIZABLE_FLAG)new QLabel("", box);

        if (N==2)
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);

            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	    vectorTable2->setColumnStretchable(0,true);
        }
        else
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),4, box);
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	     vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);
            vectorTable2->horizontalHeader()->setLabel(3,QString("W"));      vectorTable2->setColumnStretchable(3,true);
        }

        list_Table.push_back(std::make_pair(vectorTable2, ff));

        connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

    }


    if (setResize.find(vectorTable) != setResize.end())
        ff->beginEdit()->resize(vectorTable->numRows());
    else
        vectorTable->setNumRows(ff->getValue().size());

    vectorTable2->setNumRows(vectorTable->numRows());

    vector< RigidCoord<N,T> > value = ff->getValue();


    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];

        for (int j=0; j<N; j++)
        {
            oss[j] << value[i].getCenter()[j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }


        std::ostringstream oss_o;
        oss_o << value[i].getOrientation();
        std::istringstream oss_i;
        oss_i.str( oss_o.str() );
        T Tvalue;
        int count = 0;
        while (oss_i >> Tvalue)    vectorTable2->setText(i,count++,QString::number(Tvalue));
    }
    readOnlyData(vectorTable,ff);
    readOnlyData(vectorTable2,ff);

    counterWidget+=2;
    return true;
}
//********************************************************************************************************************
template< int N, class T>
bool ModifyObject::createQtTable(Data< sofa::helper::vector< RigidCoord<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2 )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0 && !EMPTY_FLAG)  return false;
        box->setColumns(1);

        if (RESIZABLE_FLAG) new QLabel("", box);
        new QLabel("Center", box);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        vectorTable->setReadOnly(false);
        list_Table.push_back(std::make_pair(vectorTable, ff));
        vectorTable->horizontalHeader()->setLabel(0,QString("X"));	    vectorTable->setColumnStretchable(0,true);
        if (N>1) { vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);}
        if (N>2) { vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);}

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        if (RESIZABLE_FLAG)new QLabel("", box);
        new QLabel("Orientation", box);
        if (RESIZABLE_FLAG)new QLabel("", box);

        if (N==2)
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);
            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	    vectorTable2->setColumnStretchable(0,true);
        }
        else
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),4, box);
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	     vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);
            vectorTable2->horizontalHeader()->setLabel(3,QString("W"));      vectorTable2->setColumnStretchable(3,true);
        }

        list_Table.push_back(std::make_pair(vectorTable2, ff));

        connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

    }

    if (setResize.find(vectorTable) != setResize.end())
        ff->beginEdit()->resize(vectorTable->numRows());
    else
        vectorTable->setNumRows(ff->getValue().size());

    vectorTable2->setNumRows(vectorTable->numRows());

    vector< RigidCoord<N,T> > value = ff->getValue();


    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];

        for (int j=0; j<N; j++)
        {
            oss[j] << value[i].getCenter()[j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }


        std::ostringstream oss_o;
        oss_o << value[i].getOrientation();
        std::istringstream oss_i;
        oss_i.str( oss_o.str() );
        T Tvalue;
        int count = 0;
        while (oss_i >> Tvalue)    vectorTable2->setText(i,count++,QString::number(Tvalue));
    }
    readOnlyData(vectorTable,ff);
    readOnlyData(vectorTable2,ff);
    counterWidget+=2;
    return true;
}
//********************************************************************************************************************
template< int N, class T>
bool ModifyObject::createQtTable(Data< sofa::helper::vector< RigidDeriv<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2 )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;
        box->setColumns(1);


        if (RESIZABLE_FLAG) new QLabel("", box);
        new QLabel("Center", box);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        vectorTable->setReadOnly(false);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
        if (N>1) { vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);}
        if (N>2) { vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);}

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        if (RESIZABLE_FLAG)new QLabel("", box);
        new QLabel("Orientation", box);
        if (RESIZABLE_FLAG)new QLabel("", box);

        if (N==2)
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);
            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	 vectorTable2->setColumnStretchable(0,true);
        }
        else
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),3, box);
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	 vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);
        }


        list_Table.push_back(std::make_pair(vectorTable2, ff));
        connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

    }


    if (setResize.find(vectorTable) != setResize.end())
        ff->beginEdit()->resize(vectorTable->numRows());
    else
        vectorTable->setNumRows(ff->getValue().size());

    vectorTable2->setNumRows(vectorTable->numRows());

    vector< RigidDeriv<N,T> > value = ff->getValue();


    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];

        for (int j=0; j<N; j++)
        {
            oss[j] << value[i].getVCenter()[j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }

        std::ostringstream oss_o;
        oss_o << value[i].getVOrientation();
        std::istringstream oss_i;
        oss_i.str( oss_o.str() );
        T Tvalue;
        int count = 0;
        while (oss_i >> Tvalue)    vectorTable2->setText(i,count++,QString::number(Tvalue));
    }
    readOnlyData(vectorTable,ff);
    readOnlyData(vectorTable2,ff);
    counterWidget+=2;
    return true;
}
//********************************************************************************************************************
template< int N, class T>
bool ModifyObject::createQtTable(DataPtr< sofa::helper::vector< RigidDeriv<N,T> > > *ff, Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2 )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;
        box->setColumns(1);


        if (RESIZABLE_FLAG) new QLabel("", box);
        new QLabel("Center", box);


        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),N);
        else vectorTable = new Q3Table(ff->getValue().size(),N, box);

        vectorTable->setReadOnly(false);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        vectorTable->horizontalHeader()->setLabel(0,QString("X"));	vectorTable->setColumnStretchable(0,true);
        if (N>1) { vectorTable->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable->setColumnStretchable(1,true);}
        if (N>2) { vectorTable->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable->setColumnStretchable(2,true);}

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

        if (RESIZABLE_FLAG)new QLabel("", box);
        new QLabel("Orientation", box);
        if (RESIZABLE_FLAG)new QLabel("", box);

        if (N==2)
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),1, box);
            vectorTable2->horizontalHeader()->setLabel(0,QString("angle"));	 vectorTable2->setColumnStretchable(0,true);
        }
        else
        {
            vectorTable2 = new Q3Table(ff->getValue().size(),3, box);
            vectorTable2->horizontalHeader()->setLabel(0,QString("X"));	 vectorTable2->setColumnStretchable(0,true);
            vectorTable2->horizontalHeader()->setLabel(1,QString("Y"));      vectorTable2->setColumnStretchable(1,true);
            vectorTable2->horizontalHeader()->setLabel(2,QString("Z"));      vectorTable2->setColumnStretchable(2,true);
        }


        list_Table.push_back(std::make_pair(vectorTable2, ff));
        connect( vectorTable2, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );

    }

    if (setResize.find(vectorTable) != setResize.end())
        ff->beginEdit()->resize(vectorTable->numRows());
    else
        vectorTable->setNumRows(ff->getValue().size());

    vectorTable2->setNumRows(vectorTable->numRows());

    vector< RigidDeriv<N,T> > value = ff->getValue();


    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[N];

        for (int j=0; j<N; j++)
        {
            oss[j] << value[i].getVCenter()[j];
            vectorTable->setText(i,j,std::string(oss[j].str()).c_str());
        }

        std::ostringstream oss_o;
        oss_o << value[i].getVOrientation();
        std::istringstream oss_i;
        oss_i.str( oss_o.str() );
        T Tvalue;
        int count = 0;
        while (oss_i >> Tvalue)    vectorTable2->setText(i,count++,QString::number(Tvalue));
    }
    readOnlyData(vectorTable,ff);
    readOnlyData(vectorTable2,ff);
    counterWidget+=2;
    return true;
}
//********************************************************************************************************************
template< class T>
void ModifyObject::storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidCoord<3,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<3,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<3,T> vec_value;
        for (int j=0; j<3; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidCoord<3,T> > new_value;
    sofa::helper::vector<sofa::helper::Quater<T> > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Quater<T> vec_value;
        for (unsigned int j=0; j<4; ++j)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidCoord<3,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}


//********************************************************************************************************************
template< class T>
void ModifyObject::storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidCoord<2,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<2,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<2,T> vec_value;
        for (int j=0; j<2; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidCoord<2,T> > new_value;

    sofa::helper::vector< T > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        T vec_value;
        vec_value = (T) atof(table->text(i,0));
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidCoord<2,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}
//********************************************************************************************************************

template< class T>
void ModifyObject::storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidCoord<3,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<3,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<3,T> vec_value;
        for (int j=0; j<3; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidCoord<3,T> > new_value;
    sofa::helper::vector<sofa::helper::Quater<T> > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Quater<T> vec_value;
        for (unsigned int j=0; j<4; ++j)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidCoord<3,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}


//********************************************************************************************************************
template< class T>
void ModifyObject::storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidCoord<2,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<2,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<2,T> vec_value;
        for (int j=0; j<2; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidCoord<2,T> > new_value;

    sofa::helper::vector< T > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        T vec_value;
        vec_value = (T) atof(table->text(i,0));
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidCoord<2,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}

template< class T>
void ModifyObject::storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidDeriv<3,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<3,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<3,T> vec_value;
        for (int j=0; j<3; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidDeriv<3,T> > new_value;
    sofa::helper::vector<Vec<3,T> > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<3,T> vec_value;
        for (unsigned int j=0; j<3; ++j)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidDeriv<3,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}


//********************************************************************************************************************
template< class T>
void ModifyObject::storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< RigidDeriv<2,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<2,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<2,T> vec_value;
        for (int j=0; j<2; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidDeriv<2,T> > new_value;

    sofa::helper::vector< T > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        T vec_value;
        vec_value = (T) atof(table->text(i,0));
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidDeriv<2,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}

template< class T>
void ModifyObject::storeQtRigid3Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidDeriv<3,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<3,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<3,T> vec_value;
        for (int j=0; j<3; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidDeriv<3,T> > new_value;
    sofa::helper::vector<Vec<3,T> > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<3,T> vec_value;
        for (unsigned int j=0; j<3; ++j)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidDeriv<3,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}


//********************************************************************************************************************
template< class T>
void ModifyObject::storeQtRigid2Table( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, DataPtr< sofa::helper::vector< RigidDeriv<2,T> > >* ff )
{
    Q3Table* table = it_list_table->first;

    sofa::helper::vector< Vec<2,T> > new_valuePosition;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        Vec<2,T> vec_value;
        for (int j=0; j<2; j++)
        {
            vec_value[j] = (T) atof(table->text(i,j));
        }
        new_valuePosition.push_back( vec_value );
    }

    it_list_table++;
    table = it_list_table->first;

    sofa::helper::vector< RigidDeriv<2,T> > new_value;

    sofa::helper::vector< T > new_valueOrientation;
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        T vec_value;
        vec_value = (T) atof(table->text(i,0));
        new_valueOrientation.push_back( vec_value );
    }
    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        new_value.push_back(RigidDeriv<2,T>(new_valuePosition[i], new_valueOrientation[i]));
    }
    ff->setValue( new_value );
}



#ifndef WIN32
//********************************************************************************************************************
template< int N, class T>
bool ModifyObject::createQtSpringTable(Data< sofa::helper::vector< typename sofa::component::forcefield::SpringForceField< StdVectorTypes<Vec<N,T>,Vec<N,T>,T> >::Spring > >   *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0 && !EMPTY_FLAG)  return false;
        box->setColumns(1);

        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),5);
        else vectorTable = new Q3Table(ff->getValue().size(),5, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        vectorTable->horizontalHeader()->setLabel(0,QString("Index 1")); vectorTable->setColumnStretchable(0,true);
        vectorTable->horizontalHeader()->setLabel(1,QString("Index 2")); vectorTable->setColumnStretchable(1,true);
        vectorTable->horizontalHeader()->setLabel(2,QString("Ks : stiffness")); vectorTable->setColumnStretchable(2,true);
        vectorTable->horizontalHeader()->setLabel(3,QString("Kd : damping")); vectorTable->setColumnStretchable(3,true);
        vectorTable->horizontalHeader()->setLabel(4,QString("Rest Length")); vectorTable->setColumnStretchable(4,true);

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }

    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    sofa::helper::vector<typename  sofa::component::forcefield::SpringForceField< StdVectorTypes<Vec<N,T>,Vec<N,T>,T> >::Spring > value = ff->getValue();

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[5];
        oss[0] << value[i].m1;      vectorTable->setText(i,0,std::string(oss[0].str()).c_str());
        oss[1] << value[i].m2;      vectorTable->setText(i,1,std::string(oss[1].str()).c_str());
        oss[2] << value[i].ks;      vectorTable->setText(i,2,std::string(oss[2].str()).c_str());
        oss[3] << value[i].kd;      vectorTable->setText(i,3,std::string(oss[3].str()).c_str());
        oss[4] << value[i].initpos; vectorTable->setText(i,4,std::string(oss[4].str()).c_str());
    }

    readOnlyData(vectorTable,ff);
    return true;
}

//********************************************************************************************************************
template< int N, class T>
void ModifyObject::storeQtSpringTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data<  sofa::helper::vector< typename sofa::component::forcefield::SpringForceField< StdVectorTypes< Vec<N,T>, Vec<N,T>, T> >::Spring > >  *ff)
{

    Q3Table* table = it_list_table->first;

    vector<typename sofa::component::forcefield::SpringForceField< StdVectorTypes< Vec<N,T>, Vec<N,T>, T> >::Spring > new_value;

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        double values[5];
        typename sofa::component::forcefield::SpringForceField< StdVectorTypes< Vec<N,T>, Vec<N,T>, T> >::Spring  current_spring;
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

//********************************************************************************************************************
template< class T>
bool ModifyObject::createQtRigidSpringTable(Data< vector< typename sofa::component::forcefield::JointSpringForceField< StdRigidTypes< 3,T > >::Spring > >  *ff, Q3GroupBox *box, Q3Table* vectorTable )
{
    if (!vectorTable)
    {
        if (ff->getValue().size() == 0  && !EMPTY_FLAG)  return false;
        box->setColumns(1);

        if (RESIZABLE_FLAG) vectorTable = addResizableTable(box,ff->getValue().size(),13);
        else vectorTable = new Q3Table(ff->getValue().size(),13, box);

        list_Table.push_back(std::make_pair(vectorTable, ff));
        vectorTable->horizontalHeader()->setLabel(0,QString("Index 1"));
        vectorTable->horizontalHeader()->setLabel(1,QString("Index 2"));
        vectorTable->horizontalHeader()->setLabel(2,QString("soft stiffness Trans"));
        vectorTable->horizontalHeader()->setLabel(3,QString("hard stiffness Trans"));
        vectorTable->horizontalHeader()->setLabel(4,QString("soft stiffness Rot"));
        vectorTable->horizontalHeader()->setLabel(5,QString("hard stiffness Rot"));

        vectorTable->horizontalHeader()->setLabel(6,QString("Kd : damping"));

        vectorTable->horizontalHeader()->setLabel(7,QString("Angle X min"));
        vectorTable->horizontalHeader()->setLabel(8,QString("Angle X max"));
        vectorTable->horizontalHeader()->setLabel(9,QString("Angle Y min"));
        vectorTable->horizontalHeader()->setLabel(10,QString("Angle Y max"));
        vectorTable->horizontalHeader()->setLabel(11,QString("Angle Z min"));
        vectorTable->horizontalHeader()->setLabel(12,QString("Angle Z max"));

        connect( vectorTable, SIGNAL( valueChanged(int,int) ), this, SLOT( changeValue() ) );
    }

    if (setResize.find(vectorTable) != setResize.end()) ff->beginEdit()->resize(vectorTable->numRows());
    else	  vectorTable->setNumRows(ff->getValue().size());

    vector<typename sofa::component::forcefield::JointSpringForceField<  StdRigidTypes< 3,T > >::Spring > value = ff->getValue();

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        std::ostringstream oss[13];
        oss[0]  << value[i].m1;			 vectorTable->setText(i ,0 ,std::string(oss[0] .str()).c_str());
        oss[1]  << value[i].m2;			 vectorTable->setText(i ,1 ,std::string(oss[1] .str()).c_str());
        oss[2]  << value[i].softStiffnessTrans;  vectorTable->setText(i ,2 ,std::string(oss[2] .str()).c_str());
        oss[3]  << value[i].hardStiffnessTrans;  vectorTable->setText(i ,3 ,std::string(oss[3] .str()).c_str());
        oss[4]  << value[i].softStiffnessRot;	 vectorTable->setText(i ,4 ,std::string(oss[4] .str()).c_str());
        oss[5]  << value[i].hardStiffnessRot;	 vectorTable->setText(i ,5 ,std::string(oss[5] .str()).c_str());
        oss[6]  << value[i].kd;			 vectorTable->setText(i ,6 ,std::string(oss[6] .str()).c_str());
        oss[7]  << value[i].limitAngles[0];	 vectorTable->setText(i ,7 ,std::string(oss[7] .str()).c_str());
        oss[8]  << value[i].limitAngles[1];	 vectorTable->setText(i ,8 ,std::string(oss[8] .str()).c_str());
        oss[9]  << value[i].limitAngles[2]; 	 vectorTable->setText(i, 9 ,std::string(oss[9] .str()).c_str());
        oss[10] << value[i].limitAngles[3];   	 vectorTable->setText(i,10 ,std::string(oss[10].str()).c_str());
        oss[11] << value[i].limitAngles[4];   	 vectorTable->setText(i,11 ,std::string(oss[11].str()).c_str());
        oss[12] << value[i].limitAngles[5];   	 vectorTable->setText(i,12 ,std::string(oss[12].str()).c_str());
    }
    readOnlyData(vectorTable,ff);
    return true;
}

//********************************************************************************************************************
template< class T>
void ModifyObject::storeQtRigidSpringTable( std::list< std::pair< Q3Table*, core::objectmodel::BaseData*> >::iterator &it_list_table, Data< sofa::helper::vector< typename sofa::component::forcefield::JointSpringForceField< StdRigidTypes<3,T> >::Spring > >  *ff )
{

    Q3Table* table = it_list_table->first;

    sofa::helper::vector<  typename sofa::component::forcefield::JointSpringForceField< StdRigidTypes<3,T> >::Spring > new_value;

    for (unsigned int i=0; i<ff->getValue().size(); i++)
    {
        double values[20];
        typename sofa::component::forcefield::JointSpringForceField< StdRigidTypes<3,T> >::Spring current_spring;
        for (int j=0; j<20; j++)
        {
            values[j] = atof(table->text(i,j));
        }
        current_spring.m1 = (int) values[0];
        current_spring.m2 = (int)values[1];

        current_spring.freeMovements[0]       = values[2];
        current_spring.freeMovements[1]       = values[3];
        current_spring.freeMovements[2]       = values[4];
        current_spring.freeMovements[3]       = values[5];
        current_spring.freeMovements[4]       = values[6];
        current_spring.freeMovements[5]       = values[7];

        current_spring.softStiffnessTrans     = values[8];
        current_spring.hardStiffnessTrans     = values[9];
        current_spring.softStiffnessRot       = values[10];
        current_spring.hardStiffnessRot       = values[11];

        current_spring.kd           = values[12];

        current_spring.initTrans[0] = values[13];
        current_spring.initTrans[1] = values[14];
        current_spring.initTrans[2] = values[15];

        current_spring.initRot[0]   = values[16];
        current_spring.initRot[1]   = values[17];
        current_spring.initRot[2]   = values[18];
        current_spring.initRot[3]   = values[19];
        new_value.push_back(current_spring);
    }
    ff->setValue( new_value );
}
#endif
//********************************************************************************************************************

Q3Table *ModifyObject::addResizableTable(Q3GroupBox *box,int number, int column)
{
    box->setColumns(2);
    QSpinBox *spinBox = new QSpinBox(0,INT_MAX, 1, box);
    Q3Table* table = new Q3Table(number,column, box);
    spinBox->setValue(number);
    resizeMap.insert(std::make_pair(spinBox, table));
    connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( resizeTable(int) ) );
    return  table;
}

const core::objectmodel::BaseData* ModifyObject::getData(const QObject *object)
{
    for (unsigned int i=0; i<objectGUI.size(); ++i)
    {
        if (objectGUI[i].second == object) return  objectGUI[i].first;
    }
    return false;
}

void ModifyObject::resizeTable(int number)
{
    QSpinBox *spinBox = (QSpinBox *) sender();
    Q3Table *table = resizeMap[spinBox];
    if (number != table->numRows())
    {
        table->setNumRows(number);
        setResize.insert(table);
    }
    updateTables();
    setResize.clear();
}

void ModifyObject::readOnlyData(Q3Table *widget, core::objectmodel::BaseData* data)
{
    widget->setReadOnly(( (data->isReadOnly()) && READONLY_FLAG));
}

void ModifyObject::readOnlyData(QWidget *widget, core::objectmodel::BaseData* data)
{
    widget->setEnabled(!( (data->isReadOnly()) && READONLY_FLAG));
}

} // namespace qt

} // namespace gui

} // namespace sofa

