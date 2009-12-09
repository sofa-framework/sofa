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

#include "ModifyObject.h"
#include "DataWidget.h"
#include "QDisplayDataWidget.h"
#include <iostream>
#ifdef SOFA_QT4
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTabWidget>
#include <QGridLayout>
#include <QTabWidget>
#include <Q3ListView>
#else
#include <qpushbutton.h>
#include <qlayout.h>
#include <qtabwidget.h>
#include <qtextedit.h>
#endif

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.inl>

#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/TransformationVisitor.h>



#include <qwt_legend.h>


#define WIDGET_BY_TAB 15


namespace sofa
{

namespace gui
{

namespace qt
{



using namespace  sofa::defaulttype;
using namespace  sofa::core::objectmodel;



#ifndef SOFA_QT4
typedef QScrollView Q3ScrollView;
#endif

ModifyObject::ModifyObject(
    void *Id,
    Q3ListViewItem* item_clicked,
    QWidget* parent,
    const ModifyObjectFlags& dialogFlags,
    const char* name,
    bool modal, Qt::WFlags f )
    :Id_(Id),
     item_(item_clicked),
     parent_(parent),
     node(NULL),
     data_(NULL),
     dialogFlags_(dialogFlags),
     visualContentModified(false),
     outputTab(NULL),
     warningTab(NULL),
     logWarningEdit(NULL),
     logOutputEdit(NULL),
     graphEnergy(NULL),
     QDialog(parent, name, modal, f)
{
    setCaption(name);
    //connect ( this, SIGNAL( objectUpdated() ), parent_, SLOT( redraw() ));
    connect ( this, SIGNAL( dialogClosed(void *) ) , parent_, SLOT( modifyUnlock(void *)));
    energy_curve[0]=NULL;
    energy_curve[1]=NULL;
    energy_curve[2]=NULL;
}

void ModifyObject::createDialog(Base* base)
{
    if(base == NULL)
    {
        return;
    }
    node = base;
    data_ = NULL;
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
    bool isNode = (dynamic_cast< Node *>(node) != NULL);
    QWidget *tabVisualization = NULL; //tab for visualization info: only created if needed ( boolean visualTab gives this answer ).

    QVBoxLayout *currentTabLayout = NULL;
    QVBoxLayout *tabPropertiesLayout=NULL;
    QVBoxLayout *tabVisualizationLayout = NULL;

    buttonUpdate = new QPushButton( this, "buttonUpdate" );
    buttonUpdate->setText("&Update");
    buttonUpdate->setEnabled(false);
    QPushButton *buttonOk = new QPushButton( this, "buttonOk" );
    buttonOk->setText( tr( "&OK" ) );

    QPushButton *buttonCancel = new QPushButton( this, "buttonCancel" );

    buttonCancel->setText( tr( "&Cancel" ) );

    // displayWidget
    if (node)
    {
        //If the current element is a node, we add a box to perform geometric transformation: translation, rotation, scaling
        if(dialogFlags_.REINIT_FLAG && isNode)
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

            transformation[7] = new WFloatLineEdit( box, "transformation[7]" );
            transformation[7]->setMinFloatValue( (float)-INFINITY );
            transformation[7]->setMaxFloatValue( (float)INFINITY );

            transformation[8] = new WFloatLineEdit( box, "transformation[8]" );
            transformation[8]->setMinFloatValue( (float)-INFINITY );
            transformation[8]->setMaxFloatValue( (float)INFINITY );


            //********************************************************************************
            //Default values
            transformation[0]->setFloatValue(0);
            transformation[1]->setFloatValue(0);
            transformation[2]->setFloatValue(0);

            transformation[3]->setFloatValue(0);
            transformation[4]->setFloatValue(0);
            transformation[5]->setFloatValue(0);

            transformation[6]->setFloatValue(1);
            transformation[7]->setFloatValue(1);
            transformation[8]->setFloatValue(1);

            connect( transformation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[3], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[4], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[5], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[6], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[7], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
            connect( transformation[8], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );

            //Option still experimental : disabled !!!!
            textScale->hide();
            transformation[6]->hide();
            transformation[7]->hide();
            transformation[8]->hide();


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
            else if ((*it).second->getGroup() == std::string("BIG"))
            {
                std::cout << (*it).first << " in new tab" << std::endl;
                emptyTab = true;
                currentTab= new QWidget();
                currentTabLayout = new QVBoxLayout( currentTab, 0, 1, QString("tabBIGLayout") + QString::number(counterWidget));
                dialogTab->addTab(currentTab, QString((*it).first.c_str()));
            }
            if (hideData(it->second)) continue;
            std::string box_name(oss.str());
            QDisplayDataWidget* displaydatawidget = new QDisplayDataWidget(currentTab,(*it).second,getFlags());
            ++i;
            if (displaydatawidget->getNumWidgets())
            {
                if (currentTab == currentTab_save && emptyTab && counterWidget/WIDGET_BY_TAB == counterTab)
                {
                    dialogTab->addTab(currentTab, QString("Properties ") + QString::number(counterWidget/WIDGET_BY_TAB));
                    ++counterTab;
                    emptyTab = false;
                }

                dataIndexTab.insert(std::make_pair((*it).second, dialogTab->count()-1));
                ++counterWidget;
                currentTabLayout->addWidget(displaydatawidget);
                counterWidget += displaydatawidget->getNumWidgets();
                connect(buttonUpdate,   SIGNAL( clicked() ), displaydatawidget, SLOT( UpdateData() ) );
                connect(displaydatawidget, SIGNAL( WidgetHasChanged(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
                connect(buttonOk, SIGNAL(clicked() ), displaydatawidget, SLOT( UpdateData() ) );
                connect(displaydatawidget, SIGNAL(DataParentNameChanged()), this, SLOT( updateListViewItem() ) );
                connect(this, SIGNAL(updateDataWidgets()), displaydatawidget, SLOT(UpdateWidgets()) );
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

        if (Node* real_node = dynamic_cast< Node* >(node))
        {
            if (dialogFlags_.REINIT_FLAG && (real_node->mass!= NULL || real_node->forceField.size()!=0 ) )
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
                new QLabel(QString(node->getName().c_str()), box);
                new QLabel(QString("Class"), box);
                new QLabel(QString(node->getClassName().c_str()), box);
                std::string namespacename = node->decodeNamespaceName(typeid(*node));
                if (!namespacename.empty())
                {
                    new QLabel(QString("Namespace"), box);
                    new QLabel(QString(namespacename.c_str()), box);
                }
                if (!node->getTemplateName().empty())
                {
                    new QLabel(QString("Template"), box);
                    new QLabel(QString(node->getTemplateName().c_str()), box);
                }

                tabLayout->addWidget( box );
            }

            //Class description
            core::ObjectFactory::ClassEntry* entry = core::ObjectFactory::getInstance()->getEntry(node->getClassName());
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
                std::map<std::string, core::ObjectFactory::Creator*>::iterator it = entry->creatorMap.find(node->getTemplateName());
                if (it != entry->creatorMap.end() && *it->second->getTarget())
                {
                    new QLabel(QString("Provided by"), box);
                    new QLabel(QString(it->second->getTarget()), box);
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


            tabLayout->addStretch();

            updateConsole();
            if (outputTab)  dialogTab->addTab(outputTab,  QString("Outputs"));
            if (warningTab) dialogTab->addTab(warningTab, QString("Warnings"));
        }

        //Adding buttons at the bottom of the dialog
        QHBoxLayout *lineLayout = new QHBoxLayout( 0, 0, 6, "Button Layout");



        lineLayout->addWidget(buttonUpdate);


        QSpacerItem *Horizontal_Spacing = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
        lineLayout->addItem( Horizontal_Spacing );


        lineLayout->addWidget(buttonOk);
        lineLayout->addWidget(buttonCancel);
        generalLayout->addLayout( lineLayout );
        //Signals and slots connections
        connect( buttonUpdate,   SIGNAL( clicked() ), this, SLOT( updateValues() ) );
        connect( buttonOk,       SIGNAL( clicked() ), this, SLOT( accept() ) );
        connect( buttonCancel,   SIGNAL( clicked() ), this, SLOT( reject() ) );
        resize( QSize(553, 130).expandedTo(minimumSizeHint()) );
    }
}

void ModifyObject::createDialog(BaseData* data)
{
    data_ = data;
    node = NULL;
    QVBoxLayout *generalLayout = new QVBoxLayout(this, 0, 1, "generalLayout");
    QHBoxLayout *lineLayout = new QHBoxLayout( 0, 0, 6, "Button Layout");
    buttonUpdate = new QPushButton( this, "buttonUpdate" );
    buttonUpdate->setText("&Update");
    buttonUpdate->setEnabled(false);
    QPushButton *buttonOk = new QPushButton( this, "buttonOk" );
    buttonOk->setText( tr( "&OK" ) );
    QPushButton *buttonCancel = new QPushButton( this, "buttonCancel" );
    buttonCancel->setText( tr( "&Cancel" ) );

    QDisplayDataWidget* displaydatawidget = new QDisplayDataWidget(this,data,getFlags());
    generalLayout->addWidget(displaydatawidget);
    lineLayout->addWidget(buttonUpdate);


    QSpacerItem *Horizontal_Spacing = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    lineLayout->addItem( Horizontal_Spacing );


    lineLayout->addWidget(buttonOk);
    lineLayout->addWidget(buttonCancel);
    generalLayout->addLayout( lineLayout );
    connect(buttonUpdate,   SIGNAL( clicked() ), displaydatawidget, SLOT( UpdateData() ) );
    connect(displaydatawidget, SIGNAL( WidgetHasChanged(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
    connect(buttonOk, SIGNAL(clicked() ), displaydatawidget, SLOT( UpdateData() ) );
    connect(displaydatawidget, SIGNAL(DataParentNameChanged()), this, SLOT( updateListViewItem() ) );
    connect( buttonOk,       SIGNAL( clicked() ), this, SLOT( accept() ) );
    connect( buttonCancel,   SIGNAL( clicked() ), this, SLOT( reject() ) );
}

//******************************************************************************************
void ModifyObject::updateConsole()
{
    //Console Warnings
    if ( !node->getWarnings().empty())
    {
        if (!logWarningEdit)
        {
            warningTab = new QWidget();
            QVBoxLayout* tabLayout = new QVBoxLayout( warningTab, 0, 1, QString("tabWarningLayout"));

            QPushButton *buttonClearWarnings = new QPushButton(warningTab, "buttonClearWarnings");
            tabLayout->addWidget(buttonClearWarnings);
            buttonClearWarnings->setText( tr("&Clear"));
            connect( buttonClearWarnings, SIGNAL( clicked()), this, SLOT( clearWarnings()));

            logWarningEdit = new Q3TextEdit( warningTab, QString("WarningEdit"));
            tabLayout->addWidget( logWarningEdit );

            logWarningEdit->setReadOnly(true);
        }

        logWarningEdit->setText(QString(node->getWarnings().c_str()));
        logWarningEdit->moveCursor(Q3TextEdit::MoveEnd, false);
        logWarningEdit->ensureCursorVisible();

    }
    //Console Outputs
    if ( !node->getOutputs().empty())
    {
        if (!logOutputEdit)
        {
            outputTab = new QWidget();
            QVBoxLayout* tabLayout = new QVBoxLayout( outputTab, 0, 1, QString("tabOutputLayout"));

            QPushButton *buttonClearOutputs = new QPushButton(outputTab, "buttonClearOutputs");
            tabLayout->addWidget(buttonClearOutputs);
            buttonClearOutputs->setText( tr("&Clear"));
            connect( buttonClearOutputs, SIGNAL( clicked()), this, SLOT( clearOutputs()));

            logOutputEdit = new Q3TextEdit( outputTab, QString("OutputEdit"));
            tabLayout->addWidget( logOutputEdit );

            logOutputEdit->setReadOnly(true);
        }

        logOutputEdit->setText(QString(node->getOutputs().c_str()));
        logOutputEdit->moveCursor(Q3TextEdit::MoveEnd, false);
        logOutputEdit->ensureCursorVisible();
    }
}

//*******************************************************************************************************************
void ModifyObject::changeValue()
{
    const QObject* s = sender();
    setUpdates.insert(getData(s));
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

const core::objectmodel::BaseData* ModifyObject::getData(const QObject *object)
{
    for (unsigned int i=0; i<objectGUI.size(); ++i)
    {
        if (objectGUI[i].second == object) return  objectGUI[i].first;
    }
    return false;
}

//*******************************************************************************************************************
void ModifyObject::updateValues()
{
    if (buttonUpdate == NULL // || !buttonUpdate->isEnabled()
       ) return;

    //Make the update of all the values
    if (node)
    {
        //If the current element is a node of the graph, we first apply the transformations
        if (dialogFlags_.REINIT_FLAG && dynamic_cast< Node *>(node))
        {
            Node* current_node = dynamic_cast< Node *>(node);
            if (!(transformation[0]->getFloatValue() == 0 &&
                    transformation[1]->getFloatValue() == 0 &&
                    transformation[2]->getFloatValue() == 0 &&
                    transformation[3]->getFloatValue() == 0 &&
                    transformation[4]->getFloatValue() == 0 &&
                    transformation[5]->getFloatValue() == 0 &&
                    transformation[6]->getFloatValue() == 1 &&
                    transformation[7]->getFloatValue() == 1 &&
                    transformation[8]->getFloatValue() == 1 ))
            {

                sofa::simulation::TransformationVisitor transform;
                transform.setTranslation(transformation[0]->getFloatValue(),transformation[1]->getFloatValue(),transformation[2]->getFloatValue());
                transform.setRotation(transformation[3]->getFloatValue(),transformation[4]->getFloatValue(),transformation[5]->getFloatValue());
                transform.setScale(transformation[6]->getFloatValue(),transformation[7]->getFloatValue(),transformation[8]->getFloatValue());
                transform.execute(current_node);

                transformation[0]->setFloatValue(0);
                transformation[1]->setFloatValue(0);
                transformation[2]->setFloatValue(0);

                transformation[3]->setFloatValue(0);
                transformation[4]->setFloatValue(0);
                transformation[5]->setFloatValue(0);

                transformation[6]->setFloatValue(1);
                transformation[7]->setFloatValue(1);
                transformation[8]->setFloatValue(1);

            }
        }

        //Special Treatment for visual flags
        for (unsigned int index_object=0; index_object < objectGUI.size(); ++index_object)
        {
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
        }

        if (dialogFlags_.REINIT_FLAG)
        {
            if (sofa::core::objectmodel::BaseObject *obj = dynamic_cast< sofa::core::objectmodel::BaseObject* >(node))
            {
                obj->reinit();
            }
            else if (Node *n = dynamic_cast< Node *>(node)) n->reinit();
        }
    }

    if (visualContentModified) updateContext(dynamic_cast< Node *>(node));

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
    if (!node->nodeInVisualGraph.empty())
    {
        node->nodeInVisualGraph->copyContext((core::objectmodel::Context&) *(node->getContext()));
        node->nodeInVisualGraph->execute< sofa::simulation::UpdateVisualContextVisitor >();
    }
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
        if (energy_curve[0]) energy_curve[0]->setRawData(&history[0],&(energy_history[0][0]), history.size());
        if (energy_curve[1]) energy_curve[1]->setRawData(&history[0],&(energy_history[1][0]), history.size());
        if (energy_curve[2]) energy_curve[2]->setRawData(&history[0],&(energy_history[2][0]), history.size());
        if (graphEnergy) graphEnergy->replot();
    }

}

void ModifyObject::updateListViewItem()
{
    if(node)
    {
        if( !dynamic_cast< Node *>(node))
        {
            std::string name=item_->text(0).ascii();
            std::string::size_type pos = name.find(' ');
            if (pos != std::string::npos)
                name.resize(pos);
            name += "  ";
            name += node->getName();
            item_->setText(0,name.c_str());
        }
        else if (dynamic_cast< Node *>(node))
            item_->setText(0,node->getName().c_str());
    }
    if(data_)
    {
        Q3ListViewItem* parent = item_->parent();
        std::string name = parent->text(0).ascii();
        std::string::size_type pos = name.find(' ');
        if (pos != std::string::npos)
            name.resize(pos);
        name += "  ";
        name += data_->getOwner()->getName();
        parent->setText(0,name.c_str());
    }

}

//**************************************************************************************************************************************
//Called each time a new step of the simulation if computed
void ModifyObject::updateTables()
{
    emit updateDataWidgets();
    if (graphEnergy) updateHistory();
}




} // namespace qt

} // namespace gui

} // namespace sofa
