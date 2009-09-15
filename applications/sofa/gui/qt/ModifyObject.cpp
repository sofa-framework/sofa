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

#include <sofa/gui/qt/ModifyObject.h>
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
#define SIZE_TEXT     75

namespace sofa
{

namespace gui
{

namespace qt
{

SOFA_LINK_CLASS(GraphDataWidget);
SOFA_LINK_CLASS(SimpleDataWidget);
SOFA_LINK_CLASS(StructDataWidget);
SOFA_LINK_CLASS(TableDataWidget);

using namespace  sofa::defaulttype;
using sofa::core::objectmodel::BaseData;


#ifndef SOFA_QT4
typedef QScrollView Q3ScrollView;
#endif

ModifyObject::ModifyObject(void *Id_, core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked,  QWidget* parent_, const char* name, bool, Qt::WFlags /*f*/ )
    : parent(parent_), node(NULL), Id(Id_),visualContentModified(false)
{
    //Title of the Dialog
    setCaption(name);

    HIDE_FLAG = true;
    READONLY_FLAG = true;
    EMPTY_FLAG = false;
    RESIZABLE_FLAG = false;
    REINIT_FLAG = true;

    outputTab = warningTab = NULL;
    energy_curve[0]=NULL;	        energy_curve[1]=NULL;	        energy_curve[2]=NULL;

    logWarningEdit=NULL; logOutputEdit=NULL;

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
            else if ((*it).second->getGroup() == std::string("BIG"))
            {
                std::cout << (*it).first << " in new tab" << std::endl;
                emptyTab = true;
                currentTab= new QWidget();
                currentTabLayout = new QVBoxLayout( currentTab, 0, 1, QString("tabBIGLayout") + QString::number(counterWidget));
                dialogTab->addTab(currentTab, QString((*it).first.c_str()));
            }

            {
// 		  if (hideData(it->second)) continue;
                std::string box_name(oss.str());
                box = new Q3GroupBox(currentTab, QString(box_name.c_str()));
                box->setColumns(4);
                box->setTitle(QString((*it).first.c_str()));

                std::string label_text=(*it).second->getHelp();

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

                DataWidget::CreatorArgument dwarg;
                dwarg.node = node;
                dwarg.name = (*it).first;
                dwarg.data = (*it).second;
                dwarg.readOnly = (dwarg.data->isReadOnly() && READONLY_FLAG);
                dwarg.dialog = this;
                dwarg.parent = box;
                std::string widget = dwarg.data->getWidget();
                box->setColumns(2);
                DataWidget* dw;
                if (widget.empty())
                    dw = DataWidgetFactory::CreateAnyObject(dwarg);
                else
                    dw = DataWidgetFactory::CreateObject(dwarg.data->getWidget(), dwarg);
                if (dw == NULL)
                {
                    box->setColumns(4);
                    std::cout << "WIDGET FAILED for data " << dwarg.name << " : " << dwarg.data->getValueTypeString() << std::endl;
                }
                if (dw != NULL)
                {
                    //std::cout << "WIDGET created for data " << dwarg.data << " : " << dwarg.name << " : " << dwarg.data->getValueTypeString() << std::endl;
                    dataWidgets[dwarg.data] = dw;
                    counterWidget+=dw->sizeWidget();
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


            tabLayout->addStretch();

            updateConsole();
            if (outputTab)  dialogTab->addTab(outputTab,  QString("Outputs"));
            if (warningTab) dialogTab->addTab(warningTab, QString("Warnings"));
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


//******************************************************************************************
void ModifyObject::updateConsole()
{
    //Console Warnings
    if ( !node->sendl.getWarnings().empty())
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

        logWarningEdit->setText(QString(node->sendl.getWarnings().c_str()));
        logWarningEdit->moveCursor(Q3TextEdit::MoveEnd, false);
        logWarningEdit->ensureCursorVisible();

    }
    //Console Outputs
    if ( !node->sendl.getOutputs().empty())
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

        logOutputEdit->setText(QString(node->sendl.getOutputs().c_str()));
        logOutputEdit->moveCursor(Q3TextEdit::MoveEnd, false);
        logOutputEdit->ensureCursorVisible();
    }
}

//*******************************************************************************************************************
void ModifyObject::changeValue()
{
    const QObject* s = sender();
    for (DataWidgetMap::iterator it = dataWidgets.begin(), itend = dataWidgets.end(); it != itend; ++it)
    {
        DataWidget* dw = it->second;
        if (dw->processChange(s))
        {
            s = NULL;
            break;
        }
    }
    if (s != NULL)
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
        std::string oldName = node->getName();
        //If the current element is a node of the graph, we first apply the transformations
        if (REINIT_FLAG && dynamic_cast< Node *>(node))
        {
            Node* current_node = dynamic_cast< Node *>(node);
            if (!(transformation[0]->getFloatValue() == 0 &&
                    transformation[1]->getFloatValue() == 0 &&
                    transformation[2]->getFloatValue() == 0 &&
                    transformation[3]->getFloatValue() == 0 &&
                    transformation[4]->getFloatValue() == 0 &&
                    transformation[5]->getFloatValue() == 0 &&
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
        for (DataWidgetMap::iterator it = dataWidgets.begin(), itend = dataWidgets.end(); it != itend; ++it)
        {
            DataWidget* dw = it->second;
            dw->writeToData();
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
        std::string newName = node->getName();
        if (newName != oldName)
        {
            if( !dynamic_cast< Node *>(node))
            {
                std::string name=item->text(0).ascii();
                std::string::size_type pos = name.find(' ');
                if (pos != std::string::npos)
                    name.resize(pos);
                name += "  ";

                name+=newName;
                item->setText(0,name.c_str());
            }
            else if (dynamic_cast< Node *>(node))
                item->setText(0,newName.c_str());
        }

        if (REINIT_FLAG)
        {
            if (sofa::core::objectmodel::BaseObject *obj = dynamic_cast< sofa::core::objectmodel::BaseObject* >(node))
            {
                obj->reinit();
            }
            else if (Node *n = dynamic_cast< Node *>(node)) n->reinit();
        }
    }

    if (visualContentModified) updateContext(dynamic_cast< Node *>(node));

    updateTables();
    emit (objectUpdated());
    buttonUpdate->setEnabled(false);
    visualContentModified = false;


    for (DataWidgetMap::iterator it = dataWidgets.begin(), itend = dataWidgets.end(); it != itend; ++it)
    {
        DataWidget* dw = it->second;
        dw->updateVisibility();
    }

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
        if (energy_curve[0]) energy_curve[0]->setRawData(&history[0],&(energy_history[0][0]), history.size());
        if (energy_curve[1]) energy_curve[1]->setRawData(&history[0],&(energy_history[1][0]), history.size());
        if (energy_curve[2]) energy_curve[2]->setRawData(&history[0],&(energy_history[2][0]), history.size());
        if (graphEnergy) graphEnergy->replot();
    }

}




void ModifyObject::updateTextEdit()
{
    updateConsole();

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

    if (graphEnergy) updateHistory();
    updateTextEdit();
    for (DataWidgetMap::iterator it = dataWidgets.begin(), itend = dataWidgets.end(); it != itend; ++it)
    {
        DataWidget* dw = it->second;
        dw->update();
    }
    std::list< std::pair< Q3Table*, BaseData*> >::iterator it_list_Table;
    for (it_list_Table = list_Table.begin(); it_list_Table != list_Table.end(); it_list_Table++)
    {

        if ( dynamic_cast < Data<sofa::component::misc::Monitor< Vec3Types >::MonitorData > *> ( (*it_list_Table).second ) )
        {
            std::list< std::pair< Q3Table*, BaseData*> >::iterator it_center = it_list_Table;
            it_list_Table++;
            std::list< std::pair< Q3Table*, BaseData*> >::iterator it_center2 = it_list_Table;
            it_list_Table++; //two times because a monitor is composed of 3 baseData
            createTable((*it_list_Table).second,NULL,(*it_center).first,(*it_center2).first, (*it_list_Table).first);
        }
        else createTable((*it_list_Table).second,NULL,(*it_list_Table).first);
    }
}


//**************************************************************************************************************************************
//create or update an existing table with the contents of a field
bool ModifyObject::createTable( BaseData* field,Q3GroupBox *box, Q3Table* vectorTable, Q3Table* vectorTable2, Q3Table* vectorTable3)
{
    //********************************************************************************************************//
    if(  Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData >   * >( field))
    {
        return createMonitorQtTable < Vec3dTypes >(ff,box,vectorTable, vectorTable2, vectorTable3);
    }

    //********************************************************************************************************//
    else if(  Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData >   * >( field))
    {
        return createMonitorQtTable < Vec3fTypes >(ff,box,vectorTable, vectorTable2, vectorTable3);
    }

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
    if (  Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3dTypes >::MonitorData > * >( it_list_table->second))
    {
        storeMonitorQtTable< Vec3dTypes >(it_list_table, ff);
    }
    //**************************************************************************************************************************************
    else if (  Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData >  *ff = dynamic_cast< Data<sofa::component::misc::Monitor< Vec3fTypes >::MonitorData > * >( it_list_table->second))
    {
        storeMonitorQtTable< Vec3fTypes >(it_list_table, ff);
    }
    //**************************************************************************************************************************************
}

//********************************************************************************************************************
//TEMPLATE FUNCTIONS
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
