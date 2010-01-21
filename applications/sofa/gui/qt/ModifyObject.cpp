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
#include <sofa/gui/qt/QDisplayDataWidget.h>
#include <sofa/gui/qt/QDataDescriptionWidget.h>
#include <sofa/gui/qt/QTabulationModifyObject.h>

#include <iostream>
#ifdef SOFA_QT4
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTabWidget>
#include <Q3ListView>
#else
#include <qpushbutton.h>
#include <qlayout.h>
#include <qtabwidget.h>
#include <qtextedit.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{


ModifyObject::ModifyObject(
    void *Id,
    Q3ListViewItem* item_clicked,
    QWidget* parent,
    const ModifyObjectFlags& dialogFlags,
    const char* name,
    bool modal, Qt::WFlags f )
    :QDialog(parent, name, modal, f),
     Id_(Id),
     item_(item_clicked),
     node(NULL),
     data_(NULL),
     dialogFlags_(dialogFlags),
     outputTab(NULL),
     logOutputEdit(NULL),
     warningTab(NULL),
     logWarningEdit(NULL),
     transformation(NULL),
     energy(NULL)
{
    setCaption(name);
}

void ModifyObject::createDialog(core::objectmodel::Base* base)
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



    bool isNode = (dynamic_cast< simulation::Node *>(node) != NULL);

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
        const std::vector< std::pair<std::string, core::objectmodel::BaseData*> >& fields = node->getFields();

        std::vector< QTabulationModifyObject* > propertyTabs;
        std::vector< QTabulationModifyObject* > visualizationTabs;

        enum TABTYPE {PROPERTYTAB, VISUALIZATIONTAB};
        for( std::vector< std::pair<std::string, core::objectmodel::BaseData*> >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {
            core::objectmodel::BaseData* data=it->second;

            //For each Data of the current Object
            //We determine where it belongs:
            TABTYPE type;
            //Find if the name starts with show or draw
            const std::string name = ((*it).first).substr(0,4);
            if (name == "show" || name == "draw") type=VISUALIZATIONTAB;
            else                                  type=PROPERTYTAB;

            QTabulationModifyObject* currentTab=NULL;
            switch(type)
            {
            case PROPERTYTAB:
            {
                if (propertyTabs.empty())
                {
                    propertyTabs.push_back(new QTabulationModifyObject(this,node, item_,propertyTabs.size()+1));
                    if(dialogFlags_.REINIT_FLAG && isNode)
                    {
                        transformation = new QTransformationWidget(propertyTabs.back(), QString("Transformation"));
                        propertyTabs.back()->layout()->add( transformation );
                        propertyTabs.back()->externalWidgetAddition(transformation->getNumWidgets());
                        connect( transformation, SIGNAL(TransformationDirty(bool)), buttonUpdate, SLOT( setEnabled(bool) ) );
                    }
                }
                else if ( propertyTabs.back()->isFull() || data->getGroup() == std::string("BIG"))
                {
                    propertyTabs.push_back(new QTabulationModifyObject(this,node, item_,propertyTabs.size()+1));
                }
                currentTab = propertyTabs.back();
                break;
            }
            case VISUALIZATIONTAB:
            {
                if (visualizationTabs.empty() )
                {
                    visualizationTabs.push_back(new QTabulationModifyObject(this,node, item_,visualizationTabs.size()+1));
                    if (isNode)
                    {
                        displayFlag = new QDisplayFlagWidget(visualizationTabs.back(),dynamic_cast< simulation::Node *>(node),QString("Visualization Flags"));
                        visualizationTabs.back()->layout()->add( displayFlag );
                        visualizationTabs.back()->externalWidgetAddition(displayFlag->getNumWidgets());

                        connect(buttonUpdate,   SIGNAL(clicked() ),          displayFlag, SLOT( applyFlags() ) );
                        connect(buttonOk,       SIGNAL(clicked() ),          displayFlag, SLOT( applyFlags() ) );
                        connect(displayFlag, SIGNAL( DisplayFlagDirty(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
                    }
                }
                else if ( visualizationTabs.back()->isFull())
                {
                    visualizationTabs.push_back(new QTabulationModifyObject(this,node, item_,visualizationTabs.size()+1));
                }

                currentTab = visualizationTabs.back();
                if ( dynamic_cast< Data<int> * >( data ) &&
                        ( (*it).first == "showVisualModels" || (*it).first == "showBehaviorModels" ||  (*it).first == "showCollisionModels" ||  (*it).first == "showBoundingCollisionModels" ||  (*it).first == "showMappings" ||  (*it).first == "showMechanicalMappings" ||  (*it).first == "showForceFields" ||  (*it).first == "showInteractionForceFields" ||  (*it).first == "showWireFrame" ||  (*it).first == "showNormals" ) ) continue;
                break;
            }
            }

            currentTab->addData(data, getFlags());
            connect(buttonUpdate,   SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
            connect(buttonOk,       SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
            connect(this,           SIGNAL(updateDataWidgets()), currentTab, SLOT( updateWidgetValue()) );

            connect(currentTab, SIGNAL( TabDirty(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
        }

        //Add Property Tabs
        for (unsigned int i=0; i<propertyTabs.size(); ++i)
        {
            QString nameTab;
            if (propertyTabs.size() == 1) nameTab="Property";
            else                          nameTab=QString("Property ")+QString::number(propertyTabs[i]->getIndex()) + "/" + QString::number(propertyTabs.size());
            dialogTab->addTab(propertyTabs[i],nameTab);
            propertyTabs[i]->addStretch();
        }
        //Add Visualization Tabs
        for (unsigned int i=0; i<visualizationTabs.size(); ++i)
        {
            QString nameTab;
            if (visualizationTabs.size() == 1) nameTab="Visualization";
            else                               nameTab=QString("Visualization ")+QString::number(visualizationTabs[i]->getIndex()) + "/" + QString::number(visualizationTabs.size());
            dialogTab->addTab(visualizationTabs[i], nameTab);
            visualizationTabs[i]->addStretch();
        }


        //Energy Widget
        if (simulation::Node* real_node = dynamic_cast< simulation::Node* >(node))
        {
            if (dialogFlags_.REINIT_FLAG && (!real_node->mass.empty() || !real_node->forceField.empty() ) )
            {
                energy = new QEnergyStatWidget(dialogTab, real_node);
                dialogTab->addTab(energy,QString("Energy Stats"));
            }
        }

        // Info Widget
        {
            QDataDescriptionWidget* description=new QDataDescriptionWidget(dialogTab, node);
            dialogTab->addTab(description, QString("Infos"));
        }

        //Console
        {
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
        resize( QSize(300, 130).expandedTo(minimumSizeHint()) );
    }
}

void ModifyObject::createDialog(core::objectmodel::BaseData* data)
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
    connect(this, SIGNAL(updateDataWidgets()), displaydatawidget, SLOT(UpdateWidgets()) );
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
void ModifyObject::updateValues()
{
    if (buttonUpdate == NULL // || !buttonUpdate->isEnabled()
       ) return;

    //Make the update of all the values
    if (node)
    {
        bool isNode =( dynamic_cast< simulation::Node *>(node) != 0);
        //If the current element is a node of the graph, we first apply the transformations
        if (dialogFlags_.REINIT_FLAG && isNode)
        {
            simulation::Node* current_node = dynamic_cast< simulation::Node *>(node);
            if (!transformation->isDefaultValues())
                transformation->applyTransformation(current_node);
            transformation->setDefaultValues();
        }

        if (isNode)
        {
            displayFlag->applyFlags();
        }

        if (dialogFlags_.REINIT_FLAG)
        {
            if (sofa::core::objectmodel::BaseObject *obj = dynamic_cast< sofa::core::objectmodel::BaseObject* >(node))
            {
                obj->reinit();
            }
            else if (simulation::Node *n = dynamic_cast< simulation::Node *>(node)) n->reinit();
        }

    }

    emit (objectUpdated());
    buttonUpdate->setEnabled(false);
}


//*******************************************************************************************************************

void ModifyObject::updateListViewItem()
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

//**************************************************************************************************************************************
//Called each time a new step of the simulation if computed
void ModifyObject::updateTables()
{
    emit updateDataWidgets();
    if (energy)
    {
        energy->step();
        if (dialogTab->currentPage() == energy) energy->updateVisualization();
    }
}




} // namespace qt

} // namespace gui

} // namespace sofa
