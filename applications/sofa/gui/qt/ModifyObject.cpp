/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "ModifyObject.h"
#include "DataWidget.h"
#include "QDisplayDataWidget.h"
#include "QDataDescriptionWidget.h"
#include "QTabulationModifyObject.h"
#include <QTextBrowser>
#include <QDesktopServices>
#include <sofa/gui/qt/QTransformationWidget.h>
#ifdef SOFA_HAVE_QWT
#include <sofa/gui/qt/QEnergyStatWidget.h>
#include <sofa/gui/qt/QMomentumStatWidget.h>
#endif

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::Message ;


#include <iostream>

#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTabWidget>
#include <QTreeWidget>
#include <QScrollArea>
#include <QApplication>
#include <QDesktopWidget>


// uncomment to show traces of GUI operations in this file
// #define DEBUG_GUI

namespace sofa
{

namespace gui
{

namespace qt
{


ModifyObject::ModifyObject(void *Id,
                           QTreeWidgetItem* item_clicked,
                           QWidget* parent,
                           const ModifyObjectFlags& dialogFlags,
                           const char* name,
                           bool modal, Qt::WindowFlags f )
    :QDialog(parent, f),
      Id_(Id),
      item_(item_clicked),
      node(NULL),
      data_(NULL),
      dialogFlags_(dialogFlags),
      messageTab(NULL),
      messageEdit(NULL),
      transformation(NULL)
    #ifdef SOFA_HAVE_QWT
    ,energy(NULL)
    ,momentum(NULL)
    #endif
{
    setWindowTitle(name);
    //setObjectName(name);
    setModal(modal);
}

void ModifyObject::createDialog(core::objectmodel::Base* base)
{
    if(base == NULL)
    {
        return;
    }
#ifdef DEBUG_GUI
    std::cout << "GUI: (base) createDialog(" << base->getClassName() << " " << base->getName() << ")" << std::endl;
#endif
#ifdef DEBUG_GUI
    std::cout << "GUI>emit beginObjectModification(" << base->getName() << ")" << std::endl;
#endif
    emit beginObjectModification(base);
#ifdef DEBUG_GUI
    std::cout << "GUI<emit beginObjectModification(" << base->getName() << ")" << std::endl;
#endif
    node = base;
    data_ = NULL;

    //Layout to organize the whole window
    QVBoxLayout *generalLayout = new QVBoxLayout(this);
    generalLayout->setObjectName("generalLayout");
    generalLayout->setMargin(0);
    generalLayout->setSpacing(1);
    //Tabulation widget
    dialogTab = new QTabWidget(this);
    //generalLayout->addWidget(dialogTab);

    //add a scrollable area for data properties
    QScrollArea* m_scrollArea = new QScrollArea();
    //    const int screenHeight = QApplication::desktop()->height();
    m_scrollArea->setMinimumSize(600,QApplication::desktop()->height() * 0.75);
    m_scrollArea->setWidgetResizable(true);
    dialogTab->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_scrollArea->setWidget(dialogTab);
    generalLayout->addWidget(m_scrollArea);

    //    generalLayout->addWidget(dialogTab);

    connect(dialogTab, SIGNAL( currentChanged(int)), this, SLOT( updateTables()));

    //    bool isNode = (dynamic_cast< simulation::Node *>(node) != NULL);

    buttonUpdate = new QPushButton( this );
    buttonUpdate->setObjectName("buttonUpdate");
    buttonUpdate->setText("&Update");
    buttonUpdate->setEnabled(false);
    QPushButton *buttonOk = new QPushButton( this );
    buttonOk->setObjectName("buttonOk");
    buttonOk->setText( tr( "&OK" ) );

    QPushButton *buttonCancel = new QPushButton( this );
    buttonCancel->setObjectName("buttonCancel");
    buttonCancel->setText( tr( "&Cancel" ) );

    // displayWidget
    if (node)
    {
        const sofa::core::objectmodel::Base::VecData& fields = node->getDataFields();
        const sofa::core::objectmodel::Base::VecLink& links = node->getLinks();

        std::map< std::string, std::vector<QTabulationModifyObject* > > groupTabulation;

        //If we operate on a Node, we have to ...
        /*if(isNode)
        {
            if (dialogFlags_.REINIT_FLAG)
            {
                //add the widgets to apply some basic transformations

                m_tabs.push_back(new QTabulationModifyObject(this,node, item_,1));
                groupTabulation[std::string("Property")].push_back(m_tabs.back());
                connect(m_tabs.back(), SIGNAL(nodeNameModification(simulation::Node *)), this, SIGNAL(nodeNameModification(simulation::Node *)));

                transformation = new QTransformationWidget(m_tabs.back(), QString("Transformation"));
                m_tabs.back()->layout()->add(transformation);
                m_tabs.back()->externalWidgetAddition(transformation->getNumWidgets());
                connect( transformation, SIGNAL(TransformationDirty(bool)), buttonUpdate, SLOT( setEnabled(bool) ) );
                connect( transformation, SIGNAL(TransformationDirty(bool)), this, SIGNAL( componentDirty(bool) ) );
            }
        }*/

        std::vector<std::string> tabNames;
        //Put first the Property Tab
        tabNames.push_back("Property");

        for( sofa::core::objectmodel::Base::VecData::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {
            core::objectmodel::BaseData* data=*it;
            if (!data)
            {
                dmsg_error("ModifyObject") << "NULL Data in '" << node->getName() << "'" ;
                continue;
            }

            if (data->getName().empty()) continue; // ignore unnamed data

            if (!data->getGroup())
            {
                dmsg_error("ModifyObject") << "NULL group for Data (" << data->getName() << "() in (" << node->getName() << "'";
                continue;
            }

            //For each Data of the current Object
            //We determine where it belongs:
            std::string currentGroup=data->getGroup();

            if (currentGroup.empty()) currentGroup="Property";

            // Ignore the data in group "Infos" so they can be putted in the real Infos panel that is
            // handled in a different way (see QDataDescriptionWidget)
            if (currentGroup == "Infos")
                continue;

#ifdef DEBUG_GUI
            std::cout << "GUI: add Data " << data->getName() << " in " << currentGroup << std::endl;
#endif
            QTabulationModifyObject* currentTab=NULL;

            std::vector<QTabulationModifyObject* > &tabs=groupTabulation[currentGroup];
            bool newTab = false;
            if (tabs.empty()) tabNames.push_back(currentGroup);
            if (tabs.empty() || tabs.back()->isFull())
            {
                newTab = true;
                m_tabs.push_back(new QTabulationModifyObject(this,node, item_,tabs.size()+1));
                tabs.push_back(m_tabs.back());
            }
            currentTab = tabs.back();
            currentTab->addData(data, getFlags());
            if (newTab)
            {
                connect(buttonUpdate,   SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
                connect(buttonOk,       SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
                connect(this,           SIGNAL(updateDataWidgets()), currentTab, SLOT( updateWidgetValue()) );

                connect(currentTab, SIGNAL( TabDirty(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
                connect(currentTab, SIGNAL( TabDirty(bool) ), this, SIGNAL( componentDirty(bool) ) );
            }

#ifdef DEBUG_GUI
            std::cout << "GUI: added Data " << data->getName() << " in " << currentGroup << std::endl;
#endif
        }
#ifdef DEBUG_GUI
        std::cout << "GUI: end Data" << std::endl;
#endif

        for( sofa::core::objectmodel::Base::VecLink::const_iterator it = links.begin(); it!=links.end(); ++it)
        {
            core::objectmodel::BaseLink* link=*it;

            if (link->getName().empty()) continue; // ignore unnamed links
            if (!link->storePath() && link->getSize() == 0) continue; // ignore empty links

            //For each Link of the current Object
            //We determine where it belongs:
            std::string currentGroup="Links";

#ifdef DEBUG_GUI
            std::cout << "GUI: add Link " << link->getName() << " in " << currentGroup << std::endl;
#endif

            QTabulationModifyObject* currentTab=NULL;

            std::vector<QTabulationModifyObject* > &tabs=groupTabulation[currentGroup];
            if (tabs.empty()) tabNames.push_back(currentGroup);
            if (tabs.empty() || tabs.back()->isFull())
            {
                m_tabs.push_back(new QTabulationModifyObject(this,node, item_,tabs.size()+1));
                tabs.push_back(m_tabs.back() );
            }
            currentTab = tabs.back();

            currentTab->addLink(link, getFlags());
            connect(buttonUpdate,   SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
            connect(buttonOk,       SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
            connect(this,           SIGNAL(updateDataWidgets()), currentTab, SLOT( updateWidgetValue()) );

            connect(currentTab, SIGNAL( TabDirty(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
            connect(currentTab, SIGNAL( TabDirty(bool) ), this, SIGNAL( componentDirty(bool) ) );
        }
#ifdef DEBUG_GUI
        std::cout << "GUI: end Link" << std::endl;
#endif

        //std::map< std::string, std::vector<QTabulationModifyObject* > >::iterator it;
        //for (it=groupTabulation.begin();it!=groupTabulation.end();++it)
        //{
        //    const std::string &groupName=it->first;
        //    std::vector<QTabulationModifyObject* > &tabs=it->second;
        for (std::vector<std::string>::const_iterator it = tabNames.begin(), itend = tabNames.end(); it != itend; ++it)
        {
            const std::string& groupName = *it;
            std::vector<QTabulationModifyObject* > &tabs=groupTabulation[groupName];

            for (unsigned int i=0; i<tabs.size(); ++i)
            {
                QString nameTab;
                if (tabs.size() == 1) nameTab=groupName.c_str();
                else                  nameTab=QString(groupName.c_str())+ " " + QString::number(tabs[i]->getIndex()) + "/" + QString::number(tabs.size());
#ifdef DEBUG_GUI
                std::cout << "GUI: add Tab " << nameTab.toStdString() << std::endl;
#endif
                dialogTab->addTab(tabs[i],nameTab);
                tabs[i]->addStretch();
            }
        }

#ifdef SOFA_HAVE_QWT
        //Energy Widget
        if (simulation::Node* real_node = dynamic_cast< simulation::Node* >(node))
        {
            if (dialogFlags_.REINIT_FLAG )
            {
                energy = new QEnergyStatWidget(dialogTab, real_node);
                dialogTab->addTab(energy,QString("Energy Stats"));
            }
        }

        //Momentum Widget
        if (simulation::Node* real_node = dynamic_cast< simulation::Node* >(node))
        {
            if (dialogFlags_.REINIT_FLAG )
            {
                momentum = new QMomentumStatWidget(dialogTab, real_node);
                dialogTab->addTab(momentum,QString("Momentum Stats"));
            }
        }
#endif

        /// Info Widget
        {
            QDataDescriptionWidget* description=new QDataDescriptionWidget(dialogTab, node);
            dialogTab->addTab(description, QString("Infos"));
        }

        /// Message widget
        {
            updateConsole();
            if (messageTab)
            {
                std::stringstream tmp;
                int numMessages = node->countLoggedMessages({Message::Info, Message::Advice, Message::Deprecated,
                                                             Message::Error, Message::Warning, Message::Fatal});
                tmp << "Messages(" << numMessages << ")" ;
                dialogTab->addTab(messageTab, QString::fromStdString(tmp.str()));
            }
        }


        //Adding buttons at the bottom of the dialog
        QHBoxLayout *lineLayout = new QHBoxLayout( 0);
        lineLayout->setMargin(0);
        lineLayout->setSpacing(6);
        lineLayout->setObjectName("Button Layout");
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
        resize( QSize(450, 130).expandedTo(minimumSizeHint()) );
    }
}

void ModifyObject::clearMessages()
{
    node->clearWarnings();
    messageEdit->clear();

    std::stringstream tmp;
    int numMessages = node->countLoggedMessages({Message::Info, Message::Advice, Message::Deprecated,
                                                 Message::Error, Message::Warning, Message::Fatal});
    tmp << "Messages(" << numMessages << ")" ;

    dialogTab->setTabText(dialogTab->indexOf(messageTab), QString::fromStdString(tmp.str()));
}



void ModifyObject::createDialog(core::objectmodel::BaseData* data)
{
    data_ = data;
    node = NULL;

#ifdef DEBUG_GUI
    std::cout << "GUI>emit beginDataModification("<<data->getName()<<")" << std::endl;
#endif
    emit beginDataModification(data);
#ifdef DEBUG_GUI
    std::cout << "GUI<emit beginDataModification("<<data->getName()<<")" << std::endl;
#endif

#ifdef DEBUG_GUI
    std::cout << "GUI: createDialog( Data<" << data->getValueTypeString() << "> " << data->getName() << ")" << std::endl;
#endif

    QVBoxLayout *generalLayout = new QVBoxLayout(this);
    generalLayout->setMargin(0);
    generalLayout->setSpacing(1);
    generalLayout->setObjectName("generalLayout");
    QHBoxLayout *lineLayout = new QHBoxLayout( 0);
    lineLayout->setMargin(0);
    lineLayout->setSpacing(6);
    lineLayout->setObjectName("Button Layout");
    buttonUpdate = new QPushButton( this );
    buttonUpdate->setObjectName("buttonUpdate");
    buttonUpdate->setText("&Update");
    buttonUpdate->setEnabled(false);
    QPushButton *buttonOk = new QPushButton( this );
    buttonOk->setObjectName("buttonOk");
    buttonOk->setText( tr( "&OK" ) );
    QPushButton *buttonCancel = new QPushButton( this );
    buttonCancel->setObjectName("buttonCancel");
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
    connect(displaydatawidget, SIGNAL( WidgetDirty(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
    connect(displaydatawidget, SIGNAL( WidgetDirty(bool) ), this, SIGNAL( componentDirty(bool) ) );
    connect(buttonOk, SIGNAL(clicked() ), displaydatawidget, SLOT( UpdateData() ) );
    connect(displaydatawidget, SIGNAL(DataOwnerDirty(bool)), this, SLOT( updateListViewItem() ) );
    connect( buttonOk,       SIGNAL( clicked() ), this, SLOT( accept() ) );
    connect( buttonCancel,   SIGNAL( clicked() ), this, SLOT( reject() ) );
    connect(this, SIGNAL(updateDataWidgets()), displaydatawidget, SLOT(UpdateWidgets()) );
}

const std::string toHtmlString(const Message::Type t)
{
    switch(t)
    {
    case Message::Info:
        return "<font color='green'>Info</font>";
    case Message::Advice:
        return "<font color='green'>Advice</font>";
    case Message::Deprecated:
        return "<font color='grey'>Deprecated</font>";
    case Message::Warning:
        return "<font color='darkcyan'>Warning</font>";
    case Message::Error:
        return "<font color='red'>Error</font>";
    case Message::Fatal:
        return "Fatal";
    default:
        return "Undefine";
    }
    return "Undefine";
}

class ClickableTextEdit : public QTextEdit
{
public:
    Q_OBJECT

public:
    ClickableTextEdit(QWidget* w) : QTextEdit(w) {

    }


};



void ModifyObject::openExternalBrowser(const QUrl &link)
{
    QDesktopServices::openUrl(link) ;
}

//******************************************************************************************
void ModifyObject::updateConsole()
{
    if (!messageEdit)
    {
        messageTab = new QWidget();
        QVBoxLayout* tabLayout = new QVBoxLayout( messageTab);
        tabLayout->setMargin(0);
        tabLayout->setSpacing(1);
        tabLayout->setObjectName("tabWarningLayout");
        QPushButton *buttonClearWarnings = new QPushButton(messageTab);
        buttonClearWarnings->setObjectName("buttonClearWarnings");
        tabLayout->addWidget(buttonClearWarnings);
        buttonClearWarnings->setText( tr("&Clear"));
        connect( buttonClearWarnings, SIGNAL( clicked()), this, SLOT( clearMessages()));

        messageEdit = new QTextBrowser(messageTab);
        //messageEdit->backwardAvailable(false);
        connect(messageEdit, SIGNAL(anchorClicked(const QUrl&)), this, SLOT(openExternalBrowser(const QUrl&)));
        messageEdit->setObjectName("WarningEdit");
        messageEdit->setOpenExternalLinks(false);
        messageEdit->setOpenLinks(false);
        tabLayout->addWidget( messageEdit );
        messageEdit->setReadOnly(true);
    }

    if (dialogTab->currentWidget() == messageTab)
    {
        std::stringstream tmp;
        tmp << "<table>";
        tmp << "<tr><td><td><td><td>" ;
        m_numMessages = 0 ;
        for(const Message& message : node->getLoggedMessages())
        {
            tmp << "<tr>";
            tmp << "<td>["<<toHtmlString(message.type())<<"]</td>" ;
            tmp << "<td><i>" << message.messageAsString() << "</i></td>" ;
            m_numMessages++;
        }
        tmp << "</table>";

        messageEdit->setHtml(QString(tmp.str().c_str()));
        messageEdit->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
        messageEdit->ensureCursorVisible();
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
        if (transformation && dialogFlags_.REINIT_FLAG && isNode)
        {
            simulation::Node* current_node = dynamic_cast< simulation::Node *>(node);
            if (!transformation->isDefaultValues())
                transformation->applyTransformation(current_node);
            transformation->setDefaultValues();
        }



        if (dialogFlags_.REINIT_FLAG)
        {
            if (sofa::core::objectmodel::BaseObject *obj = dynamic_cast< sofa::core::objectmodel::BaseObject* >(node))
            {
                obj->reinit();
            }
            else if (simulation::Node *n = dynamic_cast< simulation::Node *>(node)) n->reinit(sofa::core::ExecParams::defaultInstance());
        }

    }

#ifdef DEBUG_GUI
    std::cout << "GUI>emit objectUpdated()" << std::endl;
#endif
    emit (objectUpdated());
#ifdef DEBUG_GUI
    std::cout << "GUI<emit objectUpdated()" << std::endl;
#endif

    if (node)
    {
#ifdef DEBUG_GUI
        std::cout << "GUI>emit endObjectModification("<<node->getName()<<")" << std::endl;
#endif
        emit endObjectModification(node);
#ifdef DEBUG_GUI
        std::cout << "GUI<emit endObjectModification("<<node->getName()<<")" << std::endl;
#endif
#ifdef DEBUG_GUI
        std::cout << "GUI>emit beginObjectModification("<<node->getName()<<")" << std::endl;
#endif
        emit beginObjectModification(node);
#ifdef DEBUG_GUI
        std::cout << "GUI<emit beginObjectModification("<<node->getName()<<")" << std::endl;
#endif
    }

    buttonUpdate->setEnabled(false);
}


//*******************************************************************************************************************

void ModifyObject::updateListViewItem()
{
    QTreeWidgetItem* parent = item_->parent();
    QString currentName =parent->text(0);
    std::string name = parent->text(0).toStdString();
    std::string::size_type pos = name.find(' ');
    if (pos != std::string::npos)
        name.resize(pos);
    name += "  ";
    name += data_->getOwner()->getName();
    QString newName(name.c_str());
    if (newName != currentName) parent->setText(0,newName);
}

//**************************************************************************************************************************************
//Called each time a new step of the simulation if computed
void ModifyObject::updateTables()
{
#ifdef DEBUG_GUI
    std::cout << "GUI>emit updateDataWidgets()" << std::endl;
#endif
    emit updateDataWidgets();
#ifdef DEBUG_GUI
    std::cout << "GUI<emit updateDataWidgets()" << std::endl;
#endif
#ifdef SOFA_HAVE_QWT
    if (energy)
    {
        energy->step();
        if (dialogTab->currentWidget() == energy) energy->updateVisualization();
    }

    if (momentum)
    {
        momentum->step();
        if (dialogTab->currentWidget() == momentum) momentum->updateVisualization();
    }
#endif

    if(node)
    {
        updateConsole();
    }
}

void ModifyObject::reject   ()
{
    if (node)
    {
#ifdef DEBUG_GUI
        std::cout << "GUI>emit endObjectModification(" << node->getName() << ")" << std::endl;
#endif
        emit endObjectModification(node);
#ifdef DEBUG_GUI
        std::cout << "GUI<emit endObjectModification(" << node->getName() << ")" << std::endl;
#endif
    }

    const QString dataModifiedString = parseDataModified();
    if (!dataModifiedString.isEmpty())
    {
        emit  dataModified( dataModifiedString  );
    }

    //          else if (data) emit endDataModification(data);
    emit(dialogClosed(Id_));
    deleteLater();
    QDialog::reject();
} //When closing a window, inform the parent.

void ModifyObject::accept   ()
{
    updateValues();

    const QString dataModifiedString = parseDataModified();
    if (!dataModifiedString.isEmpty())
    {
        emit  dataModified( dataModifiedString  );
    }

    if (node)
    {
#ifdef DEBUG_GUI
        std::cout << "GUI>emit endObjectModification(" << node->getName() << ")" << std::endl;
#endif
        emit endObjectModification(node);
#ifdef DEBUG_GUI
        std::cout << "GUI<emit endObjectModification(" << node->getName() << ")" << std::endl;
#endif
    }
    //          else if (data) emit endDataModification(data);
    emit(dialogClosed(Id_));
    deleteLater();
    QDialog::accept();
} //if closing by using Ok button, update the values

QString ModifyObject::parseDataModified()
{
    QString cat;

    for (std::size_t i = 0; i < m_tabs.size(); ++i)
    {
        const QString tabString = m_tabs[i]->getDataModifiedString();
        if (!tabString.isEmpty())
        {
            cat += tabString;
            if (i != (m_tabs.size() - 1))
            {
                cat += "\n";
            }
        }
    }

    return cat;
}

} // namespace qt

} // namespace gui

} // namespace sofa
