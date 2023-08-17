/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <QTimer>
#include <sofa/gui/qt/QTransformationWidget.h>
#if SOFA_GUI_QT_HAVE_QT_CHARTS
#include <sofa/gui/qt/QEnergyStatWidget.h>
#include <sofa/gui/qt/QMomentumStatWidget.h>
#endif
#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::Message;

#include <sofa/simulation/Node.h>
#include <iostream>

#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTabWidget>
#include <QTreeWidget>
#include <QScrollArea>
#include <QApplication>
#include <QScreen>

namespace sofa::gui::qt
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
      basenode(nullptr),
      data_(nullptr),
      dialogFlags_(dialogFlags),
      messageTab(nullptr),
      messageEdit(nullptr)
    #if SOFA_GUI_QT_HAVE_QT_CHARTS
    ,energy(nullptr)
    ,momentum(nullptr)
    #endif
{
    setWindowTitle(name);
    //setObjectName(name);
    setModal(modal);
}

void ModifyObject::createDialog(core::objectmodel::Base* base)
{
    if(base == nullptr)
    {
        return;
    }
    emit beginObjectModification(base);
    basenode = base;
    data_ = nullptr;

    //Layout to organize the whole window
    QVBoxLayout *generalLayout = new QVBoxLayout(this);
    generalLayout->setObjectName("generalLayout");
    generalLayout->setContentsMargins(0,0,0,0);
    generalLayout->setSpacing(1);

    //Tabulation widget
    dialogTab = new QTabWidget(this);

    //add a scrollable area for data properties
    QScrollArea* m_scrollArea = new QScrollArea();

    //    const int screenHeight = QApplication::desktop()->height();
    QRect geometry = QGuiApplication::primaryScreen()->availableGeometry();

    m_scrollArea->setMinimumSize(600, geometry.height() * 0.75);
    m_scrollArea->setWidgetResizable(true);
    dialogTab->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_scrollArea->setWidget(dialogTab);
    generalLayout->addWidget(m_scrollArea);

    connect(dialogTab, SIGNAL( currentChanged(int)), this, SLOT( updateTables()));

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

    QPushButton *buttonRefresh = new QPushButton( this );
    buttonRefresh->setObjectName("buttonRefresh");
    buttonRefresh->setText( tr( "Refresh" ) );

    // displayWidget
    if (basenode)
    {
        const sofa::core::objectmodel::Base::VecData& fields = basenode->getDataFields();
        const sofa::core::objectmodel::Base::VecLink& links = basenode->getLinks();

        std::map< std::string, std::vector<QTabulationModifyObject* > > groupTabulation;

        std::vector<std::string> tabNames;
        //Put first the Property Tab
        tabNames.push_back("Property");

        for( sofa::core::objectmodel::Base::VecData::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {
            core::objectmodel::BaseData* data=*it;
            if (!data)
            {
                dmsg_error("ModifyObject") << "nullptr Data in '" << basenode->getName() << "'" ;
                continue;
            }

            if (data->getName().empty()) continue; // ignore unnamed data

            //For each Data of the current Object
            //We determine where it belongs:
            std::string currentGroup=data->getGroup();

            if (currentGroup.empty()) currentGroup="Property";

            // Ignore the data in group "Infos" so they can be putted in the real Infos panel that is
            // handled in a different way (see QDataDescriptionWidget)
            if (currentGroup == "Infos")
                continue;

            QTabulationModifyObject* currentTab=nullptr;

            std::vector<QTabulationModifyObject* > &tabs=groupTabulation[currentGroup];
            bool newTab = false;
            if (tabs.empty()) tabNames.push_back(currentGroup);
            if (tabs.empty() || tabs.back()->isFull())
            {
                newTab = true;
                m_tabs.push_back(new QTabulationModifyObject(this,basenode, item_,tabs.size()+1));
                tabs.push_back(m_tabs.back());
            }
            currentTab = tabs.back();
            currentTab->addData(data, getFlags());
            if (newTab)
            {
                connect(buttonUpdate,   SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
                connect(buttonOk,       SIGNAL(clicked() ),          currentTab, SLOT( updateDataValue() ) );
                connect(this,           SIGNAL(updateDataWidgets()), currentTab, SLOT( updateWidgetValue()) );

                /// The timer is deleted when the 'this' object is destroyed.
                QTimer *timer = new QTimer(this);
                connect(timer, SIGNAL(timeout()), this, SLOT(updateTables()));
                connect(timer, SIGNAL(timeout()), currentTab, SLOT(updateDataValue()));
                timer->start(10);

                connect(currentTab, SIGNAL( TabDirty(bool) ), buttonUpdate, SLOT( setEnabled(bool) ) );
                connect(currentTab, SIGNAL( TabDirty(bool) ), this, SIGNAL( componentDirty(bool) ) );
            }
        }

        for( sofa::core::objectmodel::Base::VecLink::const_iterator it = links.begin(); it!=links.end(); ++it)
        {
            core::objectmodel::BaseLink* link=*it;

            if (link->getName().empty()) continue; // ignore unnamed links
            if (!link->storePath() && link->getSize() == 0) continue; // ignore empty links

            //For each Link of the current Object
            //We determine where it belongs:
            std::string currentGroup="Links";

            QTabulationModifyObject* currentTab=nullptr;

            std::vector<QTabulationModifyObject* > &tabs=groupTabulation[currentGroup];
            if (tabs.empty()) tabNames.push_back(currentGroup);
            if (tabs.empty() || tabs.back()->isFull())
            {
                m_tabs.push_back(new QTabulationModifyObject(this,basenode, item_,tabs.size()+1));
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

        for (std::vector<std::string>::const_iterator it = tabNames.begin(), itend = tabNames.end(); it != itend; ++it)
        {
            const std::string& groupName = *it;
            std::vector<QTabulationModifyObject* > &tabs=groupTabulation[groupName];

            for (unsigned int i=0; i<tabs.size(); ++i)
            {
                QString nameTab;
                if (tabs.size() == 1) nameTab=groupName.c_str();
                else                  nameTab=QString(groupName.c_str())+ " " + QString::number(tabs[i]->getIndex()) + "/" + QString::number(tabs.size());
                dialogTab->addTab(tabs[i],nameTab);
                tabs[i]->addStretch();
            }
        }

#if SOFA_GUI_QT_HAVE_QT_CHARTS
        //Energy Widget
        if (simulation::Node* real_node = sofa::core::castTo<simulation::Node*>(basenode))
        {
            if (dialogFlags_.REINIT_FLAG)
            {
                energy = new QEnergyStatWidget(dialogTab, real_node);
                dialogTab->addTab(energy, QString("Energy Stats"));
            }
        }

        //Momentum Widget
        if (simulation::Node* real_node = sofa::core::castTo<simulation::Node*>(basenode))
        {
            if (dialogFlags_.REINIT_FLAG)
            {
                momentum = new QMomentumStatWidget(dialogTab, real_node);
                dialogTab->addTab(momentum, QString("Momentum Stats"));
            }
        }
#endif


        /// Info Widget
        {
            QDataDescriptionWidget* description=new QDataDescriptionWidget(dialogTab, basenode);
            dialogTab->addTab(description, QString("Infos"));
        }

        /// Message widget
        {
            updateConsole();
            if (messageTab)
            {
                std::stringstream tmp;
                int numMessages = basenode->countLoggedMessages({Message::Info, Message::Advice, Message::Deprecated,
                                                                 Message::Error, Message::Warning, Message::Fatal});
                tmp << "Messages(" << numMessages << ")" ;
                dialogTab->addTab(messageTab, QString::fromStdString(tmp.str()));
            }
        }

        //Adding buttons at the bottom of the dialog
        QHBoxLayout *lineLayout = new QHBoxLayout( nullptr);
        lineLayout->setContentsMargins(0,0,0,0);
        lineLayout->setSpacing(6);
        lineLayout->setObjectName("Button Layout");
        lineLayout->addWidget(buttonUpdate);
        QSpacerItem *Horizontal_Spacing = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
        lineLayout->addItem( Horizontal_Spacing );

        lineLayout->addWidget(buttonOk);
        lineLayout->addWidget(buttonCancel);
        lineLayout->addWidget(buttonRefresh);
        generalLayout->addLayout( lineLayout );

        //Signals and slots connections
        connect( buttonUpdate,   SIGNAL( clicked() ), this, SLOT( updateValues() ) );
        connect(buttonRefresh,       SIGNAL(clicked() ),          this, SLOT( updateTables() ));
        connect( buttonOk,       SIGNAL( clicked() ), this, SLOT( accept() ) );
        connect( buttonCancel,   SIGNAL( clicked() ), this, SLOT( reject() ) );

        resize( QSize(450, 130).expandedTo(minimumSizeHint()) );
    }
}

void ModifyObject::clearMessages()
{
    basenode->clearLoggedMessages();
    messageEdit->clear();

    std::stringstream tmp;
    const int numMessages = basenode->countLoggedMessages({Message::Info, Message::Advice, Message::Deprecated,
                                                           Message::Error, Message::Warning, Message::Fatal});
    tmp << "Messages(" << numMessages << ")" ;

    dialogTab->setTabText(dialogTab->indexOf(messageTab), QString::fromStdString(tmp.str()));
}



void ModifyObject::createDialog(core::objectmodel::BaseData* data)
{
    data_ = data;
    basenode = nullptr;

    emit beginDataModification(data);

    QVBoxLayout *generalLayout = new QVBoxLayout(this);
    generalLayout->setContentsMargins(0, 0, 0, 0);
    generalLayout->setSpacing(1);
    generalLayout->setObjectName("generalLayout");
    QHBoxLayout *lineLayout = new QHBoxLayout( nullptr);
    lineLayout->setContentsMargins(0, 0, 0, 0);
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
    ClickableTextEdit(QWidget* w) : QTextEdit(w) {}
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
        tabLayout->setContentsMargins(0, 0, 0, 0);
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
        for(const Message& message : basenode->getLoggedMessages())
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
    // this is controlling if we need to re-init (eg: not in the Modeller)
    if(!dialogFlags_.REINIT_FLAG)
        return;

    if (basenode == nullptr)
        return;

    simulation::Node* node = sofa::core::castTo<sofa::simulation::Node*>(basenode);
    core::objectmodel::BaseObject* object = sofa::core::castTo<core::objectmodel::BaseObject*>(basenode);

    // if the selected object is a node
    if (node)
    {
        node->reinit(sofa::core::execparams::defaultInstance());
    }
    else if (object)                 //< if the selected is an object
    {
        object->reinit();            //< we need to fully re-initialize the object to be sure it is ok.
    }
    else
    {
        throw std::runtime_error("Invalid type, only Node and BaseObject are supported. "
                                 "This is a BUG, please report to https://github.com/sofa-framework/sofa/issues");
    }

    // trigger the internal updates (eg: updateDataCallback),
    basenode->d_componentState.updateIfDirty();

    emit objectUpdated();
    emit endObjectModification(basenode);
    emit beginObjectModification(basenode);

    if (buttonUpdate)
        buttonUpdate->setEnabled(false);
}

//**************************************************************************************************************************************
//Called each time a new step of the simulation if computed
void ModifyObject::updateTables()
{
    emit updateDataWidgets();
#if SOFA_GUI_QT_HAVE_QT_CHARTS
    if (energy)
    {
        if (dialogTab->currentWidget() == energy) energy->step();
    }

    if (momentum)
    {
        if (dialogTab->currentWidget() == momentum) momentum->step();
    }
#endif

    if(basenode)
    {
        updateConsole();
    }
}

void ModifyObject::reject   ()
{
    if (basenode)
    {
        emit endObjectModification(basenode);
    }

    const QString dataModifiedString = parseDataModified();
    if (!dataModifiedString.isEmpty())
    {
        emit  dataModified( dataModifiedString  );
    }

    emit dialogClosed(Id_);
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

    if (basenode)
    {
        emit endObjectModification(basenode);
    }
    emit dialogClosed(Id_);
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

bool ModifyObject::hideData(core::objectmodel::BaseData* data) { return (!data->isDisplayed()) && dialogFlags_.HIDE_FLAG;}


} // namespace sofa::gui::qt
