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
#define SOFA_GUI_QT_DATAWIDGET_CPP

#include "DataWidget.h"
#include "ModifyObject.h"
#include <sofa/helper/Factory.inl>
#include <sofa/helper/logging/Messaging.h>

#include <QToolTip>

#define SIZE_TEXT     60


namespace sofa
{
namespace helper
{
template class SOFA_SOFAGUIQT_API Factory<std::string, sofa::gui::qt::DataWidget, sofa::gui::qt::DataWidget::CreatorArgument>;
}

namespace gui
{
namespace qt
{

using namespace core::objectmodel;

DataWidget::DataWidget(QWidget* parent,const char* name, MyData* d)
:QWidget(parent /*,name */), baseData(d), dirty(false), counter(-1)
{
    this->setObjectName(name);
}

DataWidget::~DataWidget()
{
}

void
DataWidget::setData( MyData* d)
{
    baseData = d;
    readFromData();
}

void
DataWidget::updateVisibility()
{
    parentWidget()->setVisible(baseData->isDisplayed());
}

void
DataWidget::updateDataValue()
{
    if (dirty)
    {
        const bool hasOwner = baseData->getOwner();
        std::string previousName;
        if ( hasOwner ) previousName = baseData->getOwner()->getName();
        writeToData();

        if (hasOwner)
        {
            std::string path;
            BaseNode* ownerAsNode = dynamic_cast<BaseNode*>(baseData->getOwner() );
            BaseObject* ownerAsObject = dynamic_cast<BaseObject*>(baseData->getOwner() );

            if (ownerAsNode)
            {
                path = ownerAsNode->getPathName() + "." + baseData->getName();
            }
            else if (ownerAsObject)
            {
                std::string objectPath = ownerAsObject->getName();
                sofa::core::objectmodel::BaseObject* master = ownerAsObject->getMaster();
                while (master)
                {
                    objectPath = master->getName() + "/" + objectPath;
                    master = master->getMaster();
                }
                BaseNode* n = dynamic_cast<BaseNode*>(ownerAsObject->getContext());
                if (n)
                {
                    path = n->getPathName() + std::string("/") + objectPath + std::string(".") + baseData->getName(); // TODO: compute relative path
                }
                else
                {
                    path = objectPath + "." + baseData->getName();

                    // this can happen when a DataWidget (gui associated with the data) is created while the object owner of the Data is being built by the factory and not yet attached to the graph
                    // it cannot be considered as a problem
//                    if (ownerAsObject->getContext())
//                    {
//                        Logger::mainlog( Logger::Warning, std::string("updateDataValue: ") + path + std::string(" has a context that is not a BaseNode ") + ownerAsObject->getContext()->getName(), "DataWidget" );
//                    }
//                    else
//                    {
//                        Logger::mainlog( Logger::Warning, std::string("updateDataValue: NULL context for data ") + path, "DataWidget" );
//                    }
                }
            }
            else
            {
                msg_error("DataWidget") << "updateDataValue: " << baseData->getName() << " has an owner that is neither a Node nor an Object. Something went wrong...";
            }

            const QString dataString = (path + " = " + baseData->getValueString()).c_str();
            Q_EMIT dataValueChanged(dataString);

        }

        updateVisibility();
        if(hasOwner && baseData->getOwner()->getName() != previousName)
        {
            Q_EMIT DataOwnerDirty(true);
        }
    }

    dirty = false;
    counter = baseData->getCounter();

}

void
DataWidget::updateWidgetValue()
{
    if(!dirty)
    {
        if(counter != baseData->getCounter())
        {
            readFromData();
            this->update();
        }
    }
}

void
DataWidget::setWidgetDirty(bool b)
{
    dirty = b;
    Q_EMIT WidgetDirty(b);
}

typedef sofa::helper::Factory<std::string, DataWidget, DataWidget::CreatorArgument> DataWidgetFactory;

DataWidget *DataWidget::CreateDataWidget(const DataWidget::CreatorArgument &dwarg)
{
    DataWidget *datawidget_ = 0;
    const std::string &widgetName=dwarg.data->getWidget();
    if (widgetName.empty())
        datawidget_ = DataWidgetFactory::CreateAnyObject(dwarg);
    else
        datawidget_ = DataWidgetFactory::CreateObject(widgetName, dwarg);

    return datawidget_;
}


/*QDisplayDataInfoWidget definitions */

QDisplayDataInfoWidget::QDisplayDataInfoWidget(QWidget* parent, const std::string& helper,
        core::objectmodel::BaseData* d, bool modifiable, const ModifyObjectFlags& modifyObjectFlags):QWidget(parent), data(d), numLines_(1)
{
    setMinimumHeight(25);

    QHBoxLayout* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0,0,0,0);
    layout->setSpacing(0);

    std::string final_str;
    formatHelperString(helper,final_str);
    std::string ownerClass=data->getOwnerClass();
    if (modifiable)
    {
        QPushButton *helper_button = new QPushButton(this);
        helper_button->setIcon(LinkIcon());
        helper_button->setFixedSize(QSize(16, 16));
        helper_button->setToolTip(QString(final_str.c_str()));
        helper_button->setAutoDefault(false);
        layout->addWidget(helper_button, 0, Qt::AlignLeft);
        connect(helper_button, SIGNAL( clicked() ), this, SLOT( linkModification()));
        if (!ownerClass.empty())
            helper_button->setToolTip( ("Data from "+ownerClass).c_str());
            //QToolTip::add(helper_button, ("Data from "+ownerClass).c_str());
    }
    else
    {
        /*
#ifndef SOFA_GUI_QT_NO_DATA_HELP
        QLabel* helper_label = new QLabel(this);
        helper_label->setText(QString(final_str.c_str()));
        helper_label->setMinimumWidth(20);
        layout->addWidget(helper_label);
        if (!ownerClass.empty()) QToolTip::add(helper_label, ("Data from "+ownerClass).c_str());
#else
        numLines_ = 0;
        if (!final_str.empty() || !ownerClass.empty())
        {
            if (!final_str.empty()) final_str += '\n';
            final_str += "Data from ";
            final_str += ownerClass;
            QToolTip::add(parent, final_str.c_str());
        }
#endif
        */
    }
    if(modifiable || !data->getLinkPath().empty())
    {
        linkpath_edit = new QLineEdit(this);
        linkpath_edit->setContentsMargins(2, 0, 0, 0);
        linkpath_edit->setText(QString(data->getLinkPath().c_str()));
        linkpath_edit->setReadOnly(!modifiable);
        layout->addWidget(linkpath_edit);
        linkpath_edit->setVisible(!data->getLinkPath().empty());
        if(modifyObjectFlags.PROPERTY_WIDGET_FLAG)
            connect(linkpath_edit, SIGNAL( textChanged(const QString&)), this, SIGNAL( WidgetDirty()));
        else
            connect(linkpath_edit, SIGNAL( editingFinished()), this, SLOT( linkEdited()));
    }
    else
    {
        linkpath_edit=NULL;
    }
}

void QDisplayDataInfoWidget::linkModification()
{
    if (linkpath_edit->isVisible() && linkpath_edit->text().isEmpty())
        linkpath_edit->setVisible(false);
    else
    {
        linkpath_edit->setVisible(true);
        //Open a dialog window to let the user select the data he wants to link
    }
}
void QDisplayDataInfoWidget::linkEdited()
{
//    dmsg_error("DataWidget") << "QDisplayDataInfoWidget::linkEdited " << linkpath_edit->text().toStdString() ;
    data->setParent(linkpath_edit->text().toStdString() );
}

void QDisplayDataInfoWidget::formatHelperString(const std::string& helper, std::string& final_text)
{
    std::string label_text=helper;
    numLines_ = 0;
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
            const std::size_t cut = current_sentence.size()/SIZE_TEXT;
            for (std::size_t index_cut=1; index_cut<=cut; index_cut++)
            {
                std::string::size_type numero_char=current_sentence.rfind(' ',SIZE_TEXT*index_cut);
                current_sentence = current_sentence.insert(numero_char+1,1,'\n');
                numLines_++;
            }
        }
        if (pos != std::string::npos) label_text = label_text.substr(pos+1);
        else label_text = "";
        final_text += current_sentence;
        numLines_++;
    }
}

unsigned int QDisplayDataInfoWidget::numLines(const std::string& str)
{
    std::string::size_type newline_pos;
    unsigned int numlines = 1;
    newline_pos = str.find('\n',0);
    while( newline_pos != std::string::npos )
    {
        numlines++;
        newline_pos = str.find('\n',newline_pos+1);
    }
    return numlines;
}


/* QPushButtonUpdater definitions */

void QPushButtonUpdater::setDisplayed(bool b)
{

    if (b)
    {
        this->setText(QString("Hide the values"));
    }
    else
    {
        this->setText(QString("Display the values"));
    }

}






}//qt
}//gui
}//sofa
