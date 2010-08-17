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

#include "GlobalModification.h"
#include <sofa/helper/set.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>

#ifdef SOFA_QT4
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QCompleter>
#include <QPushButton>
#include <QSpacerItem>
#else
#include <qlayout.h>
#include <qlabel.h>
#include <qpushbutton.h>
#include <qsizepolicy.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

QStringList GlobalModification::listDataName;

GlobalModification::GlobalModification()
{

    setCaption(QString("Global Modifications"));

    static bool isInitialized=false;

    //Store inside a set all the available name of Data.
    //This is a critical section. If someone makes a mistake in the desctructor, it might lead to a crash of the application!
    if (!isInitialized)
    {
        std::vector< sofa::core::ObjectFactory::ClassEntry* > allEntries;
        helper::set< std::string > allNames;

        sofa::core::ObjectFactory::getInstance()->getAllEntries(allEntries);

        for (unsigned int i=0; i<allEntries.size(); ++i)
        {
            sofa::core::ObjectFactory::ClassEntry& entry=*(allEntries[i]);

            sofa::core::ObjectFactory::Creator *creatorEntry=entry.creatorList.front().second;
            sofa::core::objectmodel::BaseObject *object=creatorEntry->createInstance(0,0);
            if (!object) continue;
            const std::vector< std::pair<std::string, sofa::core::objectmodel::BaseData*> >& datas=object->getFields();
            for (unsigned int d=0; d<datas.size(); ++d)
            {
                allNames.insert(datas[d].first);
            }
            delete object;
        }

        for (helper::set< std::string >::const_iterator it=allNames.begin(); it!=allNames.end(); ++it)
            listDataName << it->c_str();

        isInitialized=true;
    }


    //Creation of the GUI
    QVBoxLayout *globalLayout = new QVBoxLayout(this);

    //***********************************************************************************
    //Selection of the Data Name
    QWidget *dataNameWidget = new QWidget(this);
    QHBoxLayout *nameLayout = new QHBoxLayout(dataNameWidget);

    dataNameSelector = new QComboBox();

    dataNameSelector->setEditable(true);
    dataNameSelector->setAutoCompletion(true);

    dataNameSelector->insertStringList(listDataName);
#ifdef SOFA_QT4
    dataNameSelector->setCompleter(new QCompleter(listDataName,dataNameSelector));
#endif

    nameLayout->addWidget(new QLabel(QString("Select the Data to modify"),dataNameWidget));
    nameLayout->addWidget(dataNameSelector);

    //***********************************************************************************
    //New value
    QWidget *dataValueWidget = new QWidget(this);
    QHBoxLayout *valueLayout = new QHBoxLayout(dataValueWidget);

    valueModifier = new QLineEdit(dataValueWidget);


    valueLayout->addWidget(new QLabel(QString("New value: "),dataValueWidget));
    valueLayout->addWidget(valueModifier);



    //***********************************************************************************
    //Button Panel
    QWidget *buttonWidget = new QWidget(this);
    QHBoxLayout *buttonLayout = new QHBoxLayout(buttonWidget);

    QPushButton *modifyButton = new QPushButton(QString("Modify"), buttonWidget);
    QPushButton *cancelButton = new QPushButton(QString("Cancel"), buttonWidget);

    modifyButton->setAutoDefault(true);

    buttonLayout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Minimum));
    buttonLayout->addWidget(modifyButton);
    buttonLayout->addWidget(cancelButton);

    //***********************************************************************************
    globalLayout->addWidget(dataNameWidget);
    globalLayout->addWidget(dataValueWidget);
    globalLayout->addWidget(buttonWidget);


    //***********************************************************************************
    //Do the connections
    connect(modifyButton, SIGNAL(clicked()), this, SLOT(applyGlobalModification()));
    connect(cancelButton, SIGNAL(clicked()), this, SLOT(close()));
    connect(valueModifier, SIGNAL(returnPressed()), this, SLOT(applyGlobalModification()));
}

GlobalModification::~GlobalModification()
{
    delete dataNameSelector;
    delete valueModifier;
}


void GlobalModification::applyGlobalModification()
{
    QString name=dataNameSelector->currentText();
    if (listDataName.find(name) == listDataName.end())
    {
        const std::string message="Data " + std::string(name.ascii()) + " does not exist!";
        emit displayMessage(message);
        close();
        return;
    }
    emit( modifyData(name.ascii(),
            valueModifier->text().ascii()));
    close();
}
}
}
}

