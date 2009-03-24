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
#include "SofaConfiguration.h"

#include <qsizepolicy.h>
#ifdef SOFA_QT4
#include <QGroupBox>
#include <QToolBox>
#include <QSpacerItem>
#include <QPushButton>
#include <Q3Process>
#else
#include <qgroupbox.h>
#include <qtoolbox.h>
#include <qpushbutton.h>
#include <qdir.h>
#include <qprocess.h>
typedef QProcess Q3Process;
#endif

#include <map>
#include <algorithm>
#include <fstream>


namespace sofa
{

namespace gui
{

namespace qt
{

ConfigWidget::ConfigWidget(QWidget *parent, DEFINES &d):QWidget(parent), option(d)
{
    setName(QString(option.name.c_str()));
    layout=new QHBoxLayout(this);
    check=new QCheckBox(QString(option.name.c_str()), this);
    check->setChecked(option.value);
    check->setMaximumSize(300,20);

    layout->addWidget(check);

    connect(check, SIGNAL(toggled(bool)), this, SLOT(updateValue(bool)));
}

void ConfigWidget::updateValue(bool b)
{
    option.value=b;
    emit(modified());
}

TextConfigWidget::TextConfigWidget(QWidget *parent, DEFINES &d):ConfigWidget(parent,d)
{
    description=new QLineEdit(this);
    description->setText(option.description.c_str());
    description->setAlignment(Qt::AlignBottom | Qt::AlignLeft);
    connect(description, SIGNAL(textChanged(const QString&)), this, SLOT(updateValue(const QString&)));

    layout->addWidget(description);
}



void TextConfigWidget::updateValue(const QString &s)
{
    option.description=s.ascii();
    emit(modified());
}

OptionConfigWidget::OptionConfigWidget(QWidget *parent, DEFINES &d):ConfigWidget(parent,d)
{
    description=new QLabel(this);
    description->setText(option.description.c_str());
    description->setAlignment(Qt::AlignBottom | Qt::AlignLeft);

    layout->addWidget(description);
}



SofaConfiguration::SofaConfiguration(std::string p, std::vector< DEFINES >& config):QMainWindow(),path(p),data(config)

{
    resize(800, 600);

    QWidget *appli = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(appli);



    QToolBox *global = new QToolBox(appli);
    global->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

    std::string currentCategory;

    QWidget *page=NULL;
    QVBoxLayout *pageLayout=NULL;
    for (unsigned int i=0; i<config.size(); ++i)
    {
        if (currentCategory != config[i].category)
        {
            if (page)
            {
                pageLayout->addItem( new QSpacerItem(10,10,QSizePolicy::Expanding, QSizePolicy::Expanding));
            }
            currentCategory = config[i].category;
            page = new QWidget(global);
            global->addItem(page,QString(currentCategory.c_str()));
            pageLayout=new QVBoxLayout(page);
        }
        ConfigWidget *o;
        if (config[i].typeOption) o=new OptionConfigWidget(page, config[i]);
        else                      o=new TextConfigWidget(page, config[i]);

        pageLayout->addWidget(o);
        options.push_back(o);
        connect(o, SIGNAL(modified()), this, SLOT(updateOptions()));
    }

    if (page)
    {
        pageLayout->addItem( new QSpacerItem(10,10,QSizePolicy::Expanding, QSizePolicy::Expanding));
    }
    updateConditions();

    QPushButton *button = new QPushButton(QString("Save and Update Configuration"),appli);
    connect( button, SIGNAL(clicked()), this, SLOT(saveConfiguration()));
    layout->addWidget(global);

#ifdef WIN32
    projectVC = new QLineEdit(QString("ProjectVC8.bat"),appli);
    layout->addWidget(projectVC);
#endif

    layout->addWidget(button);

    this->setCentralWidget(appli);
}

bool SofaConfiguration::getValue(CONDITION &c)
{
    unsigned int i=0;
    for (; i<options.size(); ++i)
    {
        if (options[i]->name() == c.option)
        {
            bool presence = options[i]->getValue();
            if (c.presence && presence) return true;
            if (!c.presence && !presence) return true;
            return false;
        }
    }
    return false;
}

void SofaConfiguration::updateOptions()
{
    QWidget *option = (QWidget*)sender();

    if (dynamic_cast<OptionConfigWidget*>(option))
        optionsModified.insert(option);
    updateConditions();
}

void SofaConfiguration::processCondition(QWidget *w, CONDITION &c)
{
    switch(c.type)
    {
    case OPTION:
        w->setEnabled(getValue(c));
        break;
    case ARCHI:
        if (c.option == "win32")
        {
            if (c.presence)
            {
#ifndef WIN32
                w->hide();
#endif
            }
            else
            {
#ifdef WIN32
                w->hide();
#endif
            }
        }
        else if (c.option == "unix")
        {
            if (c.presence)
            {
#ifdef WIN32
                w->hide();
#endif
            }
            else
            {
#ifndef WIN32
                w->hide();
#endif
            }
        }
        break;
    }
}

void SofaConfiguration::updateConditions()
{
    for (unsigned int i=0; i<options.size(); ++i)
    {
        std::vector< CONDITION > &conditions=options[i]->option.conditions;
        for (unsigned int c=0; c<conditions.size(); ++c)
        {
            processCondition(options[i],conditions[c]);
        }
    }
}


void SofaConfiguration::saveConfiguration()
{
    std::ofstream out((path + std::string("/sofa-local.cfg")).c_str());
    std::string currentCategory;
    std::vector< CONDITION > currentConditions;
    for (unsigned int i=0; i<options.size(); ++i)
    {
        DEFINES &option=options[i]->option;


        bool differentConditions=false;
        if (currentConditions.size() != option.conditions.size()) differentConditions=true;
        else
        {
            for (unsigned int c=0; c<currentConditions.size() && !differentConditions; ++c)
            {
                if (currentConditions[c] != option.conditions[c]) differentConditions=true;
            }
        }

        if (differentConditions)
        {
            for (unsigned int c=0; c<currentConditions.size(); ++c) out << "}\n";

            currentConditions = option.conditions;
        }

        if (currentCategory != option.category)
        {
            if (!currentCategory.empty())  out << "\n\n";
            currentCategory = option.category;
            out << "########################################################################\n";
            out << "# " << currentCategory << "\n";
            out << "########################################################################\n";
        }

        if (differentConditions)
        {
            for (unsigned int c=0; c<currentConditions.size(); ++c)
            {
                if (!currentConditions[c].presence) out << "!";
                switch( currentConditions[c].type)
                {
                case OPTION:
                    out << "contains(DEFINES," << currentConditions[c].option << "){\n";
                    break;
                case ARCHI:
                    out << currentConditions[c].option << "{\n";
                    break;
                }
            }
        }


        if (option.typeOption)
        {
            std::string description=option.description;
            for (unsigned int position=0; position<description.size(); ++position)
            {
                if (description[position] == '\n') description.insert(position+1, "# ");
            }
            out << "\n# Uncomment " << description << "\n";
            if (!option.value) out << "# ";
            out << "DEFINES += " << option.name << "\n";
        }
        else
        {
            if (!option.value) out << "# ";
            out << option.name << " " << option.description << "\n";
        }
    }

    for (unsigned int c=0; c<currentConditions.size(); ++c) out << "}\n";

    out.close();

    std::set< QWidget *>::iterator it;
    if (!optionsModified.empty())
    {
        QStringList argv;
#ifndef WIN32
        argv << QString(path.c_str()) + QString("/./touchAllFilesContainingString.sh");
#else
        argv << QString(path.c_str()) + QString("/") + QString(projectVC->text());
#endif
        for (it=optionsModified.begin(); it!=optionsModified.end(); it++)
        {
            argv << QString((*it)->name());
        }
        Q3Process *p = new Q3Process(argv,this);
        p->setCommunication(0);
        p->setWorkingDirectory(QDir(QString(path.c_str())));
        p->start();


        optionsModified.clear();
    }
}

}
}
}
