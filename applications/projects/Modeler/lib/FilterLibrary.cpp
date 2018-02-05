/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "FilterLibrary.h"

#include <QToolTip>

namespace sofa
{

namespace gui
{

namespace qt
{

//-------------------------------------------------------------------------------------------------------
FilterQuery::FilterQuery(const std::string &q):query(q)
{
    decodeQuery();
}

bool FilterQuery::isValid( const ComponentLibrary* component) const
{
    //First verification: Name of the component
    if (!components.empty())
    {
        const QString componentName(component->getName().c_str());
        for (unsigned int i=0; i<components.size(); ++i)
        {
            if (!componentName.contains(components[i],Qt::CaseInsensitive)) return false;
        }
    }

    //Templates
    if (!templates.empty())
    {
        bool templateFound=false;
        const std::vector< std::string > &componentTemplates = component->getTemplates();
        for (unsigned int t=0; t<componentTemplates.size() && !templateFound; ++t)
        {
            const QString currentTemplate(componentTemplates[t].c_str());
            for (unsigned int i=0; i<templates.size(); ++i)
            {
                if (currentTemplate.contains(templates[i],Qt::CaseInsensitive)) { templateFound=true; break;}
            }
        }
        if (!templateFound) return false;
    }

    //Licenses
    if (!licenses.empty())
    {
        const QString componentLicense(component->getEntry()->license.c_str());
        for (unsigned int i=0; i<licenses.size(); ++i)
        {
            if (!componentLicense.contains(licenses[i],Qt::CaseInsensitive)) return false;
        }
    }

    //Authors
    if (!authors.empty())
    {
        const QString componentAuthor(component->getEntry()->authors.c_str());
        for (unsigned int i=0; i<authors.size(); ++i)
        {
            if (!componentAuthor.contains(authors[i],Qt::CaseInsensitive)) return false;
        }
    }

    return true;
}

void FilterQuery::decodeQuery()
{
    std::istringstream iss (query,std::istringstream::in);

    while (!iss.eof())
    {
        std::string entry; iss >> entry;
        if (entry.find("template:") != std::string::npos)
        {
            const std::string templateName=entry.substr(9);
            templates.push_back(QString(templateName.c_str()));
        }
        else if (entry.find("license:") != std::string::npos)
        {
            const std::string licenseName=entry.substr(8);
            licenses.push_back(QString(licenseName.c_str()));

        }
        else if (entry.find("author:") != std::string::npos)
        {
            const std::string authorName=entry.substr(7);
            authors.push_back(QString(authorName.c_str()));
        }
        else
        {
            components.push_back(QString(entry.c_str()));
        }

    }
}
//-------------------------------------------------------------------------------------------------------
FilterLibrary::FilterLibrary(QWidget *parent):QLineEdit(parent)
{
    connect( this, SIGNAL(textChanged(const QString&)), this, SLOT( searchText(const QString&)));
    this->setMouseTracking(true);
    help = std::string("\
To Use the filter: \n\
\t* for the components, enter directly the name\n\
\t* for the templates, enter \"template:\" before\n\
\t* for the licenses, enter \"license:\" before\n\
\t* for the authors, enter \"author:\" before\n\
Between two filters, put a 'space'");

    this->setToolTip(QString(help.c_str()));
}

void FilterLibrary::searchText(const QString &text)
{
    const std::string textString(text.toStdString());
    FilterQuery query(textString);
    emit( filterList(query) );
}

void FilterLibrary::clearText()
{
    setText(QString());
}

}
}
}
