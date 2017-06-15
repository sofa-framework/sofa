/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_FILTERLIBRARY_H
#define SOFA_FILTERLIBRARY_H

#include <iostream>
#include <sofa/core/ComponentLibrary.h>


#include <QLineEdit>


namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::core::ComponentLibrary;


class FilterQuery
{
public:
    FilterQuery( const std::string &query);

    bool isValid( const ComponentLibrary* component) const ;

protected:
    void decodeQuery();

    std::string query;

    std::vector< QString > components;
    std::vector< QString > templates;
    std::vector< QString > licenses;
    std::vector< QString > authors;
};

//***************************************************************
class FilterLibrary : public QLineEdit
{
    Q_OBJECT
public:
    FilterLibrary( QWidget* parent);

public slots:
    void searchText(const QString&);
    void clearText();
protected:
    std::string help;
signals:
    void filterList(const FilterQuery&);
};

}
}
}

#endif
