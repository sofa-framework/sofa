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
#ifndef SOFA_CATEGORYLIBRARY_H
#define SOFA_CATEGORYLIBRARY_H

#include <iostream>
#include "ComponentLibrary.h"

namespace sofa
{

namespace gui
{

namespace qt
{

typedef sofa::core::ObjectFactory::Creator    Creator;

//***************************************************************
class CategoryLibrary
{
public:
    CategoryLibrary( const std::string &categoryName);
    virtual ~CategoryLibrary() {};

    virtual ComponentLibrary *addComponent(const std::string &componentName, ClassEntry* entry, const std::vector< QString > &exampleFiles);
    virtual void endConstruction();

    virtual void setDisplayed(bool ) {};

    const std::string                      &getName()          const { return name;}
    const std::vector< ComponentLibrary* > &getComponents()    const {return components;}
    unsigned int                            getNumComponents() const {return components.size();}

    virtual QWidget *getQWidget()=0;
protected:
    virtual ComponentLibrary *createComponent(const std::string &componentName, ClassEntry* entry, const std::vector< QString > &exampleFiles)=0;

    std::string name;
    std::vector< ComponentLibrary* > components;
};

}
}
}

#endif
