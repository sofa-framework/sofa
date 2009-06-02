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

#include "QSofaLibrary.h"
#include "QCategoryLibrary.h"


namespace sofa
{

namespace gui
{

namespace qt
{

QSofaLibrary::QSofaLibrary(QWidget *parent): QWidget(parent)
{
    toolbox = new LibraryContainer(parent);
    toolbox->setCurrentIndex(-1);
    toolbox->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
}

CategoryLibrary *QSofaLibrary::createCategory(const std::string &categoryName, unsigned int numComponent)
{
    CategoryLibrary* category = new QCategoryLibrary(toolbox, categoryName, numComponent);
    toolbox->addItem( category->getQWidget(), categoryName.c_str());
    return category;
}


void QSofaLibrary::addCategory(CategoryLibrary *category)
{
    SofaLibrary::addCategory(category);
    toolbox->setItemLabel(categories.size()-1, QString(category->getName().c_str()) + QString(" [") + QString::number(category->getNumComponents()) + QString("]"));

    connect( category->getQWidget(), SIGNAL( componentDragged( std::string, std::string, std::string, ClassEntry* ) ),
            this, SLOT( componentDraggedReception( std::string, std::string, std::string , ClassEntry* )));

}

void QSofaLibrary::filter(const FilterQuery &f)
{

    numComponents=0;
    unsigned int numComponentDisplayed=0;
    unsigned int indexPage=0;
    //Look into all the categories
    for (unsigned int cat=0; cat<categories.size(); ++cat)
    {
        unsigned int numComponentDisplayedInCategory=0;
        bool needToHideCategory=true;
        //For each category, look at all the components if one has the name wanted
        const std::vector< ComponentLibrary* > &components = categories[cat]->getComponents();
        for (unsigned int comp=0; comp<components.size(); ++comp)
        {
            if (f.isValid(components[comp]))
            {
                components[comp]->setDisplayed(true);
                needToHideCategory=false;
                ++numComponentDisplayed;
                ++numComponentDisplayedInCategory;
            }
            else
            {
                components[comp]->setDisplayed(false);
            }
        }

        int idx = toolbox->indexOf(categories[cat]->getQWidget());

        if (needToHideCategory)
        {
            categories[cat]->setDisplayed(false);
            if (idx >= 0)
            {
#ifdef SOFA_QT4
                toolbox->removeItem(idx);
#else
                toolbox->removeItem(categories[cat]);
#endif

            }
        }
        else
        {

            categories[cat]->setDisplayed(true);
            if (idx < 0)
            {
                toolbox->insertItem(indexPage,categories[cat]->getQWidget(),QString(categories[cat]->getName().c_str()) );
            }
            toolbox->setItemLabel(indexPage, QString(categories[cat]->getName().c_str()) + QString(" [") + QString::number(numComponentDisplayedInCategory) + QString("]"));

            numComponents+=numComponentDisplayedInCategory;
            indexPage++;
        }
    }

}

//*********************//
// SLOTS               //
//*********************//
void QSofaLibrary::componentDraggedReception( std::string description, std::string categoryName, std::string templateName, ClassEntry* componentEntry)
{
    emit(componentDragged(description, categoryName,templateName,componentEntry));
}

}
}
}
