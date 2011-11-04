/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
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

QSofaLibrary::QSofaLibrary(QWidget *parent): QToolBox(parent)
{

    this->setCurrentIndex(1);
    this->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

    QWidget *contentContainer = new QWidget( this, "contentContainer");
    QVBoxLayout *contentLayout = new QVBoxLayout( contentContainer );
    this->addItem(contentContainer, QString());


    toolbox = new LibraryContainer(contentContainer);
    toolbox->setCurrentIndex(-1);
    toolbox->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
    contentLayout->addWidget( toolbox );
}

CategoryLibrary *QSofaLibrary::createCategory(const std::string &categoryName, unsigned int numComponent)
{
    QCategoryLibrary* category = new QCategoryLibrary(toolbox, categoryName, numComponent);
    toolbox->addItem( category, categoryName.c_str());
    return category;
}


void QSofaLibrary::addCategory(CategoryLibrary *c)
{
    QCategoryLibrary* category=static_cast<QCategoryLibrary*>(c);

    SofaLibrary::addCategory(category);
    toolbox->setItemLabel(categories.size()-1, QString(category->getName().c_str()) + QString(" [") + QString::number(category->getNumComponents()) + QString("]"));

    connect( category, SIGNAL( componentDragged( std::string, std::string, std::string, ClassEntry* ) ),
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

        QCategoryLibrary *category = static_cast<QCategoryLibrary *>(categories[cat]);
        int idx = toolbox->indexOf(category);

        if (needToHideCategory)
        {
            category->setDisplayed(false);
            if (idx >= 0)
            {
#ifdef SOFA_QT4
                toolbox->removeItem(idx);
#else
                toolbox->removeItem(category);
#endif

            }
        }
        else
        {

            category->setDisplayed(true);
            if (idx < 0)
            {
                toolbox->insertItem(indexPage,category,QString(categories[cat]->getName().c_str()) );
            }
            toolbox->setItemLabel(indexPage, QString(category->getName().c_str()) + QString(" [") + QString::number(numComponentDisplayedInCategory) + QString("]"));

            numComponents+=numComponentDisplayedInCategory;
            indexPage++;
        }
    }
    this->setItemLabel(0,QString("Sofa Components [") + QString::number(numComponentDisplayed) + QString("]"));
}

//*********************//
// SLOTS               //
//*********************//
void QSofaLibrary::componentDraggedReception( std::string description, std::string categoryName, std::string templateName, ClassEntry* componentEntry)
{
    emit(componentDragged(description, categoryName,templateName,componentEntry));
}


void QSofaLibrary::build(const std::vector< std::string >& examples)
{
    SofaLibrary::build(examples);
    this->setItemLabel(0,QString("Sofa Components [") + QString::number(getNumComponents()) + QString("]"));
}
}
}
}
