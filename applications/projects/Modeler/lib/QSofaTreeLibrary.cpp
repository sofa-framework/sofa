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

#include "QSofaTreeLibrary.h"
#include "QCategoryTreeLibrary.h"


namespace sofa
{

namespace gui
{

namespace qt
{

QSofaTreeLibrary::QSofaTreeLibrary(QWidget *parent): QTreeWidget(parent)
{
    setFocusPolicy(Qt::NoFocus);
    setIndentation(10);
    setColumnCount(2);
    setAutoFillBackground(true);
}

CategoryLibrary *QSofaTreeLibrary::createCategory(const std::string &categoryName, unsigned int numComponent)
{
    QCategoryTreeLibrary* category = new QCategoryTreeLibrary(this, categoryName, numComponent);
    return category;
}


void QSofaTreeLibrary::addCategory(CategoryLibrary *c)
{
    QCategoryTreeLibrary* category=static_cast<QCategoryTreeLibrary*>(c);

    SofaLibrary::addCategory(category);

    connect( category, SIGNAL( componentDragged( std::string, std::string, std::string, ClassEntry::SPtr ) ),
            this, SLOT( componentDraggedReception( std::string, std::string, std::string , ClassEntry::SPtr )));
}

void QSofaTreeLibrary::filter(const FilterQuery &f)
{
    numComponents=0;
    unsigned int numComponentDisplayed=0;
    unsigned int indexPage=0;
    helper::vector<QTreeWidgetItem*> categoryDisplayed;
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

        QCategoryTreeLibrary *category = static_cast<QCategoryTreeLibrary *>(categories[cat]);

        QList<QTreeWidgetItem*> found=findItems(QString(category->getName().c_str()),Qt::MatchStartsWith);

        QTreeWidgetItem* currentItem=NULL;
        int minSize=-1;
        for ( int i=0; i<found.count(); ++i)
        {
            if (minSize < 0 || minSize >found[i]->text(0).size())
            {
                currentItem=found[i];
                minSize=found[i]->text(0).size();
            }
        }
        if (!currentItem) return;

        currentItem->setText(0,QString(category->getName().c_str() ) );
        currentItem->setText(1, QString::number(numComponentDisplayedInCategory));

        if (needToHideCategory)
        {
            category->setDisplayed(false);
            setItemHidden(currentItem,true);
        }
        else
        {
            category->setDisplayed(true);
            setItemHidden(currentItem,false);
            numComponents+=numComponentDisplayedInCategory;
            indexPage++;
            categoryDisplayed.push_back(currentItem);
        }
    }

    if (indexPage <= 2 || numComponents < 15)
    {
        for (unsigned int i=0; i<categoryDisplayed.size(); ++i) categoryDisplayed[i]->setExpanded(true);
    }
    else if (indexPage == categories.size())
    {
        for (unsigned int i=0; i<categoryDisplayed.size(); ++i) categoryDisplayed[i]->setExpanded(false);
    }

    headerItem()->setText(0,QString("Sofa Components"));
    headerItem()->setText(1, QString::number(numComponentDisplayed));
}

//*********************//
// SLOTS               //
//*********************//
void QSofaTreeLibrary::componentDraggedReception( std::string description, std::string categoryName, std::string templateName, ClassEntry::SPtr componentEntry)
{
    emit(componentDragged(description, categoryName,templateName,componentEntry));
}


void QSofaTreeLibrary::build(const std::vector< std::string >& examples)
{
    SofaLibrary::build(examples);
    headerItem()->setText(0,QString("Sofa Components"));
    headerItem()->setText(1, QString::number(getNumComponents()));
    headerItem()->setTextAlignment(1, Qt::AlignHCenter|Qt::AlignVCenter|Qt::AlignCenter);

    QFont font;
    font.setBold(true);
    headerItem()->setFont(0, font);

    resizeColumnToContents(0);

    //Look into all the categories
    for (unsigned int cat=0; cat<categories.size(); ++cat)
    {
        QCategoryTreeLibrary *category = static_cast<QCategoryTreeLibrary *>(categories[cat]);
        setItemExpanded(category->getQWidget(),false);
    }

}
}
}
}
