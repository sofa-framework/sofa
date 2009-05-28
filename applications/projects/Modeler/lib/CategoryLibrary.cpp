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

#include "CategoryLibrary.h"

namespace sofa
{

namespace gui
{

namespace qt
{

//-------------------------------------------------------------------------------------------------------
CategoryLibrary::CategoryLibrary( QWidget *parent, const std::string &categoryName):QWidget(parent, categoryName.c_str()), name(categoryName)
{
}


ComponentLibrary *CategoryLibrary::addComponent(const std::string &componentName, ClassEntry* entry, const std::vector< QString > &exampleFiles)
{
    //Special case of Mapping and MechanicalMapping
    bool isMechanicalMapping = (name == "MechanicalMapping");
    bool isMapping           = (name == "Mapping");

    ComponentLibrary* component = createComponent(componentName, entry, exampleFiles);

    //Add the corresponding templates
    typedef std::list< std::pair< std::string, Creator*> >::iterator IteratorEntry;
    IteratorEntry itTemplate;

    //It exists Mappings only Mechanical or only Visual. So, we must add the component if only a creator is available for the current category
    bool componentCreationPossible=false;
    //read all the template possible, and remove unused (for Mapping processing)
    for (itTemplate=entry->creatorList.begin(); itTemplate!= entry->creatorList.end(); itTemplate++)
    {
        const std::string &templateName = itTemplate->first;
        //If the component corresponds to a MechanicalMapping, we must remove the template related to the visual mapping
        if (isMechanicalMapping)
        {
            const std::string nonMechanical = templateName.substr(0,7);
            if (nonMechanical == "Mapping") continue;
        }
        //If the component corresponds to a Mapping, we must remove the template related to the Mechanical Mapping
        else if (isMapping)
        {
            const std::string mechanical    = templateName.substr(0,17);
            if (mechanical == "MechanicalMapping") continue;
        }
        componentCreationPossible=true;
        component->addTemplate(itTemplate->first);
    }
    component->endConstruction();

    //If no constructor is available, we delete the component
    if (!componentCreationPossible)
    {
        delete component;
        component=NULL;
    }
    else
        components.push_back(component);

    return component;
}

void CategoryLibrary::endConstruction()
{
}



//-------------------------------------------------------------------------------------------------------
QCategoryLibrary::QCategoryLibrary( QWidget *parent, const std::string &categoryName, unsigned int numComponent): CategoryLibrary(parent, categoryName)
{
    //-----------------------------------------------------------------------
    //QT Creation
    //-----------------------------------------------------------------------
    layout = new CategoryLayout( this, numComponent );
}

QCategoryLibrary::~QCategoryLibrary()
{
    for (unsigned int i=0; i<components.size(); ++i)
    {
        delete components[i];
    }
    delete layout;
    components.clear();
}
ComponentLibrary *QCategoryLibrary::createComponent(const std::string &componentName, ClassEntry* entry, const std::vector< QString > &exampleFiles)
{
    ComponentLibrary* component = new QComponentLibrary(this, layout, componentName,name, entry, exampleFiles);
    return component;
}

ComponentLibrary *QCategoryLibrary::addComponent(const std::string &componentName, ClassEntry* entry, const std::vector< QString > &exampleFiles)
{
    ComponentLibrary *component = CategoryLibrary::addComponent(componentName, entry, exampleFiles);
    if (component)
    {
        layout->addWidget(component, components.size()-1,0);
        connect( component, SIGNAL( componentDragged( std::string, std::string, ClassEntry* ) ),
                this, SLOT( componentDraggedReception( std::string, std::string, ClassEntry*) ) );
    }
    return component;
}


void QCategoryLibrary::endConstruction()
{
    layout->addItem(new QSpacerItem(1,1,QSizePolicy::Minimum, QSizePolicy::Expanding ), layout->numRows(),0);
}


void QCategoryLibrary::setDisplayed(bool b)
{
    if (b) this->show();
    else   this->hide();
}


//*********************//
// SLOTS               //
//*********************//
void QCategoryLibrary::componentDraggedReception( std::string description, std::string templateName, ClassEntry* componentEntry)
{
    emit( componentDragged( description, name, templateName, componentEntry) );
}

}
}
}
