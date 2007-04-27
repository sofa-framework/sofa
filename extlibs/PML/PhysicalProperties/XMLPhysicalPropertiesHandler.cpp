/***************************************************************************
     XMLPhysicalPropertiesHandler.cpp  -  Specific properties XML handler
                             -------------------
    auto-generated       : Tuesday 6 July 2004 at 17:5:15
    copyright            : (C) 2001-2004 TIMC (E. Promayon, M. Chabanas)
    email                : Emmanuel.Promayon@imag.fr
    Date                 : $Date: 2004/07/06 16:00:23 $
    Version              : $Revision: 1.5 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "XMLPhysicalModelHandler.h"
#include "Atom.h"
#include "AtomProperties.h"
#include "Cell.h"
#include "CellProperties.h"
#include "StructuralComponent.h"
#include "StructuralComponentProperties.h"

// Read from XML. These methods are automatcially called by the PhysicalModel XML handler
// Four methods can be used to get xml attribute value:
//        std::string getStringValue(const char *);
//        int getIntValue(const char *);
//        float getFloatValue(const char *);
//        double getDoubleValue(const char *);

// atoms
void XMLPhysicalModelHandler::processAtomProperties(Atom *ptr) {
    int propValue1;
    propValue1 = getIntValue("myCustomProperty1");
    if (noProblem())
        ptr->getProperties()->setMyCustomProperty1(propValue1);

    std::string propValue2;
    propValue2 = getStringValue("myCustomProperty2");
    if (noProblem())
        ptr->getProperties()->setMyCustomProperty2(propValue2);

}

// cell
void XMLPhysicalModelHandler::processCellProperties(Cell *ptr) {
    float propValue1;
    propValue1 = getFloatValue("myCustomProperty1");
    if (noProblem())
        ptr->getProperties()->setMyCustomProperty1(propValue1);

    int propValue2;
    propValue2 = getIntValue("myCustomProperty2");
    if (noProblem())
        ptr->getProperties()->setMyCustomProperty2(propValue2);


}

// structural component
void XMLPhysicalModelHandler::processStructuralComponentProperties(StructuralComponent *ptr) {
    int propValue1;
    propValue1 = getIntValue("myCustomProperty1");
    if (noProblem())
        ptr->getProperties()->setMyCustomProperty1(propValue1);

    std::string propValue2;
    propValue2 = getStringValue("myCustomProperty2");
    if (noProblem())
        ptr->getProperties()->setMyCustomProperty2(propValue2);

}

