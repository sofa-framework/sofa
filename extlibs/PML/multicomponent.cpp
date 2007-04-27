/***************************************************************************
                          MultiComponent.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:21 $
    Version           : $Revision: 1.12 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "MultiComponent.h"

// ------------------ default constructor ---------------------
MultiComponent::MultiComponent(PhysicalModel *p) : Component(p) {
    components.clear();
}
// ------------------ constructor with name ---------------------
MultiComponent::MultiComponent(PhysicalModel *p, std::string n) : Component(p,n) {
    components.clear();
};

// ------------------ destructor ---------------------
MultiComponent::~MultiComponent() {
    deleteProperties();
    
    deleteAllSubComponents();
    
    // tell all parents that I am going away to the paradise of pointers
    removeFromParents();    
}

// ------------------ deleteAllSubComponents ---------------------
void MultiComponent::deleteAllSubComponents() {
 //   std::vector<Component *>::iterator it = components.begin();
	//std::vector<Component *>::iterator it_tmp;
 //   while (it != components.end() ) {
	//	it_tmp = it;
	//	it++;
 //       // *it is of type "Component *"
 //       delete (*it_tmp);
 //       // "au suivant !"
 //       
 //   }
	for (unsigned int i=0 ; i<components.size() ; i++)
		delete components[i];
    components.clear();
}

// --------------- xmlPrint ---------------
void MultiComponent::xmlPrint(std::ostream &o) const {
    if (getNumberOfSubComponents()>0) {
        o << "<multiComponent";
        if (getName() != "")
            o<< " name=\"" << getName().c_str() << "\" ";
        o << ">" << std::endl;
        for (unsigned int i=0; i<components.size(); i++) {
            components[i]->xmlPrint(o);
        }
        o << "</multiComponent>" << std::endl;
    }
}

// --------------- getNumberOfCells ---------------
unsigned int MultiComponent::getNumberOfCells() const {
    unsigned int nrOfCells = 0;

    // add all the cells of all the sub components
    for (unsigned int i=0; i<components.size(); i++) {
        nrOfCells += components[i]->getNumberOfCells();
    }

    return nrOfCells;
}

// --------------- getCell ---------------
Cell * MultiComponent::getCell(unsigned int cellOrderNr) const {
    bool found;
    unsigned int i;
    unsigned int startOrderNr;
    unsigned int nrOfCells;

    if (components.size() == 0)
        return NULL;

    // check in which component this cell is
    i = 0;
    startOrderNr = 0;
    do {
        nrOfCells = components[i]->getNumberOfCells();
        found = (cellOrderNr>=startOrderNr && cellOrderNr<(startOrderNr+nrOfCells));
        startOrderNr += nrOfCells;
    } while (!found && ++i<components.size());

    // get it
    return components[i]->getCell(cellOrderNr - (startOrderNr - nrOfCells));
}

// --------------- isVisible ---------------
bool MultiComponent::isVisible(const RenderingMode::Mode mode) const {
    unsigned int i;
    for ( i=0; i<components.size() && !components[i]->isVisible(mode); i++)
        ;
    return (i!=components.size());
}

// --------------- isVisible ---------------
void MultiComponent::setVisible(const RenderingMode::Mode mode, const bool b) {
    unsigned int i;
    // set all subcomponents
    for ( i=0; i<components.size(); i++)
        ;
    components[i]->setVisible(mode, b);
}

