/***************************************************************************
                           StructuralComponent.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:22 $
    Version           : $Revision: 1.24 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "StructuralComponent.h"
#include "Structure.h"
#include "StructuralComponentProperties.h"
#include "Cell.h"
#include "Atom.h"
#include <RenderingMode.h>

// ------------------ constructors ---------------------
StructuralComponent::StructuralComponent(PhysicalModel *p) : Component(p) {
    // delete the properties that has just been instanciated by
    // the Component(..) constructor
    deleteProperties();
    
    // create a new proper one
    properties = new StructuralComponentProperties(p);
    atomList = NULL;
}

StructuralComponent::StructuralComponent(PhysicalModel *p, xmlNodePtr node) : Component(p) {
    // delete the properties that has just been instanciated by
    // the Component(..) constructor
    deleteProperties();
    
    // create a new proper one
    properties = new StructuralComponentProperties(p, node);

    atomList = NULL;
}

StructuralComponent::StructuralComponent(PhysicalModel *p, std::string n) : Component(p,n) {
    // delete the properties that has just been instanciated by
    // the Component(..) constructor
    deleteProperties();
    
    // create a new proper one
    properties = new StructuralComponentProperties(p,n);
    atomList = NULL;
}

// ------------------ destructor ---------------------
StructuralComponent::~StructuralComponent() {
    deleteProperties();
    
    if (atomList)
        delete atomList;
    atomList = NULL;

    // delete all children
    deleteAllStructures();

    // tell all parents that I am going away to the paradise of pointers
    removeFromParents();
}

// ------------------ deleteAllStructures ---------------------
void StructuralComponent::deleteAllStructures() {
    std::vector <Structure *>::iterator it = structures.begin();
    while (it != structures.end()) {
        // (*it) is of type "Structure *"
        // delete (*it) only if "this" is the manager,
        // i.e. only if "this" is the first of the (*it) mySC vector
        if ((*it)->getStructuralComponent(0) == this)
            delete (*it);
        // "au suivant !"
        it ++;
    }
    structures.clear();
}

// ------------------ setColor ---------------------
void StructuralComponent::setColor(const StructuralComponentProperties::Color c) {
    ((StructuralComponentProperties *)properties)->setColor(c);
}
void StructuralComponent::setColor(const SReal r, const SReal g, const SReal b, const SReal a) {
    ((StructuralComponentProperties *)properties)->setRGBA(r,g,b,a);
}
void StructuralComponent::setColor(const SReal r, const SReal g, const SReal b) {
    ((StructuralComponentProperties *)properties)->setRGB(r,g,b);
}

// ------------------ getColor ---------------------
double * StructuralComponent::getColor() const {
    return ((StructuralComponentProperties *)properties)->getRGBA();
}

void StructuralComponent::getColor(double *r, double *g, double *b, double *a) const {
    *r = ((StructuralComponentProperties *)properties)->getRed();
    *g = ((StructuralComponentProperties *)properties)->getGreen();
    *b = ((StructuralComponentProperties *)properties)->getBlue();
    *a = ((StructuralComponentProperties *)properties)->getAlpha();
}

// ------------------ getStructuralComponentPropertiesColor ---------------------
StructuralComponentProperties::Color StructuralComponent::getStructuralComponentPropertiesColor() const {
    return ((StructuralComponentProperties *)properties)->getColor();
}


// ------------------ setMode ---------------------
void StructuralComponent::setMode(const RenderingMode::Mode m) {
    ((StructuralComponentProperties *)properties)->setMode(m);
}

// ------------------ getMode ---------------------
RenderingMode::Mode StructuralComponent::getMode() const {
    return ((StructuralComponentProperties *)properties)->getMode();
}

// ------------------ isVisible ---------------------
bool StructuralComponent::isVisible(const RenderingMode::Mode mode) const {
    return ((StructuralComponentProperties *)properties)->isVisible(mode);
}

// ------------------ setVisible ---------------------
void StructuralComponent::setVisible(const RenderingMode::Mode mode, const bool b) {
    ((StructuralComponentProperties *)properties)->setVisible(mode, b);
}

// --------------- xmlPrint ---------------
void StructuralComponent::xmlPrint(std::ostream &o) const {
    if (getNumberOfStructures()>0) {
        o << "<structuralComponent ";

        // ...the properties...
        ((StructuralComponentProperties *)properties)->xmlPrint(o);

        o << ">" << std::endl;

        // print the color as well if it is not the default one
        if (((StructuralComponentProperties *)properties)->getColor() != StructuralComponentProperties::DEFAULT) {
            o << "<color r=\"" << ((StructuralComponentProperties *)properties)->getRed();
            o << "\" g=\"" << ((StructuralComponentProperties *)properties)->getGreen();
            o << "\" b=\"" << ((StructuralComponentProperties *)properties)->getBlue();
            o << "\" a=\"" << ((StructuralComponentProperties *)properties)->getAlpha() << "\" />" << std::endl;
        }

        // optimize the memory allocation for reading
        o << "<nrOfStructures value=\"" << structures.size() << "\"/>" << std::endl;

        // print out all the structures
        for (unsigned int i=0; i<structures.size(); i++) {
            structures[i]->xmlPrint(o,this);
        }
        o << "</structuralComponent>" << std::endl;
    }
}

// --------------- getNumberOfCells ---------------
unsigned int StructuralComponent::getNumberOfCells() const {
    unsigned int nrOfCells = 0;

    // add all the cells of all the sub components
    for (unsigned int i=0; i<structures.size(); i++) {
        if (structures[i]->isInstanceOf("Cell"))
            nrOfCells++;
    }

    return nrOfCells;
}

// --------------- getCell ---------------
Cell * StructuralComponent::getCell(unsigned int cellOrderNr) const {
    unsigned int cellON;
    unsigned int i;
    Cell *c;

    if (structures.size()==0)
        return NULL;

    i = cellON = 0;
    c = NULL;
    do {
        if (structures[i]->isInstanceOf("Cell")) {
            if (cellON==cellOrderNr)
                c = (Cell *) structures[i];
            cellON++;
        }
    } while (!c && ++i<structures.size());

    return c;
}

// --------------- getCellPosition ---------------
/*
int StructuralComponent::getCellPosition(const Cell *cell) const {
	
	// look for the position of this cell in the SC structures list
	for (unsigned int i=0; i<structures.size(); i++) {
		if (structures[i]->isInstanceOf("Cell"))
		{
			if (cell->getIndex() == structures[i]->getIndex())
				return i;
		}
 
	// This cell is not in this StructuralComponent, return -1
	return -1;
}
*/

// --------------- addStructureIfNotIn ---------------
bool StructuralComponent::addStructureIfNotIn(Structure *s) {
    // loof for this Structure in the current list
    std::vector<Structure *>::iterator it;
    it = std::find(structures.begin(), structures.end(), s);

    if (it == structures.end()) {
        // not in, add it
        addStructure(s);
        return true; // structure added
    } else
        return false;
}

// --------------- getAtoms ---------------
StructuralComponent *StructuralComponent::getAtoms() {
    if (atomList)
        // already created, return it
        return atomList;

    //@@@ Define name : this name + atom List?
    //getName()
    //atomList = new StructuralComponent(myName);

    // Note: atom List is deleted in StructuralComponent destructor
    atomList = new StructuralComponent(properties->getPhysicalModel());

    // loop through all structures
    for (unsigned int i=0; i<this->getNumberOfStructures(); i++) {
        Structure *s = this->getStructure(i);

        if (s->isInstanceOf("Atom")) {
            atomList->addStructureIfNotIn(s);
        } else // s is a Cell
        {
            Cell *cell = (Cell *) s;
            for (unsigned int j=0; j<cell->getNumberOfStructures(); j++) {
                // there is only atoms in a Cell!
                atomList->addStructureIfNotIn(cell->getStructure(j));
            }
        }

    }

    return atomList;
}

// --------------- composedBy ---------------
StructuralComponent::ComposedBy StructuralComponent::composedBy() {
    if (getNumberOfStructures()==0)
        return NOTHING;
    if (getStructure(0)->isInstanceOf("Cell"))
        return CELLS;
    if (getStructure(0)->isInstanceOf("Atom"))
        return ATOMS;
    return NOTHING; // should never occurs!
}

// --------------- isCompatible ---------------
bool StructuralComponent::isCompatible(Structure *s) {
    StructuralComponent::ComposedBy cb;
    cb = composedBy();
    return ((s==NULL) ||
            (cb==NOTHING) ||
            (s->isInstanceOf("Atom") && cb==ATOMS) ||
            (s->isInstanceOf("Cell") && cb==CELLS));

}

