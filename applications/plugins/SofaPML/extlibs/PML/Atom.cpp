/***************************************************************************
                                  Atom.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2007/03/26 07:20:54 $
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

#include "Atom.h"
#include "StructuralComponent.h"
#include "PhysicalModel.h"

//----------------------- Constructors -----------------------
Atom::Atom(PhysicalModel *p) {
    properties = new AtomProperties(p);
}

Atom::Atom(PhysicalModel * p, xmlNodePtr node){
	for (xmlNodePtr child = node->xmlChildrenNode; child != NULL; child = child->next)
		if (!xmlStrcmp(child->name,(const xmlChar*)"atomProperties"))
			properties = new AtomProperties(p, child);
}

Atom::Atom(PhysicalModel *p, const SReal pos[3]) {
    properties = new AtomProperties(p, pos);
}

Atom::Atom(PhysicalModel *p, const unsigned int ind) {
    properties = new AtomProperties(p, ind);
}

Atom::Atom(PhysicalModel *p, const unsigned int ind, const SReal pos[3]) {
    properties = new AtomProperties(p, ind, pos);
}


//----------------------- Destructor -----------------------
Atom::~Atom() {
    delete (AtomProperties *) properties;
    properties = NULL;
}

//----------------------- setIndex -----------------------
bool Atom::setIndex(const unsigned int index) {
    // set the property
    Structure::setIndex(index);
    // tell the physical model about the change (and return true if insertion was ok)
    return properties->getPhysicalModel()->addGlobalIndexAtomPair(std::GlobalIndexStructurePair(index,this));
}

// --------------- xmlPrint ---------------
void Atom::xmlPrint(std::ostream &o, const StructuralComponent *sc) {
    // nothing is to be done in particular with sc, but it is here
    // because of the Structure::xmlPrint method has to be overriden

    // depending on the structure who do the calls, two cases:
    if (properties->getPhysicalModel()!=NULL && sc == properties->getPhysicalModel()->getAtoms()) {
        // - if it is the physical model atom list: the atom print its properties
        // print the atom and its properties
        o << "<atom>" << std::endl;
        ((AtomProperties *) properties)->xmlPrint(o);
        o << "</atom>" << std::endl;
    } else {
        // - if it is any other structural component: the atom print its ref
        o << "<atomRef index=\"" << getIndex() << "\" />" << std::endl;
    }
}
