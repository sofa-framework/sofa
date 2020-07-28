/***************************************************************************
                                Component.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:21 $
    Version           : $Revision: 1.5 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "Component.h"
#include "MultiComponent.h"

// -------------------- constructor --------------------
Component::Component(PhysicalModel *p, std::string n) {
    this->properties = new Properties(p,n);
    exclusive = true;
}

// -------------------- destructor --------------------
Component::~Component() {
    deleteProperties();
    removeFromParents();
}

// -------------------- deleteProperties --------------------
void Component::deleteProperties() {
    delete properties;
    properties = NULL;
}

// -------------------- removeFromParents --------------------
void Component::removeFromParents() {
    // tell all the parents that I am disappearing from memory
    // (copy the list as removeSubComponent modify my list)
    std::vector <MultiComponent *> parentMultiComponentsCopy;
    parentMultiComponentsCopy.reserve(parentMultiComponentList.size());
	//std::copy(parentMultiComponentList.begin(), parentMultiComponentList.end(), parentMultiComponentsCopy.begin());
	for (unsigned int i=0 ; i<parentMultiComponentList.size();i++)
		parentMultiComponentsCopy.push_back(parentMultiComponentList[i]);
    

    for (unsigned int i=0; i<parentMultiComponentsCopy.size(); i++)
        parentMultiComponentsCopy[i]->removeSubComponent(this);

    // here assert parentMultiComponentList.size()==0
}
