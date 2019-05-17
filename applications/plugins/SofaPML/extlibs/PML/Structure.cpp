/***************************************************************************
                                Structure.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/05/11 10:04:21 $
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

#include "Structure.h"
#include "StructureProperties.h"

//-------------------- should become inline ------------------------
unsigned int Structure::getIndex() const {
    return Structure::properties->getIndex();
}

bool Structure::setIndex(const unsigned int newIndex) {
    Structure::properties->setIndex(newIndex);
    return true;
}

StructureProperties::GeometricType Structure::getType() const {
    return Structure::properties->getType();
}

void Structure::setName(std::string n) {
    properties->setName(n);
}

std::string Structure::getName() const {
    return properties->getName();
}

/*
ostream & operator << (ostream & o, const Structure &s) {
	o << "<UndefinedStructure>" << endl;
	o << "\t" << (* s.properties) << endl;
	o << "</UndefinedStructure>" << endl;
	return o;
}
*/

