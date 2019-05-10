/***************************************************************************
                           StructureProperties.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/03/15 09:59:49 $
    Version           : $Revision: 1.9 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "StructureProperties.h"

// ------------ Constructor -------------------
StructureProperties::StructureProperties(PhysicalModel *p, const StructureProperties::GeometricType t) : Properties(p) {
    type = t;
}


// ------------ toType (static) -------------------
StructureProperties::GeometricType StructureProperties::toType(const std::string t) {
    if (t=="ATOM")
        return StructureProperties::ATOM;
    else if (t=="TETRAHEDRON")
        return StructureProperties::TETRAHEDRON;
    else if (t=="HEXAHEDRON")
        return StructureProperties::HEXAHEDRON;
    else if (t=="WEDGE")
        return StructureProperties::WEDGE;
    else if (t=="POLY_LINE")
        return StructureProperties::POLY_LINE;
    else if (t=="POLY_VERTEX")
        return StructureProperties::POLY_VERTEX;
    else if (t=="LINE")
        return StructureProperties::LINE;
    else if (t=="TRIANGLE")
        return StructureProperties::TRIANGLE;
    else if (t=="QUAD")
        return StructureProperties::QUAD;
    else
        return StructureProperties::INVALID;
}

// ------------ toString (static) -------------------
std::string StructureProperties::toString(const StructureProperties::GeometricType t) {
    std::string typeStr;

    switch (t) {
        case StructureProperties::ATOM:
            typeStr = "ATOM";
            break;
        case StructureProperties::TETRAHEDRON:
            typeStr = "TETRAHEDRON";
            break;
        case StructureProperties::HEXAHEDRON:
            typeStr = "HEXAHEDRON";
            break;
        case StructureProperties::WEDGE:
            typeStr = "WEDGE";
            break;
        case StructureProperties::POLY_LINE:
            typeStr = "POLY_LINE";
            break;
        case StructureProperties::POLY_VERTEX:
            typeStr = "POLY_VERTEX";
            break;
        case StructureProperties::TRIANGLE:
            typeStr = "TRIANGLE";
            break;
        case StructureProperties::QUAD:
            typeStr = "QUAD";
            break;
        default:
            typeStr = "INVALID";
            break;
    }
    return typeStr;
}


// ------------ xmlPrint -------------------
void StructureProperties::xmlPrint(std::ostream & o) const {
    o << "<StructureProperties  index=\"" << index << "\">" << std::endl;
    o << "<type = \"" << toString(type) << "\"/>" << std::endl;
    o << "</StructureProperties>" << std::endl;
}
