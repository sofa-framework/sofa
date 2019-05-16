/***************************************************************************
                          BasicCellProperties.cpp  -  Base of the cell properties
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
#include "BasicCellProperties.h"

//----------------------- Class member init -----------------------
void BasicCellProperties::resetUniqueIndex() {
    BasicCellProperties::maxUniqueIndex = 0;
}

BasicCellProperties::BasicCellProperties(PhysicalModel * p, const StructureProperties::GeometricType t, xmlNodePtr node)  : StructureProperties(p, t) {

	//search the name attribute
	xmlChar *pname = xmlGetProp(node, (const xmlChar*) "name");
	if(pname)
		setName((char*)pname);


	//search the index attribute
	xmlChar *pindex = xmlGetProp(node, (const xmlChar*) "index");
	if (pindex)
		index = atoi((char*)pindex);
	else
		index = maxUniqueIndex++;

	//search the unknown attributes to fill the property fields map
	xmlAttr * attrs = node->properties;
	xmlNodePtr unknownAttrs = xmlNewNode(NULL, (xmlChar*)("unknownAttrs"));
	while (attrs)
	{
		const xmlChar * pname = attrs->name;
		xmlChar * pval = attrs->children->content;
				
		if (pname && xmlStrcmp(pname, (xmlChar*)"name")
				  && xmlStrcmp(pname, (xmlChar*)"x") 
				  && xmlStrcmp(pname, (xmlChar*)"y") 
				  && xmlStrcmp(pname, (xmlChar*)"z") 
				  && xmlStrcmp(pname, (xmlChar*)"type") 
				  && xmlStrcmp(pname, (xmlChar*)"index")){
			xmlSetProp(unknownAttrs, pname, pval);
		}

		attrs = attrs->next;
	}

	//transform the unknown attributes to a property field map
	domToFields(unknownAttrs);
}

// initializing the static class member
unsigned int BasicCellProperties::maxUniqueIndex = 0;

//----------------------- Constructors -----------------------
BasicCellProperties::BasicCellProperties(PhysicalModel *p, const StructureProperties::GeometricType t)  : StructureProperties(p,t) {
    index = maxUniqueIndex++;
}

BasicCellProperties::BasicCellProperties(PhysicalModel *p, const StructureProperties::GeometricType t, const unsigned int ind)  :  StructureProperties(p,t) {
    index = ind;
    if (ind>=maxUniqueIndex)
        maxUniqueIndex = ind+1;
}

/// write the default xml properties (beginning)
void BasicCellProperties::beginXML(std::ostream & o) {
    o << "<cellProperties index=\"" << index << "\" ";
    // print the type
    switch (getType()) {
    case StructureProperties::TETRAHEDRON:
        o << "type=\"TETRAHEDRON\" ";
        break;
    case StructureProperties::HEXAHEDRON:
        o << "type=\"HEXAHEDRON\" ";
        break;
    case StructureProperties::WEDGE:
        o << "type=\"WEDGE\" ";
        break;
    case StructureProperties::POLY_LINE:
        o << "type=\"POLY_LINE\" ";
        break;
    case StructureProperties::POLY_VERTEX:
        o << "type=\"POLY_VERTEX\" ";
        break;
    case StructureProperties::LINE:
        o << "type=\"LINE\" ";
        break;
    case StructureProperties::TRIANGLE:
        o << "type=\"TRIANGLE\" ";
        break;
    case StructureProperties::QUAD:
        o << "type=\"QUAD\" ";
        break;
    default:
        o << "type=\"???\" ";
        break;
    }
    o << " ";
    if (getName()!="")
        o << "name=\"" << getName().c_str() << "\" ";

}

/// write the default xml properties (end)
void BasicCellProperties::endXML(std::ostream & o) {
    o << "/>" << std::endl;
}
