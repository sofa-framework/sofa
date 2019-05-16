/***************************************************************************
                           BasicAtomProperties.cpp 
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/07/06 16:00:21 $
    Version           : $Revision: 1.8 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#include "BasicAtomProperties.h"

//----------------------- Class member init -----------------------
void BasicAtomProperties::resetUniqueIndex() {
    BasicAtomProperties::maxUniqueIndex = 0;
}

// initializing the static class member
unsigned int BasicAtomProperties::maxUniqueIndex = 0;

//----------------------- Constructors -----------------------
BasicAtomProperties::BasicAtomProperties(PhysicalModel * p)  : StructureProperties(p, StructureProperties::ATOM) {
    setPosition(0.0, 0.0, 0.0);
    index = maxUniqueIndex++;
}

BasicAtomProperties::BasicAtomProperties(PhysicalModel * p, xmlNodePtr node)  : StructureProperties(p, StructureProperties::ATOM) {

	//search the name attribute
	xmlChar *pname = xmlGetProp(node, (const xmlChar*) "name");
	if(pname)
		setName((char*)pname);

	//search the known attributes
	xmlChar *px = xmlGetProp(node, (const xmlChar*) "x");
	xmlChar *py = xmlGetProp(node, (const xmlChar*) "y");
	xmlChar *pz = xmlGetProp(node, (const xmlChar*) "z");
	if (px && py && pz)
		setPosition(atof((char*)px), atof((char*)py), atof((char*)pz));
	else
		setPosition(0.0, 0.0, 0.0);

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
				  && xmlStrcmp(pname, (xmlChar*)"index")){
			xmlSetProp(unknownAttrs, pname, pval);
		}

		attrs = attrs->next;
	}

	//transform the unknown attributes to a property field map
	domToFields(unknownAttrs);
}

BasicAtomProperties::BasicAtomProperties(PhysicalModel *p, const SReal pos[3]) : StructureProperties(p,StructureProperties::ATOM) {
    setPosition(pos);
    index = maxUniqueIndex++;
}

BasicAtomProperties::BasicAtomProperties(PhysicalModel *p, const unsigned int ind)  :  StructureProperties(p, StructureProperties::ATOM) {
    setPosition(0.0, 0.0, 0.0);
    index = ind;
    if (ind>=maxUniqueIndex)
        maxUniqueIndex = ind+1;
}

BasicAtomProperties::BasicAtomProperties(PhysicalModel *p, const unsigned int ind, const SReal pos[3]) : StructureProperties(p, StructureProperties::ATOM) {
    setPosition(pos);
    index = ind;
    if (ind>=maxUniqueIndex)
        maxUniqueIndex = ind+1;
}

/// write the default xml properties (beginning)
void BasicAtomProperties::beginXML(std::ostream & o) {
    o << "<atomProperties index=\"" << index << "\" ";
    o << "x=\"" << X[0] << "\" y=\"" << X[1] << "\" z=\"" << X[2] << "\"  ";
    if (getName()!="")
        o << "name=\"" << getName().c_str() << "\" ";
}

/// write the default xml properties (end)
void BasicAtomProperties::endXML(std::ostream & o) {
    o << "/>" << std::endl;
}
