 /***************************************************************************
                               Properties.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/05/11 10:04:20 $
    Version           : $Revision: 1.2 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 
#include "Properties.h"

Properties::Properties(const std::string n) { 
    name = n;
    myPM = NULL;
}

Properties::Properties(PhysicalModel * p, const std::string n) { 
    name = n;
    myPM = p;
}


Properties::~Properties() {
} 

void Properties::domToFields(xmlNodePtr node)
{
	xmlAttr * attrs = node->properties;
	while (attrs)
	{
		const xmlChar * pname = attrs->name;
		xmlChar * pval = attrs->children->content;

		if (pname && pval){
			std::pair<std::string, std::string> attr(std::string((char*)pname), std::string((char*)pval));
			fields.insert(attr);
		}

		attrs = attrs->next;
	}
}
