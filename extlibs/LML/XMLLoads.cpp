/***************************************************************************
                          xmlloads.cpp  -  description
                             -------------------
    begin                : mar mar 4 2003
    copyright            : (C) 2003 by Emmanuel Promayon
    email                : Emmanuel.Promayon@imag.fr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// XML
/*#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/TransService.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
*/


#include "XMLLoads.h"

#include "Translation.h"
#include "TranslationUnit.h"
#include "Rotation.h"
#include "Force.h"
#include "ForceUnit.h"
#include "Pressure.h"
#include "RotationUnit.h"
#include "PressureUnit.h"

//---------------------- constructor --------------------
XMLLoads::XMLLoads() {
    l = NULL;
}

XMLLoads::XMLLoads(std::string fileName) : l(NULL) {
    xmlRead(fileName);
}

XMLLoads::XMLLoads(std::string fileName, Loads * allLoads) : l(allLoads){
    xmlRead(fileName);
}

//---------------------- destructor --------------------
XMLLoads::~XMLLoads() {
}

//---------------------- getLoads --------------------
Loads * XMLLoads::getLoads() {
    return l;
}

//---------------------- addLoad --------------------
void XMLLoads::addLoad(Load *load) {
    if (!l)
        l = new Loads(); // create empty list
    l->addLoad(load);
}

//---------------------- xmlRead --------------------
void XMLLoads::xmlRead(std::string fileName) {
    
    xmlFile = fileName.c_str();

	// the resulting document tree 
	xmlDocPtr doc; 
	//the pointer to the root node of the document
	xmlNodePtr root;

    doc = xmlParseFile(xmlFile);
	if (doc == NULL) {
		std::cerr << "Failed to open " << xmlFile << std::endl;
		return ;
	}
	
	root = xmlDocGetRootElement(doc);
	if (root == NULL) {
		std::cerr << "empty document" << std::endl;
		xmlFreeDoc(doc);
		return ;
	}

	//read the xml tree root, verify if it is a loads elemnt
	if (xmlStrcmp(root->name,(const xmlChar*)"loads")){
		std::cerr << "failed to read the xml tree" << std::endl;
		xmlFreeDoc(doc);
		return ;
	}

	//parse the rest of the xml tree
	for (xmlNodePtr child = root->xmlChildrenNode; child != NULL; child = child->next)
	{
		if (!xmlStrcmp(child->name,(const xmlChar*)"load")) 
			if (!parseElement(child))
				std::cerr << "failed to read xml element" << std::endl;
	}

	//free the xml
	xmlFreeDoc(doc);
	xmlCleanupParser();
	xmlMemoryDump();

}

// ------------------ parse Element ------------------
bool XMLLoads::parseElement(xmlNodePtr elem)
{
	Load * currentL;
	//get the type of load
	xmlChar *ptype = xmlGetProp(elem, (const xmlChar*) "type");
	if(ptype)
	{
		//create a new instance switch the type
		if (!xmlStrcmp(ptype,(const xmlChar*)"Translation"))
			currentL = new Translation();
		else if(!xmlStrcmp(ptype,(const xmlChar*)"Rotation"))
			currentL = new Rotation();
		else if(!xmlStrcmp(ptype,(const xmlChar*)"Force"))
			currentL = new Force();
		else
			currentL = new Pressure();

		//parse th load children to get its properties
		for (xmlNodePtr child = elem->xmlChildrenNode; child != NULL; child = child->next)
		{
			if (!xmlStrcmp(child->name,(const xmlChar*)"appliedTo")) 
				readLoadAppliedTo(child, currentL);

			if (!xmlStrcmp(child->name,(const xmlChar*)"valueEvent")) 
				readLoadValueEvent(child, currentL);

			if (!xmlStrcmp(child->name,(const xmlChar*)"direction")) 
				readLoadDirection(child, currentL);

			if (!xmlStrcmp(child->name,(const xmlChar*)"unit")) 
				readLoadUnit(child, currentL);
		}

		this->addLoad(currentL);

		return true;
	}
	else
		return false;
}


// ------------------ readLoadAppliedTo ------------------
void XMLLoads::readLoadAppliedTo(xmlNodePtr elem, Load *currentLoad)
{
	xmlChar * pData = xmlNodeGetContent(elem);
	if (pData)
		currentLoad->addTarget(std::string((char*)pData));
}

// ------------------ readLoadValueEvent ------------------
void XMLLoads::readLoadValueEvent(xmlNodePtr elem, Load *currentLoad)
{
	double value=0.0, date=0.0;

	xmlChar * pDate = xmlGetProp(elem, (const xmlChar*)"date");
	if(pDate)
		date = atof((const char*)pDate);

	xmlChar * pValue = xmlGetProp(elem, (const xmlChar*)"value");
	if(pValue)
		value = atof((const char*)pValue);

	if(pDate && pValue)
		currentLoad->addEvent(new ValueEvent(value, date));
}

// ------------------ readLoadDirection ------------------
void XMLLoads::readLoadDirection(xmlNodePtr elem, Load *currentLoad)
{
	Direction dir;

	xmlChar * px = xmlGetProp(elem, (const xmlChar*)"x");
	if(px){
		if (!xmlStrcmp(px,(const xmlChar*)"NULL")) 
			dir.setNullX();
		else
			dir.setX(atof((char *) px));
	}

	xmlChar * py = xmlGetProp(elem, (const xmlChar*)"y");
	if(py){
		if (!xmlStrcmp(py,(const xmlChar*)"NULL")) 
			dir.setNullY();
		else
			dir.setY(atof((char *) py));
	}

	xmlChar * pz = xmlGetProp(elem, (const xmlChar*)"z");
	if(pz){
		if (!xmlStrcmp(pz,(const xmlChar*)"NULL")) 
			dir.setNullZ();
		else
			dir.setZ(atof((char *) pz));
	}

	if(px && py && pz)
		currentLoad->setDirection(dir);
}

// ------------------ readLoadUnit ------------------
void XMLLoads::readLoadUnit(xmlNodePtr elem, Load *currentLoad)
{
	xmlChar * pData = xmlNodeGetContent(elem);
    
    if (currentLoad->getType() == "Translation") {
      //unit de type TranslationUnit
      if (!xmlStrcmp(pData,(const xmlChar*)"mm")) {
             currentLoad->setUnit(TranslationUnit::MM);
      }
      else if (!xmlStrcmp(pData,(const xmlChar*)"microm")) {
            currentLoad->setUnit(TranslationUnit::MICRO_M);
      }
        else if (!xmlStrcmp(pData,(const xmlChar*)"nm")) {
          currentLoad->setUnit(TranslationUnit::NM);
      }
    }
    else
    if (currentLoad->getType() == "Force") {
      if (!xmlStrcmp(pData,(const xmlChar*)"N")) {
             currentLoad->setUnit(ForceUnit::N);
      }
      else if (!xmlStrcmp(pData,(const xmlChar*)"kN")) {
            currentLoad->setUnit(ForceUnit::KN);
      }
        else if (!xmlStrcmp(pData,(const xmlChar*)"pN")) {
          currentLoad->setUnit(ForceUnit::PN);
      }
    }

    else
    if (currentLoad->getType() == "Pressure") {
      if (!xmlStrcmp(pData,(const xmlChar*)"kPa")) {
             currentLoad->setUnit(PressureUnit::KPA);
      }
      else if (!xmlStrcmp(pData,(const xmlChar*)"mmHg")) {
            currentLoad->setUnit(PressureUnit::MMHG);
      }
    }

 else
    if (currentLoad->getType() == "Rotation") {
      if (!xmlStrcmp(pData,(const xmlChar*)"radians")) {
             currentLoad->setUnit(RotationUnit::RAD);
      }
      else if (!xmlStrcmp(pData,(const xmlChar*)"degrees")) {
            currentLoad->setUnit(RotationUnit::DEG);
      }
      
    }
}
