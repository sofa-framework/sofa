/***************************************************************************
                          Loads.cpp  -  description
                             -------------------
    begin                : mar f�v 4 2003
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

#include "Loads.h"
#include "ValueEvent.h"
#include "XMLLoads.h"
#include "LoadsVersion.h"

// ------------------ constructor ------------------
Loads::Loads(std::string fileName) {
  XMLLoads xmlReader(fileName, this);
}

// ------------------ destructor ------------------
Loads::~Loads() {
  // std::vector method clear() does not delete each elements individually
  // this has to be done manually...

  // ... so here it is...

  // clear all elements from the array
  for(std::vector<Load*>::iterator it=loads.begin(); it!=loads.end(); it++)
    delete *it;    // free the element from memory
  // finally, clear all elements from the array
  loads.clear();
}

// ------------------ xmlPrint ------------------
/// print the prolog of the xml file
void  Loads::xmlPrint(std::ostream & o) const {
  o << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  o << "<!-- physical model load file -->" << std::endl;
  o << "<loads xmlns='http://www-timc.imag.fr/load'" << std::endl;
  o << "       xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" << std::endl;

  unsigned int i;
  Load * currentL;
  for (i=0; i<numberOfLoads(); i++) {
    currentL = getLoad(i);
    currentL->xmlPrint(o); // o << (*currentL) doesn't work !!!;
    o << std::endl;
  }

  o << "</loads>" << std::endl;
}

// ------------------ ansysPrint ------------------
// Print an ansys translation of the load list (not everything is implemented)
void  Loads::ansysPrint(std::ostream & o) const {
  o << "! -------------------------------------------- " << std::endl;
  o << "! translated from an physical model load file " << std::endl;
  o << "! -------------------------------------------- " << std::endl << std::endl;
  unsigned int i;
  Load * currentL;
  for (i=0; i<numberOfLoads(); i++) {
    currentL = getLoad(i);
    o << "! -- Load #" << i << " (" << currentL->getType() << ")" << std::endl;
    currentL->ansysPrint(o); 
    o << std::endl;
  }

  o << "! --- end of all selections ---" << std::endl;
  o << "ALLSEL" << std::endl;

}
    
// ------------------ operator << ------------------
std::ostream & operator << (std::ostream & o , const Loads l) {
  o << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  o << "<!-- physical model load file -->" << std::endl;
  o << "<loads xmlns='http://www-timc.imag.fr/load'" << std::endl;
  o << "       xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" << std::endl;

  unsigned int i;
  Load * currentL;
  for (i=0; i<l.numberOfLoads(); i++) {
    currentL = l.getLoad(i);
    o << (* currentL) << std::endl;
  }

  o << "</loads>" << std::endl;

  return o;
}

// ------------------ addLoad ------------------
void Loads::addLoad(Load *ld) {
  loads.push_back(ld);
}

// ------------------ getLoad ------------------
Load * Loads::getLoad(const unsigned int i) const {
  if (i<loads.size())
    return loads[i];
  else
    return NULL;
}

// ------------------ numberOfLoads ------------------
unsigned int Loads::numberOfLoads() const {
  return loads.size();
}

// ------------------ deleteLoad ------------------
void Loads::deleteLoad(const unsigned int index) {
  std::vector <Load *>::iterator it;
  it = loads.begin()+index;
  loads.erase(it);  
}

// ------------------ getFirstEventDate ------------------
/// get the first event date present in the list of loads
double Loads::getFirstEventDate() {
  double dateMin = -1.0;
  bool foundOne = false;
  ValueEvent *ev;
  
  for (unsigned int i=0; i<loads.size(); i++) {
    // as events are sorted by date, test only the first event
    // of each load
    ev = loads[i]->getValueEvent(0);
    if (ev && ((foundOne && ev->getDate()<dateMin) || !foundOne)) {
      dateMin = ev->getDate();
      foundOne=true;
    }
  }

  if (foundOne)
    return dateMin;
  else
    return -1.0;
}

// ------------------ getLastEventDate ------------------
/// get the last event date present in the list of loads
double Loads::getLastEventDate() {
  double dateMax = -1.0;
  ValueEvent *ev;

  for (unsigned int i=0; i<loads.size(); i++) {
    // as events are sorted by date, test only the last event
    // of each load
    ev = loads[i]->getValueEvent(loads[i]->numberOfValueEvents()-1);
    if (ev && ev->getDate()>dateMax)
      dateMax = ev->getDate();
  }

  return dateMax;
}

