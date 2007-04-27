/***************************************************************************
                          Loads.cpp  -  description
                             -------------------
    begin                : mar fév 4 2003
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

// --------------- static member initialization -----------------
// Version #
const std::string Loads::VERSION = "0.5 - 2 august 2005";


// ------------------ destructor ------------------
Loads::~Loads() {
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
  std::vector <Load *> newList;
  unsigned int i;
  
  if (index>=loads.size())
    return;

  // copy everything up to the index
  for (i=0; i<index; i++) {
    newList.push_back(loads[i]);
  }
  i++; // jump over the load to be deleted
  //... and copy everything up to the end
  while (i<loads.size()) {
    newList.push_back(loads[i]);
    i++;
  }

  // delete the load
  delete loads[index];
  
  // erase new list and copy everything
  loads.clear();

  // copy the new list into the new one
  for (i=0;i<newList.size();i++)
    loads.push_back(newList[i]);
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

