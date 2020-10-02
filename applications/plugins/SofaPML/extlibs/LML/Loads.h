/***************************************************************************
                          Loads.h  -  description
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

#ifndef LOADS_H
#define LOADS_H

class Load;
#include "Load.h"
#include "ValueEvent.h"
#include "Direction.h"

/** @mainpage LML library documentation
 *
 * - To learn more about LML, go to the <a href="http://www-timc.imag.fr/Emmanuel.Promayon/PML">PML/LML website</a>
 * - If you would like to help please check the \ref TODO
 * - To know the exact current LML library version use Loads::VERSION
 */

/** This class makes it possible to manage a list of "Load".
 *
 * Remember that Load is an abstract class (concrete instances are in 
 * instances of Translation, Force...etc)
 *
Usage example:
<pre>
// reading:
main() {
  Loads allLoads("myFile.lml");
  ...
  cout << allLoads;
}
 
// creating and writing:
main() {
  Loads allLoads;
  Translation *t = new Translation();
  t->setUnit(..);
  ...
  allLoads->addLoad(t);
  ...
  cout << allLoads;
}
</pre>
 *
 * All loads that are added to an object of this class are then taking over by it
 * (i.e. when an object of this class is deleted, it will delete all its loads). 
 *
 * $Revision: 51 $
 */
class Loads {

  public:
  /// default constructor
  Loads() {};

  /// build a list of load from an LML file
  Loads(std::string);

  /// destructor
  ~Loads(); 
  
  /// add a load to the list
  void addLoad(Load *ld);

  /// get a load by its index in the list
  Load * getLoad(const unsigned int i) const;

  /// delete a load and remove it from the list using its index
  void deleteLoad(const unsigned int i);
  
  /// get the number of "Load" stored in the list
  unsigned int numberOfLoads() const;

  /** print to an output stream in XML format.
  *  @see Loads.xsd
  */  
  friend std::ostream & operator << (std::ostream &, const Loads);

  /// Print to an ostream
  void xmlPrint(std::ostream &) const;

  /// Print the load list in ansys format (BEWARE: not everything is implemented)
  void ansysPrint(std::ostream &) const;
  
  /** get the first event date present in the list of loads
    * @return -1.0 if no events are found
    */
  double getFirstEventDate();
  
  /** get the last event date present in the list of loads
    * @return -1.0 if no events are found
    */
  double getLastEventDate();

  /// current version of the library
  static const std::string VERSION;
      
  private:
  /// vector of loads : these "Load" are created while the file is parsed
  std::vector <Load*> loads;
    
};


#endif //LOADS_H
