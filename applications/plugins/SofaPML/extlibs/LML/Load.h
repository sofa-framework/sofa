/***************************************************************************
                          Load.h  -  description
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
  
#ifndef LOAD_H
#define LOAD_H

#include "Direction.h"
#include "ValueEvent.h"
#include "TargetList.h"
#include "Unit.h"

#include "xmlio.h"
#include <sofa/helper/system/config.h>

/** Class that describes a load to be used in the simulation.
  * This load  can have different types Translation, Rotation, Force and Pressure.
  * This load can be created by parsing in an XML file or by load library programming
  * A load could be save in XML format as well using the << operator.
  * The load is set automatically when the method setTarget is called.
  *
  * a Load contains a Type, AppliedTo, 3 Directions x/y/z, a Unit and several ValueEvent
  * (value, date)
  *
  * All value events that are added to the load are then taking over by the load
  * (i.e. when the load is deleted, it will delete all its value event.
  *
  * $Revision: 51 $
  */
class Load {

public:
    ///Constructor
    Load();

    /// destructor is deleting all the value events (BEWARE!!!)
    virtual ~Load();

    /// return true if the load is active at time t
    bool isActive(const SReal t);

    /** The current value at date d (default: d = 0.0).
     * eg: if we have :  
     * <table>
     *    <tr>
     *        <td><b>#</b></td>
     *        <td><b>date</b></td>
     *        <td><b>value</b></td>
     *    </tr>
     *    <tr><td>0</td><td>0.5</td><td>10.0</td></tr>
     *    <tr><td>1</td><td>1.5</td><td>100.0</td></tr>
     * </table>
     * we want to have:
     * - when t<0.5,  val=0
     * - when t=0.5,  val=10
     * - when t=1.0,  val=55
     * - when t>=1.5, val=100
     *
     * Schematically:
     * <pre>
        ^
        |
     100+                        * * * * *
        |                 *
      10+         *
        |         *
        +-*-*-*-*-+-------+--------+------>
        0        0.5     1.0      1.5
        </pre>
    */
    SReal getValue (const SReal d = 0.0);

    /** Insert an event from the particular load
     * the load is set to value v when time is equal to t
     * @param ve the force to add in the list at the correct place
     */
    void addEvent (ValueEvent * ve);
    /** set the valueEvent.
     *  @param v the value
     *  @param d the date at which the value is applied
     */
    void addValueEvent(const SReal v, const SReal d);

    /// Get a the ValueEvent
    ValueEvent * getValueEvent(const unsigned int i) const;
    /// get the nr of value event
    unsigned int numberOfValueEvents() const;
    /// set all value events
    void setAllEvents(std::vector<ValueEvent *>&);

    /// get the type string, has to be impleted in subclasses
    std::string getType() const;

    /// add a lots of new targets using a description string (...)
    void addTarget (std::string currentData);
    /// add a new target
    void addTarget(unsigned int target);
    /// get the number of target
    unsigned int numberOfTargets() const;
    /** Get a target by index
     * @param target the target index in the list
     * @return the target or -1 if target index is out of bounds.
     */
    int getTarget(const unsigned int target) const;
    /// get the complete list
    TargetList getTargetList() const;
    /// set the complete list
    void setTargetList(const TargetList &);

    /// Set the direction using 3 coordinates
    void setDirection (const SReal x, const SReal y, const SReal z);
    
    /// Set the direction using another direction
    void setDirection(const Direction &);
    
    /// Get the direction
    void getDirection (SReal & x, SReal & y, SReal & z) const;
    
    /// get direction object (a copy)
    Direction getDirection() const;

    /// get the unit
    Unit getUnit() const;
    /// set the unit
    void setUnit(const Unit u);

    /** print to an output stream in XML format.
     * @see loads.xsd
     */
    friend std::ostream & operator << (std::ostream &, Load);

    /// Print  the load in ansys format (BEWARE: not everything is implemented)
    virtual void ansysPrint(std::ostream &) const;

    /// Print to an ostream
    void xmlPrint(std::ostream &) const;

    /// static methode to create a new load using a specific type (return NULL if type is unknown)
    static Load * LoadFactory(std::string type);
    
private:
    /// the list of targets
    TargetList targetList;
    /// the list of different events
    std::vector <ValueEvent *> eventList;

    /// delete all the list
    void deleteEventList();

protected:
    Direction dir;
    Unit unit;
    std::string typeString;
};

#endif //LOAD_H
