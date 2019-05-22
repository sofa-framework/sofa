/***************************************************************************
                                 Component.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:21 $
    Version           : $Revision: 1.16 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// Allow to compile the stl types without warnings on MSVC
// this line should be BEFORE the #ifndef
//#ifdef WIN32
//#pragma warning(disable:4786) // disable C4786 warning
//#endif


#ifndef COMPONENT_H
#define COMPONENT_H

#include <RenderingMode.h>
#include "Properties.h"
#include <string>
#include <vector>
#include <algorithm> // for remove
class Cell;
class MultiComponent;

/** A component is something that composed something and could also be
 *  a part of something.
 *
 *  (just in case you don't really understand, a good reference is "The hitch
 *  hiker's guide to the galaxy", Douglas Adams, 1952-2001. Thanks for reading
 *  this absolutly clear documentation!!!)
 *
 * $Revision: 1.16 $
 */
class Component {
public:
    /** Default constructor, a component needs to know the PM it is in.
      * If not given name is initialized to the empty string
      */
    Component(PhysicalModel *, std::string n="");
    /** Virtual destructor needed here as this is an abstract class (pure virtual) */
    virtual ~Component();

    /// tell if this component is exclusive or not
    bool isExclusive() const;

    /// set the exclusive flag
    void setExclusive(const bool);

    /// pure virtual method, implemented in the child-class
    virtual bool isInstanceOf(const char *) const = 0;

    /// get the name of the component
    const std::string getName() const;

    /// set the name of the component
    void setName(const std::string);

    /** print to an output stream in "pseudo" XML format.
     */
    virtual void xmlPrint(std::ostream &) const = 0;

    /// get the total nr of cell of the component
    virtual unsigned int getNumberOfCells() const = 0;

    /// conveniant method to get cell by order number (not cell index)
    virtual Cell * getCell(unsigned int) const = 0;

    /// return the state of a visibility mode
    virtual bool isVisible(const RenderingMode::Mode mode) const = 0;

    /// set the state of a visibility mode
    virtual void setVisible(const RenderingMode::Mode mode, const bool b) = 0;

    /** @name parent multi component admin
      */
    /*@{*/
    /// get the list of all the Multi Component that are using this Component
    std::vector <MultiComponent *> getAllParentMultiComponents();

    /// get the number of MultiComponent that are using this Component (= nr of parent component)
    unsigned int getNumberOfParentMultiComponents() const;

    /// get a particular MultiComponent that is using this Component (a particular parent component)
    MultiComponent * getParentMultiComponent(unsigned int);

    /// add a particular parent MultiComponent in the list
    void addParentMultiComponent(MultiComponent *);

    /// remove a particular parent MultiComponent
    void removeParentMultiComponent(MultiComponent *);
    /*@}*/

    /// set the physical model
    void setPhysicalModel(PhysicalModel *);

    /// get the physical model
    PhysicalModel * getPhysicalModel() const;

protected:
    Properties *properties;
    
    /** this tell the parent components that this component is 
     *  removed from memory.
     *  As the destructor is virtual, this method has to be called 
     *  in all sub-classes destructors.
     */
    void removeFromParents();
    
    /// delete the "properties" pointer and set it to NULL
    void deleteProperties();

private:
    bool exclusive;

    /** list of Component that are using this component
      *  (if another component is using this component, it is in this list)
      */
    std::vector <MultiComponent *> parentMultiComponentList;

};

// -------------- inline -------------
inline void Component::setExclusive(const bool b) {
    exclusive = b;
}
inline bool Component::isExclusive() const {
    return exclusive;
}
inline const std::string Component::getName() const {
    return properties->getName();
}
inline void Component::setName(const std::string n) {
    properties->setName(n);
}

// -------------- parent Multi Component admin  -------------

inline std::vector <MultiComponent *> Component::getAllParentMultiComponents() {
    return parentMultiComponentList;
}
inline unsigned int Component::getNumberOfParentMultiComponents() const {
    return parentMultiComponentList.size();
}
inline MultiComponent * Component::getParentMultiComponent(unsigned int i) {
    if (i<parentMultiComponentList.size())
        return parentMultiComponentList[i];
    else
        return NULL;
}
inline void Component::addParentMultiComponent(MultiComponent *c) {
    parentMultiComponentList.push_back(c);
}
inline void Component::removeParentMultiComponent(MultiComponent *c) {
    std::vector <MultiComponent *>::iterator it = std::find(parentMultiComponentList.begin(), parentMultiComponentList.end(), c);
    if (it!=parentMultiComponentList.end())
        parentMultiComponentList.erase(it);
}

inline void Component::setPhysicalModel(PhysicalModel *pm) {
    properties->setPhysicalModel(pm);
}

inline PhysicalModel * Component::getPhysicalModel() const {
    return properties->getPhysicalModel();
}

#endif //COMPONENT_H
