/***************************************************************************
                               MultiComponent.h
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


#ifndef MULTICOMPONENT_H
#define MULTICOMPONENT_H

#include <vector>
#include <algorithm>  
#include <cstring>

//using namespace std;

#include "Component.h"

/**
 * A multi-component stores other components, hence providing a way to have
 * an tree representation of components. Isn't that tricky?
 *
 * <b>NOTE: </b> To delete and free the memory of all the sub components, you have to
 * call the deleteAllSubComponents() method.
 *
 *@author Emmanuel Promayon
 * $Revision: 1.16 $
 */
class MultiComponent : public Component {
public:
    /** Default Constructor */
    MultiComponent(PhysicalModel *);
    /// constructor that allows to name the structure (provides a name)
    MultiComponent(PhysicalModel *, std::string);
    /// delete all the subcomponents (call the deleteAllSubComponents method)
    ~MultiComponent();

    unsigned int getNumberOfSubComponents() const;
    Component * getSubComponent(const unsigned int) const;
    void addSubComponent(Component *);
    /**
     * Remove a component from the list.
     * Becareful: this method DOES NOT delete the object and/or free the memory.
      * This method ask the component c to remove this multicomponent from the list of its parent component
     * @param c the ptr to the structure to remove
     * @see removeAllSubComponent()
     */
    void removeSubComponent(Component *c);

    /**
     * this method free all the sub-components (i.e. delete all the sub component
     *  and clear the list).
     * After this methode getNumberOfSubComponents should return 0
     */
    void deleteAllSubComponents();

    /** return true only if the parameter is equal to "MultiComponent" */
    virtual bool isInstanceOf(const char *) const;

    /** print to an output stream in "pseaudo" XML format (do nothing if there are no sub components).
      */
    void xmlPrint(std::ostream &) const;

    /// get the total nr of cell of the component
    unsigned int getNumberOfCells() const;

    /// get cell by order number (not cell index)
    Cell * getCell(unsigned int) const;

    /// conveniant method to get the sub component of the name given in parameter
    Component * getComponentByName(const std::string);
    
    /// return the state of a visibility mode in all the sub component (if at least one sub component is visible for this mode, it will return true; if none are visible it will return false).
    virtual bool isVisible(const RenderingMode::Mode mode) const;

    /// set the state of a visibility mode in all the sub component.
    virtual void setVisible(const RenderingMode::Mode mode, const bool b);

protected:
    /**
     * List of sub component
     */
    std::vector <Component *> components;
};

// -------------------- inline methods -------------------
inline 	unsigned int MultiComponent::getNumberOfSubComponents() const {
    return components.size();
}
inline Component * MultiComponent::getSubComponent(const unsigned int i) const {
    if (i<components.size())
        return components[i];
    else
        return NULL;
}
inline void MultiComponent::addSubComponent(Component * c) {
    // add c in the list
    components.push_back(c);
    // add this in the list of c's composing component
    c->addParentMultiComponent(this);
}
inline void MultiComponent::removeSubComponent(Component *c) {
    std::vector <Component *>::iterator it = std::find(components.begin(), components.end(), c);
    if (it != components.end()) {
        components.erase(it);
        c->removeParentMultiComponent(this);
    }
}
inline Component * MultiComponent::getComponentByName(const std::string n) {
    std::vector <Component *>::iterator it=components.begin();
    Component *foundC=NULL;
    while (it!=components.end() && !foundC) {
        foundC = ((*it)->getName()==n)?(*it):NULL;
        // look inside the component if it is a MultiComponent
        if (!foundC && (*it)->isInstanceOf("MultiComponent")) {
            foundC = ((MultiComponent *) (*it))->getComponentByName(n);
        }
        it++;
    }
    return foundC;
}
inline bool MultiComponent::isInstanceOf(const char *className) const {
    return (strcmp(className, "MultiComponent")==0);
}

#endif //MULTICOMPONENT_H
