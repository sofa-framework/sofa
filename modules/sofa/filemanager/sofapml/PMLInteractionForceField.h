/***************************************************************************
								PMLInteractionForceField
                             -------------------
    begin             : October 8th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2007/02/25 13:51:44 $
    Version           : $Revision: 0.2 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

//-------------------------------------------------------------------------
//						--   Description   --
//	PMLInteractionForceField create an interaction Forcefield (stiffSprings)
//  between 2 other pml Bodies. The sofa structure is translated from pml,
//  specifying the 2 bodies and the list of springs (LINES)
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef PMLINTERACTIONFORCEFIELD_H
#define PMLINTERACTIONFORCEFIELD_H

#include "PMLBody.h"
#include "sofa/component/forcefield/StiffSpringForceField.h"

#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::component::forcefield;
using namespace std;

class PMLInteractionForceField: public PMLBody
{
public :

    PMLInteractionForceField(StructuralComponent* body, PMLBody* b1, PMLBody* b2, GNode * parent);

    ~PMLInteractionForceField();

    string isTypeOf() { return "interaction"; }

    ///Inherit methods
    GNode* getPointsNode() {return NULL;}
    bool FusionBody(PMLBody*) {return false;}
    Vector3 getDOF(unsigned int ) {return Vector3();}

private :

    /// creation of the scene graph
    /// only a forcefield is created
    void createForceField();
    void createMechanicalState(StructuralComponent* ) {}
    void createTopology(StructuralComponent* ) {}
    void createMass(StructuralComponent* ) {}
    void createVisualModel(StructuralComponent* ) {}
    void createCollisionModel() {}

    void createSprings(StructuralComponent * body);


    //structure
    StiffSpringForceField<Vec3Types> *Sforcefield;
    PMLBody * body1;
    PMLBody * body2;

    //properties
    SReal  ks;			// spring stiffness
    SReal  kd;			// damping factor
};

}
}
}

#endif

