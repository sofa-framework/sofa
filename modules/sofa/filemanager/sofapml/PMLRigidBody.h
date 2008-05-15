/***************************************************************************
								PMLRigidBody
                             -------------------
    begin             : August 18th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2006/02/25 13:51:44 $
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
//	PMLRigidBody translate an indeformable object from Physical Model structure
//  to sofa structure.
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef PMLRIGIDBODY_H
#define PMLRIGIDBODY_H

#include "PMLBody.h"

#include <StructuralComponent.h>

#include "sofa/component/MechanicalObject.h"
#include "sofa/defaulttype/RigidTypes.h"
#include "sofa/defaulttype/Quat.h"


#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::defaulttype;
using namespace std;

class PMLRigidBody: public PMLBody
{
public :

    PMLRigidBody(StructuralComponent* body, GNode * parent);

    ~PMLRigidBody();

    string isTypeOf() { return "rigid"; }

    ///merge the body with current object
    bool FusionBody(PMLBody*);

    Vector3 getDOF(unsigned int index);

    GNode* getPointsNode() {return VisualNode;}

    MechanicalObject<RigidTypes> * getRefDOF() { return refDOF;}

    ///is the body totally fixed?
    bool bodyFixed;


private :

    ///creation of the scene graph
    void createMechanicalState(StructuralComponent* body);
    void createTopology(StructuralComponent* body);
    void createMass(StructuralComponent* body);
    void createVisualModel(StructuralComponent* body);
    void createForceField() {}
    void createCollisionModel();

    ///initialization of properties
    void initMass(string m);
    void initInertiaMatrix(string m);
    void initPosition(string m);
    void initVelocity(string m);


    ///mechanical model containing the reference node (gravity center)
    MechanicalObject<RigidTypes> * refDOF;
    ///mapping between refDof and mesh points
    BaseMapping * mapping;
    ///GNode containing the visual model
    GNode * VisualNode;
    ///GNode containing the collision models
    GNode * CollisionNode;
    ///barycenter coordinates of the solid
    Vector3 bary;

    //members for the mass (only one of the 2 vectors is filled)
    std::vector<SReal> massList;
    std::vector<SReal> inertiaMatrix;

    //members coding for the position
    Vector3 transPos;
    Quat rotPos;

    //members coding for the velocity
    Vector3 transVel;
    Vector3 rotVel;

};

}
}
}

#endif

