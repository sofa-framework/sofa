/***************************************************************************
								  PMLBody
                             -------------------
    begin             : August 17th, 2006
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
//	PMLBody is an abstract class representing an object and its structure,
//  imported from a PML file, to create the sofa scene graph.
//  it is inherited by PMLRigidBody and PML(ForcefieldName) classes.
//-------------------------------------------------------------------------

#ifndef PMLBODY_H
#define PMLBODY_H

#include <StructuralComponent.h>

#include "sofa/core/componentmodel/behavior/BaseMechanicalState.h"
#include "sofa/core/BaseMapping.h"
#include "sofa/core/componentmodel/topology/Topology.h"
#include "sofa/core/componentmodel/behavior/BaseMass.h"
#include "sofa/core/componentmodel/behavior/ForceField.h"
#include "sofa/component/visualmodel/OglModel.h"
#include "sofa/core/CollisionModel.h"
#include <sofa/core/componentmodel/behavior/OdeSolver.h>

#include "sofa/defaulttype/Vec3Types.h"
#include <sofa/simulation/tree/GNode.h>

//#include "sofa/component/StiffSpringForceField.h"

#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace sofa::core;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::core::componentmodel::topology;
using namespace sofa::component::visualmodel;
using namespace sofa::component;
using namespace sofa::defaulttype;
using namespace sofa::simulation::tree;
using namespace std;


class PMLBody
{
public :

    PMLBody();

    virtual ~PMLBody();

    string getName() {return name;}

    virtual string isTypeOf() =0;

    /// accessors
    BaseMechanicalState* getMechanicalState() { return mmodel; }
    BaseMass* getMass() { return mass; }
    Topology* getTopology() { return topology; }
    ForceField<Vec3Types>* getForcefield() { return forcefield; }

    bool hasCollisions() { return collisionsON; }
    virtual GNode* getPointsNode()=0;

    ///merge 2 bodies
    virtual bool FusionBody(PMLBody*)=0;

    virtual Vector3 getDOF(unsigned int index)=0;

    //link between atoms indexes (physical model) and DOFs indexes (sofa)
    map<unsigned int, unsigned int> AtomsToDOFsIndexes;

    ///the node from which the body is created
    GNode * parentNode;

protected :

    ///creation of the scene graph
    virtual void createMechanicalState(StructuralComponent* body) =0;
    virtual void createTopology(StructuralComponent* body) =0;
    virtual void createMass(StructuralComponent* body) =0;
    virtual void createVisualModel(StructuralComponent* body) =0;
    virtual void createForceField() =0;
    virtual void createCollisionModel() =0;
    void createSolver();

    //name of the object
    string name;

    ///is collisions detection activated
    bool collisionsON;

    ///objects structure
    BaseMechanicalState * mmodel;
    BaseMass * mass;
    Topology * topology;
    ForceField<Vec3Types> * forcefield;
    OdeSolver * solver;

    std::string solverName;
};

}
}
}

#endif //PMLBODY_H
