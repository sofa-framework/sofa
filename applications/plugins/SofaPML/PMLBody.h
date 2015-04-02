/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

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

#ifndef SOFAPML_PMLBODY_H
#define SOFAPML_PMLBODY_H

#include <StructuralComponent.h>

#include "sofa/core/behavior/BaseMechanicalState.h"
#include "sofa/core/BaseMapping.h"
#include "sofa/core/topology/Topology.h"
#include "sofa/core/behavior/BaseMass.h"
#include "sofa/core/behavior/ForceField.h"
#include <SofaOpenglVisual/OglModel.h>
#include "sofa/core/CollisionModel.h"
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>

#include "sofa/defaulttype/Vec3Types.h"
#include <sofa/simulation/tree/GNode.h>
#include "initSofaPML.h"

//#include "sofa/component/StiffSpringForceField.h"

#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::core::topology;
using namespace sofa::component::visualmodel;
using namespace sofa::component;
using namespace sofa::defaulttype;
using namespace sofa::simulation::tree;
using namespace std;


class SOFA_BUILD_FILEMANAGER_PML_API PMLBody
{
public :

    PMLBody();

    virtual ~PMLBody();

    string getName() {return name;}

    virtual string isTypeOf() =0;

    /// accessors
    BaseMechanicalState::SPtr getMechanicalState() { return mmodel; }
    BaseMass::SPtr getMass() { return mass; }
    Topology::SPtr getTopology() { return topology; }
    ForceField<Vec3Types>::SPtr getForcefield() { return forcefield; }

    bool hasCollisions() { return collisionsON; }
    virtual GNode::SPtr getPointsNode()=0;

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
    BaseMechanicalState::SPtr mmodel;
    BaseMass::SPtr mass;
    Topology::SPtr topology;
    ForceField<Vec3Types>::SPtr forcefield;
    OdeSolver::SPtr odeSolver;
    LinearSolver::SPtr linearSolver;

    std::string odeSolverName, linearSolverName;
};

}
}
}

#endif //PMLBODY_H
