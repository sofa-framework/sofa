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
//	PMLRigidBody translate an indeformable object from Physical Model structure
//  to sofa structure.
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef SOFAPML_PMLRIGIDBODY_H
#define SOFAPML_PMLRIGIDBODY_H

#include "PMLBody.h"
#include "initSofaPML.h"

#include <StructuralComponent.h>

#include <SofaBaseMechanics/MechanicalObject.h>
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
using namespace sofa::component::container;


class SOFA_BUILD_FILEMANAGER_PML_API PMLRigidBody: public PMLBody
{
public :

    PMLRigidBody(StructuralComponent* body, GNode * parent);

    ~PMLRigidBody();

    string isTypeOf() { return "rigid"; }

    ///merge the body with current object
    bool FusionBody(PMLBody*);

    Vector3 getDOF(unsigned int index);

    GNode::SPtr getPointsNode() {return VisualNode;}

    MechanicalObject<RigidTypes>::SPtr getRefDOF() { return refDOF;}

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
    MechanicalObject<RigidTypes>::SPtr refDOF;
    ///mapping between refDof and mesh points
    BaseMapping::SPtr mapping;
    ///GNode containing the visual model
    GNode::SPtr VisualNode;
    ///GNode containing the collision models
    GNode::SPtr CollisionNode;
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

