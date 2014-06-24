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
//	PMLInteractionForceField create an interaction Forcefield (stiffSprings)
//  between 2 other pml Bodies. The sofa structure is translated from pml,
//  specifying the 2 bodies and the list of springs (LINES)
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef SOFAPML_PMLINTERACTIONFORCEFIELD_H
#define SOFAPML_PMLINTERACTIONFORCEFIELD_H

#include "PMLBody.h"
#include <SofaDeformable/StiffSpringForceField.h>
#include "initSofaPML.h"

#include <SofaBaseMechanics/MechanicalObject.h>
#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::component::interactionforcefield;
using namespace std;
using namespace sofa::component::container;

class SOFA_BUILD_FILEMANAGER_PML_API PMLInteractionForceField: public PMLBody
{
public :

    PMLInteractionForceField(StructuralComponent* body, PMLBody* b1, PMLBody* b2, GNode * parent);

    ~PMLInteractionForceField();

    string isTypeOf() { return "interaction"; }

    ///Inherit methods
    GNode::SPtr getPointsNode() {return NULL;}
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
    StiffSpringForceField<Vec3Types>::SPtr Sforcefield;
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

