/***************************************************************************
								PMLInteractionForceField
                             -------------------
    begin             : October 8th, 2006
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

#include "PMLInteractionForceField.h"

#include "sofa/component/MechanicalObject.h"
#include "sofa/component/forcefield/StiffSpringForceField.h"
using namespace sofa::component::forcefield;

#include <PhysicalModel.h>
#include <MultiComponent.h>
#include <CellProperties.h>


namespace sofa
{

namespace filemanager
{

namespace pml
{

PMLInteractionForceField::PMLInteractionForceField(StructuralComponent* body, PMLBody* b1, PMLBody* b2, GNode * parent)
{
    parentNode = parent;

    //get the parameters
    collisionsON = false;
    name = body->getProperties()->getName();

    body1 = b1;
    body2 = b2;

    ks = body->getProperties()->getDouble("stiffness");
    kd = body->getProperties()->getDouble("damping");

    //create the structure
    createForceField();
    createSprings(body);
}

PMLInteractionForceField::~PMLInteractionForceField()
{
    if(mmodel) delete mmodel;
    if (Sforcefield) delete Sforcefield;
}


//create a TetrahedronFEMForceField
void PMLInteractionForceField::createForceField()
{
    Sforcefield = new StiffSpringForceField<Vec3dTypes>((MechanicalObject<Vec3dTypes>*)body1->getMechanicalState(), (MechanicalObject<Vec3dTypes>*)body2->getMechanicalState());
    parentNode->addObject(Sforcefield);
}


void PMLInteractionForceField::createSprings(StructuralComponent * body)
{
    Vec3dTypes::VecCoord& P1 = *((MechanicalObject<Vec3dTypes>*)body1->getMechanicalState())->getX();
    Vec3dTypes::VecCoord& P2 = *((MechanicalObject<Vec3dTypes>*)body2->getMechanicalState())->getX();
    if (kd==0.0)kd=5.0;
    if (ks==0.0)ks=500.0;
    for (unsigned int i=0; i<body->getNumberOfCells() ; i++)
    {
        Cell * cell = body->getCell(i);
        if (cell->getType() == StructureProperties::LINE)
        {
            unsigned int dof1 = body1->AtomsToDOFsIndexes[cell->getStructure(0)->getIndex()];
            unsigned int dof2 = body2->AtomsToDOFsIndexes[cell->getStructure(1)->getIndex()];
            Vec3dTypes::Deriv gap = P1[dof1] - P2[dof2];
            Sforcefield->addSpring(dof1, dof2, ks, kd, sqrt(dot(gap,gap)));
        }
    }
}


}
}
}
