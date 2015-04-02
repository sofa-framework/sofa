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

#include "PMLInteractionForceField.h"

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaDeformable/StiffSpringForceField.h>

#include <PhysicalModel.h>
#include <MultiComponent.h>
#include <PhysicalProperties/CellProperties.h>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::core::objectmodel;
using namespace sofa::component::interactionforcefield;
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
}


//create a TetrahedronFEMForceField
void PMLInteractionForceField::createForceField()
{
    Sforcefield = New<StiffSpringForceField<Vec3Types> >((MechanicalObject<Vec3Types>*)(body1->getMechanicalState().get()), (MechanicalObject<Vec3Types>*)(body2->getMechanicalState().get()));
    parentNode->addObject(Sforcefield);
}


void PMLInteractionForceField::createSprings(StructuralComponent * body)
{
    const Vec3Types::VecCoord& P1 = ((MechanicalObject<Vec3Types>*)body1->getMechanicalState().get())->read(core::ConstVecCoordId::position())->getValue();
    const Vec3Types::VecCoord& P2 = ((MechanicalObject<Vec3Types>*)body2->getMechanicalState().get())->read(core::ConstVecCoordId::position())->getValue();
    if (kd==0.0)kd=5.0;
    if (ks==0.0)ks=500.0;
    for (unsigned int i=0; i<body->getNumberOfCells() ; i++)
    {
        Cell * cell = body->getCell(i);
        if (cell->getType() == StructureProperties::LINE)
        {
            unsigned int dof1 = body1->AtomsToDOFsIndexes[cell->getStructure(0)->getIndex()];
            unsigned int dof2 = body2->AtomsToDOFsIndexes[cell->getStructure(1)->getIndex()];
            Vec3Types::Deriv gap = P1[dof1] - P2[dof2];
            Sforcefield->addSpring(dof1, dof2, ks, kd, sqrt(dot(gap,gap)));
        }
    }
}


}
}
}
