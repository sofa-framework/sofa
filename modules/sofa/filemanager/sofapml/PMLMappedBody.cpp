/***************************************************************************
								PMLMappedBody
                             -------------------
    begin             : June 12th, 2007
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2007/12/06 9:32:44 $
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

#include "PMLMappedBody.h"

#include "sofa/component/MechanicalObject.h"
#include "sofa/component/mapping/BarycentricMapping.h"
using namespace sofa::component::mapping;
using namespace sofa::component;

#include <PhysicalModel.h>
#include <MultiComponent.h>
#include <CellProperties.h>
#include <PMLTransform.h>

namespace sofa
{

namespace filemanager
{

namespace pml
{

PMLMappedBody::PMLMappedBody(StructuralComponent* body, PMLBody* fromBody, GNode * parent)
{
    parentNode = parent;
    bodyRef = fromBody;

    solverName = body->getProperties()->getString("solver");

    //create the structure
    createMechanicalState(body);
    createSolver();
}


PMLMappedBody::~PMLMappedBody()
{
    if(mapping) delete mapping;
}


Vec3d PMLMappedBody::getDOF(unsigned int index)
{
    return (*((MechanicalState<Vec3dTypes>*)mmodel)->getX())[index];
}


//creation of the mechanical model
//each pml atom constituing the body correspond to a DOF
void PMLMappedBody::createMechanicalState(StructuralComponent* body)
{
    mmodel = new MechanicalObject<Vec3dTypes>;
    StructuralComponent* atoms = body->getAtoms();
    mmodel->resize(atoms->getNumberOfStructures());
    Atom* pAtom;

    double pos[3];
    for (unsigned int i(0) ; i<atoms->getNumberOfStructures() ; i++)
    {
        pAtom = (Atom*) (atoms->getStructure(i));
        pAtom->getPosition(pos);
        AtomsToDOFsIndexes.insert(std::pair <unsigned int, unsigned int>(pAtom->getIndex(),i));
        (*((MechanicalState<Vec3dTypes>*)mmodel)->getX())[i] = Vec3d(pos[0],pos[1],pos[2]);
    }

    //creation of the mapping
    mapping = new BarycentricMapping< MechanicalMapping<MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >((MechanicalState<Vec3Types>*)bodyRef->getMechanicalState(),(MechanicalState<Vec3Types>*) mmodel);

    parentNode->addObject(mmodel);
    parentNode->addObject(mapping);

}


}
}
}
