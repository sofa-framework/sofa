/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "PMLMappedBody.h"

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/BarycentricMapping.h>

#include <PhysicalModel.h>
#include <MultiComponent.h>
#include <PhysicalProperties/CellProperties.h>
#include <PMLTransform.h>

namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace sofa::core::objectmodel;
using namespace sofa::component::mapping;
using namespace sofa::component;

PMLMappedBody::PMLMappedBody(StructuralComponent* body, PMLBody* fromBody, GNode * parent)
{
    parentNode = parent;
    bodyRef = fromBody;

    odeSolverName = body->getProperties()->getString("odesolver");
    linearSolverName = body->getProperties()->getString("linearsolver");

    //create the structure
    createMechanicalState(body);
    createSolver();
}


PMLMappedBody::~PMLMappedBody()
{
}


Vector3 PMLMappedBody::getDOF(unsigned int index)
{
    return ((MechanicalState<Vec3Types>*)mmodel.get())->read(core::ConstVecCoordId::position())->getValue()[index];
}


//creation of the mechanical model
//each pml atom constituing the body correspond to a DOF
void PMLMappedBody::createMechanicalState(StructuralComponent* body)
{
    mmodel = New<MechanicalObject<Vec3Types> >();
    StructuralComponent* atoms = body->getAtoms();
    mmodel->resize(atoms->getNumberOfStructures());
    Atom* pAtom;

    SReal pos[3];
    for (unsigned int i(0) ; i<atoms->getNumberOfStructures() ; i++)
    {
        pAtom = (Atom*) (atoms->getStructure(i));
        pAtom->getPosition(pos);
        AtomsToDOFsIndexes.insert(std::pair <unsigned int, unsigned int>(pAtom->getIndex(),i));
        ((MechanicalState<Vec3Types>*)mmodel.get())->writePositions()[i] = Vector3(pos[0],pos[1],pos[2]);
    }

    //creation of the mapping
    mapping = New<BarycentricMapping< Vec3Types, Vec3Types> > ();
    ((Mapping< Vec3Types, Vec3Types>*)mapping.get())->setModels((MechanicalState<Vec3Types>*)bodyRef->getMechanicalState().get(),(MechanicalState<Vec3Types>*) mmodel.get());

    parentNode->addObject(mmodel);
    parentNode->addObject(mapping);

}


}
}
}
