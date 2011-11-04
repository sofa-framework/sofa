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

#include "PMLMappedBody.h"

#include "sofa/component/container/MechanicalObject.h"
#include "sofa/component/mapping/BarycentricMapping.h"

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
    if(mapping) delete mapping;
}


Vector3 PMLMappedBody::getDOF(unsigned int index)
{
    return (*((MechanicalState<Vec3Types>*)mmodel)->getX())[index];
}


//creation of the mechanical model
//each pml atom constituing the body correspond to a DOF
void PMLMappedBody::createMechanicalState(StructuralComponent* body)
{
    mmodel = new MechanicalObject<Vec3Types>;
    StructuralComponent* atoms = body->getAtoms();
    mmodel->resize(atoms->getNumberOfStructures());
    Atom* pAtom;

    SReal pos[3];
    for (unsigned int i(0) ; i<atoms->getNumberOfStructures() ; i++)
    {
        pAtom = (Atom*) (atoms->getStructure(i));
        pAtom->getPosition(pos);
        AtomsToDOFsIndexes.insert(std::pair <unsigned int, unsigned int>(pAtom->getIndex(),i));
        (*((MechanicalState<Vec3Types>*)mmodel)->getX())[i] = Vector3(pos[0],pos[1],pos[2]);
    }

    //creation of the mapping
    mapping = new BarycentricMapping< MechanicalMapping<MechanicalState<Vec3Types>, MechanicalState<Vec3Types> > >((MechanicalState<Vec3Types>*)bodyRef->getMechanicalState(),(MechanicalState<Vec3Types>*) mmodel);

    parentNode->addObject(mmodel);
    parentNode->addObject(mapping);

}


}
}
}
