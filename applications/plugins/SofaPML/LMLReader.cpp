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

#include "LMLReader.h"

#include <Loads.h>

#include "LMLForce.h"
#include "LMLConstraint.h"

#include "PMLReader.h"
#include "PMLRigidBody.h"

#include "sofa/defaulttype/Vec3Types.h"
#include "sofa/defaulttype/RigidTypes.h"
#include "sofa/core/objectmodel/BaseObject.h"
#include <SofaBoundaryCondition/FixedConstraint.h>

namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;
using namespace sofa::component::projectiveconstraintset;

LMLReader::LMLReader(char* filename)
{
    lmlFile = filename;
    loadsList = NULL;
}

void LMLReader::BuildStructure(const char* filename, PMLReader * pmlreader)
{
    if (filename)
        lmlFile = filename;

    //check if there is a file specified
    if (!lmlFile)
    {
        cout<<"LMLReader error : No lml file found"<<endl;
        return;
    }

    if(loadsList) delete loadsList;
    //read the file
    loadsList = new Loads(lmlFile);
    //loadsList = data.getLoads();
    this->BuildStructure(pmlreader);
}

void LMLReader::BuildStructure(Loads * loads, PMLReader * pmlreader)
{
    loadsList = loads;
    this->BuildStructure(pmlreader);
}

void LMLReader::BuildStructure(PMLReader * pmlreader)
{
    //check if loads was read
    if (!loadsList || loadsList->numberOfLoads()<=0)
    {
        cout<<"LMLReader error : No loads found"<<endl;
        return;
    }

    std::vector<PMLBody*>::iterator it = pmlreader->bodiesList.begin();

    while(it!=pmlreader->bodiesList.end())
    {
        //find forces and constraints in the loads list
        LMLConstraint<Vec3Types> *constraints = new LMLConstraint<Vec3Types>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3Types>*)(*it)->getMechanicalState().get());
        std::cout << "Looking for a constraint" << std::endl;
        if (constraints->getTargets().size() >0)
        {
            std::cout << "  Constraint found " << std::endl;
            if ( (*it)->isTypeOf() == "rigid")
            {
                delete constraints;
                FixedConstraint<RigidTypes>::SPtr fixedConstraint = New<FixedConstraint<RigidTypes> >();
                //fixedConstraint->addConstraint(0);
                fixedConstraint->setName("loads");
                (*it)->parentNode->addObject(fixedConstraint);
                ((PMLRigidBody*)*it)->bodyFixed = true;
            }
            else
                (*it)->getPointsNode()->addObject(constraints);
        }
        else
            delete constraints;

        LMLForce<Vec3Types> *forces = new LMLForce<Vec3Types>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3Types>*)(*it)->getMechanicalState().get());
        if (forces->getTargets().size() >0)
            (*it)->getPointsNode()->addObject(forces);
        else
            delete forces;

        it++;
    }
}

void LMLReader::updateStructure(Loads * loads, PMLReader * pmlreader)
{
    loadsList = loads;
    std::vector<PMLBody *>::iterator it = pmlreader->bodiesList.begin();
    GNode * pointsNode;

    while (it != pmlreader->bodiesList.end() )
    {
        pointsNode = (*it)->getPointsNode().get();

        // //update constraints
        // for (unsigned i=0 ; i<pointsNode->constraint.size() ; i++)
        // {
        //     if (pointsNode->constraint[i]->getName() == "loads")
        //         pointsNode->removeObject ( pointsNode->constraint[i] );
        //     //delete ?
        // }
        LMLConstraint<Vec3Types> *constraints = new LMLConstraint<Vec3Types>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3Types>*)(*it)->getMechanicalState().get());
        if (constraints->getTargets().size() >0)
        {
            if( (*it)->isTypeOf() == "rigid")
            {
                delete constraints;
                FixedConstraint<RigidTypes>::SPtr fixedConstraint = New<FixedConstraint<RigidTypes> >();
                //fixedConstraint->addConstraint(0);
                (*it)->parentNode->addObject(fixedConstraint);
                fixedConstraint->setName("loads");
                ((PMLRigidBody*)*it)->bodyFixed = true;
            }
            else
                pointsNode->addObject(constraints);
        }
        else
            delete constraints;

        //update forces
        for (unsigned i=0 ; i<pointsNode->forceField.size() ; i++)
        {
            if (pointsNode->forceField[i]->getName() == "loads")
                pointsNode->removeObject ( pointsNode->forceField[i] );
            //delete ?
        }
        LMLForce<Vec3Types> *forces = new LMLForce<Vec3Types>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3Types>*)(*it)->getMechanicalState().get());
        if (forces->getTargets().size() >0)
            (*it)->getPointsNode()->addObject(forces);
        else
            delete forces;

        it++;
    }
}

void LMLReader::saveAsLML(const char * filename)
{
    if(!loadsList)
        return;

    std::ofstream outputFile(filename);
    loadsList->xmlPrint(outputFile);
}

}
}
}
