/***************************************************************************
								  LMLReader
                             -------------------
    begin             : August 9th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2006/08/09 8:58:16 $
    Version           : $Revision: 0.1 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "LMLReader.h"

#include <XMLLoads.h>

#include "LMLForce.h"
#include "LMLConstraint.h"

#include "PMLReader.h"
#include "PMLRigidBody.h"

#include "sofa/defaulttype/Vec3Types.h"
#include "sofa/defaulttype/RigidTypes.h"
using namespace sofa::defaulttype;
#include "sofa/core/objectmodel/BaseObject.h"
using namespace sofa::core::objectmodel;
#include "sofa/component/constraint/FixedConstraint.h"
using namespace sofa::component::constraint;

namespace sofa
{

namespace filemanager
{

namespace pml
{


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

    //read the file
    XMLLoads data(lmlFile);
    loadsList = data.getLoads();

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
        LMLConstraint<Vec3dTypes> *constraints = new LMLConstraint<Vec3dTypes>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3dTypes>*)(*it)->getMechanicalState());
        if (constraints->getTargets().size() >0)
            if( (*it)->isTypeOf() == "rigid")
            {
                delete constraints;
                FixedConstraint<RigidTypes> * fixedConstraint = new FixedConstraint<RigidTypes>;
                //fixedConstraint->addConstraint(0);
                fixedConstraint->setName("loads");
                (*it)->parentNode->addObject(fixedConstraint);
                ((PMLRigidBody*)*it)->bodyFixed = true;
            }
            else
                (*it)->getPointsNode()->addObject(constraints);
        else
            delete constraints;

        LMLForce<Vec3dTypes> *forces = new LMLForce<Vec3dTypes>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3dTypes>*)(*it)->getMechanicalState());
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
        pointsNode = (*it)->getPointsNode();

        //update constraints
        for (unsigned i=0 ; i<pointsNode->constraint.size() ; i++)
        {
            if (pointsNode->constraint[i]->getName() == "loads")
                pointsNode->removeObject ( pointsNode->constraint[i] );
            //delete ?
        }
        LMLConstraint<Vec3dTypes> *constraints = new LMLConstraint<Vec3dTypes>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3dTypes>*)(*it)->getMechanicalState());
        if (constraints->getTargets().size() >0)
        {
            if( (*it)->isTypeOf() == "rigid")
            {
                delete constraints;
                FixedConstraint<RigidTypes> * fixedConstraint = new FixedConstraint<RigidTypes>;
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
        LMLForce<Vec3dTypes> *forces = new LMLForce<Vec3dTypes>(loadsList, (*it)->AtomsToDOFsIndexes, (MechanicalState<Vec3dTypes>*)(*it)->getMechanicalState());
        if (forces->getTargets().size() >0)
            (*it)->getPointsNode()->addObject(forces);
        else
            delete forces;

        it++;
    }
}

}
}
}
