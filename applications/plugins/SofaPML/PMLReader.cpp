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

#include "PMLReader.h"

#include <MultiComponent.h>

#include "PMLRigidBody.h"
#include "PMLFemForceField.h"
#include "PMLStiffSpringForceField.h"
#include "PMLInteractionForceField.h"
#include "PMLMappedBody.h"

#include <SofaBaseCollision/DefaultPipeline.h>
#include <SofaBaseCollision/DefaultContactManager.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <SofaBaseCollision/BruteForceDetection.h>
#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
#include "sofa/simulation/common/VisualVisitor.h"
#include "sofa/simulation/common/Simulation.h"
#include "sofa/simulation/common/Node.h"

using sofa::core::objectmodel::New;

namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::component::collision;
using namespace sofa::simulation::tree;
using namespace sofa::simulation;

//build sofa structure from a pml file
void PMLReader::BuildStructure(const char* filename, GNode* root)
{

    if (!filename) return;

    if(pm)
    {
        delete pm;
        pm = NULL;
    }
    pm = new PhysicalModel(filename);

    if (!pm)
    {
        cerr<<"PML reader error : failed to load PML file "<<filename<<endl;
        return;
    }

    if (pm->getNumberOfAtoms()==0)
    {
        cerr<<"PML reader error : PML file "<<filename<<" not valid"<<endl;
        return;
    }

    this->BuildStructure(root);
}

//build sofa structure from a physicalModel
void PMLReader::BuildStructure(PhysicalModel * model, GNode* root)
{

    if(pm) delete pm;
    pm = model;
    this->BuildStructure(root);
}

//really build the sofa structure
void PMLReader::BuildStructure(GNode* root)
{

    bodiesList.clear();

    //get the MultiComponent "Bodies"
    MultiComponent * bodies = (MultiComponent*) (pm->getExclusiveComponents()->getComponentByName("Bodies"));
    //if none, the physical model is not properly defined, so we exit...
    if(!bodies) return;

    bool collisionsExist = false;

    //get each body
    PMLBody * body;
    for (unsigned int i=0 ; i<bodies->getNumberOfSubComponents() ; i++ )
    {
        //create the under structure (mech model, topology, etc) to create the body
        body = createBody( (StructuralComponent*) bodies->getSubComponent(i), root );

        //if no problem, we put it in the scene graph
        if (body)
        {
            bodiesList.push_back(body);
            if (body->hasCollisions())
                collisionsExist = true;
        }
    }

    //if at least one of the bodies wants to detect contacts, we create all objects to do this
    if (collisionsExist)
    {
        DefaultPipeline::SPtr ps = New<DefaultPipeline>();
        BruteForceDetection::SPtr bfd = New<BruteForceDetection>();
        NewProximityIntersection::SPtr mpi = New<NewProximityIntersection>();
        //computes the distance contact from the bounding box
        VisualComputeBBoxVisitor act(NULL);
        getSimulation()->init(root);
        root->execute(act);
        SReal dx=(act.maxBBox[0]-act.minBBox[0]);
        SReal dy=(act.maxBBox[1]-act.minBBox[1]);
        SReal dz=(act.maxBBox[2]-act.minBBox[2]);
        SReal dmax = sqrt(dx*dx + dy*dy + dz*dz);
        //maybe the ratio should be changed...
        mpi->setAlarmDistance(dmax/30);
        mpi->setContactDistance(dmax/40);
        DefaultContactManager::SPtr contactManager = New<DefaultContactManager>();
        DefaultCollisionGroupManager::SPtr cgm = New<DefaultCollisionGroupManager>();

        root->addObject(ps);
        root->addObject(bfd);
        root->addObject(mpi);
        root->addObject(contactManager);
        root->addObject(cgm);
    }

    //if there is 2 bodies with the same type and some nodes in common, we merge them
    processFusions(root);

    sofa::simulation::getSimulation()->init(root);
}

//create the body structure
//each a new body type is created, this method must be updated to take care of it.
PMLBody* PMLReader::createBody(StructuralComponent* SC, GNode * root)
{

    string type = SC->getProperties()->getString("bodyType");

    GNode::SPtr child = New<GNode>(SC->getProperties()->getName());

    if (type == "rigid" )
    {
        root->addChild((simulation::Node*)child.get());
        return new PMLRigidBody(SC, child.get());
    }
    if (type == "FEM" )
    {
        root->addChild((simulation::Node*)child.get());
        return new PMLFemForceField(SC, child.get());
    }
    if (type == "stiffSpring" )
    {
        root->addChild((simulation::Node*)child.get());
        return new PMLStiffSpringForceField(SC, child.get());
    }
    if (type == "interaction" )
    {
        std::string name1 = SC->getProperties()->getString("body1");
        std::string name2 = SC->getProperties()->getString("body2");
        PMLBody * body1=NULL;
        PMLBody * body2=NULL;

        std::vector<PMLBody*>::iterator it = bodiesList.begin();
        while((!body1 || !body2) && it != bodiesList.end() )
        {
            if ( (*it)->getName() == name1)
                body1 = *it;
            if ( (*it)->getName() == name2)
                body2 = *it;
            it++;
        }
        if (body1 && body2)
            return new PMLInteractionForceField(SC, body1, body2, root);
    }
    if (type == "mapped")
    {
        std::string name1 = SC->getProperties()->getString("bodyRef");
        PMLBody * body1=NULL;
        std::vector<PMLBody*>::iterator it = bodiesList.begin();
        while( !body1 && it != bodiesList.end() )
        {
            if ( (*it)->getName() == name1)
                body1 = *it;
            it++;
        }
        if (body1)
        {
            body1->parentNode->addChild((simulation::Node*)child.get());
            return new PMLMappedBody(SC, body1, child.get());
        }
        else
            cerr<<"mapped body : no body ref named "<<name1<<" found"<<endl;
    }

    return NULL;
}


void PMLReader::processFusions(GNode * root)
{
    std::vector<PMLBody *>::iterator it1 = bodiesList.begin();
    std::vector<PMLBody *>::iterator it2 = bodiesList.begin();

    while (it1 != bodiesList.end())
    {
        it2 = it1;
        it2++;
        while(it2 != bodiesList.end() )
        {
            bool Fusion = false;
            if ((*it1)->isTypeOf() == (*it2)->isTypeOf() )
            {
                map<unsigned int, unsigned int>::iterator ind = (*it1)->AtomsToDOFsIndexes.begin();
                while(ind != (*it1)->AtomsToDOFsIndexes.end() && !Fusion)
                {
                    if((*it2)->AtomsToDOFsIndexes.find(ind->first) != (*it2)->AtomsToDOFsIndexes.end() )
                    {
                        Fusion = (*it1)->FusionBody(*it2);
                        if (Fusion)
                        {
                            std::vector<PMLBody *>::iterator tmp = it2;
                            tmp--;
                            (*it1)->parentNode->setName((*it1)->parentNode->getName() + " & "+ (*it2)->parentNode->getName() );
                            root->removeChild( (simulation::Node*)(*it2)->parentNode );
                            bodiesList.erase(it2);
                            it2 = tmp;
                        }
                    }
                    ind++;
                }
            }
            it2++;
        }
        it1++;
    }
}


//save the current scene as a pml filename
void PMLReader::saveAsPML(const char * filename)
{
    std::vector<PMLBody*>::iterator itb = bodiesList.begin();
    StructuralComponent * atoms = pm->getAtoms();
    Atom * atom;

    while (itb != bodiesList.end())
    {
        std::map<unsigned int, unsigned int>::iterator itm = (*itb)->AtomsToDOFsIndexes.begin();
        while(itm != (*itb)->AtomsToDOFsIndexes.end() )
        {
            atom = (Atom*) atoms->getStructureByIndex( (*itm).first );
            Vector3 pos = (*itb)->getDOF((*itm).second);
            atom->setPosition(pos[0], pos[1], pos[2]);

            itm++;
        }
        itb++;
    }
    std::ofstream outputFile(filename);
    pm->xmlPrint(outputFile);
    cout<<"fichier sauvegarde"<<endl;
}


//return a point position giving a physical model atom index
Vector3 PMLReader::getAtomPos(unsigned int atomindex)
{
    Vector3 pos;
    std::vector<PMLBody*>::iterator itb = bodiesList.begin();

    while (itb != bodiesList.end())
    {
        std::map<unsigned int, unsigned int>::iterator itm = (*itb)->AtomsToDOFsIndexes.begin();
        while(itm != (*itb)->AtomsToDOFsIndexes.end())
        {
            if ( (*itm).first == atomindex )
            {
                pos = (*itb)->getDOF((*itm).second);
                return pos;
            }
            itm++;
        }
        itb++;
    }
    return Vector3();
}


//update all the physical model atoms positions
void PMLReader::updatePML()
{
    std::vector<PMLBody*>::iterator itb = bodiesList.begin();
    StructuralComponent * atoms = pm->getAtoms();
    Atom * atom;

    while (itb != bodiesList.end())
    {
        if( (*itb)->isTypeOf() != "interaction" )
        {
            if ((*itb)->isTypeOf() != "rigid" || ((PMLRigidBody*)*itb)->bodyFixed == false)
            {
                std::map<unsigned int, unsigned int>::iterator itm = (*itb)->AtomsToDOFsIndexes.begin();
                while(itm != (*itb)->AtomsToDOFsIndexes.end() )
                {
                    atom = (Atom*) atoms->getStructureByIndex( (*itm).first );
                    Vector3 pos = (*itb)->getDOF((*itm).second);
                    atom->setPosition(pos[0], pos[1], pos[2]);

                    itm++;
                }
            }
        }
        itb++;
    }
}


}
}
}
