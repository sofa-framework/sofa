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

#ifndef SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_INL
#define SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_INL

#include <sofa/component/container/ArticulatedHierarchyContainer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/common/Simulation.h>

namespace sofa
{

namespace component
{

namespace container
{

ArticulatedHierarchyContainer::ArticulationCenter::Articulation::Articulation():
    axis(initData(&axis, (Vector3) Vector3(1,0,0), "axis", "Set the rotation axis for the articulation")),
    rotation(initData(&rotation, (bool) false, "rotation", "Rotation")),
    translation(initData(&translation, (bool) false, "translation", "Translation")),
    articulationIndex(initData(&articulationIndex, (int) 0, "articulationIndex", "Articulation index"))
{
    this->addAlias(&axis, "rotationAxis");
}

ArticulatedHierarchyContainer::ArticulationCenter::ArticulationCenter():
    parentIndex(initData(&parentIndex, "parentIndex", "Parent of the center articulation")),
    childIndex(initData(&childIndex, "childIndex", "Child of the center articulation")),
    globalPosition(initData(&globalPosition, "globalPosition", "Global position of the articulation center")),
    posOnParent(initData(&posOnParent, "posOnParent", "Parent position of the articulation center")),
    posOnChild(initData(&posOnChild, "posOnChild", "Child position of the articulation center")),
    articulationProcess(initData(&articulationProcess, (int) 0, "articulationProcess", " 0 - (default) hierarchy between articulations (euler angles)\n 1- ( on Parent) no hierarchy - axis are attached to the parent\n 2- (attached on Child) no hierarchy - axis are attached to the child"))
{
}

ArticulatedHierarchyContainer::ArticulationCenter* ArticulatedHierarchyContainer::getArticulationCenterAsChild(int index)
{
    vector<ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();
    for (; ac != acEnd; ac++)
    {
        if ((*ac)->childIndex.getValue() == index)
            return (*ac);
    }
    return (*ac);
}

vector<ArticulatedHierarchyContainer::ArticulationCenter*> ArticulatedHierarchyContainer::getAcendantList(int index)
{
    unsigned int i=0;
    acendantList.clear();
    while (index !=0)
    {
        acendantList.push_back(getArticulationCenterAsChild(index));
        index = acendantList[i]->parentIndex.getValue();
        i++;
    }
    return acendantList;
}

ArticulatedHierarchyContainer::ArticulatedHierarchyContainer():
    filename(initData(&filename, "filename", "BVH File to load the articulation", false))
{
    joint = NULL;
    id = 0;
    chargedFromFile = false;
    numOfFrames = 0;
    dtbvh = 0.0;
}


void ArticulatedHierarchyContainer::buildCenterArticulationsTree(sofa::helper::io::bvh::BVHJoint* bvhjoint, int id_buf, const char* name, simulation::Node* node)
{
    std::vector<sofa::helper::io::bvh::BVHJoint*> jointChildren = bvhjoint->getChildren();
    if (jointChildren.size()==0)
        return;

    std::string str(name);
    str.append("/");
    str.append(bvhjoint->getName());

    simulation::Node* nodeOfArticulationCenters =node->createChild(str);

    ArticulationCenter::SPtr ac = sofa::core::objectmodel::New<ArticulationCenter>();
    nodeOfArticulationCenters->addObject(ac);
    articulationCenters.push_back(ac.get());

    ac->posOnParent.setValue(Vector3(bvhjoint->getOffset()->x,bvhjoint->getOffset()->y,bvhjoint->getOffset()->z)); //
    ac->posOnChild.setValue(Vector3(0,0,0));
    ac->parentIndex.setValue(id_buf);
    ac->childIndex.setValue(bvhjoint->getId()+1);

    simulation::Node* nodeOfArticulations = nodeOfArticulationCenters->createChild("articulations");

    sofa::helper::io::bvh::BVHChannels* channels = bvhjoint->getChannels();
    sofa::helper::io::bvh::BVHMotion* motion = bvhjoint->getMotion();

    serr<<"num Frames found in BVH ="<<motion->frameCount<<sendl;

    ArticulationCenter::Articulation::SPtr a;

    for (unsigned int j=0; j<channels->channels.size(); j++)
    {
        switch(channels->channels[j])
        {
        case sofa::helper::io::bvh::BVHChannels::NOP:
            break;
        case sofa::helper::io::bvh::BVHChannels::Xposition:
            a = sofa::core::objectmodel::New<ArticulationCenter::Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(Vector3(1,0,0));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Yposition:
            a = sofa::core::objectmodel::New<ArticulationCenter::Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(Vector3(0,1,0));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Zposition:
            a = sofa::core::objectmodel::New<ArticulationCenter::Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(Vector3(0,0,1));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Xrotation:
            a = sofa::core::objectmodel::New<ArticulationCenter::Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(Vector3(1,0,0));
            a->rotation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Yrotation:
            a = sofa::core::objectmodel::New<ArticulationCenter::Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(Vector3(0,1,0));
            a->rotation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Zrotation:
            a = sofa::core::objectmodel::New<ArticulationCenter::Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(Vector3(0,0,1));
            a->rotation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        }
    }

    for(unsigned int i=0; i<jointChildren.size(); i++)
    {
        buildCenterArticulationsTree(jointChildren[i], bvhjoint->getId()+1, bvhjoint->getName(), nodeOfArticulationCenters);
    }
}

void ArticulatedHierarchyContainer::init ()
{
    simulation::Node* context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node

    std::string file = filename.getFullPath();
    if ( sofa::helper::system::DataRepository.findFile (file) )
    {

        sofa::helper::io::bvh::BVHLoader loader = sofa::helper::io::bvh::BVHLoader();
        joint = loader.load(sofa::helper::system::DataRepository.getFile ( file ).c_str());
        chargedFromFile = true;
        numOfFrames = joint->getMotion()->frameCount;
        dtbvh = joint->getMotion()->frameTime;
    }

    if (joint != NULL)
    {
        simulation::Node* articulationCenters = context->createChild("ArticulationCenters");

        buildCenterArticulationsTree(joint, 0, "Root", articulationCenters);

        component::container::MechanicalObject<Vec1dTypes>* mm1 = dynamic_cast<component::container::MechanicalObject<Vec1dTypes>*>(context->getMechanicalState());
        mm1->resize(id);

        context = (context->child.begin())->get();
        component::container::MechanicalObject<RigidTypes>* mm2 = dynamic_cast<component::container::MechanicalObject<RigidTypes>*>(context->getMechanicalState());
        mm2->resize(joint->getNumJoints()+1);
    }
    else
    {
        context->getTreeObjects<ArticulationCenter>(&articulationCenters);
        vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();
        for (; ac != acEnd; ac++)
        {
            context = dynamic_cast<simulation::Node *>((*ac)->getContext());
            for (simulation::Node::ChildIterator it = context->child.begin(); it != context->child.end(); ++it)
            {
                simulation::Node* n =  it->get();
                n->getTreeObjects<ArticulationCenter::Articulation>(&(*ac)->articulations);
            }

            // for Arboris Mapping, init the transformation for each articulation center
            Quat q; // TODO: add a rotation component to the positionning on the ArticulatedHierarchyContainer
            (*ac)->H_p_pLc.set((*ac)->posOnParent.getValue(),q);
            (*ac)->H_c_cLp.set((*ac)->posOnChild.getValue(), q);
            (*ac)->H_pLc_cLp.identity();

        }
    }
}




} // namespace container

} // namespace component

} // namespace sofa

#endif
