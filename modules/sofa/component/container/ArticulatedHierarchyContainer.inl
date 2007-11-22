/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#ifndef SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_INL
#define SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_INL

#include <sofa/component/container/ArticulatedHierarchyContainer.h>

namespace sofa
{

namespace component
{

namespace container
{

ArticulatedHierarchyContainer::ArticulationCenter::Articulation::Articulation():
    axis(initData(&axis, (Vector3) Vector3(1,0,0), "rotationAxis", "Set the rotation axis for the articulation")),
    rotation(initData(&rotation, (bool) false, "rotation", "Rotation")),
    translation(initData(&translation, (bool) false, "translation", "Translation")),
    articulationIndex(initData(&articulationIndex, (int) 0, "articulationIndex", "Articulation index"))
{
}

ArticulatedHierarchyContainer::ArticulationCenter::ArticulationCenter():
    parentIndex(initData(&parentIndex, "parentIndex", "Parent of the center articulation")),
    childIndex(initData(&childIndex, "childIndex", "Child of the center articulation")),
    globalPosition(initData(&globalPosition, "globalPosition", "Global position of the articulation center")),
    posOnParent(initData(&posOnParent, "posOnParent", "Parent position of the articulation center")),
    posOnChild(initData(&posOnChild, "posOnChild", "Child position of the articulation center"))
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

vector<ArticulatedHierarchyContainer::ArticulationCenter*> ArticulatedHierarchyContainer::getArticulationCenters()
{
    return articulationCenters;
}

vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> ArticulatedHierarchyContainer::ArticulationCenter::getArticulations()
{
    return articulations;
}

ArticulatedHierarchyContainer::ArticulatedHierarchyContainer()
{
    joint = NULL;
    id = 0;
    chargedFromFile = false;
    numOfFrames = 0;
    dtbvh = 0.0;
}

void ArticulatedHierarchyContainer::parse (sofa::core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("filename"))
    {
        sofa::helper::io::bvh::BVHLoader loader = sofa::helper::io::bvh::BVHLoader();
        joint = loader.load(arg->getAttribute("filename"));
        chargedFromFile = true;
        numOfFrames = joint->getMotion()->frameCount;
        dtbvh = joint->getMotion()->frameTime;
    }
}

void ArticulatedHierarchyContainer::buildCenterArticulationsTree(sofa::helper::io::bvh::BVHJoint* bvhjoint, int id_buf, const char* name, simulation::tree::GNode* node)
{
    std::vector<sofa::helper::io::bvh::BVHJoint*> jointChildren = bvhjoint->getChildren();
    if (jointChildren.size()==0)
        return;

    std::string str(name);
    str.append("/");
    str.append(bvhjoint->getName());

    simulation::tree::GNode* nodeOfArticulationCenters = new simulation::tree::GNode(str);
    node->addChild(nodeOfArticulationCenters);

    ArticulationCenter* ac = new ArticulationCenter();
    nodeOfArticulationCenters->addObject(ac);
    articulationCenters.push_back(ac);

    ac->posOnParent.setValue(Vector3(bvhjoint->getOffset()->x,bvhjoint->getOffset()->y,bvhjoint->getOffset()->z)); //
    ac->posOnChild.setValue(Vector3(0,0,0));
    ac->parentIndex.setValue(id_buf);
    ac->childIndex.setValue(bvhjoint->getId()+1);

    simulation::tree::GNode* nodeOfArticulations = new simulation::tree::GNode("articulations");
    nodeOfArticulationCenters->addChild(nodeOfArticulations);

    sofa::helper::io::bvh::BVHChannels* channels = bvhjoint->getChannels();
    sofa::helper::io::bvh::BVHMotion* motion = bvhjoint->getMotion();

    ArticulationCenter::Articulation* a;

    for (unsigned int j=0; j<channels->channels.size(); j++)
    {
        switch(channels->channels[j])
        {
        case sofa::helper::io::bvh::BVHChannels::NOP:
            break;
        case sofa::helper::io::bvh::BVHChannels::Xposition:
            a = new ArticulationCenter::Articulation();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a);
            a->axis.setValue(Vector3(1,0,0));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Yposition:
            a = new ArticulationCenter::Articulation();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a);
            a->axis.setValue(Vector3(0,1,0));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Zposition:
            a = new ArticulationCenter::Articulation();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a);
            a->axis.setValue(Vector3(0,0,1));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Xrotation:
            a = new ArticulationCenter::Articulation();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a);
            a->axis.setValue(Vector3(1,0,0));
            a->rotation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Yrotation:
            a = new ArticulationCenter::Articulation();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a);
            a->axis.setValue(Vector3(0,1,0));
            a->rotation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Zrotation:
            a = new ArticulationCenter::Articulation();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a);
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
    simulation::tree::GNode* context = dynamic_cast<simulation::tree::GNode *>(this->getContext()); // access to current node

    if (joint != NULL)
    {
        simulation::tree::GNode* articulationCenters = new simulation::tree::GNode("ArticulationCenters");
        context->addChild(articulationCenters);

        buildCenterArticulationsTree(joint, 0, "Root", articulationCenters);

        component::MechanicalObject<Vec1dTypes>* mm1 = dynamic_cast<component::MechanicalObject<Vec1dTypes>*>(context->getMechanicalState());
        mm1->resize(id);

        context = *context->child.begin();
        component::MechanicalObject<RigidTypes>* mm2 = dynamic_cast<component::MechanicalObject<RigidTypes>*>(context->getMechanicalState());
        mm2->resize(joint->getNumJoints()+1);
    }
    else
    {
        context->getTreeObjects<ArticulationCenter>(&articulationCenters);
        vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();
        for (; ac != acEnd; ac++)
        {
            context = dynamic_cast<simulation::tree::GNode *>((*ac)->getContext());
            for (simulation::tree::GNode::ChildIterator it = context->child.begin(); it != context->child.end(); ++it)
            {
                GNode* n = *it;
                n->getTreeObjects<ArticulationCenter::Articulation>(&(*ac)->articulations);
            }
        }
    }
}

} // namespace container

} // namespace component

} // namespace sofa

#endif
