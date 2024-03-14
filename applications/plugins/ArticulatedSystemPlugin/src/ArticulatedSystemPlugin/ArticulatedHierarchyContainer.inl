/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <ArticulatedSystemPlugin/ArticulatedHierarchyContainer.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Node.h>

namespace sofa::component::container
{

Articulation::Articulation():
    axis(initData(&axis, type::Vec3(1,0,0), "axis", "Set the rotation axis for the articulation")),
    rotation(initData(&rotation, (bool) false, "rotation", "Rotation")),
    translation(initData(&translation, (bool) false, "translation", "Translation")),
    articulationIndex(initData(&articulationIndex, (int) 0, "articulationIndex", "Articulation index"))
{
    this->addAlias(&axis, "rotationAxis");
}

ArticulationCenter::ArticulationCenter():
    parentIndex(initData(&parentIndex, "parentIndex", "Parent of the center articulation")),
    childIndex(initData(&childIndex, "childIndex", "Child of the center articulation")),
    globalPosition(initData(&globalPosition, "globalPosition", "Global position of the articulation center")),
    posOnParent(initData(&posOnParent, "posOnParent", "Parent position of the articulation center")),
    posOnChild(initData(&posOnChild, "posOnChild", "Child position of the articulation center")),
    articulationProcess(initData(&articulationProcess, (int) 0, "articulationProcess", " 0 - (default) hierarchy between articulations (euler angles)\n 1- ( on Parent) no hierarchy - axis are attached to the parent\n 2- (attached on Child) no hierarchy - axis are attached to the child"))
{
}

ArticulationCenter* ArticulatedHierarchyContainer::getArticulationCenterAsChild(int index)
{
    type::vector<ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    const type::vector<ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();
    for (; ac != acEnd; ac++)
    {
        if ((*ac)->childIndex.getValue() == index)
            return (*ac);
    }
    return nullptr;
}

type::vector<ArticulationCenter*> ArticulatedHierarchyContainer::getAcendantList(int index)
{
    unsigned int i=0;
    acendantList.clear();
    while (index !=0)
    {
        ArticulationCenter* AC = getArticulationCenterAsChild(index);
        if (AC != nullptr)
            acendantList.push_back(AC);
        else {
            msg_error() << "getArticulationCenterAsChild not found for index: " << index;
            break;
        }

        index = acendantList[i]->parentIndex.getValue();
        i++;
    }
    
    return acendantList;
}

ArticulatedHierarchyContainer::ArticulatedHierarchyContainer():
    filename(initData(&filename, "filename", "BVH File to load the articulation", false))
{
    joint = nullptr;
    id = 0;
    chargedFromFile = false;
    numOfFrames = 0;
    dtbvh = 0.0;
}


void ArticulatedHierarchyContainer::buildCenterArticulationsTree(sofa::helper::io::bvh::BVHJoint* bvhjoint, int id_buf, const char* name, simulation::Node* node)
{
    const std::vector<sofa::helper::io::bvh::BVHJoint*> jointChildren = bvhjoint->getChildren();
    if (jointChildren.size()==0)
        return;

    std::string str(name);
    str.append("/");
    str.append(bvhjoint->getName());

    const simulation::Node::SPtr nodeOfArticulationCenters =node->createChild(str);

    const ArticulationCenter::SPtr ac = sofa::core::objectmodel::New<ArticulationCenter>();
    nodeOfArticulationCenters->addObject(ac);
    articulationCenters.push_back(ac.get());

    ac->posOnParent.setValue(type::Vec3(bvhjoint->getOffset()->x,bvhjoint->getOffset()->y,bvhjoint->getOffset()->z)); //
    ac->posOnChild.setValue(type::Vec3(0,0,0));
    ac->parentIndex.setValue(id_buf);
    ac->childIndex.setValue(bvhjoint->getId()+1);

    const simulation::Node::SPtr nodeOfArticulations = nodeOfArticulationCenters->createChild("articulations");

    const sofa::helper::io::bvh::BVHChannels* channels = bvhjoint->getChannels();
    const sofa::helper::io::bvh::BVHMotion* motion = bvhjoint->getMotion();

    msg_info()<<"num Frames found in BVH ="<<motion->frameCount;

    Articulation::SPtr a;

    for (unsigned int j=0; j<channels->channels.size(); j++)
    {
        switch(channels->channels[j])
        {
        case sofa::helper::io::bvh::BVHChannels::NOP:
            break;
        case sofa::helper::io::bvh::BVHChannels::Xposition:
            a = sofa::core::objectmodel::New<Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(type::Vec3(1,0,0));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Yposition:
            a = sofa::core::objectmodel::New<Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(type::Vec3(0,1,0));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Zposition:
            a = sofa::core::objectmodel::New<Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(type::Vec3(0,0,1));
            a->translation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Xrotation:
            a = sofa::core::objectmodel::New<Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(type::Vec3(1,0,0));
            a->rotation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Yrotation:
            a = sofa::core::objectmodel::New<Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(type::Vec3(0,1,0));
            a->rotation.setValue(true);
            a->articulationIndex.setValue(id);
            for (int k=0; k<motion->frameCount; k++)
                a->motion.push_back(motion->frames[k][j]);
            id++;
            break;
        case sofa::helper::io::bvh::BVHChannels::Zrotation:
            a = sofa::core::objectmodel::New<Articulation>();
            nodeOfArticulations->addObject(a);
            ac->articulations.push_back(a.get());
            a->axis.setValue(type::Vec3(0,0,1));
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
        buildCenterArticulationsTree(jointChildren[i], bvhjoint->getId()+1, bvhjoint->getName(), nodeOfArticulationCenters.get());
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

    if (joint != nullptr)
    {
        const simulation::Node::SPtr articulationCenters = context->createChild("ArticulationCenters");

        buildCenterArticulationsTree(joint, 0, "Root", articulationCenters.get());

        auto* mm1 = dynamic_cast<core::behavior::MechanicalState<defaulttype::Vec1Types>*>(context->getMechanicalState());
        mm1->resize(id);

        context = (context->child.begin())->get();
        auto* mm2 = dynamic_cast<core::behavior::MechanicalState<defaulttype::RigidTypes>*>(context->getMechanicalState());
        mm2->resize(joint->getNumJoints()+1);
    }
    else
    {
        context->getTreeObjects<ArticulationCenter>(&articulationCenters);
        msg_info() << "Found " << articulationCenters.size() << " centers";
        type::vector<ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
        const type::vector<ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();
        for (; ac != acEnd; ac++)
        {
            context = dynamic_cast<simulation::Node *>((*ac)->getContext());
            for (simulation::Node::ChildIterator it = context->child.begin(); it != context->child.end(); ++it)
            {
                simulation::Node* n =  it->get();
                n->getTreeObjects<Articulation>(&(*ac)->articulations);
            }

            // for Arboris Mapping, init the transformation for each articulation center
            type::Quat<SReal> q; // TODO: add a rotation component to the positionning on the ArticulatedHierarchyContainer
            (*ac)->H_p_pLc.set((*ac)->posOnParent.getValue(),q);
            (*ac)->H_c_cLp.set((*ac)->posOnChild.getValue(), q);
            (*ac)->H_pLc_cLp.identity();
        }
    }
}


} // namespace sofa::component::container
