/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "CompliantAttachPerformer.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/BaseMapping.h>
#include <SofaUserInteraction/MouseInteractor.h>
#include <sofa/simulation/Simulation.h>
#include <iostream>
using std::cerr;
using std::endl;

#include <Compliant/compliance/UniformCompliance.h>
#include <sofa/simulation/InitVisitor.h>




namespace sofa
{

namespace component
{

namespace collision
{


template <class DataTypes>
CompliantAttachPerformer<DataTypes>::CompliantAttachPerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i)
    , mapper(0)
    , mouseState(0)
{
    //    cerr<<"CompliantAttachPerformer<DataTypes>::CompliantAttachPerformer()" << endl;
    this->interactor->setMouseAttached(false);
    _compliance = 1e-3;
    _isCompliance = true;
    _visualmodel = false;
}


template <class DataTypes>
CompliantAttachPerformer<DataTypes>::~CompliantAttachPerformer()
{
//    cerr<<"CompliantAttachPerformer<DataTypes>::~CompliantAttachPerformer()" << endl;
    clear();
}

template <class DataTypes>
void CompliantAttachPerformer<DataTypes>::clear()
{
////    cerr<<"CompliantAttachPerformer<DataTypes>::clear()" << endl;

    if( interactionNode )
    {
        if( pickedNode )
            pickedNode->removeChild(interactionNode);
        interactionNode.reset();  // delete the subgraph if no other reference to it
    }

    if (mapper)
    {
        mapper->cleanup();
        delete mapper; mapper=NULL;
    }

    this->interactor->setDistanceFromMouse(0);
    this->interactor->setMouseAttached(false);
}



// hack to stabilize the mouse
// sometimes (when clicking quickly at the same position)
// the mouse position from mouseState->readPositions()[0]
// is wrong. So we compute it from the ray.
// And while the mouse did not move, we do not update anything
// (otherwise it would be update with the wrong position)
// As soon as the mouse is moving, the position is OK.
static defaulttype::Vec3d initialMousePos(0,0,0);


template <class DataTypes>
void CompliantAttachPerformer<DataTypes>::start()
{

//    cerr<<"CompliantAttachPerformer<DataTypes>::start()" << endl;

    if (interactionNode)  // previous interaction still holding
    {
        cerr<<"CompliantAttachPerformer<DataTypes>::start(), releasing previous interaction" << endl;
        clear();            // release it
        return;
    }
    //    cerr<<"CompliantAttachPerformer<DataTypes>::start()" << endl;


    //--------- Picked object
    BodyPicked picked=this->interactor->getBodyPicked();
    if (!picked.body && !picked.mstate) return;


    // get a picked state and a particle index
    core::behavior::MechanicalState<DataTypes>* mstateCollision=NULL;
    core::behavior::MechanicalState<MouseTypes>* mstateCollisionVec=NULL;

    if (picked.body)  // picked a collision element: create a MechanicalState mapped under the collision model, to store the picked point - NOT TESTED YET !!!!
    {
//        std::cerr<<"mapped\n";

        pickedNode = down_cast<simulation::Node>(picked.body->getContext());

        mapper = MouseContactMapper::Create(picked.body);
        if (!mapper)
        {
            this->interactor->serr << "Problem with Mouse Mapper creation : "   << this->interactor->sendl;
            return;
        }
        std::string name = "contactMouse";
        mstateCollisionVec = mapper->createMapping(name.c_str());
        mapper->resize(1);

        typename MouseTypes::Coord pointPicked;
        MouseTypes::set(pointPicked, picked.point[0], picked.point[1], picked.point[2]);
        const int idx=picked.indexCollisionElement;
        Real r=0.0;

        pickedParticleIndex = mapper->addPoint(pointPicked, idx, r);
//        mapper->addPointB(pointPicked, idx, r
//#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
//                , picked.baryCoords
//#endif
//                                 );
        mapper->update();

        // copy the tags of the collision model to the mapped state
        if (mstateCollisionVec->getContext() != picked.body->getContext())
        {

            simulation::Node *mappedNode=(simulation::Node *) mstateCollisionVec->getContext();
            simulation::Node *mainNode=(simulation::Node *) picked.body->getContext();
            core::behavior::BaseMechanicalState *mainDof=mainNode->getMechanicalState();
            const core::objectmodel::TagSet &tags=mainDof->getTags();
            for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
            {
                mstateCollisionVec->addTag(*it);
                mappedNode->mechanicalMapping->addTag(*it);
            }
            mstateCollisionVec->setName("AttachedPoint");
            mappedNode->mechanicalMapping->setName("MouseMapping");
        }

        _baseCollisionMState = mstateCollisionVec;
    }
    else // picked an existing particle
    {
//        std::cerr<<"Already\n";


//        typedef mapping::DistanceFromTargetMapping< MouseTypes,DataTypes1 >  DistanceFromTargetMappingMouse;

        pickedNode = down_cast<simulation::Node>(picked.mstate->getContext());

        mstateCollision = dynamic_cast< core::behavior::MechanicalState<DataTypes>*  >(picked.mstate);
        //        cerr<<"CompliantAttachPerformer<DataTypes>::attach, pickedParticleIndex = " << pickedParticleIndex << endl;
        if (!mstateCollision)
        {
            this->interactor->serr << "incompatible MState during Mouse Interaction " << this->interactor->sendl;
            return;
        }

        _baseCollisionMState = mstateCollision;
    }

    assert(pickedNode);
    pickedParticleIndex = picked.indexCollisionElement;
    if ((unsigned int)_baseCollisionMState->getSize()<picked.indexCollisionElement+1)
        pickedParticleIndex = 0;

    //-------- Mouse manipulator
    mouseMapping = this->interactor->core::objectmodel::BaseObject::template searchUp<sofa::core::BaseMapping>();
    this->mouseState = down_cast<Point3dState>(this->interactor->getMouseContainer());
//    typename Point3dState::ReadVecCoord xmouse = mouseState->readPositions();
//    typename Point3dState::Coord pointOnRay = mouseState->readPositions()[0];



    // set target point to closest point on the ray
    SReal distanceFromMouse=picked.rayLength;
    Ray ray = this->interactor->getMouseRayModel()->getRay(0);
    defaulttype::Vector3 pointOnRay = ray.origin() + ray.direction()*distanceFromMouse;
//    ray.setOrigin(pointOnRay);
    this->interactor->setMouseAttached(true);
    this->interactor->setDistanceFromMouse(distanceFromMouse);


    initialMousePos = DataTypes::getCPos(mouseState->readPositions()[0]);

//    cerr<<"CompliantAttachPerformer<DataTypes>::start() "<<mouseState->readPositions()[0]<<" "<<pointOnRay<< endl;

//    mouseState->writePositions()[0] = pointOnRay;

    //---------- Set up the interaction

    // look for existing interactions
    static const std::string distanceMappingName="InteractionDistanceFromTargetMapping_createdByCompliantAttachPerformer";

    interactionNode = static_cast<simulation::Node*>(_baseCollisionMState->getContext() )->createChild("InteractionDistanceNode");

    typedef component::container::MechanicalObject<DataTypes1> MechanicalObject1;
    typename MechanicalObject1::SPtr extensions = core::objectmodel::New<MechanicalObject1>();
    interactionNode->addObject(extensions);
    extensions->setName("extensionValues");




    if( mstateCollisionVec )
    {
        typedef mapping::DistanceFromTargetMapping< MouseTypes,DataTypes1 >  MyDistanceFromTargetMapping;
        typename MyDistanceFromTargetMapping::SPtr map = core::objectmodel::New<MyDistanceFromTargetMapping>();
        map->setModels(mstateCollisionVec,extensions.get());
        interactionNode->addObject( map );
        map->setName(distanceMappingName.c_str());
        typename MouseTypes::Coord pointOnRayPosition;
        MouseTypes::set(pointOnRayPosition, pointOnRay[0], pointOnRay[1], pointOnRay[2]);
        map->createTarget(/*picked.indexCollisionElement*/ pickedParticleIndex, pointOnRayPosition, /*(picked.point-pointOnRay).norm()*/ 0);
        map->d_showObjectScale.setValue( _arrowSize );
        map->d_color.setValue( _color );
        _distanceMapping = map.get();
    }
    else
    {
        typedef mapping::DistanceFromTargetMapping< DataTypes,DataTypes1 >  MyDistanceFromTargetMapping;
        typename MyDistanceFromTargetMapping::SPtr map = core::objectmodel::New<MyDistanceFromTargetMapping>();
        map->setModels(mstateCollision,extensions.get());
        interactionNode->addObject( map );
        map->setName(distanceMappingName.c_str());
        typename DataTypes::Coord pointOnRayPosition;
        DataTypes::set(pointOnRayPosition, pointOnRay[0], pointOnRay[1], pointOnRay[2]);
        map->createTarget(/*picked.indexCollisionElement*/ pickedParticleIndex, pointOnRayPosition, /*(picked.point-pointOnRay).norm()*/ 0);
        map->d_showObjectScale.setValue( _arrowSize );
        map->d_color.setValue( _color );
        _distanceMapping = map.get();
    }



    if( _visualmodel )
    {
        _vm = core::objectmodel::New<visualmodel::OglModel>();
        defaulttype::ResizableExtVector<visualmodel::OglModel::Coord>& vmpos= *_vm->m_positions.beginWriteOnly();
        vmpos.resize(2);
        vmpos[0] = visualmodel::OglModel::Coord( _baseCollisionMState->getPX(pickedParticleIndex), _baseCollisionMState->getPY(pickedParticleIndex), _baseCollisionMState->getPZ(pickedParticleIndex) );
        vmpos[1] = pointOnRay;
        _vm->m_positions.endEdit();
        defaulttype::ResizableExtVector< visualmodel::OglModel::Triangle >& vmtri= *_vm->m_triangles.beginWriteOnly();
        vmtri.resize(1);
        vmtri[0] = visualmodel::OglModel::Triangle( 0, 0, 1 );
        _vm->m_triangles.endEdit();
        interactionNode->addObject( _vm );
        _vm->setName("mouse");
//        std::cerr<<"mouse: "<<interactionNode->getPathName()<<std::endl;
    }




    //       cerr<<"CompliantAttachPerformer<DataTypes>::start(), create target of " << picked.indexCollisionElement << " at " <<  (*_baseCollisionMState->getX())[picked.indexCollisionElement] << " to " << pointOnRay << ", distance = " << (picked.point-pointOnRay).norm() << endl;

    typedef forcefield::UniformCompliance<DataTypes1> UniformCompliance1;
    typename UniformCompliance1::SPtr compliance = core::objectmodel::New<UniformCompliance1>();
    compliance->setName("pickCompliance");
    compliance->compliance.setValue(_compliance);
    compliance->isCompliance.setValue(_isCompliance);
    interactionNode->addObject(compliance);
    compliance->rayleighStiffness.setValue(_compliance!=0?0.1:0);

    interactionNode->execute<simulation::InitVisitor>(sofa::core::ExecParams::defaultInstance());

}

template <class DataTypes>
void CompliantAttachPerformer<DataTypes>::execute()
{
    if( !mouseState ) return;

    // update the distance mapping using the target position
    typename Point3dState::ReadVecCoord xmouse = mouseState->readPositions();

    // hack, while the mouse did not move, we do not update anything
    if( DataTypes::getCPos(xmouse[0]) == initialMousePos ) return;

    _distanceMapping->updateTarget(pickedParticleIndex,xmouse[0][0],xmouse[0][1],xmouse[0][2]);


    if( _visualmodel )
    {
        defaulttype::ResizableExtVector<visualmodel::OglModel::Coord>& vmpos= *_vm->m_positions.beginWriteOnly();
        vmpos[0] = visualmodel::OglModel::Coord( _baseCollisionMState->getPX(pickedParticleIndex), _baseCollisionMState->getPY(pickedParticleIndex), _baseCollisionMState->getPZ(pickedParticleIndex) );
        vmpos[1] = DataTypes::getCPos(xmouse[0]);
        _vm->m_positions.endEdit();
    //    std::cerr<<"mouse: "<<mstateCollision->getName()<<" "<<mstateCollision->getPX(pickedParticleIndex)<<std::endl;
    }




//    mouseMapping->apply(core::MechanicalParams::defaultInstance());
//    mouseMapping->applyJ(core::MechanicalParams::defaultInstance());

    this->interactor->setMouseAttached(true);
}


template <class DataTypes>
void CompliantAttachPerformer<DataTypes>::configure(configurationsetting::MouseButtonSetting* setting)
{
    //Get back parameters from the MouseButtonSetting Component
    configurationsetting::CompliantAttachButtonSetting* s = dynamic_cast<configurationsetting::CompliantAttachButtonSetting*>(setting);
    if (s)
    {
        _compliance = s->compliance.getValue();
        _isCompliance = s->isCompliance.getValue();
        _arrowSize = s->arrowSize.getValue();
        _color = s->color.getValue();
        _visualmodel = s->visualmodel.getValue();
    }
}


}
}
}
