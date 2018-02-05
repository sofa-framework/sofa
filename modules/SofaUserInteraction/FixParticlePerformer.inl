/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaUserInteraction/FixParticlePerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBoundaryCondition/FixedConstraint.h>

#include <sofa/simulation/Simulation.h>

#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/DeleteVisitor.h>

#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/CapsuleModel.h>


namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
void FixParticlePerformer<DataTypes>::start()
{
    const BodyPicked &picked=this->interactor->getBodyPicked();

    helper::vector<unsigned int > points;
    typename DataTypes::Coord fixPoint;
    MouseContainer* mstateCollision=getFixationPoints(picked, points, fixPoint);

    if (!mstateCollision || points.empty())
    {
        msg_error("FixParticlePerformer") << "Model not supported!" ;
        return;
    }

    simulation::Node* nodeCollision = static_cast<simulation::Node*>(mstateCollision->getContext());
    simulation::Node::SPtr nodeFixation = nodeCollision->createChild("FixationPoint");
    fixations.push_back( nodeFixation.get() );

    //Create the Container of points
    typename MouseContainer::SPtr mstateFixation = sofa::core::objectmodel::New< MouseContainer >();

    mstateFixation->resize(1);
    {
        helper::WriteAccessor<Data<VecCoord> > xData = *mstateFixation->write(core::VecCoordId::position());
        xData.wref()[0] = fixPoint;
    }
    nodeFixation->addObject(mstateFixation);


    //Fix all the points
    typename projectiveconstraintset::FixedConstraint<DataTypes>::SPtr fixFixation = sofa::core::objectmodel::New< projectiveconstraintset::FixedConstraint<DataTypes> >();
    fixFixation->d_fixAll.setValue(true);
    nodeFixation->addObject(fixFixation);

    //Add Interaction ForceField
    typename MouseForceField::SPtr distanceForceField = sofa::core::objectmodel::New< MouseForceField >(mstateFixation.get(), mstateCollision);
    const double friction=0.0;
    const double coeffStiffness=1/(double)points.size();
    for (unsigned int i=0; i<points.size(); ++i)
        distanceForceField->addSpring(0,points[i], stiffness*coeffStiffness, friction, 0);
    nodeFixation->addObject(distanceForceField);

    nodeFixation->execute<simulation::InitVisitor>(sofa::core::ExecParams::defaultInstance());
}

template <class DataTypes>
void FixParticlePerformer<DataTypes>::execute()
{
};




template <class DataTypes>
void FixParticlePerformer<DataTypes>::draw(const core::visual::VisualParams* /*vparams*/)
{
    /// @todo fix draw
//    for (unsigned int i=0; i<fixations.size(); ++i)
//    {
//        bool b = vparams->displayFlags().getShowBehaviorModels();
//        core::visual::DisplayFlags* flags = const_cast<core::visual::DisplayFlags*>(&vparams->displayFlags());
//        flags->setShowBehaviorModels(true);
//        simulation::getSimulation()->draw(const_cast<core::visual::VisualParams*>(vparams),fixations[i]);
//        flags->setShowBehaviorModels(b);
//    }
}


template <class DataTypes>
FixParticlePerformer<DataTypes>::FixParticlePerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i)
{
}


template <class DataTypes>
sofa::component::container::MechanicalObject< DataTypes >* FixParticlePerformer<DataTypes>::getFixationPoints(const BodyPicked &b, helper::vector<unsigned int> &points, typename DataTypes::Coord &fixPoint)
{
    const int idx=b.indexCollisionElement;
    MouseContainer* collisionState=0;

    if (b.body)
    {
        collisionState = dynamic_cast<MouseContainer*>(b.body->getContext()->getMechanicalState());

        if (SphereModel *sphere = dynamic_cast<SphereModel*>(b.body))
        {
            Sphere s(sphere, idx);
            fixPoint = s.p();
            points.push_back(s.getIndex());
        }
        else if(TriangleModel *triangle = dynamic_cast<TriangleModel*>(b.body))
        {
            Triangle t(triangle, idx);
            fixPoint = (t.p1()+t.p2()+t.p3())/3.0;
            points.push_back(t.p1Index());
            points.push_back(t.p2Index());
            points.push_back(t.p3Index());
        }
        else if(CapsuleModel *capsule = dynamic_cast<CapsuleModel*>(b.body)){
            fixPoint = (capsule->point1(idx) + capsule->point2(idx))/2.0;
            points.push_back(capsule->point1Index(idx));
            points.push_back(capsule->point2Index(idx));
        }
        else if(dynamic_cast<RigidSphereModel*>(b.body)||dynamic_cast<OBBModel*>(b.body)){
            collisionState = dynamic_cast<MouseContainer*>(b.mstate);
            fixPoint = (collisionState->read(core::ConstVecCoordId::position())->getValue())[idx];
            points.push_back(idx);
        }
    }
    else if (b.mstate)
    {
        collisionState = dynamic_cast<MouseContainer*>(b.mstate);
        fixPoint = (collisionState->read(core::ConstVecCoordId::position())->getValue())[idx];
        points.push_back(idx);
    }


    return collisionState;
}

}
}
}
