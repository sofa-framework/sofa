/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/component/collision/FixParticlePerformer.h>
#include <sofa/component/collision/MouseInteractor.h>
#include <sofa/component/constraint/FixedConstraint.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>

namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
void FixParticlePerformer<DataTypes>::start()
{

    core::componentmodel::behavior::MechanicalState<DataTypes>* mstateCollision=NULL;
    int index;
    typename DataTypes::Coord pointPicked;

    BodyPicked picked=this->interactor->getBodyPicked();
    if (picked.body)
    {
        if (mapper) delete mapper;
        mapper = MouseContactMapper::Create(picked.body);
        if (!mapper)
        {
            this->interactor->serr << "Problem with Mouse Mapper creation : " << this->interactor->sendl;
            return;
        }
        std::string name = "contactMouse";
        mstateCollision = mapper->createMapping(name.c_str());
        mapper->resize(1);

        const int idx=picked.indexCollisionElement;
        pointPicked=(*(mstateCollision->getX()))[idx];
        typename DataTypes::Real r=0.0;

        index = mapper->addPoint(pointPicked, idx, r);
        mapper->update();

        if (mstateCollision->getContext() != picked.body->getContext())
        {

            simulation::Node *mappedNode=(simulation::Node *) mstateCollision->getContext();
            simulation::Node *mainNode=(simulation::Node *) picked.body->getContext();
            core::componentmodel::behavior::BaseMechanicalState *mainDof=dynamic_cast<core::componentmodel::behavior::BaseMechanicalState *>(mainNode->getMechanicalState());
            const core::objectmodel::TagSet &tags=mainDof->getTags();
            for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
            {
                mstateCollision->addTag(*it);
                mappedNode->mechanicalMapping->addTag(*it);
            }
            mstateCollision->setName("AttachedPoint");
            mappedNode->mechanicalMapping->setName("MouseMapping");
        }
    }
    else
    {
        mstateCollision = dynamic_cast< core::componentmodel::behavior::MechanicalState<DataTypes>*  >(picked.mstate);
        index = picked.indexCollisionElement;
        pointPicked=(*(mstateCollision->getX()))[index];
        if (!mstateCollision)
        {
            this->interactor->serr << "uncompatible MState during Mouse Interaction " << this->interactor->sendl;
            return;
        }
    }



    std::string name = "contactMouse";
    simulation::Node* nodeCollision = static_cast<simulation::Node*>(mstateCollision->getContext());
    simulation::Node* nodeFixation = simulation::getSimulation()->newNode("FixationPoint");
    fixations.push_back( nodeFixation );
    MouseContainer* mstateFixation = new MouseContainer();
    mstateFixation->setIgnoreLoader(true);
    mstateFixation->resize(1);
    (*mstateFixation->getX())[0] = pointPicked;
    nodeFixation->addObject(mstateFixation);
    constraint::FixedConstraint<DataTypes> *fixFixation = new constraint::FixedConstraint<DataTypes>();


    nodeFixation->addObject(fixFixation);
    MouseForceField *distanceForceField = new MouseForceField(mstateFixation, mstateCollision);
    const double friction=0.0;
    distanceForceField->addSpring(0,index, stiffness, friction, 0);
    nodeFixation->addObject(distanceForceField);

    nodeCollision->addChild(nodeFixation);
    nodeFixation->updateContext();
    nodeFixation->execute<simulation::InitVisitor>();
}

template <class DataTypes>
void FixParticlePerformer<DataTypes>::execute()
{
};

template <class DataTypes>
void FixParticlePerformer<DataTypes>::draw()
{
    for (unsigned int i=0; i<fixations.size(); ++i)
    {
        bool b = fixations[i]->getContext()->getShowBehaviorModels();
        fixations[i]->getContext()->setShowBehaviorModels(true);
        simulation::getSimulation()->draw(fixations[i]);
        fixations[i]->getContext()->setShowBehaviorModels(b);
    }
}


template <class DataTypes>
FixParticlePerformer<DataTypes>::FixParticlePerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i), mapper(NULL)
{
}
template <class DataTypes>
FixParticlePerformer<DataTypes>::~FixParticlePerformer()
{
    if (mapper) delete mapper;
    fixations.clear();
};

}
}
}
