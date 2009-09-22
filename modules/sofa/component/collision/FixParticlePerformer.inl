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

    BodyPicked picked=this->interactor->getBodyPicked();
    if (!picked.body) return;
    MouseContactMapper *mapFixation;
    if (mapperFixations.find(picked.body) == mapperFixations.end())
    {
        mapFixation = MouseContactMapper::Create(picked.body);
        mapperFixations.insert(std::make_pair(picked.body, mapFixation));
    }
    else mapFixation=mapperFixations[picked.body];

    if (!mapFixation)
    {
        std::cerr << "Problem with Mouse MapFixation creation : " << DataTypes::Name() << std::endl;
        return;
    }
    std::string name = "contactMouse";
    core::componentmodel::behavior::MechanicalState<DataTypes>* mstateCollision = mapFixation->createMapping(name.c_str());
    mapFixation->resize(1);
    const typename DataTypes::Coord pointPicked=picked.point;
    const int idx=picked.indexCollisionElement;
    typename DataTypes::Real r=0.0;
    const int index = mapFixation->addPoint(pointPicked, idx, r);
    mapFixation->update();
    simulation::Node* nodeCollision = static_cast<simulation::Node*>(mstateCollision->getContext());
    simulation::Node* nodeFixation = simulation::getSimulation()->newNode("FixationPoint");
    fixations.push_back( nodeFixation );
    MouseContainer* mstateFixation = new MouseContainer();
    mstateFixation->setIgnoreLoader(true);
    mstateFixation->resize(1);
    (*mstateFixation->getX())[0] = pointPicked;
    nodeFixation->addObject(mstateFixation);
    constraint::FixedConstraint<DataTypes> *fixFixation = new constraint::FixedConstraint<DataTypes>();
    fixationConstraint.push_back(fixFixation);


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
    if (!fixationConstraint.empty())
    {
        for (unsigned int i=0; i<fixationConstraint.size(); ++i)
        {
            bool b = fixationConstraint[i]->getContext()->getShowBehaviorModels();
            fixationConstraint[i]->getContext()->setShowBehaviorModels(true);
            fixationConstraint[i]->draw();
            fixationConstraint[i]->getContext()->setShowBehaviorModels(b);
        }
    }
}
template <class DataTypes>
FixParticlePerformer<DataTypes>::FixParticlePerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i)
{
}



template <class DataTypes>
FixParticlePerformer<DataTypes>::~FixParticlePerformer()
{

    while (!fixations.empty())
    {
        simulation::Node *node=*fixations.begin();
        node->detachFromGraph();
        node->execute<simulation::DeleteVisitor>();
        delete node;
        fixations.erase(fixations.begin());
    }

    while (!mapperFixations.empty())
    {
        MouseContactMapper *mapFixation = mapperFixations.begin()->second;
        mapFixation->cleanup();
        delete mapFixation;
        mapperFixations.erase(mapperFixations.begin());
    }

    fixationConstraint.clear();
};


#ifdef WIN32
#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3fTypes> >  FixParticlePerformerVec3fClass("FixParticle",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3dTypes> >  FixParticlePerformerVec3dClass("FixParticle",true);
#endif
#endif
}
}
}
