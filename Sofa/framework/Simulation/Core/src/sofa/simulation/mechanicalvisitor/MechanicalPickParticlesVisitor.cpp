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

#include <sofa/simulation/mechanicalvisitor/MechanicalPickParticlesVisitor.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/BaseMapping.h>

namespace sofa::simulation::mechanicalvisitor
{

Visitor::Result MechanicalPickParticlesVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    if (mm->hasTag(tagNoPicking)) // picking disabled for this model
        return RESULT_CONTINUE;

    //We deactivate the Picking with static objects (not simulated)
    core::CollisionModel *c {nullptr};
    mm->getContext()->get(c, core::objectmodel::BaseContext::Local);
    if (c && !c->isSimulated()) //If it is an obstacle, we don't try to pick
    {
        return RESULT_CONTINUE;
    }
    mm->pickParticles(this->params, rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPickParticlesVisitor::fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
{
    if (mm->hasTag(tagNoPicking)) // picking disabled for this model
        return RESULT_CONTINUE;

    if (node->mechanicalMapping  && !node->mechanicalMapping->isMechanical())
        return RESULT_PRUNE;
    mm->pickParticles(this->params, rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPickParticlesVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!map->isMechanical())
        return RESULT_PRUNE;
    return RESULT_CONTINUE;
}

/// get the closest pickable particle
void MechanicalPickParticlesVisitor::getClosestParticle( core::behavior::BaseMechanicalState*& mstate, sofa::Index& indexCollisionElement, type::Vec3& point, SReal& rayLength )
{
    mstate = nullptr;

    if( particles.empty() ) return;

    rayLength = std::numeric_limits<SReal>::max();

    core::behavior::BaseMechanicalState* mstatei;
    unsigned int indexCollisionElementi;
    type::Vec3 pointi;
    SReal rayLengthi;

    // particles are sorted from their distance to the ray
    // threshold for valid particles is the shortest distance + small tolerance relative to ray length
    const SReal dmax = particles.begin()->first + radius0*1e-10;

    for( Particles::const_iterator it=particles.begin(), itend=particles.end() ; it!=itend ; ++it )
    {
        if( it->first > dmax ) break; // from now on, particles are too far from the ray

        // get current valid particle
        mstatei = it->second.first;
        indexCollisionElementi = it->second.second;
        pointi[0] = mstatei->getPX(indexCollisionElementi);
        pointi[1] = mstatei->getPY(indexCollisionElementi);
        pointi[2] = mstatei->getPZ(indexCollisionElementi);
        rayLengthi = (pointi-rayOrigin)*rayDirection;

        if( rayLengthi < rayLength ) // keep the closest particle to the ray origin
        {
            mstate = mstatei;
            indexCollisionElement = indexCollisionElementi;
            point = pointi;
            rayLength = rayLengthi;
        }
    }
}

}