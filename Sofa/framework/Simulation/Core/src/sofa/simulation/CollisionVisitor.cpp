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
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/collision/Pipeline.h>


namespace sofa::simulation
{



void BaseCollisionVisitor::processCollisionPipeline(simulation::Node*
#ifdef SOFA_DUMP_VISITOR_INFO
                                                    node
#endif
                                                , core::collision::Pipeline* obj)
{
    //msg_info()<<"CollisionVisitor::processCollisionPipeline"<<std::endl;
#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionReset");
    ctime_t t0=begin(node, obj);
#endif
    obj->computeCollisionReset();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, obj,t0);
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionDetection");
    t0=begin(node, obj);
#endif
    obj->computeCollisionDetection();

    if (obj->getNarrowPhaseDetection())
        m_primitiveTestCount += obj->getNarrowPhaseDetection()->getPrimitiveTestCount();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, obj,t0);
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionResponse");
    t0=begin(node, obj);
#endif
    obj->computeCollisionResponse();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, obj,t0);
#endif
}



Visitor::Result BaseCollisionVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->collisionPipeline, &BaseCollisionVisitor::processCollisionPipeline);
    return RESULT_CONTINUE;
}


void CollisionVisitor::fwdConstraintSet(simulation::Node*
#ifdef SOFA_DUMP_VISITOR_INFO
                                            node
#endif
                                        , core::behavior::BaseConstraintSet* c)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionDetectionInConstraints");
    ctime_t t0=begin(node, c);
#endif
    c->processGeometricalData();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, c,t0);
#endif
}

void ProcessGeometricalDataVisitor::fwdConstraintSet(simulation::Node*
#ifdef SOFA_DUMP_VISITOR_INFO
                                            node
#endif
                                        , core::behavior::BaseConstraintSet* c)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionDetectionInConstraints");
    ctime_t t0=begin(node, c);
#endif
    c->processGeometricalData();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, c,t0);
#endif
}

Visitor::Result ProcessGeometricalDataVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each<ProcessGeometricalDataVisitor,simulation::Node, NodeSequence<sofa::core::behavior::BaseConstraintSet>,core::behavior::BaseConstraintSet>(this, node, node->constraintSet, &ProcessGeometricalDataVisitor::fwdConstraintSet);
    return RESULT_CONTINUE;
}

Visitor::Result CollisionVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each<CollisionVisitor,simulation::Node, NodeSequence<sofa::core::behavior::BaseConstraintSet>,core::behavior::BaseConstraintSet>(this, node, node->constraintSet, &CollisionVisitor::fwdConstraintSet);
    for_each<CollisionVisitor,simulation::Node, NodeSingle<sofa::core::collision::Pipeline>,sofa::core::collision::Pipeline>(this, node, node->collisionPipeline, &CollisionVisitor::processCollisionPipeline);
    return RESULT_CONTINUE;
}



void CollisionResetVisitor::processCollisionPipeline(simulation::Node*
#ifdef SOFA_DUMP_VISITOR_INFO
        node
#endif
        , core::collision::Pipeline* obj)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionReset");
    ctime_t t0=begin(node, obj);
#endif
    obj->computeCollisionReset();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, obj,t0);
#endif
}

void CollisionDetectionVisitor::processCollisionPipeline(simulation::Node*
#ifdef SOFA_DUMP_VISITOR_INFO
        node
#endif
        , core::collision::Pipeline* obj)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionDetection");
    ctime_t t0=begin(node, obj);
#endif
    obj->computeCollisionDetection();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, obj,t0);
#endif
}

void CollisionResponseVisitor::processCollisionPipeline(simulation::Node*
#ifdef SOFA_DUMP_VISITOR_INFO
        node
#endif
        , core::collision::Pipeline* obj)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    printComment("computeCollisionResponse");
    ctime_t t0=begin(node, obj);
#endif
    obj->computeCollisionResponse();
#ifdef SOFA_DUMP_VISITOR_INFO
    end(node, obj,t0);
#endif
}


} // namespace sofa::simulation



