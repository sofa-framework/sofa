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
#include <sofa/component/collision/detection/algorithm/CollisionPipeline.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>

#include <sofa/simulation/Node.h>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/simulation/Visitor.h>
#endif

#include <sofa/helper/ScopedAdvancedTimer.h>
using sofa::helper::ScopedAdvancedTimer ;

#include <sofa/helper/AdvancedTimer.h>


namespace sofa::component::collision::detection::algorithm
{

using namespace core;
using namespace core::objectmodel;
using namespace core::collision;
using namespace sofa::defaulttype;

void registerCollisionPipeline(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("The default collision detection and modeling pipeline.")
        .add< CollisionPipeline >());
}

const int CollisionPipeline::defaultDepthValue = 6;

CollisionPipeline::CollisionPipeline()
    : d_doPrintInfoMessage(initData(&d_doPrintInfoMessage, false, "verbose",
                                    "Display extra information at each computation step. (default=false)"))
    , d_doDebugDraw(initData(&d_doDebugDraw, false, "draw",
                             "Draw the detected collisions. (default=false)"))

    //TODO(dmarchal 2017-05-16) Fix the min & max value with response from a github issue. Remove in 1 year if not done.
    , d_depth(initData(&d_depth, defaultDepthValue, "depth",
               ("Max depth of bounding trees. (default=" + std::to_string(defaultDepthValue) + ", min=?, max=?)").c_str()))
{
    
    m_subCollisionPipeline = sofa::core::objectmodel::New<SubCollisionPipeline>();
    m_subCollisionPipeline->d_depth.setParent(&this->d_depth);
    m_subCollisionPipeline->l_broadPhaseDetection.set(this->broadPhaseDetection);
    m_subCollisionPipeline->l_narrowPhaseDetection.set(this->narrowPhaseDetection);
    m_multiCollisionPipeline = sofa::core::objectmodel::New<CompositeCollisionPipeline>();
    m_multiCollisionPipeline->l_subCollisionPipelines.add(m_subCollisionPipeline.get());
    
    this->addSlave(m_subCollisionPipeline);
    this->addSlave(m_multiCollisionPipeline);
}

void CollisionPipeline::init()
{
    Inherit1::init();
    
    msg_info() << "CollisionPipeline is now a wrapper to CompositeCollisionPipeline with a single SubCollisionPipeline.";
    msg_info() << "If you want more flexibility, use directly the components CompositeCollisionPipeline and SubCollisionPipeline, with their respective Data.";

    auto context = this->getContext();
    assert(context);
    // set the whole collision models list to the sub collision pipeline
    sofa::type::vector<sofa::core::CollisionModel::SPtr> collisionModels;
    context->get<sofa::core::CollisionModel, sofa::type::vector<sofa::core::CollisionModel::SPtr>>(&collisionModels, BaseContext::SearchRoot);
        
    for(auto collisionModel : collisionModels)
    {
        m_subCollisionPipeline->l_collisionModels.add(collisionModel.get());
    }
    
    // set the other component to the sub collision pipeline (which is implcitely searched/set by PipelineImpl)
    m_subCollisionPipeline->l_intersectionMethod.set(this->intersectionMethod);
    m_subCollisionPipeline->l_broadPhaseDetection.set(this->broadPhaseDetection);
    m_subCollisionPipeline->l_narrowPhaseDetection.set(this->narrowPhaseDetection);
    m_subCollisionPipeline->l_contactManager.set(this->contactManager);
    
    m_subCollisionPipeline->init();
    m_multiCollisionPipeline->init();
    
    /// Insure that all the value provided by the user are valid and report message if it is not.
    checkDataValues() ;
}

void CollisionPipeline::checkDataValues()
{
    if(d_depth.getValue() < 0)
    {
        msg_warning() << "Invalid value 'depth'=" << d_depth.getValue() << "." << msgendl
                      << "Replaced with the default value = " << defaultDepthValue;
        d_depth.setValue(defaultDepthValue) ;
    }
}

void CollisionPipeline::doCollisionReset()
{
    m_multiCollisionPipeline->doCollisionReset();
}

void CollisionPipeline::doCollisionDetection(const type::vector<core::CollisionModel*>& collisionModels)
{
    m_multiCollisionPipeline->doCollisionDetection(collisionModels);
}

void CollisionPipeline::doCollisionResponse()
{
    m_multiCollisionPipeline->doCollisionResponse();
}

std::set< std::string > CollisionPipeline::getResponseList() const
{
    return m_multiCollisionPipeline->getResponseList();
}

} // namespace sofa::component::collision::detection::algorithm
