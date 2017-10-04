#include "SimpleAnimationLoop.h"


#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/IntegrateBeginEvent.h>
#include <sofa/simulation/IntegrateEndEvent.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/helper/cast.h>

namespace sofa
{

namespace simulation
{

SOFA_DECL_CLASS(SimpleAnimationLoop)

int _ = core::RegisterObject("a truly simple animation loop")
    .add< SimpleAnimationLoop >()
    ;



class SOFA_Compliant_API SimpleAnimateVisitor : public Visitor {

  protected :
    SReal dt;

  public:
    SimpleAnimateVisitor(const core::ExecParams* params, SReal dt)
        : Visitor(params), dt(dt) { }
    

  protected:
    
    virtual void on_collision_pipeline(simulation::Node* node,
                                       core::collision::Pipeline* /*obj*/) {

        sofa::helper::ScopedAdvancedTimer step("collision");

        // note: AnimateVisitor propagates a CollisionBegin event here
        
        CollisionVisitor act(this->params);
        node->execute(&act);    

        // note: AnimateVisitor propagates a CollisionEnd event here
    }


  public:

    // README: any pass you might add to this function should be
    // (de)enabled through a boolean data in the animation loop,
    // possibly true by default, with a comment stating why this is
    // useful
    virtual Result processNodeTopDown(simulation::Node* node) {

        // early stop
        if(!node->isActive()) return Visitor::RESULT_PRUNE;
        if(node->isSleeping()) return Visitor::RESULT_PRUNE;

        // dt
        node->setDt(dt);
        
        // collision pipelines
        if( node->collisionPipeline ) {
            on_collision_pipeline(node, node->collisionPipeline);
        }
        
        // wtf? multiple solvers? are y'all crazy?!
        if( !node->solver.empty() ) {

            sofa::helper::ScopedAdvancedTimer step("time integration");
            
            sofa::core::MechanicalParams mparams(*this->params);
            mparams.setDt(dt);
            

            for(unsigned i = 0, n = node->solver.size(); i < n; ++i ) {
                node->solver[i]->solve(params, dt);
            }

            const SReal next = node->getTime() + dt;
            using namespace sofa::core;
            
            // this is needed
            MechanicalProjectPositionAndVelocityVisitor project(&mparams,
                                                                next,
                                                                VecCoordId::position(),
                                                                VecDerivId::velocity());
            project.execute(node);
            MechanicalPropagateOnlyPositionAndVelocityVisitor propagate(&mparams,
                                                                        next,
                                                                        VecCoordId::position(),
                                                                        VecDerivId::velocity(),
                                                                        true);
            propagate.execute(node);

            // stop after first solver
            return RESULT_PRUNE;
        }


        // note: AnimateVisitor does something with
        // interactionforcefields at this point, this could be needed.
        
        return RESULT_CONTINUE;
    }


    
};





SimpleAnimationLoop::SimpleAnimationLoop():
    extra_steps(initData(&extra_steps, unsigned(0),
                         "extra_steps",
                         "perform additional simulation steps during one animation step")) {
    
}


// README: any pass you might add to this function should be
// (de)enabled through a boolean data, possibly true by default, with
// a comment stating why this is useful
void SimpleAnimationLoop::step(const core::ExecParams* params, SReal dt)
{
    sofa::simulation::Node* gnode = down_cast<sofa::simulation::Node>(this->getContext()->getRootContext());
    
    if (dt == 0) {
        // make sure we don't do silly shit
        dt = gnode->getDt();
    }

    const SReal start = gnode->getTime();
    
    for(unsigned i = 0, n = 1 + extra_steps.getValue(); i < n; ++i) {
        
        {
            AnimateBeginEvent ev ( dt );
            PropagateEventVisitor act ( params, &ev );
            gnode->execute ( act );
        }
        
    
        {
            sofa::helper::ScopedAdvancedTimer step("animate visitor");
            AnimateVisitor act(params, dt);
            gnode->execute ( act );
        }

        // note: only the root node is updated, which is probably
        // reasonable anyways
        gnode->setTime ( start + (i + 1) * dt );
        
        // note: DefaultAnimationLoop executes UpdateSimulationContextVisitor here
        {
            AnimateEndEvent ev ( dt );
            PropagateEventVisitor act ( params, &ev );
            gnode->execute ( act );
        }
    }    
    
    gnode->execute< UpdateMappingVisitor >(params);
    
    // note: DefaultAnimationLoop propagates an UpdateMappingEvent here
        
}    


}
}
