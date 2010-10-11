#ifndef SOFA_SIMULATION_COMMON_VISITOREXECUTE_H
#define SOFA_SIMULATION_COMMON_VISITOREXECUTE_H


#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

namespace sofa
{
namespace simulation
{
namespace common
{

struct VisitorExecuteFunc
{
protected:
    core::objectmodel::BaseContext& ctx;
public:
    VisitorExecuteFunc(core::objectmodel::BaseContext& ctx):ctx(ctx) {};

    template< class Visitor >
    void operator()(Visitor* pv, bool prefetch = false )
    {
        prepareVisitor(pv);
        pv->execute(&ctx, prefetch );
    }
    template< class Visitor >
    void operator()(Visitor v, bool prefetch = false )
    {
        prepareVisitor(&v);
        v.execute(&ctx, prefetch );
    }
protected:
    void prepareVisitor( sofa::simulation::Visitor* v)
    {
        v->setTags( ctx.getTags() );
    }
    void prepareVisitor( sofa::simulation::BaseMechanicalVisitor* mv)
    {
        /*     if (v->writeNodeData())
        v->setNodeMap(this->getWriteNodeMap());
        else
        v->setNodeMap(this->getNodeMap());
        */
        prepareVisitor( (sofa::simulation::Visitor*)mv );
    }
};
}
}
}

#endif // SOFA_SIMULATION_COMMON_VISITOREXECUTE_H
