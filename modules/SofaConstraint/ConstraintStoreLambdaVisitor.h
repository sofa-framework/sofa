#ifndef SOFA_CONSTRAINT_CONSTRAINTSTORELAMBDAVISITOR_H
#define SOFA_CONSTRAINT_CONSTRAINTSTORELAMBDAVISITOR_H
#include "config.h"

#include <sofa/simulation/MechanicalVisitor.h>


namespace sofa
{
namespace simulation
{

class SOFA_CONSTRAINT_API ConstraintStoreLambdaVisitor : public BaseMechanicalVisitor
{
public:
    ConstraintStoreLambdaVisitor(const sofa::core::ConstraintParams* cParams, const sofa::defaulttype::BaseVector* lambda);

    Visitor::Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet) override;

    void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;

    bool stopAtMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;

private:
    const sofa::core::ConstraintParams* m_cParams;
    const sofa::defaulttype::BaseVector* m_lambda;
};


}

}

#endif 
