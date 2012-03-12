#include "ComplianceSolver.h"
#include "Compliance.h"
#include <sofa/core/ObjectFactory.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;

int ComplianceSolverClass = core::RegisterObject("A simple explicit time integrator")
        .add< ComplianceSolver >()
        ;

SOFA_DECL_CLASS(ComplianceSolver);


ComplianceSolver::ComplianceSolver()
{
}

void ComplianceSolver::solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
//    cerr<<"ComplianceSolver::solve" << endl;
    ComputeMatrixSizesVisitor computeSizes(params);
    this->getContext()->executeVisitor(&computeSizes);
    cerr<<"ComplianceSolver::solve, sizeM = " << computeSizes.sizeM <<", sizeJ = "<< computeSizes.sizeJ << endl;
}

void ComplianceSolver::ComputeMatrixSizesVisitor::processNodeBottomUp(simulation::Node* node)
{
    // ==== process compliances
    vector<BaseCompliance::SPtr> compliances;
    node->getNodeObjects<BaseCompliance>(&compliances);
//    cerr<<"node " << node->getName() << ", compliances: " << endl;
//    for(unsigned i=0; i<compliances.size(); i++ ){
//        cerr<< compliances[i]->getName() <<", ";
//    }
//    cerr<<endl;
    if( compliances.size()>0 )
    {
        assert(node->mechanicalState);
        sizeJ += node->mechanicalState->getMatrixSize();
    }
    assert(compliances.size()<2);

    // ==== process independent DOFs
    if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
    {
//        cerr<<"node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
        sizeM += node->mechanicalState->getMatrixSize();
    }

}


}
}
}
