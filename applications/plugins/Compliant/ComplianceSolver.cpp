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

SOFA_DECL_CLASS(ComplianceSolver);
int ComplianceSolverClass = core::RegisterObject("A simple explicit time integrator").add< ComplianceSolver >();



ComplianceSolver::ComplianceSolver()
{
}

void ComplianceSolver::solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    //    cerr<<"ComplianceSolver::solve" << endl;

    MatrixAssemblyVisitor assembly(params,this);
    this->getContext()->executeVisitor(&assembly);
    cerr<<"ComplianceSolver::solve, sizeM = " << assembly.sizeM <<", sizeC = "<< assembly.sizeC << endl;

    matM.resize(assembly.sizeM,assembly.sizeM);
    matC.resize(assembly.sizeC,assembly.sizeC);
    matJ.resize(assembly.sizeC,assembly.sizeM);

    assembly.pass++; // second pass
    this->getContext()->executeVisitor(&assembly);



}

simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown(simulation::Node* node)
{
    if( pass==1 )
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //        cerr<<"node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            m_offset[node->mechanicalState] = sizeM;
            sizeM += node->mechanicalState->getMatrixSize();
        }


        // ==== process compliances
        vector<BaseCompliance*> compliances;
        node->getNodeObjects<BaseCompliance>(&compliances);
        //    cerr<<"node " << node->getName() << ", compliances: " << endl;
        //    for(unsigned i=0; i<compliances.size(); i++ ){
        //        cerr<< compliances[i]->getName() <<", ";
        //    }
        //    cerr<<endl;
        if( compliances.size()>0 )
        {
            assert(node->mechanicalState);
            c_offset[compliances[0]] = sizeC;
            sizeC += node->mechanicalState->getMatrixSize();
        }
        assert(compliances.size()<2);

    }
    else if (pass==2)
    {

    }
    else
    {
        cerr<<"ComplianceSolver::ComputeMatrixSizesVisitor::processNodeTopDown, unknown pass " << pass << endl;
    }

    return RESULT_CONTINUE;

}



}
}
}
