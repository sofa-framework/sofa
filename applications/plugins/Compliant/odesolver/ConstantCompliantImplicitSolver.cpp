#include "ConstantCompliantImplicitSolver.h"

#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

#include "../assembly/AssemblyVisitor.h"
#include "../constraint/ConstraintValue.h"



namespace sofa {
namespace component {
namespace odesolver {

int ConstantCompliantImplicitSolverClass = core::RegisterObject("Pre-assembled AssembedSolver").add< ConstantCompliantImplicitSolver >();

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;



void ConstantCompliantImplicitSolver::reinit()
{
    CompliantImplicitSolver::reinit();
    if( assemblyVisitor )
    {
        delete assemblyVisitor;
        assemblyVisitor = NULL;
    }
}


void ConstantCompliantImplicitSolver::perform_assembly( const core::MechanicalParams *mparams, system_type& sys )
{
    if( assemblyVisitor ) return;

    assemblyVisitor = new simulation::AssemblyVisitor(mparams);

    // fetch nodes/data
    send( *assemblyVisitor );

    // assemble system
    assemblyVisitor->assemble(sys);
}



}
}
}
