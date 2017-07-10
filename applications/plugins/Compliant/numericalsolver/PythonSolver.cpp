#include "PythonSolver.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {


SOFA_DECL_CLASS(PythonSolver)

static int PythonSolverClass = core::RegisterObject("Python solver").add< PythonSolver >();

PythonSolver::PythonSolver()
    : factor_callback(initData(&factor_callback, "factor_callback", "durrr")),
      solve_callback(initData(&solve_callback, "solve_callback", "durrr")),
      correct_callback(initData(&correct_callback, "correct_callback", "durrr"))            
{
    
}

void PythonSolver::fetch_blocks(std::vector<block>& res, const system_type& sys) {

    res.clear();
    
    for(unsigned i = 0, off = 0, n = sys.compliant.size(); i < n; ++i) {
        
        system_type::dofs_type* const dofs = sys.compliant[i];
        const system_type::constraint_type& constraint = sys.constraints[i];
        
        const unsigned dim = dofs->getDerivDimension();
        
        for(unsigned k = 0, kmax = dofs->getSize(); k < kmax; ++k) {
            block b;

            b.offset = off;
            b.size = dim;
            b.projector = constraint.projector.get();

            assert( !b.projector || !b.projector->mask || 
                    b.projector->mask->empty() || b.projector->mask->size() == kmax );
            
            res.push_back( b );
            off += dim;
        }
    }

}


void PythonSolver::project(real* vec, const block* b) {
    if(b->projector) {
        bool correct = false;
        b->projector->project(vec + b->offset, b->size, 0, correct);
    }
}

void PythonSolver::factor(const system_type& sys) {

    // TODO pass projected sub-kkt instead ?
    if(!factor_callback.getValue()) {
        serr << "factor callback not set, aborting" << sendl;
        return;
    }

    fetch_blocks(blocks, sys);

    python::vec<block> pyblocks = python::vec<block>::map(blocks);
    data_type data = {&sys, &pyblocks, project};

    factor_callback.getValue().data(&data);
}


void PythonSolver::solve(vec& x,
                         const system_type& sys,
                         const vec& rhs) const {
    
    if(!solve_callback.getValue()) {
        serr << "solve callback not set, aborting" << sendl;
        return;
    }

    python::vec<block> pyblocks = python::vec<block>::map(blocks);
    data_type data = {&sys, &pyblocks, project};
    solve_callback.getValue().data(&x, &data, &rhs);
}


void PythonSolver::correct(vec& x,
                           const system_type& sys,
                           const vec& rhs,
                           real damping) const {
    
    if(!correct_callback.getValue()) {
        serr << "correct callback not set, aborting" << sendl;
        return;
    }

    python::vec<block> pyblocks = python::vec<block>::map(blocks);
    data_type data = {&sys, &pyblocks, project};
    correct_callback.getValue().data(&x, &data, &rhs, damping);
}




}
}
}
