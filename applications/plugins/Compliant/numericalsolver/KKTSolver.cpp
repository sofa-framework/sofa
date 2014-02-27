#include "KKTSolver.h"

#include "../utils/bench.h"

namespace sofa {
namespace component {
namespace linearsolver {
			


    KKTSolver::KKTSolver()
       : debug(initData(&debug,false,"debug","print debug info"))
       , _preconditioner( NULL )
		 // , benchmarkPath(initData(&benchmarkPath, "benchmarkPath", "path to save convergence benchmark output"))
    {}


    void KKTSolver::init()
    {
        // look for an optional preconditioner
        _preconditioner = this->getContext()->get<preconditioner_type>(core::objectmodel::BaseContext::Local);

        // if( !benchmarkPath.getValue().empty() )
        //     _benchmark = new Benchmark( benchmarkPath.getValue() );
        // else
        //     _benchmark = new BaseBenchmark(); // does nothing and should be optimized by the compiler not to add any overcost
		
		// max: there *IS* overhead due to indirection
    }
	

}
}
}
