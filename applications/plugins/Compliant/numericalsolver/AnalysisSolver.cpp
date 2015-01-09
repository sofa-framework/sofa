#include "AnalysisSolver.h"

#include <sofa/core/ObjectFactory.h>

#include <Eigen/SVD>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(AnalysisSolver)
int AnalysisSolverClass = core::RegisterObject("Analysis solver: runs other KKTSolvers successively on a given problem and performs extra analysis on KKT system").add< AnalysisSolver >();

AnalysisSolver::AnalysisSolver()

    : condest(initData(&condest, false, "condest", "compute condition number with svd")) {

}

void AnalysisSolver::init() {

	solvers.clear();
	this->getContext()->get<KKTSolver>( &solvers, core::objectmodel::BaseContext::Local );
	
	if( solvers.size() < 2 ) {
		std::cerr << "warning: no other kkt solvers found" << std::endl;
	} else {
		std::cout << "AnalysisSolver: dynamics/correction will use " 
				  << solvers.back()->getName() 
				  << std::endl;
	}
}

void AnalysisSolver::factor(const system_type& system) {

	// start at 1 since 0 is current solver
    for( unsigned i = 1, n = solvers.size() ; i < n; ++i ) {
		solvers[i]->factor(system);
	}

    if( condest.getValue() ) {

        typedef system_type::dmat dmat;
        dmat kkt;

        kkt.setZero(system.size(), system.size());

        kkt <<
            dmat(system.H), dmat(-system.J.transpose()),
            dmat(-system.J), dmat(-system.C);

        const system_type::vec eigen = kkt.jacobiSvd().singularValues();

        const real min = eigen.tail<1>()(0);
        const real max = eigen(0);

        if( min < std::numeric_limits<real>::epsilon() ) std::cout << "AnalysisSolver: singular system"<<std::endl;
        else
        {
            const real cond = max/min;
            std::cout << "condition number: " << cond << "("<<max<<"/"<<min<<")"<<std::endl;
            std::cout << "required precision:  "<<log(cond)<<" bits"<<std::endl;
        }
    }

    // TODO add more as needed
}


// solution is that of the first solver
void AnalysisSolver::correct(vec& res,
                             const system_type& sys,
                             const vec& rhs,
                             real damping) const {
	assert( solvers.size() > 1 );
	solvers.back()->correct(res, sys, rhs, damping);
}

// solution is that of the last solver
void AnalysisSolver::solve(vec& res,
							const system_type& sys,
							const vec& rhs) const {

    vec backup = res; // backup initial solution

	// start at 1 since 0 is current solver
    for( unsigned i = 1, n = solvers.size(); i < n; ++i ) {
		res = backup;
		solvers[i]->solve(res, sys, rhs);
	}
	
}






}
}
}
