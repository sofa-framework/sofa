#ifndef ASSEMBLED_SYSTEM_H
#define ASSEMBLED_SYSTEM_H

#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <sofa/helper/system/config.h>

namespace sofa {
namespace component {
namespace linearsolver {
			
// Assembly of all the relevant data in the scene graph. Used by
// KKTSolver and AssemblyVisitor.

class AssembledSystem {
public:
	typedef SReal real;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
	typedef Eigen::SparseMatrix<real, Eigen::RowMajor> mat;
				
	// TODO protect m/n ?

	// independent dofs
	unsigned m;

	// compliant dofs
	unsigned n;

	// total size
	unsigned size() const;

	AssembledSystem(unsigned m = 0, unsigned n = 0);
				
	real dt;
				
	// mass, stiffness, compliance, mapping and projection
	// matrices
	mat H, // M, K,
		C, J, P;
	
	// force, velocity and deformation vectors
	vec p, f, v, phi; 					// should we have lambda ?
	
	// unilateral flags
	vec unilateral;
};


}
}
}

#endif
