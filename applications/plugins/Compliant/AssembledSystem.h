#ifndef ASSEMBLED_SYSTEM_H
#define ASSEMBLED_SYSTEM_H

#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <sofa/helper/system/config.h>

namespace sofa {
namespace core {
namespace behavior {
class BaseMechanicalState;
}
}

namespace component {
namespace linearsolver {
			
// Assembly of all the relevant data in the scene graph. Used by
// KKTSolver and AssemblyVisitor.

// TODO it might be more efficient to store column-major only,
// especially store JT as a column-major matrix, as we sometimes have
// to compute Hinv * JT which requires converting back and forth.

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
        C, J,
        P;
	
	// force, velocity and deformation vectors
	vec p, f, v, phi, lambda;
	
	// master/compliant dofs, sorted consistently with the above
	typedef core::behavior::BaseMechanicalState dofs_type;
	std::vector< dofs_type* > master, compliant;
	
	void debug(SReal threshold = 0) const;

    // return true iff the magnitude of every diagonal entries are larger or equal than the sum of the magnitudes of all the non-diagonal entries in the same row
    bool isDiagonalDominant() const;
	
};


}
}
}

#endif
