#ifndef ASSEMBLED_SYSTEM_H
#define ASSEMBLED_SYSTEM_H

#include "SolverFlags.h"

#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <sofa/helper/system/config.h>

namespace sofa {
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
		C, J, P;
	
	// force, velocity and deformation vectors
	vec p, f, v, phi; 					// should we have lambda ?
	 

	void debug(SReal threshold = 0) const;

	// only for compliant dofs for now
	typedef Eigen::Matrix< SolverFlags::value_type, Eigen::Dynamic, 1> flags_type;
	flags_type flags;
	
	struct block {
		block(unsigned off, unsigned size);
		
		unsigned offset, size;

		typedef core::objectmodel::Base* data_type;
		data_type data;
		
	};

	typedef std::vector<block> blocks_type;
	blocks_type blocks;
};


}
}
}

#endif
