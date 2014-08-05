#ifndef ASSEMBLED_SYSTEM_H
#define ASSEMBLED_SYSTEM_H

#include "initCompliant.h"
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <sofa/helper/system/config.h>
#include <sofa/simulation/common/VectorOperations.h>

#include "../constraint/BaseConstraintValue.h"
#include "../constraint/Constraint.h"

namespace sofa {
namespace core {
namespace behavior {
class BaseMechanicalState;
}
}

namespace component {
namespace linearsolver {
			
// TODO this should go in numericalsolver/linearsolver

// Assembly of all the relevant data in the scene graph. Used by
// KKTSolver and AssemblyVisitor.

// TODO it might be more efficient to store column-major only,
// especially store JT as a column-major matrix, as we sometimes have
// to compute Hinv * JT which requires converting back and forth.

class SOFA_Compliant_API AssembledSystem {
public:
	typedef SReal real;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
	
	typedef Eigen::SparseMatrix<real, Eigen::RowMajor> rmat;
	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dmat; ///< dense matrix

	// makes it easier to filter constraint rows
	typedef rmat mat;
	
	// TODO protect m/n ?

	// independent dofs
	unsigned m;

	// compliant dofs
	unsigned n;

	// total size
	unsigned size() const;

	AssembledSystem(unsigned m = 0, unsigned n = 0);
				
	real dt;
				
    // ode matrix, compliance, mapping and projection
	// matrices
    mat H,
        C, J,
        P;

    /// false iff there are projective constraints (useful to eventually replace H with P^T.H.P in the numerical solver)
    /// cheaper to evaluate than effectively testing if P==I
    bool isPIdentity;

	// master/compliant dofs, sorted consistently with the above
	typedef core::behavior::BaseMechanicalState dofs_type;
	std::vector< dofs_type* > master, compliant;

    // constraint components (projector and value)
    struct constraint_type {
        constraint_type() : projector(0), value(0) {}
        linearsolver::Constraint::SPtr projector;
        odesolver::BaseConstraintValue::SPtr value;
    };
    std::vector< constraint_type > constraints; // of the size of compliant dof
	
	void debug(SReal threshold = 0) const;

    // // return true iff the magnitude of every diagonal entries are larger or equal than the sum of the magnitudes of all the non-diagonal entries in the same row
    // bool isDiagonalDominant() const;


    /// Copy a state vector from the scene graph to this system. Only the independent DOFs are copied.
    void copyFromMultiVec(vec& target, core::ConstMultiVecId sourceId );
    /// Copy a state vector from this system to the scene graph. Only the independent DOFs are copied.
    void copyToMultiVec( core::MultiVecId targetId, const vec& source );
    /// Add a state vector from this system to the scene graph. Only the independent DOFs are added.
    void addToMultiVec( core::MultiVecId targetId, const vec& source );


    /// Copy a state vector from the scene graph to this system. Only the compliant DOFs are copied.
    /// Note the given target can be either of the size n and will then used entierly OR
    /// it could be of size m+n and it will be used starting at index m
    void copyFromCompliantMultiVec(vec& target, core::ConstMultiVecId sourceId );
    /// Copy a state vector from this system to the scene graph. Only the compliant DOFs are copied.
    /// Note the given target can be either of the size n and will then used entierly OR
    /// it could be of size m+n and it will be used starting at index m
    void copyToCompliantMultiVec( core::MultiVecId targetId, const vec& source );
    /// Add a state vector from this system to the scene graph. Only the compliant DOFs are added.
    /// Note the given target can be either of the size n and will then used entierly OR
    /// it could be of size m+n and it will be used starting at index m
    void addToCompliantMultiVec( core::MultiVecId targetId, const vec& source );

};


}
}
}

#endif
