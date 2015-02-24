#ifndef PYTHONMULTIMAPPING_H
#define PYTHONMULTIMAPPING_H


#include "AssembledMultiMapping.h"
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace mapping {

/** 
	a very general mapping defined on the python side: f(x) = value, df(x) = jacobian

	jacobian is given row-major in a single vector, value as a veccoord.

	this is mostly useful to python scripts that need to compute
	arbitrary multimappings.
	
	this class can be used to set arbitrary velocity constraints, by
	setting 'hard_positions': in this case, y = b but the velocity is
	still mapped as dy = A dx. With zero compliance, this corresponds
	to a kinematic velocity constraint of the form:  A dx = -b / dt

	Hence b corresponds to (current_position - desired_position)

	@author Maxime Tournier
	
*/

template<class TIn, class TOut>
class SOFA_Compliant_API PythonMultiMapping : public AssembledMultiMapping<TIn, TOut> {
	typedef PythonMultiMapping self;
	

  public:
	SOFA_CLASS(SOFA_TEMPLATE2(PythonMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));
	
	typedef vector< typename TIn::Real > matrix_type;
	typedef typename TOut::VecCoord value_type;

	Data<matrix_type> matrix;
	Data<value_type> value;		
	Data<bool> hard_positions;
	
	PythonMultiMapping() :
		matrix(initData(&matrix, "jacobian", "jacobian for the mapping (row-major)")),
		value(initData(&value, "value", "mapping value")) {
	}


	virtual void assemble( const vector<typename self::in_pos_type>& in )  {
		// initialize jacobians

		typedef typename self::jacobian_type::CompressedMatrix jack_type;

		// each input mstate
		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			jack_type& jack = this->jacobian(j).compressedMatrix;

			unsigned dim = this->from(j)->getMatrixSize();
			
			jack.resize(value.getValue().size(), dim );
			jack.setZero();
		}

		// each out dof
		unsigned off = 0;
			
		// each input mstate
		for(unsigned i = 0, n = value.getValue().size(); i < n; ++i) {
			
			// each input mstate
			for(unsigned j = 0, m = in.size(); j < m; ++j) {
				jack_type& jack = this->jacobian(j).compressedMatrix;
				
				unsigned dim = this->from(j)->getMatrixSize();
				
				unsigned r = i;
				jack.startVec(r);

				// each input mstate dof
				for(unsigned k = 0, p = in[j].size(); k < p; ++k) {
					
					// each dof dimension
					for(unsigned u = 0; u < self::Nin; ++u) {
						unsigned c = k * self::Nin + u;
						SReal value = matrix.getValue()[off + c];
						if( value ) jack.insertBack(r, c) = value;
					}					
				}
				off += dim;
			}
			
		}
		assert( off == matrix.getValue().size() );

		// each input mstate
		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			jack_type& jack = this->jacobian(j).compressedMatrix;
			
			jack.finalize();
		}
		
		
	}


    virtual void apply(typename self::out_pos_type& out, 
					   const vector<typename self::in_pos_type>& in ) {
		
		// let's be paranoid
		assert( out.size() == value.getValue().size() );
		assert( matrix.getValue().size() % value.getValue().size() == 0 );
		
		// each out dof
		unsigned off = 0;
		for(unsigned i = 0, n = out.size(); i < n; ++i) {
			out[i] = value.getValue()[i];
		}
		
	}
	
};


}
}
}



#endif
