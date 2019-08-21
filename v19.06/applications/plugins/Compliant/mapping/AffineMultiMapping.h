#ifndef AFFINEMULTIMAPPING_H
#define AFFINEMULTIMAPPING_H


#include "AssembledMultiMapping.h"
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace mapping {

/** 
    OBSELETE: use PythonMultiMapping instead

	a very general affine multi mapping: y = A x + b
	
	where x is the concatenation of input dofs, in the order specified
	in fromModel().

	A is given row-major in a single vector, b as a vector.

	this is mostly useful to python scripts that need to compute
	arbitrary multimappings.
	
	this class can be used to set arbitrary velocity constraints, by
	setting 'hard_positions': in this case, y = b but the velocity is
	still mapped as dy = A dx. With zero compliance, this corresponds
	to a kinematic velocity constraint of the form:  A dx = -b / dt

	Hence b corresponds to (current_position - desired_position)

	@author Maxime Tournier
	
*/

// TODO make it work for any vector output dofs (only 1d dofs for now)

template<class TIn, class TOut>
class SOFA_Compliant_API AffineMultiMapping : public AssembledMultiMapping<TIn, TOut> {
	typedef AffineMultiMapping self;
	

  public:
	SOFA_CLASS(SOFA_TEMPLATE2(AffineMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));
	
    typedef helper::vector< typename TIn::Real > matrix_type;
	
    typedef helper::vector< typename TOut::Real > value_type;

	Data<matrix_type> matrix; ///< matrix for the mapping (row-major)
	Data<value_type> value; ///< offset value
	Data<bool> hard_positions;
	
	AffineMultiMapping() :
		matrix(initData(&matrix, "matrix", "matrix for the mapping (row-major)")),
		value(initData(&value, "value", "offset value")),
		hard_positions(initData(&hard_positions, 
								false, 
								"hard_positions", 
								"skip matrix multiplication in apply call: the output value will be hard set to @value")) {
		
		// hard positions allows to build arbitrary constraints

		assert( self::Nout == 1 );
	}


    virtual void assemble( const helper::vector<typename self::in_pos_type>& in )  {
		// initialize jacobians

		typedef typename self::jacobian_type::CompressedMatrix jack_type;

		// each input mstate
        unsigned size = 0;
        const unsigned rows = value.getValue().size() * TOut::Deriv::total_size;

		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			jack_type& jack = this->jacobian(j).compressedMatrix;

			const unsigned cols = this->from(j)->getMatrixSize();
			
            jack.resize(rows, cols );
            size += rows * cols;
		}

        if(matrix.getValue().size() != size) {

            if( matrix.getValue().size() ) serr << "matrix size incorrect" << sendl;
                
            return;
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
                        
						const SReal value = matrix.getValue()[off + c];
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
                       const helper::vector<typename self::in_pos_type>& in ) {
		
		// let's be paranoid
		assert( out.size() == value.getValue().size() );
		assert( matrix.getValue().size() % value.getValue().size() == 0 );
		
		// each out dof
		unsigned off = 0;
		for(unsigned i = 0, n = out.size(); i < n; ++i) {
			out[i] = value.getValue()[i];

			if( !hard_positions.getValue() ) {
				
				// each input mstate
				for(unsigned j = 0, m = in.size(); j < m; ++j) {
					unsigned dim = this->from(j)->getMatrixSize();
					
					// each input mstate dof
					for(unsigned k = 0, p = in[j].size(); k < p; ++k) {

						const typename TIn::Real* data = &matrix.getValue()[off] + k * self::Nin;
						
						using namespace utils;
						out[i][0] += map(in[j][k]).dot(map<self::Nin>(data));
					}
					
					off += dim;
				}
				
			}
		}
		
	}
	
};


}
}
}



#endif
