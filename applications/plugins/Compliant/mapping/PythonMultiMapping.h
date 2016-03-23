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
	
	@author Maxime Tournier
	
*/

// TODO also fill a mask Data from python to be able to setup frommasks

template<class TIn, class TOut>
class SOFA_Compliant_API PythonMultiMapping : public AssembledMultiMapping<TIn, TOut> {
	typedef PythonMultiMapping self;
	

  public:
	SOFA_CLASS(SOFA_TEMPLATE2(PythonMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));
	
    typedef helper::vector< typename TIn::Real > matrix_type;
	typedef typename TOut::VecCoord value_type;

	Data<matrix_type> matrix;
	Data<value_type> value;		
	
	PythonMultiMapping() :
		matrix(initData(&matrix, "jacobian", "jacobian for the mapping (row-major)")),
		value(initData(&value, "value", "mapping value")) {
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
			jack.setZero();
            size += rows * cols;
		}

        if(matrix.getValue().size() != size) {

            if( matrix.getValue().size() ) serr << "matrix size incorrect" << sendl;
            else {
                // std::cout << "derp" << std::endl;
                return;
            }
        }


		// each out dof
		unsigned off = 0;
			
		// each output mstate
		for(unsigned i = 0, n = value.getValue().size(); i < n; ++i) {

            for(unsigned v = 0; v < self::Nout; ++v) {

                // each input mstate
                for(unsigned j = 0, m = in.size(); j < m; ++j) {
                    jack_type& jack = this->jacobian(j).compressedMatrix;
				
                    const unsigned dim = this->from(j)->getMatrixSize();
				
                    const unsigned r = self::Nout * i + v;
                    jack.startVec(r);

                    // each input mstate dof
                    for(unsigned k = 0, p = in[j].size(); k < p; ++k) {
					
                        // each dof dimension
                        for(unsigned u = 0; u < self::Nin; ++u) {
                            const unsigned c = k * self::Nin + u;
                            const SReal value = matrix.getValue()[off + c];
                            if( value ) jack.insertBack(r, c) = value;
                        }					
                    }
                    off += dim;
                }
                
            }
			
		}
		assert( off == matrix.getValue().size() );

		// each input mstate
		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			jack_type& jack = this->jacobian(j).compressedMatrix;
			
			jack.finalize();
            // std::cout << jack << std::endl;
		}
		
		
	}


    virtual void apply(typename self::out_pos_type& out, 
                       const helper::vector<typename self::in_pos_type>& /*in*/ ) {
		
		// let's be paranoid
		assert( out.size() == value.getValue().size() );
		assert( matrix.getValue().size() % value.getValue().size() == 0 );
		
		// each out dof
//		unsigned off = 0;
		for(unsigned i = 0, n = out.size(); i < n; ++i) {
			out[i] = value.getValue()[i];
		}
		
	}
	
};


}
}
}



#endif
