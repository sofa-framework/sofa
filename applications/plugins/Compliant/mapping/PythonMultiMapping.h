#ifndef PYTHONMULTIMAPPING_H
#define PYTHONMULTIMAPPING_H


#include "AssembledMultiMapping.h"
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace mapping {

// TODO move this elsewhere if reusable
struct with_py_callback {
    typedef void* (*py_callback_type)(int);
    py_callback_type py_callback;
    
    with_py_callback();
    virtual ~with_py_callback();
};


/** 
	a very general mapping defined on the python side: f(x) = value, df(x) = jacobian

	jacobian is given row-major in a single vector, value as a veccoord.

	this is mostly useful to python scripts that need to compute
	arbitrary multimappings.
	
	@author Maxime Tournier
	
*/

// TODO also fill a mask Data from python to be able to setup frommasks

template<class TIn, class TOut>
class SOFA_Compliant_API PythonMultiMapping : public AssembledMultiMapping<TIn, TOut>,
                                              public with_py_callback {
	typedef PythonMultiMapping self;
    
 public:
	SOFA_CLASS(SOFA_TEMPLATE2(PythonMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));
	
    typedef helper::vector< typename TIn::Real > matrix_type;
	typedef typename TOut::VecCoord value_type;

	Data<matrix_type> matrix;
	Data<value_type> value;

	Data<matrix_type> gs_matrix;
    Data<bool> use_gs;
	Data<typename TOut::VecDeriv> out_force;
    
	PythonMultiMapping() :
        matrix(initData(&matrix, "jacobian", "jacobian for the mapping (row-major)")),
        value(initData(&value, "value", "mapping value")),
        gs_matrix(initData(&gs_matrix, "geometric_stiffness", "mapping geometric stiffness matrix (row-major)")),
        use_gs(initData(&use_gs, true, "use_geometric_stiffness", "mapping geometric stiffness matrix (row-major)")),
        out_force(initData(&out_force, "out_force", "output force used to compute geometric stiffness (read-only)")) {        
        
    }
    
    
    enum {
        out_deriv_size = TOut::Deriv::total_size,
        in_deriv_size = TIn::Deriv::total_size,

        out_coord_size = TOut::Coord::total_size,
        in_coord_size = TIn::Coord::total_size
    };

    enum {
        // indicate state for python callback
        apply_state = 0,
        gs_state = 1
    };
    
 public:
    
    template<class T>
    static T& set(const Data<T>& data) {
        return const_cast<T&>(data.getValue());
    }
    
 protected:

	virtual void assemble_geometric( const helper::vector<typename self::in_pos_type>& in,
                                     const typename self::const_out_deriv_type& out) {
        
        if(use_gs.getValue()) {
            // copy force in data
            set(out_force) = out.ref();

            // std::cout << "c++ out_force: " << set(out_force) << std::endl;
            
            // hand over to python
            if(this->py_callback) {
                this->py_callback( gs_state );
            }

            // assemble sparse jacobian from data
            unsigned rows = 0;

            for(unsigned i = 0, n = in.size(); i < n; ++i) {
                rows += this->from(i)->getMatrixSize();
            }

            typedef typename self::geometric_type::CompressedMatrix gs_type;
            gs_type& dJ = this->geometric.compressedMatrix;
            
            dJ.resize(rows, rows);
            dJ.setZero();

            const unsigned size = rows * rows;

            const matrix_type& gs = gs_matrix.getValue();            
            if( gs.size() != size ) {
                
                if( gs.size() ) {
                    serr << "assemble: incorrect geometric stiffness size, treating as zero";
                }
                
                return;
            }

            for(unsigned i = 0; i < rows; ++i) {
                dJ.startVec(i);
                
                for(unsigned j = 0; j < rows; ++j) {
                    const SReal value = gs[ rows * i + j];
                    if(value ) dJ.insertBack(i, j) = value;
                }
                
            }

            dJ.finalize();

            // std::cout << "c++ dJT" << std::endl
            //           << dJ << std::endl;
        }
	}
	
	
    virtual void assemble( const helper::vector<typename self::in_pos_type>& in )  {
		// initialize jacobians

        const value_type& value = this->value.getValue();
        const matrix_type& matrix = this->matrix.getValue();
        
		typedef typename self::jacobian_type::CompressedMatrix block_type;

		// resize jacobian blocks
        unsigned size = 0;
        
        const unsigned rows = value.size() * out_deriv_size;

		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			block_type& block = this->jacobian(j).compressedMatrix;

            const unsigned cols = this->from(j)->getMatrixSize();
			block.resize(rows, cols);
			block.setZero();
            size += rows * cols;
		}

        if(matrix.size() != size) {

            // note: empty 'jacobian' will be silently treated as zero
            if( matrix.size() ) {
                serr << "assemble: incorrect jacobian size, treating as zero" << sendl;
            }
            
            return;
        }


		// each out dof
		unsigned off = 0;
			
		// each output mstate
		for(unsigned i = 0, n = value.size(); i < n; ++i) {

            for(unsigned v = 0; v < out_deriv_size; ++v) {

                // each input mstate
                for(unsigned j = 0, m = in.size(); j < m; ++j) {
                    block_type& block = this->jacobian(j).compressedMatrix;
				
                    const unsigned dim = this->from(j)->getMatrixSize();
				
                    const unsigned r = out_deriv_size * i + v;
                    block.startVec(r);

                    // each input mstate dof
                    for(unsigned k = 0, p = in[j].size(); k < p; ++k) {
					
                        // each dof dimension
                        for(unsigned u = 0; u < in_deriv_size; ++u) {
                            const unsigned c = k * in_deriv_size + u;
                            const SReal value = matrix[off + c];
                            if( value ) block.insertBack(r, c) = value;
                        }					
                    }
                    off += dim;
                }
                
            }
			
		}
		assert( off == matrix.size() );

		// each input mstate
		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			block_type& block = this->jacobian(j).compressedMatrix;
			
			block.finalize();
		}
		
	}
    
    virtual void apply(typename self::out_pos_type& out, 
                       const helper::vector<typename self::in_pos_type>& /*in*/ ) {
        
        if(this->py_callback) {
            this->py_callback( apply_state );
        }
        
        const value_type& value = this->value.getValue();
        
        if( out.size() != value.size() ) {
            serr << "apply: size for data 'value' does not match output, auto-resizing" << sendl;
            const_cast<value_type&>(value).resize( out.size() );
        }
		
		for(unsigned i = 0, n = out.size(); i < n; ++i) {
			out[i] = value[i];
		}
		
	}
	
};


}
}
}



#endif
