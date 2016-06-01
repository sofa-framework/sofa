#ifndef PYTHONMULTIMAPPING_INL
#define PYTHONMULTIMAPPING_INL

#ifdef NDEBUG
#undef NDEBUG
#warning NDEBUG
#endif

#include "PythonMultiMapping.h"

namespace sofa {
namespace component {
namespace mapping {

template<class TIn, class TOut>  
PythonMultiMapping<TIn, TOut>::PythonMultiMapping() :
    matrix(initData(&matrix, "jacobian", "jacobian for the mapping (row-major)")),
    value(initData(&value, "value", "mapping value")),
    gs_matrix(initData(&gs_matrix, "geometric_stiffness", "mapping geometric stiffness matrix (row-major)")),
    use_gs(initData(&use_gs, true, "use_geometric_stiffness", "mapping geometric stiffness matrix (row-major)")),
    out_force(initData(&out_force, "out_force", "output force used to compute geometric stiffness (read-only)")) {        
        
}
    
template<class TIn, class TOut>
void PythonMultiMapping<TIn, TOut>::assemble_geometric( const helper::vector<typename self::in_pos_type>& in,
														const typename self::const_out_deriv_type& out) {
        
    if(use_gs.getValue()) {

        if( set(out_force).size() != out.ref().size() ) {
            serr << "assemble_geometric: force size error, ignoring" << sendl;
        } else {
            // copy output force into data for the python side to
            // assemble gs, being careful not to cause reallocation
            std::copy(out.ref().begin(), out.ref().end(), set(out_force).begin());
        }
        
        // set(out_force) = out.ref();

        // std::cout << "c++ out_force: " << set(out_force) << std::endl;
            
        // hand over to python
        if(this->py_callback) {
            // std::cerr << "callback gs" << std::endl;
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
    
template<class TIn, class TOut>
void PythonMultiMapping<TIn, TOut>::assemble( const helper::vector<typename self::in_pos_type>& in )  {
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
        size += rows * cols;
    }

    if(matrix.size() != size) {

        // note: empty 'jacobian' will be silently treated as zero
        if( matrix.size() ) {
            serr << "assemble: incorrect jacobian size, treating as zero (solve *will* fail !)"
                 << sendl;
        }
            
        return;
    }


    // each out dof
    unsigned off = 0;

    unsigned nnz = 0;
    
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

                        if( value ) {
                            block.insertBack(r, c) = value;
                            ++nnz;
                        }
                    }					
                }
                off += dim;
            }
                
        }
			
    }
    assert( off == matrix.size() );

    if(!nnz) {
        serr << "assemble: zero jacobian, solve *will* fail !" << sendl;
    }
    
    // each input mstate
    for(unsigned j = 0, m = in.size(); j < m; ++j) {
        block_type& block = this->jacobian(j).compressedMatrix;
			
        block.finalize();
    }
		
}

template<class TIn, class TOut>
void PythonMultiMapping<TIn, TOut>::apply(typename self::out_pos_type& out, 
                                          const helper::vector<typename self::in_pos_type>& /*in*/ ){
        
    if(this->py_callback) {
        // std::cerr << "callback apply" << std::endl;
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


 
}
}
}

#endif

