#ifndef PYTHONMULTIMAPPING_INL
#define PYTHONMULTIMAPPING_INL

#include "PythonMultiMapping.h"

namespace sofa {
namespace component {
namespace mapping {



template<class TIn, class TOut>  
PythonMultiMapping<TIn, TOut>::PythonMultiMapping() :
    apply_callback(initData(&apply_callback, "apply_callback", "apply callback")),
    jacobian_callback(initData(&jacobian_callback, "jacobian_callback", "jacobian callback")),
    gs_callback(initData(&gs_callback, "gs_callback", "geometric stiffness callback"))
{        
    
}
    
template<class TIn, class TOut>
void PythonMultiMapping<TIn, TOut>::assemble_geometric(const helper::vector<typename self::in_pos_type>& in,
                                                       const typename self::const_out_deriv_type& out) {

    if(!gs_callback.getValue().data) return;

    typedef typename self::geometric_type::CompressedMatrix gs_type;

    gs_type& dJ = this->geometric.compressedMatrix;
    out_vec f = out_vec::map(out.ref());
    
    in_vec at[in.size()];
    int size = 0;
    for(unsigned i = 0, n = in.size(); i < n; ++i) {
        at[i] = in_vec::map(in[i].ref());
        size += this->from(i)->getMatrixSize();
    }

    dJ.resize( size, size );
    gs_callback.getValue().data(&dJ, at, in.size(), f);
    
}
    
template<class TIn, class TOut>
void PythonMultiMapping<TIn, TOut>::assemble( const helper::vector<typename self::in_pos_type>& in )  {

    if(!jacobian_callback.getValue().data) {
        serr << "no jacobian callback, solve *will* fail !" << sendl;
        return;
    }


    in_vec at[in.size()];
    out_csr_matrix* js[in.size()];
    

    for(unsigned i = 0, n = in.size(); i < n; ++i) {
        at[i] = in_vec::map(in[i].ref());

        const int rows = this->to()->getMatrixSize(), cols = this->from(i)->getMatrixSize();
        this->jacobian(i).compressedMatrix.resize(rows, cols);
        
        js[i] = &this->jacobian(i).compressedMatrix;
    }
    
    jacobian_callback.getValue().data(js, at, in.size());

}


template<class TIn, class TOut>
void PythonMultiMapping<TIn, TOut>::apply(typename self::out_pos_type& out, 
                                          const helper::vector<typename self::in_pos_type>& in ){

    if(!apply_callback.getValue().data) {
        serr << "no apply callback" << sendl;
        return;
    }

    
    in_vec at[in.size()];
    for(unsigned i = 0, n = in.size(); i < n; ++i) {
        at[i] = in_vec::map(in[i].ref());
    }
    
    out_vec to = out_vec::map(out.ref());
    apply_callback.getValue().data(to, at, in.size());
    
}


}
}
}

#endif

