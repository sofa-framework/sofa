#ifndef PYTHONMULTIMAPPING_H
#define PYTHONMULTIMAPPING_H


#include "AssembledMultiMapping.h"
#include "../utils/map.h"

#include "../python/python.h"

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

    template<class Real>
    struct vec {
        std::size_t outer;
        std::size_t inner;

        Real* data;
        
        template<class T>
        static vec map(const std::vector<T>& value) {
            return {value.size(),
                    T::total_size,
                    // yeah i know
                    const_cast<Real*>(&value[0][0]) };
        }
    };

    typedef python::vec<typename TIn::Real> in_vec;
    typedef python::vec<typename TOut::Real> out_vec;    
    
    typedef Eigen::SparseMatrix<typename TIn::Real, Eigen::RowMajor> in_csr_matrix;
    typedef Eigen::SparseMatrix<typename TOut::Real, Eigen::RowMajor> out_csr_matrix;    

    typedef void (*apply_callback_type)(out_vec* out, in_vec* in, std::size_t n);
    typedef void (*jacobian_callback_type)(out_csr_matrix** out, in_vec* in, std::size_t n);
    typedef void (*gs_callback_type)(in_csr_matrix* out, in_vec* in, std::size_t n, out_vec* f);
    typedef void (*draw_callback_type)();
    
    Data< python::opaque< apply_callback_type > > apply_callback;
    Data< python::opaque< jacobian_callback_type > >jacobian_callback;
    Data< python::opaque< gs_callback_type > > gs_callback;
    Data< python::opaque< draw_callback_type > > draw_callback;    
    
	PythonMultiMapping();

public:
    virtual void draw(const core::visual::VisualParams*);
    
 protected:

	virtual void assemble_geometric( const helper::vector<typename self::in_pos_type>& in,
                                     const typename self::const_out_deriv_type& out);
	
	
    virtual void assemble( const helper::vector<typename self::in_pos_type>& in );
    
    virtual void apply(typename self::out_pos_type& out, 
                       const helper::vector<typename self::in_pos_type>& /*in*/ );


  private:
    std::vector<in_vec> at;
    std::vector<out_csr_matrix*> js;
};


}
}
}



#endif
