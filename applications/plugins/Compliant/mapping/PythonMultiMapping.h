#ifndef PYTHONMULTIMAPPING_H
#define PYTHONMULTIMAPPING_H


#include "AssembledMultiMapping.h"
#include "../utils/map.h"
#include "../misc/python.h"

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
    
	PythonMultiMapping();
	
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
                                     const typename self::const_out_deriv_type& out);
	
	
    virtual void assemble( const helper::vector<typename self::in_pos_type>& in );
    
    virtual void apply(typename self::out_pos_type& out, 
                       const helper::vector<typename self::in_pos_type>& /*in*/ );
	
};


}
}
}



#endif
