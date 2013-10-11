#ifndef COMPLIANT_MAPPING_RIGIDJOINTMULTIMAPPING_H
#define COMPLIANT_MAPPING_RIGIDJOINTMULTIMAPPING_H

#include "AssembledMultiMapping.h"
#include "initCompliant.h"

#include "utils/se3.h"
#include "utils/map.h"
#include "utils/edit.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {

namespace component {

namespace mapping {

/** 

    Maps rigid bodies to the (log) relative coordinates between the
    two: 
    
     \[ f(p, c) = \log(p^{-1} c) \]

     Optionally, mapped dofs can be restricted using @dofs mask.

     TODO .inl

     @author: maxime.tournier@inria.fr

*/

template <class TIn, class TOut >
class RigidJointMultiMapping : public AssembledMultiMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(RigidJointMultiMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));

	typedef defaulttype::Vec<2, unsigned> index_pair;

	typedef vector< index_pair > pairs_type;
	Data< pairs_type > pairs;
	
	typedef defaulttype::Vec<6, SReal> vec6;
	typedef vector<vec6> dofs_type;
	Data< dofs_type > dofs;

	typedef AssembledMultiMapping<TIn, TOut> base;
	typedef RigidJointMultiMapping self;

protected:
	typedef SE3< typename TIn::Real > se3;
	typedef typename se3::coord_type coord_type;
	

	RigidJointMultiMapping()
		: pairs(initData(&pairs, "pairs", "index pairs for each joint")),
		  dofs(initData(&dofs, "dofs", "dof mask for each joint (default: all, last value extended if needed)") ) {
		edit(dofs)->push_back( vec6(1, 1, 1, 1, 1, 1) );
	}
	
	void init() { }


	void apply(typename self::out_pos_type& out,
	           const vector< typename self::in_pos_type >& in ) {
		assert( this->getFrom().size() == 2 );

		const pairs_type& p = pairs.getValue();
		const dofs_type& d = dofs.getValue();
		
		assert(out.size() == p.size());

		for( unsigned i = 0, n = p.size(); i < n; ++i) {
			coord_type delta = se3::coord( se3::inv(in[0][ p[i].first ] ),
			                               in[1][p[i].second] );
			
			unsigned index = std::min(i, d.size() - 1);
			coord_type value = se3::product_log( delta );
			map(out[i]) = map(d[index]).cwiseProduct( map(value) );
		}
		
	}
	

	void assemble(const vector< typename self::in_pos_type >& in ) {
		assert(this->getFrom()[0] != this->getFrom()[1]);


	}
	


};


}
}
}

#endif
