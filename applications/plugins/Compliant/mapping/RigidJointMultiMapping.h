#ifndef COMPLIANT_MAPPING_RIGIDJOINTMULTIMAPPING_H
#define COMPLIANT_MAPPING_RIGIDJOINTMULTIMAPPING_H

#include "AssembledMultiMapping.h"
#include "initCompliant.h"

#include "utils/se3.h"
#include "utils/map.h"
#include "utils/edit.h"

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
	SOFA_CLASS(SOFA_TEMPLATE2(RigidJointMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));
	
	typedef defaulttype::Vec<2, unsigned> index_pair;

	typedef vector< index_pair > pairs_type;
	Data< pairs_type > pairs;

	typedef typename TIn::Real in_real;
	typedef typename TOut::Real out_real;
	
	typedef AssembledMultiMapping<TIn, TOut> base;
	typedef RigidJointMultiMapping self;

protected:
	typedef SE3< in_real > se3;
	typedef typename se3::coord_type coord_type;
	

	RigidJointMultiMapping()
		: pairs(initData(&pairs, "pairs", "index pairs for each joint"))
		{
			
	}
	
	void apply(typename self::out_pos_type& out,
	           const vector< typename self::in_pos_type >& in ) {
		assert( this->getFrom().size() == 2 );

		const pairs_type& p = pairs.getValue();
		
		assert(out.size() == p.size());

		for( unsigned i = 0, n = p.size(); i < n; ++i) {

			coord_type parent = in[0][ p[i](0) ];
			coord_type child = in[1][ p[i](1) ];
		
			coord_type delta = se3::prod( se3::inv(parent), child);

			typename se3::deriv_type value = se3::product_log( delta );
            utils::map(out[i]).template head<3>() =  utils::map(value.getLinear()).template cast<out_real>();
            utils::map(out[i]).template tail<3>() =  utils::map(value.getAngular()).template cast<out_real>();
			
		}
		
	}
	

	void assemble(const vector< typename self::in_pos_type >& in ) {
		assert(this->getFrom()[0] != this->getFrom()[1]);

		const pairs_type& p = pairs.getValue();
		
		typedef typename se3::mat66 mat66;
		typedef typename se3::mat33 mat33;
		
		// resize/clean jacobians
		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			typename self::jacobian_type::CompressedMatrix& J = this->jacobian(j).compressedMatrix;
			J.resize( 6 * p.size(), 
			          6 * in[j].size() );
			J.setZero();
		}
		
		// each pair
		for(unsigned i = 0, n = p.size(); i < n; ++i) {
			
			coord_type parent = in[0][ p[i](0) ];
			coord_type child = in[1][ p[i](1) ];
			
			coord_type delta = se3::prod( se3::inv(parent), child);
			
			// each parent mstate
			for(unsigned j = 0, m = in.size(); j < m; ++j) {

				typename self::jacobian_type::CompressedMatrix& J = this->jacobian(j).compressedMatrix;
				
				mat33 Rp = se3::rotation(parent).toRotationMatrix();
				mat33 Rc = se3::rotation(child).toRotationMatrix();
//				mat33 Rdelta = se3::rotation(delta).toRotationMatrix();
				mat33 dlog = se3::dlog( se3::rotation(delta) );
				
				mat66 ddelta; 

				if( j ) {
					// child
					ddelta << 
						Rp.transpose(), mat33::Zero(),
						mat33::Zero(), dlog * Rc.transpose();
				} else {
					// parent
						ddelta << 
						-Rp.transpose(), mat33::Zero(),
							mat33::Zero(), -dlog * Rc.transpose();
				}
				

				mat66 block = ddelta;
				
				// each row
				for( unsigned u = 0; u < 6; ++u) {
					unsigned row = 6 * i + u;
					J.startVec( row );
					
					for( unsigned v = 0; v < 6; ++v) {
						unsigned col = 6 * p[i][j] + v; 
						J.insertBack(row, col) = block(u, v);
					} 
				}		 
				
			}
		}

		// finalize at the end of operations otherwise indices lookup table gets computed.
		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			typename self::jacobian_type::CompressedMatrix& J = this->jacobian(j).compressedMatrix;
			J.finalize();
		}
	}
	


};


}
}
}

#endif
