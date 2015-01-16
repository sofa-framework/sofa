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

    Data<int> geometricStiffness;

protected:
	typedef SE3< in_real > se3;
	typedef typename se3::coord_type coord_type;
	

	RigidJointMultiMapping()
		: pairs(initData(&pairs, "pairs", "index pairs (parent, child) for each joint")),
          geometricStiffness(initData(&geometricStiffness,
                                      0,
                                      "geometricStiffness",
                                      "assemble (and use) geometric stiffness (0=no GS, 1=non symmetric, 2=symmetrized)"))
        {}
	
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


    void assemble_geometric(const typename self::in_pos_type& in_pos,
                            const typename self::out_force_type& out_force) {



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
			
			const coord_type parent = in[0][ p[i](0) ];
			const coord_type child = in[1][ p[i](1) ];
			
			const coord_type delta = se3::prod( se3::inv(parent), child);
			
			// each parent mstate
			for(unsigned j = 0, m = in.size(); j < m; ++j) {

				typename self::jacobian_type::CompressedMatrix& J = this->jacobian(j).compressedMatrix;
				
				const mat33 Rp = se3::rotation(parent).toRotationMatrix();
				const mat33 Rc = se3::rotation(child).toRotationMatrix();
//				mat33 Rdelta = se3::rotation(delta).toRotationMatrix();
				const typename se3::vec3 s = se3::translation(child) - se3::translation(parent);

                // dlog in spatial coordinates
                const mat33 dlog = se3::dlog( se3::rotation(delta) ) * Rc.transpose() * Rp;
                // const mat33 dlog = mat33::Identity();
                 
				mat66 ddelta; 

				if( j ) {
					// child
					ddelta << 
						Rp.transpose(), mat33::Zero(),
						mat33::Zero(), dlog * Rp.transpose();
				} else {
					// parent
                    ddelta << 
                        -Rp.transpose(), Rp.transpose() * se3::hat(s),
                        mat33::Zero(), -dlog * Rp.transpose();
				}
				

				const mat66& block = ddelta;
				
				// each row
				for( unsigned u = 0; u < 6; ++u) {
					const unsigned row = 6 * i + u;
					J.startVec( row );
					
					for( unsigned v = 0; v < 6; ++v) {
						const unsigned col = 6 * p[i][j] + v; 
						if( block(u, v) ) J.insertBack(row, col) = block(u, v);
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
