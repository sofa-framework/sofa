#ifndef RIGIDJOINTMAPPING_H
#define RIGIDJOINTMAPPING_H

#include "AssembledMapping.h"
#include "initCompliant.h"

#include "utils/se3.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{
/** 
    Maps rigid bodies to the logarithm coordinates of a joint between
    the two: let the joint have coordinates p and c, the mapping computes:

    \[ f(p, c) = \log\left( p^{-1} c) \]
    
    this is mostly used to place a compliance on the end
 */

template <class TIn, class TOut >
class RigidJointMapping : public AssembledMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(RigidJointMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));
	
	typedef defaulttype::Vec<2, unsigned> index_pair;
	typedef vector< index_pair > pairs_type;
	
	Data< pairs_type > pairs;
	Data< bool > skip_rotation;
	Data< bool > out_joint_angle;
	
	RigidJointMapping() 
		: pairs(initData(&pairs, "pairs", "pairs of rigid frames defining joint in source dofs" )),
		  skip_rotation(initData(&skip_rotation, false, "skip_rotation", "leaves rotation part zero" )),
		  out_joint_angle(initData(&out_joint_angle, false, "out_joint_angle", "output joint angle to std::cerr(unsigned rad)"))
		{
			
		}

protected:
	
	typedef SE3< typename TIn::Real > se3;
	
	typedef RigidJointMapping self;
	virtual void assemble( const typename self::in_pos_type& in_pos ) {
		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;

		J.resize(6 * pairs.getValue().size(),
		         in_pos.size() * 6);

		J.setZero();
		
		pairs_type& p = *pairs.beginEdit();

		typedef typename se3::mat66 mat66;
		std::vector< mat66 > blocks(2);			

		for(unsigned i = 0, n = p.size(); i < n; ++i) {

			// FIXME this means the lowest index is always the parent, derp
			if( p[i][1] < p[i][0] ) std::swap( p[i][1], p[i][0] );
			
			typename se3::coord_type diff = se3::prod( se3::inv( in_pos[ p[i][0] ] ), 
			                                           in_pos[ p[i][1] ] );
			
			mat66 dlog = mat66::Zero();
			
			dlog.template topLeftCorner<3, 3>() = se3::rotation(diff).toRotationMatrix();
			dlog.template bottomRightCorner<3, 3>() = se3::dlog( se3::rotation(diff) );
			
			if( skip_rotation.getValue() ) dlog.template bottomRows<3>().setZero();
			
			blocks[0] = -dlog * se3::Ad( se3::inv(diff) ) * se3::body(in_pos[ p[i][0] ]);
			blocks[1] = dlog * se3::body(in_pos[ p[i][1]]);
			
			for( unsigned u = 0; u < 6; ++u) {
				unsigned row = 6 * i + u;
				J.startVec( row );
					
				for( unsigned j = 0; j < 2; ++j) {
						
					unsigned index = p[i][j];					
						
					for( unsigned v = 0; v < 6; ++v) {
						unsigned col = 6 * index + v;
						J.insertBack(row, col) = blocks[j](u, v);
					}
				} 
			}		 
		}
		
		J.finalize();
		
		pairs.endEdit();
	} 
	
	virtual void apply(typename self::out_pos_type& out,
	                   const typename self::in_pos_type& in ) {
		pairs_type& p = *pairs.beginEdit();
		
		assert( out.size() == p.size() );				        

		for(unsigned i = 0, n = p.size(); i < n; ++i) {
			
			// FIXME this means the lowest index is always the parent, derp
			if( p[i][1] < p[i][0] ) std::swap( p[i][1], p[i][0] );
			
			out[i] = se3::product_log( se3::prod( se3::inv( in[ p[i][0] ] ), 
			                                      in[ p[i][1] ] ) ).getVAll();
			
			if( out_joint_angle.getValue() ) output( out[i] );
			                                             
			if( skip_rotation.getValue() ) {
				out[i][3] = 0;
				out[i][4] = 0;
				out[i][5] = 0;
			}
		}
		 
		pairs.endEdit();
	}
public:
	void output(typename TOut::Coord out) const {
		out[0] = 0;
		out[1] = 0;
		out[2] = 0;
		std::cerr << this->getContext()->getTime() << ", " << out.norm() << std::endl;
	}
};
}
}
}

#endif
