#ifndef RIGIDJOINTMAPPING_H
#define RIGIDJOINTMAPPING_H

#include "AssembledMapping.h"
#include <Compliant/config.h>

#include "../utils/se3.h"
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
    
    this is mostly used to place a compliance on the end. Use
    @rotation and @translation data to restrict mapping to rotation
    and translation parts, in case you only want to apply a compliance
    on these.

    @author maxime.tournier@inria.fr

    TODO @rotation / @translation should be set per output dofs
    TODO specialize for vec3 output dofs in case (rotation ^ translation) is set ?
    
 */

template <class TIn, class TOut >
class RigidJointMapping : public AssembledMapping<TIn, TOut> {
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidJointMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
	typedef defaulttype::Vec<2, unsigned> index_pair;
    typedef helper::vector< index_pair > pairs_type;
    typedef typename TIn::Real Real;
	
	Data< pairs_type > pairs; ///< pairs of rigid frames defining joint in source dofs
	Data< bool > rotation; ///< compute relative rotation
	Data< bool > translation; ///< compute relative translation
//	Data< bool > out_joint_angle;
	
	Data< bool > exact_dlog;

	RigidJointMapping() 
		: pairs(initData(&pairs, "pairs", "pairs of rigid frames defining joint in source dofs" )),
		  rotation(initData(&rotation, true, "rotation", "compute relative rotation" )),
		  translation(initData(&translation, true, "translation", "compute relative translation" )),
//		  out_joint_angle(initData(&out_joint_angle, false, "out_joint_angle", "output joint angle to std::cerr(unsigned rad)")),
		  exact_dlog(initData(&exact_dlog, false, "exact_dlog",
							  "compute exact rotation dlog. more precise if you know what you're doing, but gets unstable past half turn. for 1d and isotropic 3d springs, you don't need this"))
		{
			
		}

protected:
	
	typedef SE3< typename TIn::Real > se3;
	typedef typename se3::coord_type coord_type;


	static coord_type delta(const coord_type& p, const coord_type& c) {
// 		coord_type res;
        //res.getOrientation() = p.getOrientation().inverse() * c.getOrientation();
        //res.getCenter() = c.getCenter() - p.getCenter();
        //return res;

		return se3::prod( se3::inv(p), c);
	}
	

	typedef RigidJointMapping self;
	virtual void assemble( const typename self::in_pos_type& in_pos ) {
		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;


        const pairs_type& p = pairs.getValue();
        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();
        bool exact_dlog = this->exact_dlog.getValue();

        J.resize(6 * p.size(), in_pos.size() * 6);
        if( !translation ) J.reserve( 18*p.size() );
        else if( !rotation ) J.reserve( 27*p.size() );
        else J.reserve( 45*p.size() );


		typedef typename se3::mat66 mat66;
		typedef typename se3::mat33 mat33;
		
#ifdef _MSC_VER
		mat66* blocks;
		void* blockPtr = ::_aligned_malloc(sizeof(mat66)*2, 16);
		blocks = (mat66*) blockPtr;
#else
		std::vector< mat66 , Eigen::aligned_allocator<mat66> > blocks(2);	
#endif

// 		if( translation.getValue() ) {
// 			blocks[0] = -mat66::Identity();
// 			blocks[1] = mat66::Identity();
// 		} else {
// 			blocks[0] = mat66::Zero();
// 			blocks[1] = mat66::Zero();
// 		}
		
		for(unsigned i = 0, n = p.size(); i < n; ++i) {

            const coord_type& parent = in_pos[ p[i][0] ];
            const coord_type& child  = in_pos[ p[i][1] ];
			const coord_type diff = delta(parent, child);


            mat33 Rp_T = (se3::rotation(parent).normalized().toRotationMatrix()).transpose();

            const typename se3::vec3 s = se3::translation(child) - se3::translation(parent);

            mat33 chunk;

            if( rotation )
            {
                if( exact_dlog ) {
                    mat33 Rc = se3::rotation(child).normalized().toRotationMatrix();
                    // note: dlog is in spatial coordinates !
                    chunk = se3::dlog( se3::rotation(diff).normalized() ) * Rc.transpose();
                } else {
                    chunk = Rp_T;
                }
            }
            else
            {
                chunk = mat33::Zero();
            }

            if(!translation)
                Rp_T = mat33::Zero();

            // parent
            blocks[0] <<
                -Rp_T, Rp_T * se3::hat(s),
                mat33::Zero(), -chunk;

            // child
            blocks[1] <<
                Rp_T, mat33::Zero(),
                mat33::Zero(), chunk;

            bool reverse =  p[i][0] > p[i][1];
            for( unsigned u = 0; u < 6; ++u) {
                unsigned row = 6 * i + u;
                J.startVec( row );

                for( unsigned j = 0; j < 2; ++j) {

                    unsigned ordered = reverse ? 1 - j : j;
                    unsigned index = p[i][ ordered ];

                    for( unsigned v = 0; v < 6; ++v) {
                        unsigned col = 6 * index + v;
                        J.insertBack(row, col) = blocks[ ordered ](u, v);
                    }
                }
            }
		}
		
		J.finalize();

#if _MSC_VER
		::_aligned_free(blockPtr);
#endif
	} 
	
	virtual void apply(typename self::out_pos_type& out,
	                   const typename self::in_pos_type& in ) {
        const pairs_type& p = pairs.getValue();
        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();
		
		assert( out.size() == p.size() );				        

		for(unsigned i = 0, n = p.size(); i < n; ++i) {
			
			// // FIXME this means the lowest index is always the parent, derp
			// if( p[i][1] < p[i][0] ) std::swap( p[i][1], p[i][0] );
			
			// out[i] = se3::product_log( se3::prod( se3::inv( in[ p[i][0] ] ), 
			//                                       in[ p[i][1] ] ) ).getVAll();

            const coord_type diff = delta( in[ p[i][0] ], in[ p[i][1] ] );
			                                             
            if( !rotation ) {
                out[i][0] = diff.getCenter()[0];
                out[i][1] = diff.getCenter()[1];
                out[i][2] = diff.getCenter()[2];
                out[i][3] = 0;
                out[i][4] = 0;
                out[i][5] = 0;
			}
            else if( !translation ) {
				out[i][0] = 0;
				out[i][1] = 0;
				out[i][2] = 0;

                typename se3::vec3 l = se3::log( se3::rotation(diff) );
                out[i][3] = l[0];
                out[i][4] = l[1];
                out[i][5] = l[2];
			}
            else
            {
                out[i] = se3::product_log( diff ).getVAll();
            }
			

//            if( out_joint_angle.getValue() ) output( out[i] );

        }
    }

//	void output(typename TOut::Coord out) const {
//		out[0] = 0;
//		out[1] = 0;
//		out[2] = 0;
//		std::cerr << this->getContext()->getTime() << ", " << out.norm() << std::endl;
//	}

    virtual void updateForceMask()
    {
        const pairs_type& p = pairs.getValue();

        for( size_t i = 0, iend = p.size(); i < iend; ++i )
        {
            if( this->maskTo->getEntry(i) )
            {
                const index_pair& indices = p[i];
                this->maskFrom->insertEntry(indices[0]);
                this->maskFrom->insertEntry(indices[1]);
            }
        }
    }
};
}
}
}

#endif
