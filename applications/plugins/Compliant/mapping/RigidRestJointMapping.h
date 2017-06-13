#ifndef RIGIDRESTJOINTMAPPING_H
#define RIGIDRESTJOINTMAPPING_H

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
    Maps to the logarithm coordinates of a joint between
    the current position and the rest position of a rigid body:
    let the joint have coordinates p and c, the mapping computes:

    \[ f(p, c) = \log\left( p^{-1} c) \]
    

    @author Matthieu Nesme
    @date 2016


 */

template <class TIn, class TOut >
class RigidRestJointMapping : public AssembledMapping<TIn, TOut> {
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidRestJointMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));

    typedef typename TIn::Real Real;

    Data< bool > rotation; ///< compute relative rotation
    Data< bool > translation; ///< compute relative translation
	
	Data< bool > exact_dlog;

    RigidRestJointMapping()
        : rotation(initData(&rotation, true, "rotation", "compute relative rotation" )),
		  translation(initData(&translation, true, "translation", "compute relative translation" )),
		  exact_dlog(initData(&exact_dlog, false, "exact_dlog",
							  "compute exact rotation dlog. more precise if you know what you're doing, but gets unstable past half turn. for 1d and isotropic 3d springs, you don't need this"))
		{
			
		}

protected:
	
	typedef SE3< typename TIn::Real > se3;
	typedef typename se3::coord_type coord_type;


	static coord_type delta(const coord_type& p, const coord_type& c) {
		return se3::prod( se3::inv(p), c);
	}
	

    typedef RigidRestJointMapping self;
	virtual void assemble( const typename self::in_pos_type& in_pos ) {
        typename self::jacobian_type& J = this->jacobian;

        const size_t size = in_pos.size();

        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();
        bool exact_dlog = this->exact_dlog.getValue();

        J.resizeBlocks( size, size );
        if( !translation ) J.compressedMatrix.reserve( 18*size );
        else if( !rotation ) J.compressedMatrix.reserve( 9*size );
        else J.compressedMatrix.reserve( 27*size );

        typedef typename se3::mat66 mat66;
        typedef typename se3::mat33 mat33;

        typename self::in_pos_type restPos( this->fromModel.get()->read(core::VecCoordId::restPosition()) );


        for(unsigned i = 0 ; i < size ; ++i) {

            const coord_type& parent = in_pos[i];
            const coord_type& child  = restPos[i];
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


            mat66 block;
            block <<
                -Rp_T, Rp_T * se3::hat(s),
                mat33::Zero(), -chunk;

            for( unsigned u = 0; u < 6; ++u) {
                unsigned row = 6 * i + u;
                J.compressedMatrix.startVec( row );
                for( unsigned v = 0; v < 6; ++v) {
                    unsigned col = 6 * i + v;
                    J.compressedMatrix.insertBack(row, col) = block(u, v);
                }
            }
        }
		
        J.compressedMatrix.finalize();

	} 
	
	virtual void apply(typename self::out_pos_type& out,
                       const typename self::in_pos_type& in ) {

        const size_t size = in.size();

        typename self::in_pos_type restPos( this->fromModel.get()->read(core::VecCoordId::restPosition()) );

        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();
		
        assert( out.size() == size );

        for(unsigned i = 0; i < size; ++i) {
			
            const coord_type diff = delta( in[i], restPos[i] );
			                                             
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
        }
    }



};
}
}
}

#endif
