#ifndef RIGIDJOINTFROMTARGETMAPPING_H
#define RIGIDJOINTFROMTARGETMAPPING_H

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
    rigids c and given frame 'targets'

    \[ f(c) = \log\left( target^{-1} c) \]
    
    this is mostly used to place a compliance on the end. Use
    @rotation and @translation data to restrict mapping to rotation
    and translation parts, in case you only want to apply a compliance
    on these.

    @author Matthieu Nesme

    @todo display target frame
    
 */

template <class TIn, class TOut >
class RigidJointFromTargetMapping : public AssembledMapping<TIn, TOut> {
protected:

    typedef SE3< typename TIn::Real > se3;
    typedef typename se3::coord_type coord_type;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidJointFromTargetMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));

    typedef typename TIn::Real Real;

    typedef helper::vector< coord_type > targets_type;
    Data< targets_type > targets; ///< target positions which who computes deltas

    Data< bool > rotation; ///< compute relative rotation
    Data< bool > translation; ///< compute relative translation
	Data< bool > exact_dlog;

    RigidJointFromTargetMapping()
          : targets( initData(&targets, "targets", "target positions which who computes deltas") )
          , rotation(initData(&rotation, true, "rotation", "compute relative rotation" ))
          , translation(initData(&translation, true, "translation", "compute relative translation" ))
          , exact_dlog(initData(&exact_dlog, false, "exact_dlog",
							  "compute exact rotation dlog. more precise if you know what you're doing, but gets unstable past half turn. for 1d and isotropic 3d springs, you don't need this"))
		{
		}

protected:

	static coord_type delta(const coord_type& p, const coord_type& c) {
		return se3::prod( se3::inv(p), c);
	}

    typedef RigidJointFromTargetMapping self;
	virtual void assemble( const typename self::in_pos_type& in_pos ) {
		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;


        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();
        const helper::vector< coord_type >& targets = this->targets.getValue();
        bool exact_dlog = this->exact_dlog.getValue();

        assert( in_pos.size() == targets.size() );


        J.resize(in_pos.size() * 6, in_pos.size() * 6);
        J.reserve( 36*in_pos.size() );

		typedef typename se3::mat66 mat66;
		typedef typename se3::mat33 mat33;

        mat66 block;

        for(unsigned i = 0, n = in_pos.size(); i < n; ++i) {

            coord_type parent = targets[i];
            if (!rotation) {
                parent[3]=0.;parent[4]=0.;parent[5]=0.;parent[6]=1.;
            }

            const coord_type& child  = in_pos[i];
            const coord_type diff = delta(parent, child); // note that parent^{-1} is precomputable but it would be a mess to handle when target is changing dynamically


            mat33 Rp_T = (se3::rotation(parent).normalized().toRotationMatrix()).transpose();

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

            // child
            block <<
                Rp_T, mat33::Zero(),
                mat33::Zero(), chunk;

            for( unsigned u = 0; u < 6; ++u)
            {
                unsigned row = 6 * i + u;
                J.startVec( row );

                for( unsigned v = 0; v < 6; ++v)
                {
                    unsigned col = 6 * i + v;
                    J.insertBack(row, col) = block(u, v);
                }
            }
        }
		
        J.finalize();
	} 
	
	virtual void apply(typename self::out_pos_type& out,
	                   const typename self::in_pos_type& in ) {

        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();
        const helper::vector< coord_type >& targets = this->targets.getValue();

        for(unsigned i = 0, n = in.size(); i < n; ++i) {
            coord_type parent = targets[i];
            if (!rotation) {
                parent[3]=0.;parent[4]=0.;parent[5]=0.;parent[6]=1.;
            }
            const coord_type diff = delta( parent, in[i] );

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


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

/**
    Maps rigid bodies to the logarithm coordinates of a joint between
    rigids c and the world frame W=(0,0,0, 0,0,0,1)

    \[ f(c) = \log\left( W c) \]

    this is mostly used to place a compliance on the end. Use
    @rotation and @translation data to restrict mapping to rotation
    and translation parts, in case you only want to apply a compliance
    on these.

    @author Matthieu Nesme

 */

template <class TIn, class TOut >
class RigidJointFromWorldFrameMapping : public AssembledMapping<TIn, TOut> {
protected:

    typedef SE3< typename TIn::Real > se3;
    typedef typename se3::coord_type coord_type;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidJointFromWorldFrameMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));

    typedef typename TIn::Real Real;

    static const coord_type s_worldFrame;

    Data< bool > rotation; ///< compute relative rotation
    Data< bool > translation; ///< compute relative translation
//    Data< bool > exact_dlog;

    RigidJointFromWorldFrameMapping()
          : rotation(initData(&rotation, true, "rotation", "compute relative rotation" ))
          , translation(initData(&translation, true, "translation", "compute relative translation" ))
//          , exact_dlog(initData(&exact_dlog, false, "exact_dlog",
//                              "compute exact rotation dlog. more precise if you know what you're doing, but gets unstable past half turn. for 1d and isotropic 3d springs, you don't need this"))
        {
        }

protected:


    typedef RigidJointFromWorldFrameMapping self;
    virtual void assemble( const typename self::in_pos_type& in_pos ) {
        typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;


        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();


        J.resize(in_pos.size() * 6, in_pos.size() * 6);
        if( !translation || !rotation ) J.reserve( 9*in_pos.size() );
        else J.reserve( 18*in_pos.size() );

        typedef typename se3::mat66 mat66;
        typedef typename se3::mat33 mat33;

        mat66 block;

        for(unsigned i = 0, n = in_pos.size(); i < n; ++i) {

//            const coord_type& child  = in_pos[i];
//            const coord_type& diff = child;


            mat33 Rp_T = mat33::Identity();

            mat33 chunk;

            if( rotation )
            {
               /* if( exact_dlog.getValue() ) {
                    mat33 Rc = se3::rotation(child).normalized().toRotationMatrix();
                    // note: dlog is in spatial coordinates !
                    chunk = se3::dlog( se3::rotation(diff).normalized() ) * Rc.transpose();
                } else*/ {
                    chunk = Rp_T;
                }

            }
            else
            {
                chunk = mat33::Zero();
            }

            if(!translation)
                Rp_T = mat33::Zero();

            // child
            block <<
                Rp_T, mat33::Zero(),
                mat33::Zero(), chunk;

            for( unsigned u = 0; u < 6; ++u)
            {
                unsigned row = 6 * i + u;
                J.startVec( row );

                for( unsigned v = 0; v < 6; ++v)
                {
                    unsigned col = 6 * i + v;
                    J.insertBack(row, col) = block(u, v);
                }
            }
        }

        J.finalize();
    }

    virtual void apply(typename self::out_pos_type& out,
                       const typename self::in_pos_type& in ) {

        bool rotation = this->rotation.getValue();
        bool translation = this->translation.getValue();

        for(unsigned i = 0, n = in.size(); i < n; ++i) {

            const coord_type& diff = in[i];

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
