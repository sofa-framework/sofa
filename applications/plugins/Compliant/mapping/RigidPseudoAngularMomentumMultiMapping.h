#ifndef RIGID_PSEUDOAM_MULTIMAPPING_H
#define RIGID_PSEUDOAM_MULTIMAPPING_H

#include "AssembledMultiMapping.h"

#include <Compliant/config.h>
#include <Compliant/utils/se3.h>

namespace sofa {

namespace component {

namespace mapping {

/**
   
   maps a kind of centroidal angular momentum from a set of rigid bodies. only
   positions are accounted for.

   author: maxime.tournier@anatoscope.com
 */

template <class TIn, class TOut >
class RigidPseudoAngularMomentumMultiMapping : public AssembledMultiMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(RigidPseudoAngularMomentumMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));

    typedef helper::vector<SReal> mass_type;

    Data<mass_type> mass;
    Data<typename TOut::VecCoord> inertia;
    
    Data<typename TIn::VecCoord> previous;
	Data<bool> use_rotations;
    
	typedef AssembledMultiMapping<TIn, TOut> base;
	typedef RigidPseudoAngularMomentumMultiMapping self;

    using real = typename TIn::Real;
    using se3 = SE3<real>;
    
	RigidPseudoAngularMomentumMultiMapping()
		: mass(initData(&mass, "mass", "individual mass for each incoming dof, in the same order")),
          inertia(initData(&inertia, "inertia", "body-fixed inertia")),          
          previous(initData(&previous, "previous", "previous value for input dofs positions")),
          use_rotations(initData(&use_rotations, true, "use_rotations", "use rotational part")),
		  total(0) {
        
		assert( self::Nin == 6 );
		assert( self::Nout == 3 );
	}

protected:

	SReal total;

    using vec3 = typename se3::vec3;
    
    std::vector< vec3 > delta, dp;

    struct error { };
    
    template<class In>
    void check_preconditions(const In& in) {
        const std::size_t size = previous.getValue().size();
        if(size != in.size()) {
            msg_error() << "previous has wrong size: " << size << " != " << in.size() << ",  aborting";
            throw error();
        }
        
    }
    
	virtual void apply(typename self::out_pos_type& out, 
                       const helper::vector<typename self::in_pos_type>& in ) {
        try{ check_preconditions(in); } catch(error& e) { return; }
        
        const mass_type& m = mass.getValue();
	
        se3::map(out[0]).setZero();
		
        vec3 com = vec3::Zero();
		total = 0;
        
        delta.clear();
        dp.clear();        
            
        for( unsigned i = 0, n = m.size(); i < n; ++i) {
            assert(in[i].size() == 1);
            // const vec3 qi = se3::map(previous.getValue()[i].getCenter());
            const vec3 pi = se3::map(in[i][0].getCenter());

            com += m[i] * pi;
            total += m[i];
        }

        com /= total;
        
		for( unsigned i = 0, n = m.size(); i < n; ++i) {
            assert(in[i].size() == 1);

            const vec3 qi = se3::map(previous.getValue()[i].getCenter());
            const vec3 pi = se3::map(in[i][0].getCenter());
            const vec3 di = pi - qi;

            se3::map(out[0]) += m[i] * (pi - com).cross(di);
            
            delta.push_back(pi - com);
            dp.push_back(di);


            //
            if(use_rotations.getValue()) {
                const typename se3::quat hi = se3::orient(previous.getValue()[i]);
                const typename se3::quat gi = se3::orient(in[i][0]);
            
                const vec3 s = se3::map(inertia.getValue()[i]);
                
                const vec3 omega = se3::log(hi.conjugate() * gi);
                const vec3 mu = hi * s.cwiseProduct( omega );
                
                se3::map(out[0]) += mu;
            }
		}


        se3::map(out[0]) /= total;
        
	}

    void assemble(const helper::vector< typename self::in_pos_type >& in ) {
        try{ check_preconditions(in); } catch(error& e) { return; }        
        
        const mass_type& m = mass.getValue();
		
        // resize/clean jacobians
		for(unsigned i = 0, n = in.size(); i < n; ++i) {
            assert(in[i].size() == 1);
            
			typename self::jacobian_type::CompressedMatrix& J = this->jacobian(i).compressedMatrix;
            
			J.resize( self::Nout, self::Nin );

            const auto factor = m[i] / total;
            
            typename se3::mat33 left_block =
                se3::hat( (delta[i] + (factor - 1.0) * dp[i]) * factor );

            const typename se3::quat hi = se3::orient(previous.getValue()[i]);
            
            const typename se3::mat33 Ri = hi.toRotationMatrix();
            const vec3 s = se3::map(inertia.getValue()[i]);

            // TODO use dlog
            typename se3::mat33 right_block;

            if(use_rotations.getValue()) {
                right_block = (Ri * s.asDiagonal() * Ri.transpose()) / total;
            }
            
			for(unsigned k = 0; k < self::Nout; ++k) {
				const unsigned row = k;
				J.startVec(row);

				for(unsigned j = 0; j < 3; ++j) {
					const unsigned col = j;
					J.insertBack(row, col) = left_block(k, j);
				}

                if( use_rotations.getValue() ) {
                    for(unsigned j = 0; j < 3; ++j) {
                        const unsigned col = 3 + j;
                        J.insertBack(row, col) = right_block(k, j);
                    }
                }
                
			}
			
			J.finalize();
		}
		
	}


};

}
}
}

#endif
