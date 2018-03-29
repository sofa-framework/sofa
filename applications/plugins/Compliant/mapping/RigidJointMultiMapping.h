#ifndef COMPLIANT_MAPPING_RIGIDJOINTMULTIMAPPING_H
#define COMPLIANT_MAPPING_RIGIDJOINTMULTIMAPPING_H

#include "AssembledMultiMapping.h"
#include <Compliant/config.h>

#include "../utils/se3.h"
#include "../utils/map.h"

namespace sofa {

namespace component {

namespace mapping {



// static dispatch wizardry
namespace impl {

template<class U>
static bool use_dlog( ::sofa::defaulttype::RigidCoord<3, U>* ) { return false; }

template<class U>
static bool use_dlog( ::sofa::defaulttype::Vec<6, U>* ) { return true; }

// dispatch on output type
template<class U, class V>
void fill( ::sofa::defaulttype::RigidCoord<3, U>& out,
           const ::sofa::defaulttype::RigidCoord<3, V>& in) {
    out = in;
}

// use log coords for vec6
template<class U, class V>
void fill( ::sofa::defaulttype::Vec<6, U>& out,
           const ::sofa::defaulttype::RigidCoord<3, V>& in) {
    
    typedef SE3<V> se3;
    
    typename se3::deriv_type value = se3::product_log( in );
    
    utils::map(out).template head<3>() = utils::map(value.getLinear()).template cast<U>();
    utils::map(out).template tail<3>() = utils::map(value.getAngular()).template cast<U>();
    
}


}

/** 

    Maps rigid bodies to the relative transform between the
    two: 
    
     \[ f(p, c) = p^{-1} c \]

     logarithm coordinates are used for Vec6 output dofs
     
     TODO .inl

     @author: Maxime Tournier

*/



template <class TIn, class TOut >
class RigidJointMultiMapping : public AssembledMultiMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(RigidJointMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));
	
	typedef defaulttype::Vec<2, unsigned> index_pair;

    typedef helper::vector< index_pair > pairs_type;
	Data< pairs_type > pairs; ///< index pairs (parent, child) for each joint

	typedef typename TIn::Real in_real;
	typedef typename TOut::Real out_real;
	
	typedef AssembledMultiMapping<TIn, TOut> base;
	typedef RigidJointMultiMapping self;

    Data<int> geometricStiffness;

protected:

    static const bool use_dlog;
    
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
               const helper::vector< typename self::in_pos_type >& in ) {
		// assert( this->getFrom().size() == 2 );

		const pairs_type& p = pairs.getValue();
		
		assert(out.size() == p.size());

		for( unsigned i = 0, n = p.size(); i < n; ++i) {

            const coord_type& parent = in[0][ p[i](0) ];
            const coord_type& child = in[1][ p[i](1) ];
			const coord_type delta = se3::prod( se3::inv(parent), child);
            
            impl::fill(out[i], delta);
            
			
		}
		
	}



    void assemble_geometric(const helper::vector<typename self::const_in_coord_type>& in_pos,
                            const typename self::const_out_deriv_type& out_force) {
        // we're done lol
        if( true || ! geometricStiffness.getValue() ) return;
        
        // assert( this->getFromModels().size() == 2 );
        // assert( this->getFromModels()[0] != this->getFromModels()[1] );

        // assert( this->getToModels().size() == 1 );
        // assert( this->getToModels()[0]->getSize() == 1 );

        typedef typename self::geometric_type::CompressedMatrix matrix_type;
        matrix_type& dJ = this->geometric.compressedMatrix;

        // out wrench
        const typename TOut::Deriv& mu = out_force[0];

        // TODO put these in se3::angular/linear
        typedef typename se3::vec3 vec3;
        const vec3 f = se3::map(mu).template head<3>();
        const vec3 tau = se3::map(mu).template tail<3>();
        
        // matrix sizes
        const unsigned size_p = this->from(0)->getMatrixSize();
        const unsigned size_c = this->from(1)->getMatrixSize();

        const unsigned size = size_p + size_c;
        
        dJ.resize( size, size );
        dJ.setZero();

        // lets process pairs
        const pairs_type& p = pairs.getValue();

        // TODO we really need sorted pairs here, is this even possible ?
        // assert( p.size() == 1 && "not sure if work");
        
        // alright, let's do this
        for(unsigned i = 0, n = p.size(); i < n; ++i) {

            const unsigned index_p = p[i](0);
            const unsigned index_c = p[i](1);

			const coord_type parent = in_pos[0][ index_p ];
			const coord_type child = in_pos[1][ index_c ];

            typedef typename se3::mat33 mat33;
            typedef typename se3::mat66 mat66;
            typedef typename se3::vec3 vec3;
            
            const mat33 Rp = se3::rotation(parent).toRotationMatrix();

            const vec3 Rp_f = Rp * f;
            const vec3 Rp_tau = Rp * tau;

            const mat33 Rp_f_hat = se3::hat(Rp_f);
            const mat33 Rp_tau_hat = se3::hat(Rp_tau);

            const vec3 s = se3::translation(child) - se3::translation(parent);

            // parent rows
            {
                // (p, p) block
                mat66 block_pp;

                block_pp <<
                    mat33::Zero(), Rp_f_hat,
                    -Rp_f_hat, se3::hat(s) * Rp_f_hat + Rp_tau_hat;
            
                // (p, c) block
                typename se3::mat66 block_pc;

                block_pc <<
                    se3::mat36::Zero(),
                    Rp_f_hat, mat33::Zero();

                // fill parent rows
                for(unsigned i = 0; i < 6; ++i) {
                    const unsigned row = 6 * index_p + i;

                    dJ.startVec(row);
                
                    // pp
                    for(unsigned j = 0; j < 6; ++j) {
                        const unsigned col = 6 * index_p + j;
                        if(block_pp(i, j)) dJ.insertBack(row, col) = block_pp(i, j);
                    }

                    // pc
                    for(unsigned j = 0; j < 6; ++j) {
                        const unsigned col = size_p + 6 * index_c + j;
                        if(block_pc(i, j)) dJ.insertBack(row, col) = block_pc(i, j);
                    }
                }
            }

            // child rows
            {
                // (c, p) block
                mat66 block_cp;
                block_cp <<
                    mat33::Zero(), -Rp_f_hat,
                    mat33::Zero(), -Rp_tau_hat;

                // fill child rows
                for(unsigned i = 0; i < 6; ++i) {
                    const unsigned row = size_p + 6 * index_c + i;
                
                    dJ.startVec(row);
                
                    for(unsigned j = 0; j < 6; ++j) {
                        const unsigned col = 6 * index_p + j;
                        if(block_cp(i, j)) dJ.insertBack(row, col) = block_cp(i, j);
                    }
                }
            }

        }

        dJ.finalize();
        
        if( geometricStiffness.getValue() == 2 ) {
            dJ = (dJ + matrix_type(dJ.transpose())) / 2.0;
        }
        
    }

    void assemble(const helper::vector< typename self::in_pos_type >& in ) {
		assert(this->getFrom()[0] != this->getFrom()[1]);

		const pairs_type& p = pairs.getValue();
		
		typedef typename se3::mat66 mat66;
		typedef typename se3::mat33 mat33;
		
		// resize/clean jacobians
		for(unsigned j = 0, m = in.size(); j < m; ++j) {
			typename self::jacobian_type::CompressedMatrix& J = this->jacobian(j).compressedMatrix;
            J.resize( 6 * p.size(), 6 * in[j].size() );
            J.reserve( 27 * p.size() );
		}
		
		// each pair
		for(unsigned i = 0, n = p.size(); i < n; ++i) {
			
            const coord_type& parent = in[0][ p[i](0) ];
            const coord_type& child = in[1][ p[i](1) ];
			
			const coord_type delta = se3::prod( se3::inv(parent), child);
			
			// each parent mstate
			for(unsigned j = 0, m = in.size(); j < m; ++j) {

				typename self::jacobian_type::CompressedMatrix& J = this->jacobian(j).compressedMatrix;
				
                const mat33 Rp_T = (se3::rotation(parent).normalized().toRotationMatrix()).transpose();

//				mat33 Rdelta = se3::rotation(delta).toRotationMatrix();
				const typename se3::vec3 s = se3::translation(child) - se3::translation(parent);

                mat33 chunk;

                if( use_dlog ) {
                    const mat33 Rc = se3::rotation(child).normalized().toRotationMatrix();
                    // note: dlog is in spatial coordinates !
                    chunk = se3::dlog( se3::rotation(delta).normalized() ) * Rc.transpose();
                } else {
                    chunk = Rp_T;
                }
                 
				mat66 ddelta; 

				if( j ) {
					// child
					ddelta << 
                        Rp_T, mat33::Zero(),
						mat33::Zero(), chunk;
				} else {
					// parent
                    ddelta << 
                        -Rp_T, Rp_T * se3::hat(s),
                        mat33::Zero(), -chunk;
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
	
    virtual void updateForceMask()
    {
        const pairs_type& p = pairs.getValue();

        for( size_t i = 0, iend = p.size(); i < iend; ++i )
        {
            if( this->maskTo[0]->getEntry(i) )
            {
                const index_pair& indices = p[i];
                this->maskFrom[0]->insertEntry(indices[0]);
                this->maskFrom[1]->insertEntry(indices[1]);
            }
        }
    }

};

template<class In, class Out>
const bool RigidJointMultiMapping<In, Out>::use_dlog = impl::use_dlog( (typename Out::Coord*) 0 );





}
}
}

#endif
