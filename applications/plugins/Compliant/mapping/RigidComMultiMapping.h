#ifndef COMMULTIMAPPING_H
#define COMMULTIMAPPING_H

#include "AssembledMultiMapping.h"
#include <Compliant/config.h>

#include "../utils/map.h"

namespace sofa {

namespace component {

namespace mapping {

/**
   maps a bunch of rigid dofs to their center of mass. individual
   masses must be for each incoming dof, in the same order.

   author: maxime.tournier@inria.fr
 */

template <class TIn, class TOut >
class RigidComMultiMapping : public AssembledMultiMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(RigidComMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));

    typedef helper::vector<SReal> mass_type;
	Data<mass_type> mass;
	
	typedef AssembledMultiMapping<TIn, TOut> base;
	typedef RigidComMultiMapping self;

	RigidComMultiMapping()
		: mass(initData(&mass,
						"mass",
						"individual mass for each incoming dof, in the same order")),
		  total(0) {

		assert( self::Nin == 6 );
		assert( self::Nout == 3 );
	}

protected:

	SReal total;

	virtual void apply(typename self::out_pos_type& out, 
                       const helper::vector<typename self::in_pos_type>& in ) {
		const mass_type& m = mass.getValue();
	
		using namespace utils;

        map(out[0]).setZero();
		
		unsigned off = 0;
		total = 0;
		
		for( unsigned i = 0, n = m.size(); i < n; ++i) {
			
			for(unsigned j = 0, jend = in[i].size(); j < jend; ++j) {
				assert( off < m.size() );
                map(out[0]) += m[off] * map(in[i][j].getCenter());
				total += m[off];
				++off;
			}
		}

        map(out[0]) /= total;
		
		assert( off == m.size() );
	}

    void assemble(const helper::vector< typename self::in_pos_type >& in ) {
		const mass_type& m = mass.getValue();
		
        // resize/clean jacobians
		unsigned off = 0;
		for(unsigned i = 0, n = in.size(); i < n; ++i) {
			typename self::jacobian_type::CompressedMatrix& J = this->jacobian(i).compressedMatrix;
			J.resize( self::Nout, 
			          self::Nin * in[i].size() );
			J.setZero();
			
			for(unsigned k = 0; k < self::Nout; ++k) {
				unsigned row = k;
				J.startVec(row);
				
				unsigned sub_off = off;
				for(unsigned j = 0, jend = in[i].size(); j < jend; ++j) {
					unsigned col = self::Nin * j + k;

					J.insertBack(row, col) = m[sub_off] / total;
					
					++sub_off;
				}

			}
			
			off += in[i].size();

			J.finalize();
		}
		
	}


};

}
}
}

#endif
