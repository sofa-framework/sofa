#ifndef COMPLIANT_MAPPING_FILTERMAPPING_H
#define COMPLIANT_MAPPING_FILTERMAPPING_H

#include "AssembledMapping.h"
#include <Compliant/config.h>
#include "../utils/map.h"

namespace sofa {

namespace component {

namespace mapping {

/**
  
  A simple mask-filtering mapping, used to filter unconstrained dofs
  in rigid body constraints. it used to be part of RigidJointMapping,
  but needed to be separated otherwise no forcefield can be applied on
  joint dofs, since corresponding J line is zero.

  @author: maxime.tournier@inria.fr

*/

template <class TIn, class TOut >
class MaskMapping : public AssembledMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(MaskMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
	typedef typename TIn::Real in_real;
	typedef typename TOut::Real out_real;

	// should only be used with vector dofs
    typedef helper::vector< typename TIn::Coord > dofs_type;
	Data< dofs_type > dofs;

	typedef AssembledMapping<TIn, TOut> base;
	typedef MaskMapping self;

	MaskMapping()
		: dofs(initData(&dofs, "dofs", 
						"mask for each input dof"))  {

		assert( base::Nout == 1 );
		
	}
	
protected:
	
	std::vector< std::pair<unsigned, unsigned> > index;
	
	virtual void assemble( const typename self::in_pos_type& in) {
		// note: index is filled in @apply
		
//		const dofs_type& d = dofs.getValue();
		
		// resize/clear jacobian
		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
		J.resize( base::Nout * index.size(),
				  base::Nin * in.size() );
		
		J.setZero();

		for( unsigned i = 0, n = index.size(); i < n; ++i) {
			unsigned row = i;
			J.startVec(row);

			unsigned col = base::Nin * index[i].first + index[i].second;

			J.insertBack(row, col) = 1;
		}

		J.finalize();
	}
	
	virtual void apply(typename self::out_pos_type& out, 
					   const typename self::in_pos_type& in ) {
		const dofs_type& d = dofs.getValue();

		// build array of (dof index, coeff index)
		index.clear();

		for(unsigned i = 0, n = d.size(); i < n; ++i) {
			for( unsigned j = 0; j < self::Nin; ++j) {
				if( d[i][j] ) {
					index.push_back( std::make_pair(i, j) );
				}
			}
		}
		
		// automatic output resize yo !
		this->getToModel()->resize( index.size() );
		
		for(unsigned i = 0, n = index.size(); i < n; ++i) {
			utils::map(out[i])(0) = utils::map(in[ index[i].first ])(index[i].second);
		}
	
	}
	
};

}
}
}

#endif

