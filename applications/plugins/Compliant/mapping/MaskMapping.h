#ifndef COMPLIANT_MAPPING_FILTERMAPPING_H
#define COMPLIANT_MAPPING_FILTERMAPPING_H

#include "AssembledMapping.h"
#include "initCompliant.h"
#include "utils/map.h"
#include "utils/edit.h"

namespace sofa {

namespace component {

namespace mapping {

/*
  
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
	typedef vector< typename TOut::Coord > dofs_type;
	Data< dofs_type > dofs;

	typedef AssembledMapping<TIn, TOut> base;
	typedef MaskMapping self;

	MaskMapping()
		: dofs(initData(&dofs, "dofs", 
						"mask for each input dof (default: all. last value extended if needed)"))  {
		typename TOut::Coord coord;
		for(unsigned i = 0; i < base::Nout; ++i) {
			coord[i] = 1;
		}
		
		edit(dofs)->push_back( coord );
	}
	
protected:
	
	virtual void assemble( const typename self::in_pos_type& in) {
		assert( base::Nin == base::Nout );
		
		const dofs_type& d = dofs.getValue();
		
		// resize/clear jacobian
		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
		J.resize( base::Nout * in.size(), 
				  base::Nin * in.size() );
		
		J.setZero();

		for( unsigned i = 0, n = in.size(); i < n; ++i) {
			unsigned index = std::min<int>(d.size() - 1, i);
			
			for( unsigned u = 0; u < base::Nin; ++u) {
				unsigned row = base::Nout * i + u;
				J.startVec( row );
				
				if( d[index][u] ) {
					J.insertBack(row, row) = d[index][u];
				}
			}
		}
		
		J.finalize();
		
	}
	
	virtual void apply(typename self::out_pos_type& out, 
					   const typename self::in_pos_type& in ) {
		assert(in.size() == out.size());
//		assert(d.size());
		
		const dofs_type& d = dofs.getValue();
	
		for( unsigned i = 0, n = in.size(); i < n; ++i) {
			unsigned index = std::min<int>(d.size() - 1, i);

			map(out[i]) = map(in[i]).template cast<out_real>().cwiseProduct( map(d[index]) );
		}
		
	}
	
};

}
}
}

#endif

