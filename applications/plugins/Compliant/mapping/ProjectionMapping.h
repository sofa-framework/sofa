#ifndef PROJECTIONMAPPING_H
#define PROJECTIONMAPPING_H

#include "AssembledMapping.h"
#include <Compliant/config.h>

#include "../utils/map.h"
#include <sofa/helper/pair.h>

namespace sofa {

namespace component {

namespace mapping {

/*
  
  A simple projection-mapping: projects input orthogonally (canonical
  dot-product) onto given (input dof index, deriv) pairs. This is a kind of
  general-purpose mapping. Input: any vector dof. Output: 1d dofs.

  @author: maxime.tournier@inria.fr

*/


template <class TIn, class TOut >
class ProjectionMapping : public AssembledMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(ProjectionMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
	typedef typename TIn::Real in_real;
	typedef typename TOut::Real out_real;

    typedef std::pair<unsigned, typename TIn::Coord> set_type;
    Data< helper::vector< set_type > > set;
	
    Data< helper::vector<SReal> > offset;

	typedef AssembledMapping<TIn, TOut> base;
	typedef ProjectionMapping self;

	ProjectionMapping()
		: set(initData(&set, "set", 
					   "(index, coord) vector describing projection space")),
		  offset(initData(&offset, "offset", 
						"scalar offset for each projection (if empty: 0, otherwise last value is repeated as needed)")){
		assert( base::Nout == 1 );
        assert( (unsigned)TIn::deriv_total_size == (unsigned)TIn::coord_total_size && "not vector input dofs !" );
	}
	

    virtual void init()
    {
        this->getToModel()->resize( set.getValue().size() );
        AssembledMapping<TIn, TOut>::init();
    }


protected:
	
	virtual void assemble( const typename self::in_pos_type& in) {
		
        const helper::vector<set_type>& s = set.getValue();
		
		// resize/clear jacobian
		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
		J.resize( base::Nout * s.size(), 
				  base::Nin * in.size() );

        J.reserve( base::Nout * s.size() * self::Nin );

		for( unsigned i = 0, n = s.size(); i < n; ++i) {
			unsigned row = i;

			J.startVec( row );
			
			for( unsigned j = 0, m = self::Nin; j < m; ++j) {
                unsigned col = self::Nin * s[i].first + j;

                J.insertBack(row, col) = s[i].second[j];
			}
	
		}
		
		J.finalize();
		
	}
	
	virtual void apply(typename self::out_pos_type& out, 
					   const typename self::in_pos_type& in ) {
        const helper::vector<set_type>& s = set.getValue();
        const helper::vector<SReal>& off = offset.getValue();
		
		assert(s.size() == out.size());	

		for( unsigned i = 0, n = s.size(); i < n; ++i) {
			
			SReal delta = off.empty() ? 0 : off[ std::min<int>(off.size() - 1, i) ];
			
            utils::map(out[i])(0) = utils::map(in[s[i].first]).dot( utils::map(s[i].second ) ) - delta;
		}
		
	}

    virtual void updateForceMask()
    {
        const helper::vector<set_type>& s = set.getValue();

        for( size_t i = 0, iend = s.size(); i < iend; ++i )
        {
            if( this->maskTo->getEntry(i) )
            {
                this->maskFrom->insertEntry(s[i].first);
            }
        }
    }
	
};

}
}
}



#endif
