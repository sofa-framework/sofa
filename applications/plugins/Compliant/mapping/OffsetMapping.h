#ifndef OffsetMapping_H
#define OffsetMapping_H

#include "../initCompliant.h"

#include "AssembledMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{


/**
 Maps a vec to its offsetted value:

 (p, offset) -> p - offset

 This is used in compliant constraints to constraint a violation to a given value (the offset) by constraining the offseted value to be null.
*/
template <class TIn, class TOut >
class SOFA_Compliant_API OffsetMapping : public AssembledMapping<TIn, TOut>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(OffsetMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));

    typedef OffsetMapping Self;

    enum {Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size };

    typedef typename TOut::Coord OutCoord;
    typedef vector< OutCoord > offsets_type;
    Data< offsets_type > offsets;

	
    OffsetMapping()
        : offsets( initData(&offsets, "offsets", "optional offsets (removed to given values)") )
    {}


    virtual void apply(typename Self::out_pos_type& out, const typename Self::in_pos_type& in )
    {
		assert( this->Nout == this->Nin );

        out.wref() = in.ref();

        const offsets_type& o = offsets.getValue();

        if( o.empty() ) return;

        for( size_t j = 0 ; j < in.size() ; ++j )
        {
            out[j] -= o[std::min(o.size()-1,j)];
		}

	}

    virtual void assemble( const typename Self::in_pos_type& in )
    {
        typename Self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
        J.resize( Nin * in.size(), Nin * in.size());
        J.setIdentity();
	}

	
};





}
}
}


#endif
