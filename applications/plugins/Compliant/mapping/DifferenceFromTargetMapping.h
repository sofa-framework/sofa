#ifndef DIFFERENCEFROMTARGETMAPPING_H
#define DIFFERENCEFROMTARGETMAPPING_H

#include "../initCompliant.h"

#include "AssembledMapping.h"
#include "AssembledMultiMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{


/**
 Maps a vec to its difference with a target:

 (p, target) -> p - target

 This is used in compliant constraints to obtain relative
 violation dofs, on which a compliance may be applied
 (ie conversion to a holonomic constraint)
*/
template <class TIn, class TOut >
class SOFA_Compliant_API DifferenceFromTargetMapping : public AssembledMapping<TIn, TOut>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(DifferenceFromTargetMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
    typedef DifferenceFromTargetMapping Self;

    typedef typename TIn::Coord InCoord;
    typedef vector< InCoord > targets_type;
    Data< targets_type > targets;

    Data< bool > inverted;

	
    DifferenceFromTargetMapping()
        : targets( initData(&targets, "targets", "target positions which who computes deltas") )
        , inverted( initData(&inverted, false, "inverted", "target-p (rather than p-target)") )
    {

        // backward compatibility with OffsetMapping, a previous identical mapping
        this->addAlias(&targets, "offsets");
    }

    enum {Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size };

    virtual void apply(typename Self::out_pos_type& out,
                       const typename Self::in_pos_type& in )
    {
        assert( this->Nout == this->Nin );

        // automatic output resize
        this->getToModel()->resize( in.size() );

        out.wref() = in.ref();

        const targets_type& t = targets.getValue();

        if( t.empty() ) return;

        if( inverted.getValue() )
            for( size_t j = 0 ; j < in.size() ; ++j )
            {
                out[j] = t[std::min(t.size()-1,j)] - out[j];
            }
        else
            for( size_t j = 0 ; j < in.size() ; ++j )
            {
                out[j] -= t[std::min(t.size()-1,j)];
            }
	}

    virtual void assemble( const typename Self::in_pos_type& in )
    {
        typename Self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
        J.resize( Nout * in.size(), Nin * in.size());
        J.setIdentity();
        if( inverted.getValue() ) J *= -1;
	}

	
};




}
}
}


#endif
