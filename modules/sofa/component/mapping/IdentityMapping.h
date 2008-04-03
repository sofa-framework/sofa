#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_H
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class IdentityMapping : public BasicMapping
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;

    typedef typename In::DataTypes InDataTypes;
    typedef typename InDataTypes::Real Real;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;

    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::SparseDeriv OutSparseDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename OutDataTypes::Real OutReal;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;

    enum { N=Coord::static_size };

    IdentityMapping(In* from, Out* to)
        : Inherit(from, to)
    {
    }

    virtual ~IdentityMapping()
    {
    }

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
