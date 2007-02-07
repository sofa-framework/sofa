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

template <class BaseMapping>
class IdentityMapping : public BaseMapping
{
public:
    typedef BaseMapping Inherit;
    typedef typename BaseMapping::In In;
    typedef typename BaseMapping::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;

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
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
