#ifndef SOFA_COMPONENTS_IDENTITYMAPPING_H
#define SOFA_COMPONENTS_IDENTITYMAPPING_H

#include "Sofa/Core/MechanicalMapping.h"
#include "Sofa/Core/MechanicalModel.h"
#include <vector>

namespace Sofa
{

namespace Components
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

    IdentityMapping(In* from, Out* to, const std::string& /*name*/)
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

} // namespace Components

} // namespace Sofa

#endif
