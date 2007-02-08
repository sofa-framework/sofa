#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL

#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class BaseMapping>
void IdentityMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(in.size());
    //if (this->fromModel->getContext()->getTopology()!=NULL)
    //	this->toModel->getContext()->setTopology(this->fromModel->getContext()->getTopology());
    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] = in[i];
    }
}

template <class BaseMapping>
void IdentityMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(in.size());
    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] = in[i];
    }
}

template <class BaseMapping>
void IdentityMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] += in[i];
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
