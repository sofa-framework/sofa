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

template <class BasicMapping>
void IdentityMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(in.size());
    //if (this->fromModel->getContext()->getTopology()!=NULL)
    //	this->toModel->getContext()->setTopology(this->fromModel->getContext()->getTopology());
    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] = in[i];
    }
}

template <class BasicMapping>
void IdentityMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(in.size());
    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] = in[i];
    }
}

template <class BasicMapping>
void IdentityMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] += in[i];
    }
}

template <class BaseMapping>
void IdentityMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    out.clear();
    out.resize(in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const OutSparseDeriv cIn = in[i][j];
            out[i][j].index = cIn.index;
            out[i][j].data = cIn.data;
        }
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
