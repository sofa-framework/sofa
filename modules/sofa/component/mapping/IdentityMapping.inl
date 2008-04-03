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
        for (unsigned int j=0; j < N; ++j)
            out[i][j] = (OutReal)in[i][j];
    }
}

template <class BasicMapping>
void IdentityMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(in.size());
    for(unsigned int i=0; i<out.size(); i++)
    {
        for (unsigned int j=0; j < N; ++j)
            out[i][j] = (OutReal)in[i][j];
    }
}

template <class BasicMapping>
void IdentityMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    for(unsigned int i=0; i<out.size(); i++)
    {
        for (unsigned int j=0; j < N; ++j)
            out[i][j] = (OutReal)in[i][j];
    }
}

template <class BaseMapping>
void IdentityMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    out.clear();
    out.resize(in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        typename In::SparseVecDeriv& o = out[i];
        o.reserve(in[i].size());
        for(unsigned int j=0; j<in[i].size(); j++)
        {
            const typename Out::SparseDeriv& cIn = in[i][j];
            typename In::Deriv value;
            for (unsigned int k=0; k<N; ++k)
                value[k] = cIn.data[k];

            o.push_back( typename In::SparseDeriv(cIn.index, value) );
        }
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
