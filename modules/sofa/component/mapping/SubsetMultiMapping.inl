#ifndef SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL

#include <sofa/component/mapping/SubsetMultiMapping.h>

#include <sofa/core/MultiMapping.inl>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::core;


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::init()
{
    Inherit::init();
    unsigned int total = computeTotalInputPoints();
    this->toModels[0]->resize(total);
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::addPoint( const core::State<In>* fromModel, int index)
{
    typename std::map<const core::State<In>*, IndexArray>::iterator cur  = m_indices.find(fromModel);

    if ( cur != m_indices.end() )
    {
        (*cur).second.push_back(index);
    }
    else
    {
        IndexArray ptSubset;
        ptSubset.push_back(index);
        m_indices[fromModel] = ptSubset;
    }
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::apply(const vecOutVecCoord& outPos, const vecConstInVecCoord& inPos)
{
    typename std::map<const core::State<In>* , IndexArray >::iterator iterMap;
    core::State<Out>* output = this->toModels[0];
    unsigned int total = computeTotalInputPoints();
    output->resize(total);

    OutVecCoord* outVecCoord = outPos[0];
    unsigned int size = 0;
    for( unsigned int i = 0; i < inPos.size() ; i++)
    {
        core::State<In>* current = this->fromModels[i];
        const InVecCoord* currentVecCoord = inPos[i];
        iterMap = m_indices.find( current );
        if ( iterMap != m_indices.end() )
        {
            IndexArray indices = (*iterMap).second;
            for( unsigned int j = 0; j < indices.size() ; j++)
            {
                (*outVecCoord)[size+j] = (*currentVecCoord)[ indices[j] ];
            }
            size += indices.size();

        }
    }
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const InVecDeriv*>& inDeriv)
{
    core::State<Out>* output = this->toModels[0];
    unsigned int total = computeTotalInputPoints();
    output->resize(total);

    typename std::map<const core::State<In>* , IndexArray >::iterator iterMap;
    OutVecDeriv* outVecDeriv = outDeriv[0];
    unsigned int size = 0;

    for( unsigned int i = 0; i < inDeriv.size() ; i++)
    {
        core::State<In>* current = this->fromModels[i];
        const InVecDeriv* currentVecDeriv = inDeriv[i];
        iterMap = m_indices.find( current );
        if ( iterMap != m_indices.end() )
        {
            IndexArray indices = (*iterMap).second;
            for( unsigned int j = 0; j < indices.size() ; j++)
            {
                (*outVecDeriv)[size+j] = (*currentVecDeriv)[ indices[j] ];
            }
            size += indices.size();

        }
    }
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT(const helper::vector<InVecDeriv*>& outDeriv , const helper::vector<const OutVecDeriv*>& inDeriv )
{
    typename std::map<const core::State<In>* , IndexArray >::iterator iterMap;
    const OutVecDeriv* mappedVecDeriv = inDeriv[0];

    unsigned int size = 0;
    for( unsigned int i = 0; i < outDeriv.size() ; i++)
    {
        core::State<In>* current = this->fromModels[i];
        InVecDeriv* currentVecDeriv = outDeriv[i];
        iterMap = m_indices.find( current );
        if ( iterMap != m_indices.end() )
        {
            IndexArray indices = (*iterMap).second;
            for( unsigned int j = 0; j < indices.size() ; j++)
            {
                (*currentVecDeriv)[ indices[j] ] += (*mappedVecDeriv)[size + j] ;
            }
            size += indices.size();
        }
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL
