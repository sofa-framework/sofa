#ifndef SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL

#include <sofa/component/mapping/SubsetMultiMapping.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::core;


template < class BasicMapping>
void SubsetMultiMapping<BasicMapping>::init()
{
    Inherit::init();
    unsigned int total = computeTotalInputPoints();
    this->toModels[0]->resize(total);
}

template < class BasicMapping>
void SubsetMultiMapping<BasicMapping>::addPoint( const In* fromModel, int index)
{
    typename std::map<const In*, IndexArray>::iterator cur  = _indices.find(fromModel);

    if ( cur != _indices.end() )
    {
        (*cur).second.push_back(index);
    }
    else
    {
        IndexArray ptSubset;
        ptSubset.push_back(index);
        _indices[fromModel] = ptSubset;
    }
}



template < class BasicMapping>
void SubsetMultiMapping<BasicMapping>::apply(const helper::vector<OutVecCoord*>& outPos, const helper::vector<const InVecCoord*>& inPos )
{
    typename std::map<const In* , IndexArray >::iterator iterMap;
    Out* output = this->toModels[0];
    unsigned int total = computeTotalInputPoints();
    output->resize(total);

    OutVecCoord* outVecCoord = outPos[0];
    unsigned int size = 0;
    for( unsigned int i = 0; i < inPos.size() ; i++)
    {
        In* current = this->fromModels[i];
        const InVecCoord* currentVecCoord = inPos[i];
        iterMap = _indices.find( current );
        if ( iterMap != _indices.end() )
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

template < class BasicMapping>
void SubsetMultiMapping<BasicMapping>::applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const InVecDeriv*>& inDeriv)
{
    Out* output = this->toModels[0];
    unsigned int total = computeTotalInputPoints();
    output->resize(total);

    typename std::map<const In* , IndexArray >::iterator iterMap;
    OutVecDeriv* outVecDeriv = outDeriv[0];
    unsigned int size = 0;

    for( unsigned int i = 0; i < inDeriv.size() ; i++)
    {
        In* current = this->fromModels[i];
        const InVecDeriv* currentVecDeriv = inDeriv[i];
        iterMap = _indices.find( current );
        if ( iterMap != _indices.end() )
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

template < class BasicMapping>
void SubsetMultiMapping<BasicMapping>::applyJT(const helper::vector<InVecDeriv*>& outDeriv , const helper::vector<const OutVecDeriv*>& inDeriv )
{
    typename std::map<const In* , IndexArray >::iterator iterMap;
    const OutVecDeriv* mappedVecDeriv = inDeriv[0];

    unsigned int size = 0;
    for( unsigned int i = 0; i < outDeriv.size() ; i++)
    {
        In* current = this->fromModels[i];
        InVecDeriv* currentVecDeriv = outDeriv[i];
        iterMap = _indices.find( current );
        if ( iterMap != _indices.end() )
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
