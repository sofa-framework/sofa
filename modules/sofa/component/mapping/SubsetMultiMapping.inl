#include <sofa/component/mapping/SubsetMultiMapping.h>


using namespace sofa::core;

namespace sofa
{
namespace component
{
namespace mapping
{

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
void SubsetMultiMapping<BasicMapping>::apply(const helper::vector<typename Out::VecCoord*>& OutPos, const helper::vector<const typename In::VecCoord*>& InPos )
{
    typename std::map<const In* , IndexArray >::iterator iterMap;
    Out* output = this->toModels[0];
    unsigned int total = computeTotalInputPoints();
    output->resize(total);

    OutVecCoord* outVecCoord = OutPos[0];
    unsigned int size = 0;
    for( unsigned int i = 0; i < InPos.size() ; i++)
    {
        In* current = this->fromModels[i];
        const InVecCoord* currentVecCoord = InPos[i];
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
void SubsetMultiMapping<BasicMapping>::applyJ(const helper::vector< typename Out::VecDeriv*>& OutDeriv, const helper::vector<const typename In::VecDeriv*>& InDeriv)
{
    Out* output = this->toModels[0];
    unsigned int total = computeTotalInputPoints();
    output->resize(total);

    typename std::map<const In* , IndexArray >::iterator iterMap;
    OutVecDeriv* outVecDeriv = OutDeriv[0];
    unsigned int size = 0;

    for( unsigned int i = 0; i < InDeriv.size() ; i++)
    {
        In* current = this->fromModels[i];
        const InVecDeriv* currentVecDeriv = InDeriv[i];
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
void SubsetMultiMapping<BasicMapping>::applyJT(const helper::vector<typename In::VecDeriv*>& OutDeriv , const helper::vector<const typename Out::VecDeriv*>& InDeriv )
{
    typename std::map<const In* , IndexArray >::iterator iterMap;
    const OutVecDeriv* mappedVecDeriv = InDeriv[0];

    unsigned int size = 0;
    for( unsigned int i = 0; i < OutDeriv.size() ; i++)
    {
        In* current = this->fromModels[i];
        InVecDeriv* currentVecDeriv = OutDeriv[i];
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

}
}
}
