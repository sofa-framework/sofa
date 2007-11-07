#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_INL

#include <sofa/component/topology/HexahedronSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/simulation/tree/PropagateEventVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////HexahedronSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


const unsigned int hexahedronEdgeArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};

template<class DataTypes>
class HexahedronSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;
    HexahedronSetTopologyModifier<DataTypes> *tstm;

    HexahedronSetTopologyLoader(HexahedronSetTopologyModifier<DataTypes> *tm) :PointSetTopologyLoader<DataTypes>(), tstm(tm)
    {
    }

    virtual void addHexahedron(int p1, int p2, int p3,int p4, int p5, int p6, int p7,int p8)
    {
        tstm->addHexahedron(Hexahedron(helper::make_array<unsigned int>((unsigned int)p1,(unsigned int)p2,(unsigned int) p3,(unsigned int) p4, (unsigned int)p5,(unsigned int)p6,(unsigned int) p7,(unsigned int) p8)));
    }
};
template<class DataTypes>
void HexahedronSetTopologyModifier<DataTypes>::addHexahedron(Hexahedron t)
{

    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    container->m_hexahedron.push_back(t);

}
template<class DataTypes>
bool HexahedronSetTopologyModifier<DataTypes>::load(const char *filename)
{

    HexahedronSetTopologyLoader<DataTypes> loader(this);
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}

template<class DataTypes>
void HexahedronSetTopologyModifier<DataTypes>::addHexahedraProcess(const sofa::helper::vector< Hexahedron > &hexahedra)
{
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    if (container->m_hexahedron.size()>0)
    {
        unsigned int hexahedronIndex;
        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getHexahedronVertexShellArray();
        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tesa=container->getHexahedronEdgeShellArray();
        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &ttsa=container->getHexahedronQuadShellArray();

        unsigned int j;


        for (unsigned int i = 0; i < hexahedra.size(); ++i)
        {
            const Hexahedron &t = hexahedra[i];
            // check if the 8 vertices are different
            assert(t[0]!=t[1]); assert(t[0]!=t[2]); assert(t[0]!=t[3]); assert(t[0]!=t[4]); assert(t[0]!=t[5]); assert(t[0]!=t[6]); assert(t[0]!=t[7]);
            assert(t[1]!=t[2]); assert(t[1]!=t[3]); assert(t[1]!=t[4]); assert(t[1]!=t[5]); assert(t[1]!=t[6]); assert(t[1]!=t[7]);
            assert(t[2]!=t[3]); assert(t[2]!=t[4]); assert(t[2]!=t[5]); assert(t[2]!=t[6]); assert(t[2]!=t[7]);
            assert(t[3]!=t[4]); assert(t[3]!=t[5]); assert(t[3]!=t[6]); assert(t[3]!=t[7]);
            assert(t[4]!=t[5]); assert(t[4]!=t[6]); assert(t[4]!=t[7]);
            assert(t[5]!=t[6]); assert(t[5]!=t[7]);
            assert(t[6]!=t[7]);

            // check if there already exists a hexahedron with the same indices
            assert(container->getHexahedronIndex(t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7])== -1);
            container->m_hexahedron.push_back(t);
            hexahedronIndex=container->m_hexahedron.size() - 1 ;
            if (tvsa.size()>0)
            {
                container->getHexahedronVertexShellForModification( t[0] ).push_back( hexahedronIndex );
                container->getHexahedronVertexShellForModification( t[1] ).push_back( hexahedronIndex );
                container->getHexahedronVertexShellForModification( t[2] ).push_back( hexahedronIndex );
                container->getHexahedronVertexShellForModification( t[3] ).push_back( hexahedronIndex );
                container->getHexahedronVertexShellForModification( t[4] ).push_back( hexahedronIndex );
                container->getHexahedronVertexShellForModification( t[5] ).push_back( hexahedronIndex );
                container->getHexahedronVertexShellForModification( t[6] ).push_back( hexahedronIndex );
                container->getHexahedronVertexShellForModification( t[7] ).push_back( hexahedronIndex );

            }
            if (container->m_hexahedronEdge.size()>0)
            {
                int edgeIndex;
                for (j=0; j<12; ++j)
                {
                    edgeIndex=container->getEdgeIndex(hexahedronEdgeArray[j][0],
                            hexahedronEdgeArray[j][1]);
                    assert(edgeIndex!= -1);
                    container->m_hexahedronEdge[hexahedronIndex][j]= edgeIndex;
                }
            }
            if (container->m_hexahedronQuad.size()>0)
            {
                int quadIndex;

                // Quad 0 :
                quadIndex=container->getQuadIndex(t[0],t[3],t[2],t[1]);
                assert(quadIndex!= -1);
                container->m_hexahedronQuad[hexahedronIndex][0]=quadIndex;
                // Quad 1 :
                quadIndex=container->getQuadIndex(t[4],t[5],t[6],t[7]);
                assert(quadIndex!= -1);
                container->m_hexahedronQuad[hexahedronIndex][1]=quadIndex;
                // Quad 2 :
                quadIndex=container->getQuadIndex(t[0],t[1],t[5],t[4]);
                assert(quadIndex!= -1);
                container->m_hexahedronQuad[hexahedronIndex][2]=quadIndex;
                // Quad 3 :
                quadIndex=container->getQuadIndex(t[1],t[2],t[6],t[5]);
                assert(quadIndex!= -1);
                container->m_hexahedronQuad[hexahedronIndex][3]=quadIndex;
                // Quad 4 :
                quadIndex=container->getQuadIndex(t[2],t[3],t[7],t[6]);
                assert(quadIndex!= -1);
                container->m_hexahedronQuad[hexahedronIndex][4]=quadIndex;
                // Quad 5 :
                quadIndex=container->getQuadIndex(t[3],t[0],t[4],t[7]);
                assert(quadIndex!= -1);
                container->m_hexahedronQuad[hexahedronIndex][5]=quadIndex;

            }
            if (tesa.size()>0)
            {
                for (j=0; j<7; ++j)
                {
                    container->m_hexahedronEdgeShell[container->m_hexahedronEdge[hexahedronIndex][j]].push_back( hexahedronIndex );
                }
            }
            if (ttsa.size()>0)
            {
                for (j=0; j<6; ++j)
                {
                    container->m_hexahedronQuadShell[container->m_hexahedronQuad[hexahedronIndex][j]].push_back( hexahedronIndex );
                }
            }


        }
    }
}



template<class DataTypes>
void HexahedronSetTopologyModifier<DataTypes>::addHexahedraWarning(const unsigned int nHexahedra, const sofa::helper::vector< Hexahedron >& hexahedraList,
        const sofa::helper::vector< unsigned int >& hexahedraIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that hexahedra just got created
    HexahedraAdded *e=new HexahedraAdded(nHexahedra, hexahedraList,hexahedraIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}




template<class DataTypes>
void HexahedronSetTopologyModifier<DataTypes>::removeHexahedraWarning( sofa::helper::vector<unsigned int> &hexahedra )
{
    /// sort vertices to remove in a descendent order
    std::sort( hexahedra.begin(), hexahedra.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    HexahedraRemoved *e=new HexahedraRemoved(hexahedra);
    this->addTopologyChange(e);
}



template<class DataTypes>
void HexahedronSetTopologyModifier<DataTypes>::removeHexahedraProcess( const sofa::helper::vector<unsigned int> &indices,const bool )
{
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    if (container->m_hexahedron.size()>0)
    {

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            Hexahedron &t = container->m_hexahedron[ indices[i] ];
            // first check that the hexahedron vertex shell array has been initialized
            if (container->m_hexahedronVertexShell.size()>0)
            {

                sofa::helper::vector< unsigned int > &shell0 = container->m_hexahedronVertexShell[ t[0] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronVertexShell[ t[1] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronVertexShell[ t[2] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronVertexShell[ t[3] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronVertexShell[ t[4] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell4.begin(), shell4.end(), indices[i] ) !=shell4.end());
                shell4.erase( std::find( shell4.begin(), shell4.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronVertexShell[ t[5] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell5.begin(), shell5.end(), indices[i] ) !=shell5.end());
                shell5.erase( std::find( shell5.begin(), shell5.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell6 = container->m_hexahedronVertexShell[ t[6] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell6.begin(), shell6.end(), indices[i] ) !=shell6.end());
                shell6.erase( std::find( shell6.begin(), shell6.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell7 = container->m_hexahedronVertexShell[ t[7] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell7.begin(), shell7.end(), indices[i] ) !=shell7.end());
                shell7.erase( std::find( shell7.begin(), shell7.end(), indices[i] ) );

            }

            /** first check that the hexahedron edge shell array has been initialized */
            if (container->m_hexahedronEdgeShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][0]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][1]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][2]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][3]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][4]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell4.begin(), shell4.end(), indices[i] ) !=shell4.end());
                shell4.erase( std::find( shell4.begin(), shell4.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][5]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell5.begin(), shell5.end(), indices[i] ) !=shell5.end());
                shell5.erase( std::find( shell5.begin(), shell5.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell6 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][6]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell6.begin(), shell6.end(), indices[i] ) !=shell6.end());
                shell6.erase( std::find( shell6.begin(), shell6.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell7 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][7]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell7.begin(), shell7.end(), indices[i] ) !=shell7.end());
                shell7.erase( std::find( shell7.begin(), shell7.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell8 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][8]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell8.begin(), shell8.end(), indices[i] ) !=shell8.end());
                shell8.erase( std::find( shell8.begin(), shell8.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell9 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][9]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell9.begin(), shell9.end(), indices[i] ) !=shell9.end());
                shell9.erase( std::find( shell9.begin(), shell9.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell10 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][10]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell10.begin(), shell10.end(), indices[i] ) !=shell10.end());
                shell10.erase( std::find( shell10.begin(), shell10.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell11 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][11]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell11.begin(), shell11.end(), indices[i] ) !=shell11.end());
                shell11.erase( std::find( shell11.begin(), shell11.end(), indices[i] ) );

            }
            /** first check that the hexahedron quad shell array has been initialized */
            if (container->m_hexahedronQuadShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][0]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][1]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][2]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][3]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][4]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell4.begin(), shell4.end(), indices[i] ) !=shell4.end());
                shell4.erase( std::find( shell4.begin(), shell4.end(), indices[i] ) );

                sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][5]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell5.begin(), shell5.end(), indices[i] ) !=shell5.end());
                shell5.erase( std::find( shell5.begin(), shell5.end(), indices[i] ) );

            }

            // removes the hexahedron from the hexahedronArray
            container->m_hexahedron[ indices[i] ] = container->m_hexahedron[ container->m_hexahedron.size() - 1 ]; // overwriting with last valid value.

            if (container->m_hexahedronEdge.size()>0)
            {
                // removes the hexahedronEdges from the hexahedronEdgeArray
                container->m_hexahedronEdge[ indices[i] ] = container->m_hexahedronEdge[ container->m_hexahedron.size() - 1 ]; // overwriting with last valid value.
                container->m_hexahedronEdge.resize( container->m_hexahedronEdge.size() - 1 ); // resizing to erase multiple occurence of the edge.
            }

            if (container->m_hexahedronQuad.size()>0)
            {
                // removes the hexahedronQuads from the hexahedronQuadArray
                container->m_hexahedronQuad[ indices[i] ] = container->m_hexahedronQuad[ container->m_hexahedron.size() - 1 ]; // overwriting with last valid value.
                container->m_hexahedronQuad.resize( container->m_hexahedronQuad.size() - 1 ); // resizing to erase multiple occurence of the edge.
            }
            container->m_hexahedron.resize( container->m_hexahedron.size() - 1 ); // resizing to erase multiple occurence of the edge.


            // now updates the shell information of the edge formely at the end of the array
            // first check that the edge shell array has been initialized
            if ( indices[i] < container->m_hexahedron.size() )
            {
                unsigned int oldHexahedronIndex=container->m_hexahedron.size();
                t = container->m_hexahedron[ indices[i] ];
                if (container->m_hexahedronVertexShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 = container->m_hexahedronVertexShell[ t[0] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldHexahedronIndex ) !=shell0.end());
                    sofa::helper::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronVertexShell[ t[1] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldHexahedronIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronVertexShell[ t[2] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldHexahedronIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronVertexShell[ t[3] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldHexahedronIndex ) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronVertexShell[ t[4] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell4.begin(), shell4.end(), oldHexahedronIndex ) !=shell4.end());
                    it=std::find( shell4.begin(), shell4.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronVertexShell[ t[5] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell5.begin(), shell5.end(), oldHexahedronIndex ) !=shell5.end());
                    it=std::find( shell5.begin(), shell5.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell6 = container->m_hexahedronVertexShell[ t[6] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell6.begin(), shell6.end(), oldHexahedronIndex ) !=shell6.end());
                    it=std::find( shell6.begin(), shell6.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell7 = container->m_hexahedronVertexShell[ t[7] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell7.begin(), shell7.end(), oldHexahedronIndex ) !=shell7.end());
                    it=std::find( shell7.begin(), shell7.end(), oldHexahedronIndex );
                    (*it)=indices[i];


                }
                if (container->m_hexahedronEdgeShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][0]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldHexahedronIndex) !=shell0.end());
                    sofa::helper::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][1]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldHexahedronIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][2]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldHexahedronIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell3 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][3]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldHexahedronIndex ) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell4 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][4]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell4.begin(), shell4.end(), oldHexahedronIndex ) !=shell4.end());
                    it=std::find( shell4.begin(), shell4.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell5 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][5]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell5.begin(), shell5.end(), oldHexahedronIndex ) !=shell5.end());
                    it=std::find( shell5.begin(), shell5.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell6 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][6]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell6.begin(), shell6.end(), oldHexahedronIndex ) !=shell6.end());
                    it=std::find( shell6.begin(), shell6.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell7 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][7]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell7.begin(), shell7.end(), oldHexahedronIndex ) !=shell7.end());
                    it=std::find( shell7.begin(), shell7.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell8 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][8]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell8.begin(), shell8.end(), oldHexahedronIndex ) !=shell8.end());
                    it=std::find( shell8.begin(), shell8.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell9 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][9]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell9.begin(), shell9.end(), oldHexahedronIndex ) !=shell9.end());
                    it=std::find( shell9.begin(), shell9.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell10 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][10]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell10.begin(), shell10.end(), oldHexahedronIndex ) !=shell10.end());
                    it=std::find( shell10.begin(), shell10.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell11 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][11]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell11.begin(), shell11.end(), oldHexahedronIndex ) !=shell11.end());
                    it=std::find( shell11.begin(), shell11.end(), oldHexahedronIndex );
                    (*it)=indices[i];


                }
                if (container->m_hexahedronQuadShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][0]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldHexahedronIndex ) !=shell0.end());
                    sofa::helper::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][1]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldHexahedronIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][2]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldHexahedronIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell3 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][3]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldHexahedronIndex ) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell4 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][4]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell4.begin(), shell4.end(), oldHexahedronIndex ) !=shell4.end());
                    it=std::find( shell4.begin(), shell4.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell5 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][5]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell5.begin(), shell5.end(), oldHexahedronIndex ) !=shell5.end());
                    it=std::find( shell5.begin(), shell5.end(), oldHexahedronIndex );
                    (*it)=indices[i];

                }
            }
        }
    }
}



template<class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs );

    // now update the local container structures.
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_hexahedronVertexShell.resize( container->m_hexahedronVertexShell.size() + nPoints );
}


template<class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

    // now update the local container structures.
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_hexahedronEdgeShell.resize( container->m_hexahedronEdgeShell.size() + edges.size() );
}

template<class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::addQuadsProcess(const sofa::helper::vector< Quad > &quads)
{
    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::addQuadsProcess( quads );

    // now update the local container structures.
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_hexahedronQuadShell.resize( container->m_hexahedronQuadShell.size() + quads.size() );
}



template< class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::removePointsProcess( sofa::helper::vector<unsigned int> &indices)
{
    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::removePointsProcess(  indices );

    // now update the local container structures
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    int vertexIndex;

    assert(container->m_hexahedronVertexShell.size()>0);

    unsigned int lastPoint = container->m_hexahedronVertexShell.size() - 1;

    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        sofa::helper::vector<unsigned int>::iterator itt=container->m_hexahedronVertexShell[lastPoint].begin();
        for (; itt!=container->m_hexahedronVertexShell[lastPoint].end(); ++itt)
        {

            vertexIndex=container->getVertexIndexInHexahedron(container->m_hexahedron[(*itt)],lastPoint);
            assert(vertexIndex!= -1);
            container->m_hexahedron[(*itt)][(unsigned int)vertexIndex]=indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_hexahedronVertexShell[ indices[i] ] = container->m_hexahedronVertexShell[ lastPoint ];

        --lastPoint;
    }

    container->m_hexahedronVertexShell.resize( container->m_hexahedronVertexShell.size() - indices.size() );
}

template< class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,const bool )
{
    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::removeEdgesProcess(  indices );

    // now update the local container structures
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    if (container->m_hexahedronEdgeShell.size()>0)
    {
        unsigned int lastEdge = container->m_hexahedronEdgeShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            // updating the edge shell itself (change the old index for the new one)
            container->m_hexahedronEdgeShell[ indices[i] ] = container->m_hexahedronEdgeShell[ lastEdge ];

            --lastEdge;
        }

        container->m_hexahedronEdgeShell.resize( container->m_hexahedronEdgeShell.size() - indices.size() );
    }
}

template< class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::removeQuadsProcess(  const sofa::helper::vector<unsigned int> &indices,const bool )
{
    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::removeQuadsProcess( indices );

    // now update the local container structures
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    if (container->m_hexahedronQuadShell.size()>0)
    {
        unsigned int lastQuad = container->m_hexahedronQuadShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            // updating the quad shell itself (change the old index for the new one)
            container->m_hexahedronQuadShell[ indices[i] ] = container->m_hexahedronQuadShell[ lastQuad ];

            --lastQuad;
        }

        container->m_hexahedronQuadShell.resize( container->m_hexahedronQuadShell.size() - indices.size() );
    }
}


template< class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index)
{
    // start by calling the standard method
    QuadSetTopologyModifier< DataTypes >::renumberPointsProcess( index );

    // now update the local container structures.
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    sofa::helper::vector< sofa::helper::vector< unsigned int > > hexahedronVertexShell_cp = container->m_hexahedronVertexShell;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        container->m_hexahedronVertexShell[i] = hexahedronVertexShell_cp[ index[i] ];
    }

    for (unsigned int i = 0; i < container->m_hexahedron.size(); ++i)
    {
        container->m_hexahedron[i][0]  = index[ container->m_hexahedron[i][0]  ];
        container->m_hexahedron[i][1]  = index[ container->m_hexahedron[i][1]  ];
        container->m_hexahedron[i][2]  = index[ container->m_hexahedron[i][2]  ];
        container->m_hexahedron[i][3]  = index[ container->m_hexahedron[i][3]  ];
        container->m_hexahedron[i][4]  = index[ container->m_hexahedron[i][4]  ];
        container->m_hexahedron[i][5]  = index[ container->m_hexahedron[i][5]  ];
        container->m_hexahedron[i][6]  = index[ container->m_hexahedron[i][6]  ];
        container->m_hexahedron[i][7]  = index[ container->m_hexahedron[i][7]  ];

    }


}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


/// Cross product for 3-elements vectors.
template<typename real>
inline real tripleProduct(const Vec<3,real>& a, const Vec<3,real>& b,const Vec<3,real> &c)
{
    return dot(a,cross(b,c));
}

/// area from 2-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<2,real>& , const Vec<2,real>& ,const Vec<2,real> &)
{
    assert(false);
    return (real)0;
}
/// area for 1-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<1,real>& , const Vec<1,real>& ,const Vec<1,real> &)
{
    assert(false);
    return (real)0;
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronVolume( const unsigned int i) const
{
    //HexahedronSetTopology< DataTypes > *topology = dynamic_cast<HexahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    //assert (topology != 0);
    //HexahedronSetTopologyContainer * container = static_cast< HexahedronSetTopologyContainer* >(topology->getTopologyContainer());
    //const Hexahedron &t=container->getHexahedron(i);
    //const VecCoord& p = *topology->getDOF()->getX();
    Real volume=(Real)(0.0); // todo
    return volume;
}
template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeRestHexahedronVolume( const unsigned int i) const
{
    //HexahedronSetTopology< DataTypes > *topology = dynamic_cast<HexahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    //assert (topology != 0);
    //HexahedronSetTopologyContainer * container = static_cast< HexahedronSetTopologyContainer* >(topology->getTopologyContainer());
    //const Hexahedron &t=container->getHexahedron(i);
    //const VecCoord& p = *topology->getDOF()->getX0();
    Real volume=(Real)(0.0); // todo
    return volume;

}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronVolume( BasicArrayInterface<Real> &ai) const
{
    HexahedronSetTopology< DataTypes > *topology = dynamic_cast<HexahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast< HexahedronSetTopologyContainer* >(topology->getTopologyContainer());
    //const sofa::helper::vector<Hexahedron> &ta=container->getHexahedronArray();
    //const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    unsigned int i;
    for (i=0; i<container->getNumberOfHexahedra(); ++i) //ta.size();++i) {
    {
        //const Hexahedron &t=container->getHexahedron(i); //ta[i];
        ai[i]=(Real)(0.0); // todo
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////HexahedronSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void HexahedronSetTopology<DataTypes>::init()
{
}
template<class DataTypes>
HexahedronSetTopology<DataTypes>::HexahedronSetTopology(MechanicalObject<DataTypes> *obj) : PointSetTopology<DataTypes>( obj,(PointSetTopology<DataTypes> *)0)
{
    this->m_topologyContainer= new HexahedronSetTopologyContainer(this);
    this->m_topologyModifier= new HexahedronSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new HexahedronSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new HexahedronSetGeometryAlgorithms<DataTypes>(this);
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
