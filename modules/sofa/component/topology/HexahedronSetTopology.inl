#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_INL

#include <sofa/component/topology/HexahedronSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
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

    virtual void addCube(int p1, int p2, int p3,int p4, int p5, int p6, int p7,int p8)
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
void HexahedronSetTopologyModifier<DataTypes>::writeMSHfile(const char *filename)
{

    std::ofstream myfile;
    myfile.open (filename);

    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);

    PointSetTopology< Vec3Types >* psp = dynamic_cast< PointSetTopology< Vec3Types >* >( topology );
    PointSetTopologyContainer * c_psp = static_cast< PointSetTopologyContainer* >(psp->getTopologyContainer());

    sofa::helper::vector< sofa::defaulttype::Vec<3,double> > p = *psp->getDOF()->getX();

    myfile << "$NOD\n";
    myfile << c_psp->getNumberOfVertices() <<"\n";

    for (unsigned int i=0; i<c_psp->getNumberOfVertices(); ++i)
    {

        double x = (double) p[i][0];
        double y = (double) p[i][1];
        double z = (double) p[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    HexahedronSetTopologyContainer * container = static_cast< HexahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const sofa::helper::vector<Hexahedron> hea=container->getHexahedronArray();

    myfile << hea.size() <<"\n";

    for (unsigned int i=0; i<hea.size(); ++i)
    {
        myfile << i+1 << " 5 1 1 8 " << hea[i][4]+1 << " " << hea[i][5]+1 << " " << hea[i][1]+1 << " " << hea[i][0]+1
                << hea[i][7]+1 << " " << hea[i][6]+1 << " " << hea[i][2]+1 << " " << hea[i][3]+1 <<"\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();

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

                for (j=0; j<8; ++j)
                {

                    sofa::helper::vector< unsigned int > &shell = container->getHexahedronVertexShellForModification( t[j] );
                    shell.push_back( hexahedronIndex );
                    sort(shell.begin(), shell.end());
                }

            }
            if (container->m_hexahedronQuad.size()>0)
            {
                int quadIndex;

                // Quad 0 :
                quadIndex=container->getQuadIndex(t[0],t[3],t[2],t[1]);
                //assert(quadIndex!= -1);
                if(quadIndex == -1)
                {
                    // first create the quad
                    sofa::helper::vector< Quad > v;
                    Quad e1 (t[0],t[3],t[2],t[1]);
                    v.push_back(e1);

                    addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                    quadIndex=container->getQuadIndex(t[0],t[3],t[2],t[1]);
                    sofa::helper::vector< unsigned int > quadIndexList;
                    quadIndexList.push_back(quadIndex);
                    this->addQuadsWarning( v.size(), v,quadIndexList);
                }
                container->m_hexahedronQuad.resize(quadIndex+1);
                container->m_hexahedronQuad[hexahedronIndex][0]=quadIndex;
                // Quad 1 :
                quadIndex=container->getQuadIndex(t[4],t[5],t[6],t[7]);
                //assert(quadIndex!= -1);
                if(quadIndex == -1)
                {
                    // first create the quad
                    sofa::helper::vector< Quad > v;
                    Quad e1 (t[4],t[5],t[6],t[7]);
                    v.push_back(e1);

                    addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                    quadIndex=container->getQuadIndex(t[4],t[5],t[6],t[7]);
                    sofa::helper::vector< unsigned int > quadIndexList;
                    quadIndexList.push_back(quadIndex);
                    this->addQuadsWarning( v.size(), v,quadIndexList);
                }
                container->m_hexahedronQuad.resize(quadIndex+1);
                container->m_hexahedronQuad[hexahedronIndex][1]=quadIndex;
                // Quad 2 :
                quadIndex=container->getQuadIndex(t[0],t[1],t[5],t[4]);
                //assert(quadIndex!= -1);
                if(quadIndex == -1)
                {
                    // first create the quad
                    sofa::helper::vector< Quad > v;
                    Quad e1 (t[0],t[1],t[5],t[4]);
                    v.push_back(e1);

                    addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                    quadIndex=container->getQuadIndex(t[0],t[1],t[5],t[4]);
                    sofa::helper::vector< unsigned int > quadIndexList;
                    quadIndexList.push_back(quadIndex);
                    this->addQuadsWarning( v.size(), v,quadIndexList);
                }
                container->m_hexahedronQuad.resize(quadIndex+1);
                container->m_hexahedronQuad[hexahedronIndex][2]=quadIndex;
                // Quad 3 :
                quadIndex=container->getQuadIndex(t[1],t[2],t[6],t[5]);
                //assert(quadIndex!= -1);
                if(quadIndex == -1)
                {
                    // first create the quad
                    sofa::helper::vector< Quad > v;
                    Quad e1 (t[1],t[2],t[6],t[5]);
                    v.push_back(e1);

                    addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                    quadIndex=container->getQuadIndex(t[1],t[2],t[6],t[5]);
                    sofa::helper::vector< unsigned int > quadIndexList;
                    quadIndexList.push_back(quadIndex);
                    this->addQuadsWarning( v.size(), v,quadIndexList);
                }
                container->m_hexahedronQuad.resize(quadIndex+1);
                container->m_hexahedronQuad[hexahedronIndex][3]=quadIndex;
                // Quad 4 :
                quadIndex=container->getQuadIndex(t[2],t[3],t[7],t[6]);
                //assert(quadIndex!= -1);
                if(quadIndex == -1)
                {
                    // first create the quad
                    sofa::helper::vector< Quad > v;
                    Quad e1 (t[2],t[3],t[7],t[6]);
                    v.push_back(e1);

                    addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                    quadIndex=container->getQuadIndex(t[2],t[3],t[7],t[6]);
                    sofa::helper::vector< unsigned int > quadIndexList;
                    quadIndexList.push_back(quadIndex);
                    this->addQuadsWarning( v.size(), v,quadIndexList);
                }
                container->m_hexahedronQuad.resize(quadIndex+1);
                container->m_hexahedronQuad[hexahedronIndex][4]=quadIndex;
                // Quad 5 :
                quadIndex=container->getQuadIndex(t[3],t[0],t[4],t[7]);
                //assert(quadIndex!= -1);
                if(quadIndex == -1)
                {
                    // first create the quad
                    sofa::helper::vector< Quad > v;
                    Quad e1 (t[3],t[0],t[4],t[7]);
                    v.push_back(e1);

                    addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                    quadIndex=container->getQuadIndex(t[3],t[0],t[4],t[7]);
                    sofa::helper::vector< unsigned int > quadIndexList;
                    quadIndexList.push_back(quadIndex);
                    this->addQuadsWarning( v.size(), v,quadIndexList);
                }
                container->m_hexahedronQuad.resize(quadIndex+1);
                container->m_hexahedronQuad[hexahedronIndex][5]=quadIndex;

            }
            if (container->m_hexahedronEdge.size()>0)
            {
                int edgeIndex;
                for (j=0; j<12; ++j)
                {
                    edgeIndex=container->getEdgeIndex(hexahedronEdgeArray[j][0],
                            hexahedronEdgeArray[j][1]);
                    assert(edgeIndex!= -1);
                    container->m_hexahedronEdge.resize(edgeIndex+1);
                    container->m_hexahedronEdge[hexahedronIndex][j]= edgeIndex;
                }
            }

            if (tesa.size()>0)
            {
                for (j=0; j<12; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_hexahedronEdgeShell[container->m_hexahedronEdge[hexahedronIndex][j]];
                    shell.push_back( hexahedronIndex );
                    sort(shell.begin(), shell.end());
                }
            }
            if (ttsa.size()>0)
            {
                for (j=0; j<6; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_hexahedronQuadShell[container->m_hexahedronQuad[hexahedronIndex][j]];
                    shell.push_back( hexahedronIndex );
                    sort(shell.begin(), shell.end());
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
void HexahedronSetTopologyModifier<DataTypes>::removeHexahedraProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    container->getHexahedronEdgeShellArray();

    container->getHexahedronQuadShellArray();

    container->getHexahedronVertexShellArray();

    container->getQuadVertexShellArray();
    container->getQuadEdgeShellArray();

    container->getEdgeVertexShellArray();

    if (container->m_hexahedron.size()>0)
    {

        sofa::helper::vector<unsigned int> quadToBeRemoved;
        sofa::helper::vector<unsigned int> edgeToBeRemoved;
        sofa::helper::vector<unsigned int> vertexToBeRemoved;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            Hexahedron &t = container->m_hexahedron[ indices[i] ];
            // first check that the hexahedron vertex shell array has been initialized
            if (container->m_hexahedronVertexShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_hexahedronVertexShell[ t[0] ];
                shell0.erase(remove(shell0.begin(), shell0.end(), indices[i]), shell0.end());
                if ((removeIsolatedItems) && (shell0.size()==0))
                    vertexToBeRemoved.push_back(t[0]);

                sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronVertexShell[ t[1] ];
                shell1.erase(remove(shell1.begin(), shell1.end(), indices[i]), shell1.end());
                if ((removeIsolatedItems) && (shell1.size()==0))
                    vertexToBeRemoved.push_back(t[1]);

                sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronVertexShell[ t[2] ];
                shell2.erase(remove(shell2.begin(), shell2.end(), indices[i]), shell2.end());
                if ((removeIsolatedItems) && (shell2.size()==0))
                    vertexToBeRemoved.push_back(t[2]);

                sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronVertexShell[ t[3] ];
                shell3.erase(remove(shell3.begin(), shell3.end(), indices[i]), shell3.end());
                if ((removeIsolatedItems) && (shell3.size()==0))
                    vertexToBeRemoved.push_back(t[3]);

                sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronVertexShell[ t[4] ];
                shell4.erase(remove(shell4.begin(), shell4.end(), indices[i]), shell4.end());
                if ((removeIsolatedItems) && (shell4.size()==0))
                    vertexToBeRemoved.push_back(t[4]);

                sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronVertexShell[ t[5] ];
                shell5.erase(remove(shell5.begin(), shell5.end(), indices[i]), shell5.end());
                if ((removeIsolatedItems) && (shell5.size()==0))
                    vertexToBeRemoved.push_back(t[5]);

                sofa::helper::vector< unsigned int > &shell6 = container->m_hexahedronVertexShell[ t[6] ];
                shell6.erase(remove(shell6.begin(), shell6.end(), indices[i]), shell6.end());
                if ((removeIsolatedItems) && (shell6.size()==0))
                    vertexToBeRemoved.push_back(t[6]);

                sofa::helper::vector< unsigned int > &shell7 = container->m_hexahedronVertexShell[ t[7] ];
                shell7.erase(remove(shell7.begin(), shell7.end(), indices[i]), shell7.end());
                if ((removeIsolatedItems) && (shell7.size()==0))
                    vertexToBeRemoved.push_back(t[7]);

            }

            /** first check that the hexahedron edge shell array has been initialized */
            if (container->m_hexahedronEdgeShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][0]];
                shell0.erase(remove(shell0.begin(), shell0.end(), indices[i]), shell0.end());
                if ((removeIsolatedItems) && (shell0.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][0]);

                sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][1]];
                shell1.erase(remove(shell1.begin(), shell1.end(), indices[i]), shell1.end());
                if ((removeIsolatedItems) && (shell1.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][1]);

                sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][2]];
                shell2.erase(remove(shell2.begin(), shell2.end(), indices[i]), shell2.end());
                if ((removeIsolatedItems) && (shell2.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][2]);

                sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][3]];
                shell3.erase(remove(shell3.begin(), shell3.end(), indices[i]), shell3.end());
                if ((removeIsolatedItems) && (shell3.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][3]);

                sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][4]];
                shell4.erase(remove(shell4.begin(), shell4.end(), indices[i]), shell4.end());
                if ((removeIsolatedItems) && (shell4.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][4]);

                sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][5]];
                shell5.erase(remove(shell5.begin(), shell5.end(), indices[i]), shell5.end());
                if ((removeIsolatedItems) && (shell5.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][5]);

                sofa::helper::vector< unsigned int > &shell6 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][6]];
                shell6.erase(remove(shell6.begin(), shell6.end(), indices[i]), shell6.end());
                if ((removeIsolatedItems) && (shell6.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][6]);

                sofa::helper::vector< unsigned int > &shell7 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][7]];
                shell7.erase(remove(shell7.begin(), shell7.end(), indices[i]), shell7.end());
                if ((removeIsolatedItems) && (shell7.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][7]);

                sofa::helper::vector< unsigned int > &shell8 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][8]];
                shell8.erase(remove(shell8.begin(), shell8.end(),indices[i]), shell8.end());
                if ((removeIsolatedItems) && (shell8.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][8]);

                sofa::helper::vector< unsigned int > &shell9 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][9]];
                shell9.erase(remove(shell9.begin(), shell9.end(),indices[i]), shell9.end());
                if ((removeIsolatedItems) && (shell9.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][9]);

                sofa::helper::vector< unsigned int > &shell10 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][10]];
                shell10.erase(remove(shell10.begin(), shell10.end(),indices[i]), shell10.end());
                if ((removeIsolatedItems) && (shell10.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][10]);

                sofa::helper::vector< unsigned int > &shell11 = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][11]];
                shell11.erase(remove(shell11.begin(), shell11.end(),indices[i]), shell11.end());
                if ((removeIsolatedItems) && (shell11.size()==0))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][11]);

            }
            /** first check that the hexahedron quad shell array has been initialized */
            if (container->m_hexahedronQuadShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][0]];
                shell0.erase(remove(shell0.begin(), shell0.end(), indices[i]), shell0.end());
                if ((removeIsolatedItems) && (shell0.size()==0))
                    quadToBeRemoved.push_back(container->m_hexahedronQuad[indices[i]][0]);

                sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][1]];
                shell1.erase(remove(shell1.begin(), shell1.end(), indices[i]), shell1.end());
                if ((removeIsolatedItems) && (shell1.size()==0))
                    quadToBeRemoved.push_back(container->m_hexahedronQuad[indices[i]][1]);

                sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][2]];
                shell2.erase(remove(shell2.begin(), shell2.end(), indices[i]), shell2.end());
                if ((removeIsolatedItems) && (shell2.size()==0))
                    quadToBeRemoved.push_back(container->m_hexahedronQuad[indices[i]][2]);

                sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][3]];
                shell3.erase(remove(shell3.begin(), shell3.end(), indices[i]), shell3.end());
                if ((removeIsolatedItems) && (shell3.size()==0))
                    quadToBeRemoved.push_back(container->m_hexahedronQuad[indices[i]][3]);

                sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][4]];
                shell4.erase(remove(shell4.begin(), shell4.end(), indices[i]), shell4.end());
                if ((removeIsolatedItems) && (shell4.size()==0))
                    quadToBeRemoved.push_back(container->m_hexahedronQuad[indices[i]][4]);

                sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][5]];
                shell5.erase(remove(shell5.begin(), shell5.end(), indices[i]), shell5.end());
                if ((removeIsolatedItems) && (shell5.size()==0))
                    quadToBeRemoved.push_back(container->m_hexahedronQuad[indices[i]][5]);

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
                    replace(shell0.begin(), shell0.end(), oldHexahedronIndex, indices[i]);
                    sort(shell0.begin(), shell0.end());

                    sofa::helper::vector< unsigned int > &shell1 = container->m_hexahedronVertexShell[ t[1] ];
                    replace(shell1.begin(), shell1.end(), oldHexahedronIndex, indices[i]);
                    sort(shell1.begin(), shell1.end());

                    sofa::helper::vector< unsigned int > &shell2 = container->m_hexahedronVertexShell[ t[2] ];
                    replace(shell2.begin(), shell2.end(), oldHexahedronIndex, indices[i]);
                    sort(shell2.begin(), shell2.end());

                    sofa::helper::vector< unsigned int > &shell3 = container->m_hexahedronVertexShell[ t[3] ];
                    replace(shell3.begin(), shell3.end(), oldHexahedronIndex, indices[i]);
                    sort(shell3.begin(), shell3.end());

                    sofa::helper::vector< unsigned int > &shell4 = container->m_hexahedronVertexShell[ t[4] ];
                    replace(shell4.begin(), shell4.end(), oldHexahedronIndex, indices[i]);
                    sort(shell4.begin(), shell4.end());

                    sofa::helper::vector< unsigned int > &shell5 = container->m_hexahedronVertexShell[ t[5] ];
                    replace(shell5.begin(), shell5.end(), oldHexahedronIndex, indices[i]);
                    sort(shell5.begin(), shell5.end());

                    sofa::helper::vector< unsigned int > &shell6 = container->m_hexahedronVertexShell[ t[6] ];
                    replace(shell6.begin(), shell6.end(), oldHexahedronIndex, indices[i]);
                    sort(shell6.begin(), shell6.end());

                    sofa::helper::vector< unsigned int > &shell7 = container->m_hexahedronVertexShell[ t[7] ];
                    replace(shell7.begin(), shell7.end(), oldHexahedronIndex, indices[i]);
                    sort(shell7.begin(), shell7.end());
                }
                if (container->m_hexahedronEdgeShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][0]];
                    replace(shell0.begin(), shell0.end(), oldHexahedronIndex, indices[i]);
                    sort(shell0.begin(), shell0.end());

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][1]];
                    replace(shell1.begin(), shell1.end(), oldHexahedronIndex, indices[i]);
                    sort(shell1.begin(), shell1.end());

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][2]];
                    replace(shell2.begin(), shell2.end(), oldHexahedronIndex, indices[i]);
                    sort(shell2.begin(), shell2.end());

                    sofa::helper::vector< unsigned int > &shell3 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][3]];
                    replace(shell3.begin(), shell3.end(), oldHexahedronIndex, indices[i]);
                    sort(shell3.begin(), shell3.end());

                    sofa::helper::vector< unsigned int > &shell4 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][4]];
                    replace(shell4.begin(), shell4.end(), oldHexahedronIndex, indices[i]);
                    sort(shell4.begin(), shell4.end());

                    sofa::helper::vector< unsigned int > &shell5 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][5]];
                    replace(shell5.begin(), shell5.end(), oldHexahedronIndex, indices[i]);
                    sort(shell5.begin(), shell5.end());

                    sofa::helper::vector< unsigned int > &shell6 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][6]];
                    replace(shell6.begin(), shell6.end(), oldHexahedronIndex, indices[i]);
                    sort(shell6.begin(), shell6.end());

                    sofa::helper::vector< unsigned int > &shell7 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][7]];
                    replace(shell7.begin(), shell7.end(), oldHexahedronIndex, indices[i]);
                    sort(shell7.begin(), shell7.end());

                    sofa::helper::vector< unsigned int > &shell8 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][8]];
                    replace(shell8.begin(), shell8.end(), oldHexahedronIndex, indices[i]);
                    sort(shell8.begin(), shell8.end());

                    sofa::helper::vector< unsigned int > &shell9 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][9]];
                    replace(shell9.begin(), shell9.end(), oldHexahedronIndex, indices[i]);
                    sort(shell9.begin(), shell9.end());

                    sofa::helper::vector< unsigned int > &shell10 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][10]];
                    replace(shell10.begin(), shell10.end(), oldHexahedronIndex, indices[i]);
                    sort(shell10.begin(), shell10.end());

                    sofa::helper::vector< unsigned int > &shell11 =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][11]];
                    replace(shell11.begin(), shell11.end(), oldHexahedronIndex, indices[i]);
                    sort(shell11.begin(), shell11.end());
                }
                if (container->m_hexahedronQuadShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][0]];
                    replace(shell0.begin(), shell0.end(), oldHexahedronIndex, indices[i]);
                    sort(shell0.begin(), shell0.end());

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][1]];
                    replace(shell1.begin(), shell1.end(), oldHexahedronIndex, indices[i]);
                    sort(shell1.begin(), shell1.end());

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][2]];
                    replace(shell2.begin(), shell2.end(), oldHexahedronIndex, indices[i]);
                    sort(shell2.begin(), shell2.end());

                    sofa::helper::vector< unsigned int > &shell3 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][3]];
                    replace(shell3.begin(), shell3.end(), oldHexahedronIndex, indices[i]);
                    sort(shell3.begin(), shell3.end());

                    sofa::helper::vector< unsigned int > &shell4 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][4]];
                    replace(shell4.begin(), shell4.end(), oldHexahedronIndex, indices[i]);
                    sort(shell4.begin(), shell4.end());

                    sofa::helper::vector< unsigned int > &shell5 =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][5]];
                    replace(shell5.begin(), shell5.end(), oldHexahedronIndex, indices[i]);
                    sort(shell5.begin(), shell5.end());
                }
            }
        }

        if ( (quadToBeRemoved.size()>0) || (edgeToBeRemoved.size()>0) || (vertexToBeRemoved.size()>0))   //
        {

            if (quadToBeRemoved.size()>0)
            {
                /// warn that quads will be deleted
                this->removeQuadsWarning(quadToBeRemoved);
            }


            if (edgeToBeRemoved.size()>0)
            {
                /// warn that edges will be deleted
                this->removeEdgesWarning(edgeToBeRemoved);
            }

            //if (vertexToBeRemoved.size()>0){
            //	this->removePointsWarning(vertexToBeRemoved);
            //}

            /// propagate to all components

            topology->propagateTopologicalChanges();


            if (quadToBeRemoved.size()>0)
            {
                /// actually remove quads without looking for isolated vertices

                this->removeQuadsProcess(quadToBeRemoved,false,false);

            }

            if (edgeToBeRemoved.size()>0)
            {
                /// actually remove edges without looking for isolated vertices
                this->removeEdgesProcess(edgeToBeRemoved,false);
            }


            if (vertexToBeRemoved.size()>0)
            {
                this->removePointsWarning(vertexToBeRemoved);
                topology->propagateTopologicalChanges();
            }

            if (vertexToBeRemoved.size()>0)
            {
                this->removePointsProcess(vertexToBeRemoved);
            }

        }

    }
}



template<class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs, const bool addDOF)
{
    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    // now update the local container structures.
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_hexahedronVertexShell.resize( container->m_hexahedronVertexShell.size() + nPoints );
}

template<class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::addNewPoint(unsigned int i, const sofa::helper::vector< double >& x)
{
    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::addNewPoint(i,x);

    // now update the local container structures.
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_hexahedronVertexShell.resize( i+1 );
}


template<class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

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
void HexahedronSetTopologyModifier< DataTypes >::removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF)
{
    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // start by calling the standard method.

    // now update the local container structures
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    //if (container->m_hexahedron.size()>0)
    container->getHexahedronVertexShellArray();

    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::removePointsProcess(  indices, removeDOF );

    int vertexIndex;

    //assert(container->m_hexahedronVertexShell.size()>0);

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
void HexahedronSetTopologyModifier< DataTypes >::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    // now update the local container structures
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    //if (container->m_hexahedronEdge.size()>0)
    container->getHexahedronEdgeShellArray();

    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::removeEdgesProcess(  indices, removeIsolatedItems );

    if (container->m_hexahedronEdge.size()>0)
    {

        unsigned int edgeIndex;
        unsigned int lastEdge = container->m_hexahedronEdgeShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            sofa::helper::vector<unsigned int>::iterator itt=container->m_hexahedronEdgeShell[lastEdge].begin();
            for (; itt!=container->m_hexahedronEdgeShell[lastEdge].end(); ++itt)
            {

                edgeIndex=container->getEdgeIndexInHexahedron(container->m_hexahedronEdge[(*itt)],lastEdge);
                assert((int)edgeIndex!= -1);
                container->m_hexahedronEdge[(*itt)][(unsigned int)edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            container->m_hexahedronEdgeShell[ indices[i] ] = container->m_hexahedronEdgeShell[ lastEdge ];

            --lastEdge;
        }

        container->m_hexahedronEdgeShell.resize( container->m_hexahedronEdgeShell.size() - indices.size() );
    }
}

template< class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::removeQuadsProcess(  const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedEdges, const bool removeIsolatedPoints)
{
    // now update the local container structures
    HexahedronSetTopology<DataTypes> *topology = dynamic_cast<HexahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyContainer * container = static_cast<HexahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    //if (container->m_hexahedronQuad.size()>0)
    container->getHexahedronQuadShellArray();

    // start by calling the standard method.
    QuadSetTopologyModifier< DataTypes >::removeQuadsProcess( indices, removeIsolatedEdges, removeIsolatedPoints );

    if (container->m_hexahedronQuad.size()>0)
    {

        unsigned int quadIndex;
        unsigned int lastQuad = container->m_hexahedronQuadShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            sofa::helper::vector<unsigned int>::iterator itt=container->m_hexahedronQuadShell[lastQuad].begin();
            for (; itt!=container->m_hexahedronQuadShell[lastQuad].end(); ++itt)
            {

                quadIndex=container->getQuadIndexInHexahedron(container->m_hexahedronQuad[(*itt)],lastQuad);
                assert((int)quadIndex!= -1);
                container->m_hexahedronQuad[(*itt)][(unsigned int)quadIndex]=indices[i];
            }

            // updating the quad shell itself (change the old index for the new one)
            container->m_hexahedronQuadShell[ indices[i] ] = container->m_hexahedronQuadShell[ lastQuad ];

            --lastQuad;
        }

        container->m_hexahedronQuadShell.resize( container->m_hexahedronQuadShell.size() - indices.size() );
    }

}


template< class DataTypes >
void HexahedronSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index, const bool renumberDOF)
{
    // start by calling the standard method
    QuadSetTopologyModifier< DataTypes >::renumberPointsProcess( index, inv_index, renumberDOF );

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
        container->m_hexahedron[i][0]  = inv_index[ container->m_hexahedron[i][0]  ];
        container->m_hexahedron[i][1]  = inv_index[ container->m_hexahedron[i][1]  ];
        container->m_hexahedron[i][2]  = inv_index[ container->m_hexahedron[i][2]  ];
        container->m_hexahedron[i][3]  = inv_index[ container->m_hexahedron[i][3]  ];
        container->m_hexahedron[i][4]  = inv_index[ container->m_hexahedron[i][4]  ];
        container->m_hexahedron[i][5]  = inv_index[ container->m_hexahedron[i][5]  ];
        container->m_hexahedron[i][6]  = inv_index[ container->m_hexahedron[i][6]  ];
        container->m_hexahedron[i][7]  = inv_index[ container->m_hexahedron[i][7]  ];

    }


}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
void HexahedronSetTopologyAlgorithms< DataTypes >::removeHexahedra(sofa::helper::vector< unsigned int >& hexahedra)
{
    HexahedronSetTopology< DataTypes > *topology = dynamic_cast<HexahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyModifier< DataTypes >* modifier  = static_cast< HexahedronSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    /// add the topological changes in the queue

    //HexahedronSetTopologyContainer * container = static_cast< HexahedronSetTopologyContainer* >(topology->getTopologyContainer());

    modifier->removeHexahedraWarning(hexahedra);


    // inform other objects that the quads are going to be removed

    topology->propagateTopologicalChanges();

    // now destroy the old hexahedra.

    modifier->removeHexahedraProcess(  hexahedra ,true);

    assert(topology->getHexahedronSetTopologyContainer()->checkTopology());

}

template<class DataTypes>
void HexahedronSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeHexahedra(items);
}

template<class DataTypes>
void HexahedronSetTopologyAlgorithms< DataTypes >::writeMSH(const char *filename)
{

    HexahedronSetTopology< DataTypes > *topology = dynamic_cast<HexahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyModifier< DataTypes >* modifier  = static_cast< HexahedronSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);

    modifier->writeMSHfile(filename);
}

template<class DataTypes>
void  HexahedronSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index)
{

    HexahedronSetTopology< DataTypes > *topology = dynamic_cast<HexahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    HexahedronSetTopologyModifier< DataTypes >* modifier  = static_cast< HexahedronSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    /// add the topological changes in the queue
    modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();
    // now renumber the points
    modifier->renumberPointsProcess(index, inv_index);

    assert(topology->getHexahedronSetTopologyContainer()->checkTopology());
}

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
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronVolume( const unsigned int /*i*/) const
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
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeRestHexahedronVolume( const unsigned int /*i*/) const
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
HexahedronSetTopology<DataTypes>::HexahedronSetTopology(MechanicalObject<DataTypes> *obj) : QuadSetTopology<DataTypes>( obj), f_m_topologyContainer(new DataPtr< HexahedronSetTopologyContainer >(new HexahedronSetTopologyContainer(), "Hexahedron Container"))
{
    /*
    this->m_topologyContainer= new HexahedronSetTopologyContainer(this);
    this->m_topologyModifier= new HexahedronSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new HexahedronSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new HexahedronSetGeometryAlgorithms<DataTypes>(this);
    */

    this->m_topologyContainer=f_m_topologyContainer->beginEdit();
    this->m_topologyContainer->setTopology(this);
    this->m_topologyModifier= new HexahedronSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new HexahedronSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new HexahedronSetGeometryAlgorithms<DataTypes>(this);

    this->addField(f_m_topologyContainer, "hexahedroncontainer");
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
