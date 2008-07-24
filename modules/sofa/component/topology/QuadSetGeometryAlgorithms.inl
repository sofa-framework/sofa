/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_INL

#include <sofa/component/topology/QuadSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl> // PointSetTopologyLoader

#include <algorithm>
#include <functional>

namespace sofa
{
namespace component
{
namespace topology
{

using namespace sofa::defaulttype;

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
QuadSetTopology<DataTypes>::QuadSetTopology(MechanicalObject<DataTypes> *obj)
    : EdgeSetTopology<DataTypes>( obj)
{
}

template<class DataTypes>
void QuadSetTopology<DataTypes>::createComponents()
{
    this->m_topologyContainer = new QuadSetTopologyContainer(this);
    this->m_topologyModifier= new QuadSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new QuadSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new QuadSetGeometryAlgorithms<DataTypes>(this);
}

template<class DataTypes>
void QuadSetTopology<DataTypes>::init()
{
    EdgeSetTopology<DataTypes>::init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetTopologyAlgorithms////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
QuadSetTopologyAlgorithms< DataTypes >::QuadSetTopologyAlgorithms(core::componentmodel::topology::BaseTopology *top)
    : EdgeSetTopologyAlgorithms<DataTypes>(top)
{ }

template<class DataTypes>
void QuadSetTopologyAlgorithms< DataTypes >::removeQuads(sofa::helper::vector< unsigned int >& quads,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyModifier< DataTypes >* modifier = topology->getQuadSetTopologyModifier();

    /// add the topological changes in the queue
    modifier->removeQuadsWarning(quads);
    // inform other objects that the quads are going to be removed
    topology->propagateTopologicalChanges();
    // now destroy the old quads.

    modifier->removeQuadsProcess( quads, removeIsolatedEdges, removeIsolatedPoints);

    topology->getQuadSetTopologyContainer()->checkTopology();
}

template<class DataTypes>
void QuadSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeQuads(items, true, true);
}

template<class DataTypes>
void QuadSetTopologyAlgorithms< DataTypes >::writeMSH(const char *filename)
{
    getQuadSetTopology()->getQuadSetTopologyModifier()->writeMSHfile(filename);
}

template<class DataTypes>
void  QuadSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyModifier< DataTypes >* modifier = topology->getQuadSetTopologyModifier();

    /// add the topological changes in the queue
    modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();
    // now renumber the points
    modifier->renumberPointsProcess(index, inv_index);

    topology->getQuadSetTopologyContainer()->checkTopology();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template< class DataTypes>
QuadSetGeometryAlgorithms< DataTypes >::QuadSetGeometryAlgorithms(core::componentmodel::topology::BaseTopology *top)
    : EdgeSetGeometryAlgorithms<DataTypes>(top)
{ }

template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeQuadArea( const unsigned int i) const
{
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer * container = topology->getQuadSetTopologyContainer();

    const Quad &t = container->getQuad(i);
    const VecCoord& p = *(topology->getDOF()->getX());
    Real area = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
            + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    return area;
}

template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeRestQuadArea( const unsigned int i) const
{
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer * container = topology->getQuadSetTopologyContainer();

    const Quad &t = container->getQuad(i);
    const VecCoord& p = *(topology->getDOF()->getX0());
    Real area = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
            + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    return area;
}

template<class DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::computeQuadArea( BasicArrayInterface<Real> &ai) const
{
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer * container = topology->getQuadSetTopologyContainer();

    //const sofa::helper::vector<Quad> &ta=container->getQuadArray();
    unsigned int nb_quads = container->getNumberOfQuads();
    const typename DataTypes::VecCoord& p = *(topology->getDOF()->getX());

    for(unsigned int i=0; i<nb_quads; ++i)
    {
        // ta.size()
        const Quad &t = container->getQuad(i);  //ta[i];
        ai[i] = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
                + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    }
}

// Computes the normal vector of a quad indexed by ind_q (not normed)
template<class DataTypes>
Vec<3,double> QuadSetGeometryAlgorithms< DataTypes >::computeQuadNormal(const unsigned int ind_q)
{
    // HYP :  The quad indexed by ind_q is planar
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer * container = topology->getQuadSetTopologyContainer();

    const Quad &q = container->getQuad(ind_q);
    const typename DataTypes::VecCoord& vect_c = *(topology->getDOF()->getX());

    const typename DataTypes::Coord& c0=vect_c[q[0]];
    const typename DataTypes::Coord& c1=vect_c[q[1]];
    const typename DataTypes::Coord& c2=vect_c[q[2]];
    //const typename DataTypes::Coord& c3=vect_c[q[3]];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    //Vec<3,Real> p3;
    //p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    Vec<3,Real> normal_q=(p1-p0).cross( p2-p0);

    return ((Vec<3,double>) normal_q);
}


// Test if a quad indexed by ind_quad (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
template<class DataTypes>
bool QuadSetGeometryAlgorithms< DataTypes >::is_quad_in_plane(const unsigned int ind_q,
        const unsigned int ind_p,
        const Vec<3,Real>&plane_vect)
{
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer * container = topology->getQuadSetTopologyContainer();

    const Quad &q = container->getQuad(ind_q);

    // HYP : ind_p==q[0] or ind_q==t[1] or ind_q==t[2] or ind_q==q[3]

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    unsigned int ind_1;
    unsigned int ind_2;
    unsigned int ind_3;

    if(ind_p==q[0])
    {
        ind_1=q[1];
        ind_2=q[2];
        ind_3=q[3];
    }
    else if(ind_p==q[1])
    {
        ind_1=q[2];
        ind_2=q[3];
        ind_3=q[0];
    }
    else if(ind_p==q[2])
    {
        ind_1=q[3];
        ind_2=q[0];
        ind_3=q[1];
    }
    else
    {
        // ind_p==q[3]
        ind_1=q[0];
        ind_2=q[1];
        ind_3=q[2];
    }

    const typename DataTypes::Coord& c0 = vect_c[ind_p];
    const typename DataTypes::Coord& c1 = vect_c[ind_1];
    const typename DataTypes::Coord& c2 = vect_c[ind_2];
    const typename DataTypes::Coord& c3 = vect_c[ind_3];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    Vec<3,Real> p3;
    p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    return((p1-p0)*( plane_vect)>=0.0 && (p2-p0)*( plane_vect)>=0.0 && (p3-p0)*( plane_vect)>=0.0);
}

/// Cross product for 3-elements vectors.
template< class Real>
Real areaProduct(const Vec<3,Real>& a, const Vec<3,Real>& b)
{
    return Vec<3,Real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x()).norm();
}

/// area from 2-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b )
{
    return a[0]*b[1] - a[1]*b[0];
}

/// area for 1-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  )
{
    //	assert(false);
    return (Real)0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
class QuadSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    QuadSetTopologyLoader(QuadSetTopologyModifier<DataTypes> *tm)
        : PointSetTopologyLoader<DataTypes>(),
          tstm(tm)
    { }

    virtual void addQuad(int p1, int p2, int p3, int p4)
    {
        tstm->addQuad(Quad(helper::make_array<unsigned int>((unsigned int)p1,(unsigned int)p2,(unsigned int) p3,(unsigned int) p4)));
    }

public:
    VecCoord pointArray;
    QuadSetTopologyModifier<DataTypes> *tstm;
};

template<class DataTypes>
bool QuadSetTopologyModifier<DataTypes>::load(const char *filename)
{
    QuadSetTopologyLoader<DataTypes> loader(this);
    if(!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::writeMSHfile(const char *filename)
{
    std::ofstream myfile;
    myfile.open (filename);

    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer *container = topology->getQuadSetTopologyContainer();

    //PointSetTopology< Vec3Types >* psp = dynamic_cast< PointSetTopology< Vec3Types >* >( topology );
    //PointSetTopologyContainer * c_psp = static_cast< PointSetTopologyContainer* >(psp->getTopologyContainer());

    Vec3Types::VecCoord &p = *((dynamic_cast< PointSetTopology< Vec3Types >* > (topology))->getDOF()->getX());

    myfile << "$NOD\n";
    myfile << p.size() <<"\n";

    for(unsigned int i=0; i<p.size(); ++i)
    {
        double x = (double) p[i][0];
        double y = (double) p[i][1];
        double z = (double) p[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    const sofa::helper::vector<Quad>& qa = container->getQuadArray();

    myfile << qa.size() <<"\n";

    for(unsigned int i=0; i<qa.size(); ++i)
    {
        myfile << i+1 << " 3 1 1 4 " << qa[i][0]+1 << " " << qa[i][1]+1 << " " << qa[i][2]+1 << " " << qa[i][3]+1 << "\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::addQuad(Quad t)
{
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();

#ifndef NDEBUG
    // check if the 4 vertices are different
    if((t[0]==t[1]) || (t[0]==t[2]) || (t[0]==t[3])
       || (t[1]==t[2]) || (t[1]==t[3]) || (t[2]==t[3]))
    {
        cout << "Error: [QuadSetTopologyModifier::addQuad] : invalid quad: "
                << t[0] << ", " << t[1] << ", " << t[2] <<  ", " << t[3] <<  endl;

        return;
    }

    // check if there already exists a quad with the same indices
    // Important: getEdgeIndex creates the quad vertex shell array
    if(container->hasQuadVertexShell())
    {
        if(container->getQuadIndex(t[0],t[1],t[2],t[3]) != -1)
        {
            cout << "Error: [QuadSetTopologyModifier::addQuad] : Quad "
                    << t[0] << ", " << t[1] << ", " << t[2] <<  ", " << t[3] << " already exists." << endl;
            return;
        }
    }
#endif

    const unsigned int quadIndex = container->m_quad.size();

    if(container->hasQuadVertexShell())
    {
        for(unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->getQuadVertexShellForModification( t[j] );
            shell.push_back( quadIndex );
        }
    }

    if(container->hasEdges())
    {
        for(unsigned int j=0; j<4; ++j)
        {
            int edgeIndex = container->getEdgeIndex(t[(j+1)%4], t[(j+2)%4]);

            if(edgeIndex == -1)
            {
                // first create the edges
                sofa::helper::vector< Edge > v(1);
                Edge e1 (t[(j+1)%4], t[(j+2)%4]);
                v[0] = e1;

                addEdgesProcess((const sofa::helper::vector< Edge > &) v);

                edgeIndex = container->getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
                sofa::helper::vector< unsigned int > edgeIndexList;
                edgeIndexList.push_back((unsigned int) edgeIndex);
                this->addEdgesWarning( v.size(), v, edgeIndexList);
            }

            if(container->hasQuadEdges())
            {
                container->m_quadEdge.resize(quadIndex+1);
                container->m_quadEdge[quadIndex][j]= edgeIndex;
            }

            if(container->hasQuadEdgeShell())
            {
                sofa::helper::vector< unsigned int > &shell = container->m_quadEdgeShell[container->m_quadEdge[quadIndex][j]];
                shell.push_back( quadIndex );
            }
        }
    }

    container->m_quad.push_back(t);
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::addQuadsProcess(const sofa::helper::vector< Quad > &quads)
{
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();
    container->m_quad.reserve(container->m_quad.size() + quads.size());

    for(unsigned int i=0; i<quads.size(); ++i)
    {
        addQuad(quads[i]);
    }
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::addQuadsWarning(const unsigned int nQuads,
        const sofa::helper::vector< Quad >& quadsList,
        const sofa::helper::vector< unsigned int >& quadsIndexList)
{
    // Warning that quads just got created
    QuadsAdded *e = new QuadsAdded(nQuads, quadsList, quadsIndexList);
    this->addTopologyChange(e);
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::addQuadsWarning(const unsigned int nQuads,
        const sofa::helper::vector< Quad >& quadsList,
        const sofa::helper::vector< unsigned int >& quadsIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that quads just got created
    QuadsAdded *e=new QuadsAdded(nQuads, quadsList,quadsIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::removeQuadsWarning( sofa::helper::vector<unsigned int> &quads)
{
    /// sort vertices to remove in a descendent order
    std::sort( quads.begin(), quads.end(), std::greater<unsigned int>() );

    // Warning that these quads will be deleted
    QuadsRemoved *e=new QuadsRemoved(quads);
    this->addTopologyChange(e);
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::removeQuadsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer *container = topology->getQuadSetTopologyContainer();

    if(!container->hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyModifier::removeQuadsProcess] quad array is empty." << endl;
#endif
        container->createQuadSetArray();
    }

    if(container->hasEdges() && removeIsolatedEdges)
    {
        if(!container->hasQuadEdges())
            container->createQuadEdgeArray();

        if(!container->hasQuadEdgeShell())
            container->createQuadEdgeShellArray();
    }

    if(removeIsolatedPoints)
    {
        if(!container->hasQuadVertexShell())
            container->createQuadVertexShellArray();
    }

    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    for(unsigned int i = 0; i<indices.size(); ++i)
    {
        const unsigned int lastQuad = container->m_quad.size() - 1;
        Quad &t = container->m_quad[ indices[i] ];
        Quad &q = container->m_quad[ lastQuad ];

        // first check that the quad vertex shell array has been initialized
        if(container->hasQuadVertexShell())
        {
            for(unsigned int v=0; v<4; ++v)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_quadVertexShell[ t[v] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedPoints) && shell.empty())
                    vertexToBeRemoved.push_back(t[v]);
            }
        }

        /** first check that the quad edge shell array has been initialized */
        if(container->hasQuadEdgeShell())
        {
            for(unsigned int e=0; e<4; ++e)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][e]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedEdges) && shell.empty())
                    edgeToBeRemoved.push_back(container->m_quadEdge[indices[i]][e]);
            }
        }

        if(indices[i] < lastQuad)
        {
            // now updates the shell information of the quad at the end of the array
            if(container->hasQuadVertexShell())
            {
                for(unsigned int v=0; v<4; ++v)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_quadVertexShell[ q[v] ];
                    replace(shell.begin(), shell.end(), lastQuad, indices[i]);
                }
            }

            if(container->hasQuadEdgeShell())
            {
                for(unsigned int e=0; e<4; ++e)
                {
                    sofa::helper::vector< unsigned int > &shell =  container->m_quadEdgeShell[ container->m_quadEdge[lastQuad][e]];
                    replace(shell.begin(), shell.end(), lastQuad, indices[i]);
                }
            }
        }

        // removes the quadEdges from the quadEdgesArray
        if(container->hasQuadEdges())
        {
            container->m_quadEdge[ indices[i] ] = container->m_quadEdge[ lastQuad ]; // overwriting with last valid value.
            container->m_quadEdge.resize( lastQuad ); // resizing to erase multiple occurence of the quad.
        }

        // removes the quad from the quadArray
        container->m_quad[ indices[i] ] = container->m_quad[ lastQuad ]; // overwriting with last valid value.
        container->m_quad.resize( lastQuad ); // resizing to erase multiple occurence of the quad.
    }

    if(!edgeToBeRemoved.empty())
    {
        /// warn that edges will be deleted
        this->removeEdgesWarning(edgeToBeRemoved);
        topology->propagateTopologicalChanges();
        /// actually remove edges without looking for isolated vertices
        this->removeEdgesProcess(edgeToBeRemoved,false);
    }

    if(!vertexToBeRemoved.empty())
    {
        this->removePointsWarning(vertexToBeRemoved);
        /// propagate to all components
        topology->propagateTopologicalChanges();
        this->removePointsProcess(vertexToBeRemoved);
    }
}

template<class DataTypes >
void QuadSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints, const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if edges exist, otherwise call the PointSet method
    EdgeSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, addDOF );

    // now update the local container structures.
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();

    if(container->hasQuadVertexShell())
        container->m_quadVertexShell.resize( container->m_quadVertexShell.size() + nPoints );
}

template<class DataTypes >
void QuadSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if edges exist, otherwise call the PointSet method
    EdgeSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    // now update the local container structures.
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();

    if(container->hasQuadVertexShell())
        container->m_quadVertexShell.resize( container->m_quadVertexShell.size() + nPoints );
}

template<class DataTypes >
void QuadSetTopologyModifier< DataTypes >::addNewPoint(unsigned int i, const sofa::helper::vector< double >& x)
{
    // start by calling the parent's method.
    // TODO : only if edges exist, otherwise call the PointSet method
    EdgeSetTopologyModifier< DataTypes >::addNewPoint(i,x);

    // now update the local container structures.
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();

    if(container->hasQuadVertexShell())
        container->m_quadVertexShell.resize( i+1 );
}

template<class DataTypes >
void QuadSetTopologyModifier< DataTypes >::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // now update the local container structures.
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();

    if(!container->hasEdges())
    {
        container->createEdgeSetArray();
    }

    // start by calling the parent's method.
    EdgeSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

    if(container->hasQuadEdgeShell())
        container->m_quadEdgeShell.resize( container->m_quadEdgeShell.size() + edges.size() );
}

template< class DataTypes >
void QuadSetTopologyModifier< DataTypes >::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)

    // now update the local container structures
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();

    // force the creation of the quad vertex shell array before any point is deleted
    if(!container->hasQuadVertexShell())
        container->createQuadVertexShellArray();

    unsigned int lastPoint = container->getNumberOfVertices() - 1;
    for(unsigned int i=0; i<indices.size(); ++i, --lastPoint)
    {
        // updating the quads connected to the point replacing the removed one:
        // for all quads connected to the last point

        sofa::helper::vector<unsigned int> &shell = container->m_quadVertexShell[lastPoint];
        for(unsigned int j=0; j<shell.size(); ++j)
        {
            const unsigned int q = shell[j];
            for(unsigned int k=0; k<4; ++k)
            {
                if(container->m_quad[q][k] == lastPoint)
                    container->m_quad[q][k] = indices[i];
            }
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_quadVertexShell[ indices[i] ] = container->m_quadVertexShell[ lastPoint ];
    }

    container->m_quadVertexShell.resize( container->m_quadVertexShell.size() - indices.size() );

    // call the parent's method.
    // TODO : only if edges exist, otherwise call PointSetMethod
    EdgeSetTopologyModifier< DataTypes >::removePointsProcess( indices, removeDOF );
}

template< class DataTypes >
void QuadSetTopologyModifier< DataTypes >::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    // TODO : clarify what exactly has to happen here (what if an edge is removed from an existing quad?)

    // now update the local container structures
    QuadSetTopology<DataTypes> *topology = getQuadSetTopology();
    QuadSetTopologyContainer *container = topology->getQuadSetTopologyContainer();

    if(!container->hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyModifier::removeEdgesProcess] edge array is empty." << endl;
#endif
        container->createEdgeSetArray();
    }

    if(!container->hasQuadEdgeShell())
        container->createQuadEdgeShellArray();

    if(!container->hasQuadEdges())
        container->createQuadEdgeArray();

    unsigned int edgeIndex;
    unsigned int lastEdge = container->getNumberOfEdges() - 1;
    for(unsigned int i = 0; i < indices.size(); ++i, --lastEdge)
    {
        // updating the quads connected to the edge replacing the removed one:
        // for all quads connected to the last point
        for(sofa::helper::vector<unsigned int>::iterator itt = container->m_quadEdgeShell[lastEdge].begin();
            itt != container->m_quadEdgeShell[lastEdge].end(); ++itt)
        {
            edgeIndex = container->getEdgeIndexInQuad(container->m_quadEdge[(*itt)], lastEdge);
            assert((int)edgeIndex!= -1);
            container->m_quadEdge[(*itt)][(unsigned int)edgeIndex] = indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_quadEdgeShell[ indices[i] ] = container->m_quadEdgeShell[ lastEdge ];
    }

    container->m_quadEdgeShell.resize( container->m_quadEdgeShell.size() - indices.size() );

    // call the parent's method.
    EdgeSetTopologyModifier< DataTypes >::removeEdgesProcess(indices, removeIsolatedItems);
}

template< class DataTypes >
void QuadSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    // now update the local container structures.
    QuadSetTopologyContainer *container = getQuadSetTopology()->getQuadSetTopologyContainer();

    if(!container->hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyModifier::renumberPointsProcess] quad array is empty." << endl;
#endif
        container->createQuadSetArray();
    }

    if(container->hasQuadVertexShell())
    {
        sofa::helper::vector< sofa::helper::vector< unsigned int > > quadVertexShell_cp = container->m_quadVertexShell;
        for(unsigned int i=0; i<index.size(); ++i)
        {
            container->m_quadVertexShell[i] = quadVertexShell_cp[ index[i] ];
        }
    }

    for(unsigned int i=0; i<container->m_quad.size(); ++i)
    {
        container->m_quad[i][0]  = inv_index[ container->m_quad[i][0]  ];
        container->m_quad[i][1]  = inv_index[ container->m_quad[i][1]  ];
        container->m_quad[i][2]  = inv_index[ container->m_quad[i][2]  ];
        container->m_quad[i][3]  = inv_index[ container->m_quad[i][3]  ];
    }

    // call the parent's method
    if(container->hasEdges())
        EdgeSetTopologyModifier< DataTypes >::renumberPointsProcess( index, inv_index, renumberDOF );
    else
        PointSetTopologyModifier< DataTypes >::renumberPointsProcess( index, inv_index, renumberDOF );
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_QUADSETTOPOLOGY_INL
