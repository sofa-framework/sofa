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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_H

#include <sofa/component/topology/EdgeSetTopology.h>
#include <map>

namespace sofa
{
namespace component
{
namespace topology
{
// forward declarations
template <class DataTypes>
class QuadSetTopology;

class QuadSetTopologyContainer;

template <class DataTypes>
class QuadSetTopologyModifier;

template < class DataTypes >
class QuadSetTopologyAlgorithms;

template < class DataTypes >
class QuadSetGeometryAlgorithms;

template <class DataTypes>
class QuadSetTopologyLoader;

class QuadsAdded;
class QuadsRemoved;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::QuadID QuadID;
typedef BaseMeshTopology::Quad Quad;
typedef BaseMeshTopology::SeqQuads SeqQuads;
typedef BaseMeshTopology::VertexQuads VertexQuads;
typedef BaseMeshTopology::EdgeQuads EdgeQuads;
typedef BaseMeshTopology::QuadEdges QuadEdges;

template< class Real>
bool is_point_in_quad(const defaulttype::Vec<3,Real>& p, const defaulttype::Vec<3,Real>& a,
        const defaulttype::Vec<3,Real>& b, const defaulttype::Vec<3,Real>& c,
        const defaulttype::Vec<3,Real>& d);

void snapping_test_quad(double epsilon, double alpha0, double alpha1, double alpha2, double alpha3,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2, bool& is_snap_3);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<3,Real>& a, const defaulttype::Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  );

/////////////////////////////////////////////////////////
/// QuadSetTopology objects
/////////////////////////////////////////////////////////

/** Describes a topological object that consists as a set of points and quads connected these points */
template<class DataTypes>
class QuadSetTopology : public EdgeSetTopology <DataTypes>
{
public:
    QuadSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~QuadSetTopology() {}

    virtual void init();

    /** \brief Returns the QuadSetTopologyContainer object of this QuadSetTopology.
    */
    QuadSetTopologyContainer *getQuadSetTopologyContainer() const
    {
        return static_cast<QuadSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the QuadSetTopologyModifier object of this QuadSetTopology.
    */
    QuadSetTopologyModifier<DataTypes> *getQuadSetTopologyModifier() const
    {
        return static_cast<QuadSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the QuadSetTopologyAlgorithms object of this QuadSetTopology.
    */
    QuadSetTopologyAlgorithms<DataTypes> *getQuadSetTopologyAlgorithms() const
    {
        return static_cast<QuadSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the QuadSetTopologyAlgorithms object of this QuadSetTopology.
    */
    QuadSetGeometryAlgorithms<DataTypes> *getQuadSetGeometryAlgorithms() const
    {
        return static_cast<QuadSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /// BaseMeshTopology API
    /// @{

    const SeqQuads& getQuads()   { return getQuadSetTopologyContainer()->getQuadArray(); }
    /// Returns the set of edges adjacent to a given quad.
    const QuadEdges& getEdgeQuadShell(QuadID i) { return getQuadSetTopologyContainer()->getQuadEdge(i); }
    /// Returns the set of quads adjacent to a given vertex.
    const VertexQuads& getQuadVertexShell(PointID i) { return getQuadSetTopologyContainer()->getQuadVertexShell(i); }
    /// Returns the set of quads adjacent to a given edge.
    const EdgeQuads& getQuadEdgeShell(EdgeID i) { return getQuadSetTopologyContainer()->getQuadEdgeShell(i); }

    /// @}

public:
    DataPtr< QuadSetTopologyContainer > *f_m_topologyContainer;
};


/**
* A class that performs topology algorithms on an QuadSet.
*/
template < class DataTypes >
class QuadSetTopologyAlgorithms : public EdgeSetTopologyAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;

    QuadSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top);

    virtual ~QuadSetTopologyAlgorithms() {}

    QuadSetTopology< DataTypes >* getQuadSetTopology() const
    {
        return static_cast<QuadSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /** \brief Remove a set  of quads
    @param quads an array of quad indices to be removed (note that the array is not const since it needs to be sorted)
    *
    @param removeIsolatedEdges if true isolated edges are also removed
    @param removeIsolatedPoints if true isolated vertices are also removed
    *
    */
    virtual void removeQuads(sofa::helper::vector< unsigned int >& quads,
            const bool removeIsolatedEdges,
            const bool removeIsolatedPoints);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& items);

    /** \brief Generic method to write the current mesh into a msh file
    */
    virtual void writeMSH(const char *filename);

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int>& index,
            const sofa::helper::vector<unsigned int>& inv_index);
};

/**
* A class that provides geometry information on an QuadSet.
*/
template < class DataTypes >
class QuadSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    QuadSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top);

    virtual ~QuadSetGeometryAlgorithms() {}

    QuadSetTopology< DataTypes >* getQuadSetTopology() const
    {
        return static_cast<QuadSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /** \brief Computes the area of quad no i and returns it
    *
    */
    Real computeQuadArea(const unsigned int i) const;

    /** \brief Computes the quad area of all quads are store in the array interface
    *
    */
    void computeQuadArea( BasicArrayInterface<Real> &ai) const;

    /** \brief Computes the initial area  of quad no i and returns it
    *
    */
    Real computeRestQuadArea(const unsigned int i) const;

    /** \brief Computes the normal vector of a quad indexed by ind_q (not normed)
    *
    */
    defaulttype::Vec<3,double> computeQuadNormal(const unsigned int ind_q);

    /** \brief Tests if a quad indexed by ind_q (and incident to the vertex indexed by ind_p)
    * is included or not in the plane defined by (ind_p, plane_vect)
    *
    */
    bool is_quad_in_plane(const unsigned int ind_q, const unsigned int ind_p,
            const defaulttype::Vec<3,Real>& plane_vect);

};

/**
* A class that modifies the topology by adding and removing quads
*/
template<class DataTypes>
class QuadSetTopologyModifier : public EdgeSetTopologyModifier <DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    QuadSetTopologyModifier(core::componentmodel::topology::BaseTopology *top)
        : EdgeSetTopologyModifier<DataTypes>(top)
    { }

    virtual ~QuadSetTopologyModifier() {}

    QuadSetTopology< DataTypes >* getQuadSetTopology() const
    {
        return static_cast<QuadSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /** \brief Build a quad set topology from a file : also modifies the MechanicalObject
    *
    */
    virtual bool load(const char *filename);

    /** \brief Write the current mesh into a msh file
    *
    */
    virtual void writeMSHfile(const char *filename);

    /** \brief Sends a message to warn that some quads were added in this topology.
    *
    * \sa addQuadsProcess
    */
    void addQuadsWarning(const unsigned int nQuads,
            const sofa::helper::vector< Quad >& quadsList,
            const sofa::helper::vector< unsigned int >& quadsIndexList);

    /** \brief Sends a message to warn that some quads were added in this topology.
    *
    * \sa addQuadsProcess
    */
    void addQuadsWarning(const unsigned int nQuads,
            const sofa::helper::vector< Quad >& quadsList,
            const sofa::helper::vector< unsigned int >& quadsIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList)
    {
        EdgeSetTopologyModifier<DataTypes>::addEdgesWarning( nEdges, edgesList, edgesIndexList);
    }

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
    {
        EdgeSetTopologyModifier<DataTypes>::addEdgesWarning( nEdges, edgesList, edgesIndexList, ancestors, baryCoefs);
    }

    /** \brief Actually Add some quads to this topology.
    *
    * \sa addQuadsWarning
    */
    virtual void addQuadsProcess(const sofa::helper::vector< Quad > &quads);

    /** \brief Sends a message to warn that some quads are about to be deleted.
    *
    * \sa removeQuadsProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeQuadsWarning( sofa::helper::vector<unsigned int> &quads);

    /** \brief Remove a subset of  quads. Eventually remove isolated edges and vertices
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeQuadsWarning
    *
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeQuadsProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges=false,
            const bool removeIsolatedPoints=false);

    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems=false);

    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    * Important : the points are actually added to the mechanical object's state vectors iff (addDOF == true)
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints, const bool addDOF = true);

    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    * Important : the points are actually added to the mechanical object's state vectors iff (addDOF == true)
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool addDOF = true);

    /** \brief Add a new point (who has no ancestors) to this topology.
    *
    * \sa addPointsWarning
    */
    virtual void addNewPoint(unsigned int i,  const sofa::helper::vector< double >& x);

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed from the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    virtual void removePointsProcess(sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);


    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int>& index,
            const sofa::helper::vector<unsigned int>& inv_index,
            const bool renumberDOF = true);

    //protected:
    /** \brief Load a quad.
    */
    void addQuad(Quad e);

public:
    //template <class DataTypes>
    friend class QuadSetTopologyLoader<DataTypes>;

};


/** Object that stores a set of quads and provides access
to each quad and its edges and vertices */
class QuadSetTopologyContainer : public EdgeSetTopologyContainer
{
    template< typename DataTypes >
    friend class QuadSetTopologyModifier;
public:
    QuadSetTopologyContainer(core::componentmodel::topology::BaseTopology *top=NULL);

    QuadSetTopologyContainer(core::componentmodel::topology::BaseTopology *top,
            const sofa::helper::vector< Quad >& quads );

    virtual ~QuadSetTopologyContainer() {}

    template< typename DataTypes >
    QuadSetTopology< DataTypes >* getQuadSetTopology() const
    {
        return static_cast<QuadSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /** \brief Returns the Quad array.
    *
    */
    const sofa::helper::vector<Quad> &getQuadArray();

    /** \brief Returns the Quad Vertex Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getQuadVertexShellArray();

    /** \brief Returns the QuadEdges array (ie provide the 4 edge indices for each quad)
    *
    */
    const sofa::helper::vector< QuadEdges > &getQuadEdgeArray() ;

    /** \brief Returns the Quad Edge Shells array (ie provides the quads adjacent to each edge)
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getQuadEdgeShellArray() ;

    /** \brief Returns the ith Quad.
    *
    */
    const Quad &getQuad(const unsigned int i);

    /** \brief Returns the number of quads in this topology.
    *
    */
    unsigned int getNumberOfQuads() ;

    /** \brief Returns the set of quads adjacent to a given vertex.
    *
    */
    const sofa::helper::vector< unsigned int > &getQuadVertexShell(const unsigned int i) ;


    /** \brief Returns the 4 edges adjacent to a given quad.
    *
    */
    const QuadEdges &getQuadEdge(const unsigned int i) ;


    /** \brief Returns the set of quads adjacent to a given edge.
    *
    */
    const sofa::helper::vector< unsigned int > &getQuadEdgeShell(const unsigned int i) ;


    /** Returns the indices of a quad given four vertex indices : returns -1 if none */
    int getQuadIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3, const unsigned int v4);

    /** returns the index (either 0, 1, 2, 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInQuad(Quad &t,unsigned int vertexIndex) const;

    /** returns the index (either 0, 1, 2, 3) of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInQuad(QuadEdges &t,unsigned int edheIndex) const;

    /** \brief Checks if the Quad Set Topology is coherent
    *
    * Check if the Quad and the Quad Shell arrays are coherent
    */
    virtual bool checkTopology() const;

    inline friend std::ostream& operator<< (std::ostream& out, const QuadSetTopologyContainer& t)
    {
        out << t.m_quad.size() << " " << t.m_quad << " "
            << t.m_quadEdge.size() << " " << t.m_quadEdge << " "
            << t.m_quadVertexShell.size();
        for (unsigned int i=0; i<t.m_quadVertexShell.size(); i++)
        {
            out << " " << t.m_quadVertexShell[i].size();
            out << " " <<t.m_quadVertexShell[i] ;
        }
        out  << " " << t.m_quadEdgeShell.size();
        for (unsigned int i=0; i<t.m_quadEdgeShell.size(); i++)
        {
            out  << " " << t.m_quadEdgeShell[i].size();
            out  << " " << t.m_quadEdgeShell[i];
        }

        return out;
    }

    /// Needed to be compliant with Datas.
    inline friend std::istream& operator>>(std::istream& in, QuadSetTopologyContainer& t)
    {
        unsigned int s;
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            Quad T; in >> T;
            t.m_quad.push_back(T);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            QuadEdges T; in >> T;
            t.m_quadEdge.push_back(T);
        }

        unsigned int sub;
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> sub;
            sofa::helper::vector< unsigned int > v;
            for (unsigned int j=0; j<sub; j++)
            {
                unsigned int value;
                in >> value;
                v.push_back(value);
            }
            t.m_quadVertexShell.push_back(v);
        }

        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> sub;
            sofa::helper::vector< unsigned int > v;
            for (unsigned int j=0; j<sub; j++)
            {
                unsigned int value;
                in >> value;
                v.push_back(value);
            }
            t.m_quadEdgeShell.push_back(v);
        }

        return in;
    }

protected:
    /** \brief Creates the QuadSet array.
    *
    * This function must be implemented by derived classes to create a list of quads from a set of tetrahedra for instance
    */
    virtual void createQuadSetArray();

    /** \brief Creates the EdgeSet array.
    *
    * Create the set of edges when needed.
    */
    virtual void createEdgeSetArray();

    bool hasQuads() const;

    bool hasQuadEdges() const;

    bool hasQuadVertexShell() const;

    bool hasQuadEdgeShell() const;

    void clearQuads();

    void clearQuadEdges();

    void clearQuadVertexShell();

    void clearQuadEdgeShell();

private:
    /** \brief Creates the array of edge indices for each quad
    *
    * This function is only called if the QuadEdge array is required.
    * m_quadEdge[i] contains the 4 indices of the 4 edges opposite to the ith vertex
    */
    void createQuadEdgeArray();

    /** \brief Creates the Quad Vertex Shell Array
    *
    * This function is only called if the QuadVertexShell array is required.
    * m_quadVertexShell[i] contains the indices of all quads adjacent to the ith vertex
    */
    void createQuadVertexShellArray();

    /** \brief Creates the Quad Edge Shell Array
    *
    * This function is only called if the QuadVertexShell array is required.
    * m_quadEdgeShell[i] contains the indices of all quads adjacent to the ith edge
    */
    void createQuadEdgeShellArray();

    /** \brief Returns a non-const quad vertex shell given a vertex index for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getQuadVertexShellForModification(const unsigned int vertexIndex);

    /** \brief Returns a non-const quad edge shell given the index of an edge for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getQuadEdgeShellForModification(const unsigned int edgeIndex);

protected:
    /// provides the set of quads
    sofa::helper::vector<Quad> m_quad;
    /// provides the 4 edges in each quad
    sofa::helper::vector<QuadEdges> m_quadEdge;
    /// for each vertex provides the set of quads adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadVertexShell;
    /// for each edge provides the set of quads adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadEdgeShell;
};

/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////

/** indicates that some quads were added */
class QuadsAdded : public core::componentmodel::topology::TopologyChange
{
public:
    QuadsAdded(const unsigned int nT)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::QUADSADDED),
          nQuads(nT)
    { }

    QuadsAdded(const unsigned int nT,
            const sofa::helper::vector< Quad >& _quadArray,
            const sofa::helper::vector< unsigned int >& quadsIndex)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex)
    { }

    QuadsAdded(const unsigned int nT,
            const sofa::helper::vector< Quad >& _quadArray,
            const sofa::helper::vector< unsigned int >& quadsIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }

    unsigned int getNbAddedQuads() const
    {
        return nQuads;
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return quadIndexArray;
    }

    const Quad &getQuad(const unsigned int i)
    {
        return quadArray[i];
    }

public:
    unsigned int nQuads;
    sofa::helper::vector< Quad > quadArray;
    sofa::helper::vector< unsigned int > quadIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
};

/** indicates that some quads are about to be removed */
class QuadsRemoved : public core::componentmodel::topology::TopologyChange
{
public:
    QuadsRemoved(const sofa::helper::vector<unsigned int> _qArray)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::QUADSREMOVED),
          removedQuadsArray(_qArray)
    { }

    unsigned int getNbRemovedQuads() const
    {
        return removedQuadsArray.size();
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedQuadsArray;
    }

    unsigned int &getQuadIndices(const unsigned int i)
    {
        return removedQuadsArray[i];
    }

protected:
    sofa::helper::vector<unsigned int> removedQuadsArray;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
