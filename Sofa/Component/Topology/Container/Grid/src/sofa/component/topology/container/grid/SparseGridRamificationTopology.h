/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/component/topology/container/grid/config.h>

#include <sofa/component/topology/container/grid/SparseGridTopology.h>

namespace sofa::component::topology::container::grid
{

/// a SparseGridTopology where each resulting cube contains only one independant connexe component (nodes can be multiplied by using virtual nodes)
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_GRID_API SparseGridRamificationTopology : public SparseGridTopology
{
public:
    SOFA_CLASS(SparseGridRamificationTopology,SparseGridTopology);
protected:
    SparseGridRamificationTopology(bool _isVirtual=false);
    ~SparseGridRamificationTopology() override;
public:
    void init() override;

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    Index findCube(const type::Vec3 &pos, SReal &fx, SReal &fy, SReal &fz) override;

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    Index findNearestCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz) override;

    /// one per connexion, in order to compute findCube by beginning by the finnest and by going up until the coarsest parent
    void findCoarsestParents();

    /// when linking similar particules between neighbors, propagate changes to all the sames particles
    void changeIndices(Index oldidx, Index newidx);

    /// surcharge of functions defined in SparseGridTopology
    void buildAsFinest() override;
    void buildFromFiner() override;
    void buildVirtualFinerLevels() override;

    /// find the connexion graph between the finest hexahedra
    void findConnexionsAtFinestLevel();
    /// Once the finest connectivity is computed, some nodes can be dobled
    void buildRamifiedFinestLevel();
    /// do 2 neighbors cubes share triangles ?
    bool sharingTriangle(helper::io::Mesh* mesh, Index cubeIdx, Index neighborIdx, unsigned where);

    /// debug printings
    void printNeighborhood();
    void printNeighbors();
    void printNbConnexions();
    void printParents();
    void printHexaIdx();

    // just to remember
    enum {UP,DOWN,RIGHT,LEFT,BEFORE,BEHIND,NUM_CONNECTED_NODES};

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<bool> _finestConnectivity;
    

    // Does the connectivity test have to be done at the finest level? (more precise but slow)
    Data<bool> d_finestConnectivity; ///< Test for connectivity at the finest level? (more precise but slower by testing all intersections between the model mesh and the faces between boundary cubes)


    /// a connexion corresponds to a connexe component in each regular hexa (each non-void hexa has at less one connexion)
    struct Connexion
    {
        Connexion():_parent(nullptr), _coarsestParent(0), _hexaIdx(0), _nonRamifiedHexaIdx(0), _tmp(0) {};

        type::fixed_array< std::set<Connexion*>,NUM_CONNECTED_NODES >	_neighbors;	// the connexion graph at a given level (it can have several neighbors in each direction)

        typedef std::pair<unsigned,Connexion*> Children; // the unsigned indicates the fine place 0->7 in the coarse element
        std::list<Children> _children;	// the hierarchical graph to finer level
        Connexion* _parent;	// the hierarchical graph to coarser level

        unsigned int _coarsestParent; //in order to compute findCube by beginning by the finnest, by going up and give the coarsest parent

        Index _hexaIdx; // idx of the corresponding hexa in the resulting Topology::seqHexahedra
        Index _nonRamifiedHexaIdx; // idx of the corresponding hexa in the initial, regular list SparseGrid::hexahedra

        int _tmp; // warning: useful to several algos (as a temporary variable) but it is not an identification number

        /// each similar connexion will have a number (saved in _tmp), this number must be given to all connected connexions)
        void propagateConnexionNumberToNeighbors( int connexionNumber, const type::vector<Connexion*>& allFineConnexions )
        {
            if (_tmp!=-1) return; // already in an existing connexion number

            _tmp = connexionNumber;
            for(int i=0; i<NUM_CONNECTED_NODES; ++i)
                for(std::set<Connexion*>::iterator n = _neighbors[i].begin(); n!=_neighbors[i].end(); ++n)
                    if( find( allFineConnexions.begin(),allFineConnexions.end(),*n ) != allFineConnexions.end() ) // the neighbors is in the good child of the coarse hexa
                        (*n)->propagateConnexionNumberToNeighbors(connexionNumber,allFineConnexions);
        }
    };

protected:

    type::vector<type::vector<Connexion*> > _connexions; // for each initial, regular SparseGrid::hexa -> a list of independant connexion


    std::map<int, std::pair<type::vector<Connexion*>,int> > _mapHexa_Connexion; // a hexa idx -> the corresponding connexion

    bool intersectionSegmentTriangle(type::Vec3 s0, type::Vec3 s1, type::Vec3 t0, type::Vec3 t1, type::Vec3 t2);

public :

    type::vector<type::vector<Connexion*> >* getConnexions() {return &_connexions;}


    typedef std::vector<type::fixed_array<type::vector<Index>,8> > HierarchicalCubeMapRamification; ///< a cube indice -> corresponding child indices (possible more than 8 for Ramification)
    HierarchicalCubeMapRamification _hierarchicalCubeMapRamification;


};

} // namespace sofa::component::topology::container::grid
