/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_SparseGridRamificationTopology_H
#define SOFA_COMPONENT_TOPOLOGY_SparseGridRamificationTopology_H
#include "config.h"

#include <SofaBaseTopology/SparseGridTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{

/// a SparseGridTopology where each resulting cube contains only one independant connexe component (nodes can be multiplied by using virtual nodes)
class SOFA_NON_UNIFORM_FEM_API SparseGridRamificationTopology : public SparseGridTopology
{
public:
    SOFA_CLASS(SparseGridRamificationTopology,SparseGridTopology);
protected:
    SparseGridRamificationTopology(bool _isVirtual=false);
    virtual ~SparseGridRamificationTopology();
public:
    virtual void init();

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findCube(const Vector3 &pos, SReal &fx, SReal &fy, SReal &fz);
// 				virtual int findCube(const Vector3 &pos);

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findNearestCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz);
// 				virtual int findNearestCube(const Vector3& pos);


    /// one per connexion, in order to compute findCube by beginning by the finnest and by going up until the coarsest parent
    void findCoarsestParents();


    /// when linking similar particules between neighbors, propagate changes to all the sames particles
    void changeIndices(unsigned oldidx,unsigned newidx);


    /// surcharge of functions defined in SparseGridTopology
    virtual void buildAsFinest();
    virtual void buildFromFiner();
    virtual void buildVirtualFinerLevels();

    /// find the connexion graph between the finest hexahedra
    void findConnexionsAtFinestLevel();
    /// Once the finest connectivity is computed, some nodes can be dobled
    void buildRamifiedFinestLevel();
    /// do 2 neighbors cubes share triangles ?
    bool sharingTriangle(helper::io::Mesh* mesh, int cubeIdx, int neighborIdx, unsigned where);

    /// debug printings
    void printNeighborhood();
    void printNeighbors();
    void printNbConnexions();
    void printParents();
    void printHexaIdx();

    // just to remember
    enum {UP,DOWN,RIGHT,LEFT,BEFORE,BEHIND,NUM_CONNECTED_NODES};

    // Does the connectivity test have to be done at the finest level? (more precise but slow)
    Data<bool> _finestConnectivity;


    /// a connexion corresponds to a connexe component in each regular hexa (each non-void hexa has at less one connexion)
    struct Connexion
    {
        Connexion():_parent(NULL), _coarsestParent(0), _hexaIdx(0), _nonRamifiedHexaIdx(0), _tmp(0) {};

        helper::fixed_array< std::set<Connexion*>,NUM_CONNECTED_NODES >	_neighbors;	// the connexion graph at a given level (it can have several neighbors in each direction)

        typedef std::pair<unsigned,Connexion*> Children; // the unsigned indicates the fine place 0->7 in the coarse element
        std::list<Children> _children;	// the hierarchical graph to finer level
        Connexion* _parent;	// the hierarchical graph to coarser level

        unsigned int _coarsestParent; //in order to compute findCube by beginning by the finnest, by going up and give the coarsest parent

        unsigned int _hexaIdx; // idx of the corresponding hexa in the resulting Topology::seqHexahedra
        unsigned int _nonRamifiedHexaIdx; // idx of the corresponding hexa in the initial, regular list SparseGrid::hexahedra

        int _tmp; // warning: useful to several algos (as a temporary variable) but it is not an identification number

        /// each similar connexion will have a number (saved in _tmp), this number must be given to all connected connexions)
        void propagateConnexionNumberToNeighbors( int connexionNumber, const helper::vector<Connexion*>& allFineConnexions )
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

    helper::vector<helper::vector<Connexion*> > _connexions; // for each initial, regular SparseGrid::hexa -> a list of independant connexion


    std::map<int, std::pair<helper::vector<Connexion*>,int> > _mapHexa_Connexion; // a hexa idx -> the corresponding connexion

    bool intersectionSegmentTriangle(Vector3 s0, Vector3 s1, Vector3 t0, Vector3 t1, Vector3 t2);

public :

    helper::vector<helper::vector<Connexion*> >* getConnexions() {return &_connexions;}


    typedef std::vector<helper::fixed_array<helper::vector<int>,8> > HierarchicalCubeMapRamification; ///< a cube indice -> corresponding child indices (possible more than 8 for Ramification)
    HierarchicalCubeMapRamification _hierarchicalCubeMapRamification;


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
