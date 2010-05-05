/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_SIMPLETESELATEDTETRATOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_SIMPLETESELATEDTETRATOPOLOGICALMAPPING_H

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <set>

#include <sofa/core/BaseMapping.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/component/topology/EdgeData.h>
#include <sofa/component/topology/TetrahedronData.h>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::component::topology;
using namespace sofa::core::topology;
using namespace sofa::core;

/**
 * This class, called SimpleTesselatedTetraTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = TetrahedronSetTopology
 * OUTPUT TOPOLOGY = Set of TetrahedronSetTopologies, as the boundary of the INPUT TOPOLOGY
 *
 * Each tetrahedron in the input Topology will be divided in eight tetrahedrom in the output topology
 *
 * SimpleTesselatedTetraTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

class SimpleTesselatedTetraTopologicalMapping : public TopologicalMapping
{
public:
    SOFA_CLASS(SimpleTesselatedTetraTopologicalMapping,TopologicalMapping);

    /** \brief Constructor.
         *
     * @param from the topology issuing TopologyChange objects (the "source").
     * @param to   the topology for which the TopologyChange objects must be translated (the "target").
     */
    SimpleTesselatedTetraTopologicalMapping ( In* from=NULL, Out* to=NULL );

    /** \brief Destructor.
     *
         * Does nothing.
         */
    virtual ~SimpleTesselatedTetraTopologicalMapping() {};

    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    virtual void init();

    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     */
    virtual void updateTopologicalMappingTopDown();

    /** \brief Translates the TopologyChange objects from the target to the source.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the second topology changes on the first topology.
     *
     */
    virtual void updateTopologicalMappingBottomUp();

    /// Return true if this mapping is able to propagate topological changes from input to output model
    virtual bool propagateFromInputToOutputModel() { return true; }

    /// Return true if this mapping is able to propagate topological changes from output to input model
    virtual bool propagateFromOutputToInputModel() { return true; }

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output topology types of this
    /// mapping.
    template<class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        if ( arg->findObject ( arg->getAttribute ( "object1","../.." ) ) == NULL )
            context->serr << "Cannot create "<<className ( obj ) <<" as object1 is missing." << context->sendl;

        if ( arg->findObject ( arg->getAttribute ( "object2",".." ) ) == NULL )
            context->serr << "Cannot create "<<className ( obj ) <<" as object2 is missing." << context->sendl;

        if ( arg->findObject ( arg->getAttribute ( "object1","../.." ) ) == NULL || arg->findObject ( arg->getAttribute ( "object2",".." ) ) == NULL )
            return false;

        BaseMeshTopology* topoIn;
        BaseMeshTopology* topoOut;

        ( dynamic_cast<sofa::core::objectmodel::BaseObject*> ( arg->findObject ( arg->getAttribute ( "object1","../.." ) ) ) )->getContext()->get ( topoIn );
        ( dynamic_cast<sofa::core::objectmodel::BaseObject*> ( arg->findObject ( arg->getAttribute ( "object2",".." ) ) ) )->getContext()->get ( topoOut );

        if ( dynamic_cast<In*> ( topoIn ) == NULL )
            return false;

        if ( dynamic_cast<Out*> ( topoOut ) == NULL )
            return false;

        return BaseMapping::canCreate ( obj, context, arg );
    }

    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes to
    /// find the input and output topologies of this mapping.
    template<class T>
    static void create ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        BaseMeshTopology* topoIn=NULL;
        BaseMeshTopology* topoOut=NULL;
        if ( arg )
        {
            if ( arg->findObject ( arg->getAttribute ( "object1","../.." ) ) != NULL )
                ( dynamic_cast<sofa::core::objectmodel::BaseObject*> ( arg->findObject ( arg->getAttribute ( "object1","../.." ) ) ) )->getContext()->get ( topoIn );

            if ( arg->findObject ( arg->getAttribute ( "object2",".." ) ) != NULL )
                ( dynamic_cast<sofa::core::objectmodel::BaseObject*> ( arg->findObject ( arg->getAttribute ( "object2",".." ) ) ) )->getContext()->get ( topoOut );
        }
        obj = new T (
            ( arg?dynamic_cast<In*> ( topoIn ) :NULL ),
            ( arg?dynamic_cast<Out*> ( topoOut ) :NULL ) );

        if ( context ) context->addObject ( obj );

        if ( ( arg ) && ( arg->getAttribute ( "object1" ) ) )
        {
            obj->object1.setValue ( arg->getAttribute ( "object1" ) );
            arg->removeAttribute ( "object1" );
        }

        if ( ( arg ) && ( arg->getAttribute ( "object2" ) ) )
        {
            obj->object2.setValue ( arg->getAttribute ( "object2" ) );
            arg->removeAttribute ( "object2" );
        }

        if ( arg ) obj->parse ( arg );
    }

    const helper::vector<int>& getPointMappedFromPoint() const { return d_pointMappedFromPoint.getValue(); }
    const helper::vector<int>& getPointMappedFromEdge() const { return d_pointMappedFromEdge.getValue(); }
    const helper::vector<int>& getPointSource() const { return d_pointSource.getValue(); }

protected:
    Data< std::string > object1;
    Data< std::string > object2;

    TetrahedronData< fixed_array<int, 8> > tetrahedraMappedFromTetra; ///< Each Tetrahedron of the input topology is mapped to the 8 tetrahedrons in which it can be divided.


    TetrahedronData<int> tetraSource; ///<Which tetra from the input topology map to a given tetra in the output topology (-1 if none)

    Data< helper::vector<int> > d_pointMappedFromPoint; ///< Each point of the input topology is mapped to the same point.
    Data< helper::vector<int> > d_pointMappedFromEdge; ///< Each edge of the input topology is mapped to his midpoint.
    Data< helper::vector<int> > d_pointSource; ///< Which input topology element map to a given point in the output topology : 0 -> none, > 0 -> point index + 1, < 0 , - edge index -1

    void swapOutputPoints(int i1, int i2);
    void removeOutputPoints( const sofa::helper::vector<unsigned int>& tab );
    void renumberOutputPoints( const sofa::helper::vector<unsigned int>& tab );

    void swapOutputTetrahedra(int i1, int i2);
    void removeOutputTetrahedra( const sofa::helper::vector<unsigned int>& tab );

    void setPointSource(int i, int source)
    {
        helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
        helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromPointData = d_pointMappedFromPoint;
        helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromEdgeData = d_pointMappedFromEdge;


        if (i != -1)
            pointSourceData[i] = source;
        if (source > 0)
        {
            pointMappedFromPointData[source-1] = i;
        }
        else if (source < 0)
        {
            pointMappedFromEdgeData[-source-1] = i;
        }
    }
    std::set<unsigned int> tetrahedraToRemove;


    void swapInputPoints(int i1, int i2);
    void removeInputPoints( const sofa::helper::vector<unsigned int>& tab );
    void renumberInputPoints( const sofa::helper::vector<unsigned int>& tab );
    void swapInputEdges(int i1, int i2);
    void removeInputEdges( const sofa::helper::vector<unsigned int>& tab );

    void swapInputTetrahedra(int i1, int i2);
    void removeInputTetrahedra( const sofa::helper::vector<unsigned int>& tab );

};

} // namespace topology
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TETRA2TRIANGLETOPOLOGICALMAPPING_H
