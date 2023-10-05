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

#include <sofa/component/engine/analyze/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/SVector.h>
#include <sofa/helper/set.h>
#include <sofa/helper/map.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/component/topology/container/dynamic/DynamicSparseGridTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/DynamicSparseGridGeometryAlgorithms.h>
#include <utility>

#define TYPE_GEODESIC 0
#define TYPE_HARMONIC 1
#define TYPE_STIFFNESS_DIFFUSION 2
#define TYPE_VORONOI 3
#define TYPE_HARMONIC_STIFFNESS 4

namespace sofa::component::engine::analyze
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class DistancesInternalData
{
public:
};

/**
 * This class computes distances between to set of mechanical objects.
 */
template <class DataTypes>
class Distances : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(Distances,DataTypes),core::DataEngine);

    typedef std::pair< core::topology::BaseMeshTopology::HexaID, double> Distance;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename sofa::type::vector< VecCoord > VecVecCoord;
    typedef type::SVector<type::SVector<double> > VVD;

protected:
    DistancesInternalData<DataTypes> data;
    friend class DistancesInternalData<DataTypes>;

    Distances ( sofa::component::topology::container::dynamic::DynamicSparseGridTopologyContainer* hexaTopoContainer, core::behavior::MechanicalState<DataTypes>* targetPointSet );

    ~Distances() override {}

public:
    Data<unsigned int> showMapIndex; ///< Frame DOF index on which display values.
    Data<bool> showDistanceMap; ///< show the dsitance for each point of the target point set.
    Data<bool> showGoalDistanceMap; ///< show the dsitance for each point of the target point set.
    Data<double> showTextScaleFactor; ///< Scale to apply on the text.
    Data<bool> showGradientMap; ///< show gradients for each point of the target point set.
    Data<double> showGradientsScaleFactor; ///< scale for the gradients displayed.
    Data<Coord> offset; ///< translation offset between the topology and the point set.
    Data<sofa::helper::OptionsGroup> distanceType; ///< type of distance to compute for inserted frames.
    Data<bool> initTarget; ///< initialize the target MechanicalObject from the grid.
    Data<int> initTargetStep; ///< initialize the target MechanicalObject from the grid using this step.
    Data<std::map<unsigned int, unsigned int> > zonesFramePair; ///< Correspondance between the segmented value and the frames.
    Data<double> harmonicMaxValue; ///< Max value used to initialize the harmonic distance grid.

    void init() override;

    void reinit() override;

    void doUpdate() override;

    /** \brief Compute the distance map depending ion the distance type.
    *
    * @param elt the point from which the distances are computed.
    * @param beginElts distance until we stop propagating.
    * @param distMax distance until we stop propagating.
    */
    void computeDistanceMap ( VecCoord beginElts = VecCoord(), const double& distMax = 0 );

    /** \brief Add a 'from' element and recompute the map of distances.
    *
    * @param elt the point from which the distances are computed.
    * @param beginElts distance until we stop propagating.
    * @param distMax distance until we stop propagating.
    */
    void addElt ( const Coord& elt, VecCoord beginElts = VecCoord(), const double& distMax = 0 );

    /** \brief Get the distance for a point set using the computed map.
    *
    * @param distances distance for each point of the topology.
    * @param gradients gradient of the distance for each point in the topology.
    * @param point the point from which the distances are computed.
    */
    void getDistances ( VVD& distances, VecVecCoord& gradients, const VecCoord& goals );

    void draw(const core::visual::VisualParams* vparams) override;

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output model types of this
    /// mapping.
    template<class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        bool error = false;
        if (arg->findObject(arg->getAttribute("hexaContainerPath", "../..")) == nullptr)
        {
            arg->logError("Cannot create " + T::GetClass()->className + " as the hexas container is missing.");
            error = true;
        }
        else if ( dynamic_cast<sofa::component::topology::container::dynamic::DynamicSparseGridTopologyContainer*> ( arg->findObject ( arg->getAttribute ( "hexaContainerPath","../.." ) ) ) == nullptr )
        {
            arg->logError("Object pointed by data attribute 'hexaContainerPath' is not of type "
                + sofa::component::topology::container::dynamic::DynamicSparseGridTopologyContainer::GetClass()->className);
            error = true;
        }

        if (arg->findObject(arg->getAttribute("targetPath", "..")) == nullptr)
        {
            arg->logError("Cannot create " + T::GetClass()->className + " as the target point set is missing.");
            error = true;
        }
        else if ( dynamic_cast<core::behavior::MechanicalState<DataTypes>*> ( arg->findObject ( arg->getAttribute ( "targetPath",".." ) ) ) == nullptr )
        {
            arg->logError("Data attribute 'targetPath' does not point to a mechanical state of data type '" + std::string(DataTypes::Name()) +"'.");
            error = true;
        }
        return !error && core::DataEngine::canCreate(obj, context, arg);
    }
    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes to
    /// find the input and output models of this mapping.
    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>(
                ( arg?dynamic_cast<sofa::component::topology::container::dynamic::DynamicSparseGridTopologyContainer*> ( arg->findObject ( arg->getAttribute ( "hexaContainerPath","../.." ) ) ) :nullptr ),
                ( arg?dynamic_cast<core::behavior::MechanicalState<DataTypes>*> ( arg->findObject ( arg->getAttribute ( "targetPath",".." ) ) ) :nullptr ) );

        if ( context ) context->addObject ( obj );

        if ( arg )
        {
            if ( arg->getAttribute ( "hexaContainerPath" ) )
            {
                obj->hexaContainerPath.setValue ( arg->getAttribute ( "hexaContainerPath" ) );
                arg->removeAttribute ( "hexaContainerPath" );
            }
            if ( arg->getAttribute ( "targetPath" ) )
            {
                obj->targetPath.setValue ( arg->getAttribute ( "targetPath" ) );
                arg->removeAttribute ( "targetPath" );
            }
            obj->parse ( arg );
        }

        return obj;
    }

private:
    Data<std::string> fileDistance; ///< file containing the result of the computation of the distances
    Data<std::string> targetPath; ///< path to the goal point set topology
    core::behavior::MechanicalState<DataTypes>* target;

    Data<std::string> hexaContainerPath; ///< path to the grid used to compute the distances
    sofa::component::topology::container::dynamic::DynamicSparseGridTopologyContainer* hexaContainer;
    sofa::component::topology::container::dynamic::DynamicSparseGridGeometryAlgorithms< DataTypes >* hexaGeoAlgo;
    const unsigned char * densityValues; // Density values
    const unsigned char * segmentIDData; // Density values

    VVD distanceMap; // distance for each hexa of the grid (topology order)

    /*************************/
    /*   Compute distances   */
    /*************************/
    inline void computeGeodesicalDistance ( const unsigned int& mapIndex, const VecCoord& beginElts, const bool& diffuseAccordingToStiffness, const double& distMax = 0 );
    // Store harmonic coords in the distanceMap structure of the class depending on the fixed values 'hfrom'
    inline void computeHarmonicCoords ( const unsigned int& mapIndex, const type::vector<core::topology::BaseMeshTopology::HexaID>& hfrom, const bool& useStiffnessMap );
    inline void computeVoronoiDistances( const unsigned int& mapIndex, const VecCoord& beginElts, const double& distMax = 0 );


    /*************************/
    /*         Utils         */
    /*************************/
    inline void findCorrespondingHexas ( type::vector<core::topology::BaseMeshTopology::HexaID>& hexas, const VecCoord& pointSet ); // Find indices from coord.
    inline void find1DCoord ( unsigned int& hexaID, const Coord& point );
    void getNeighbors ( const core::topology::BaseMeshTopology::HexaID& hexaID, std::set<core::topology::BaseMeshTopology::HexaID>& neighbors ) const;
    void computeGradients ( const unsigned int mapIndex, type::vector<double>& distances, VecCoord& gradients, const type::vector<core::topology::BaseMeshTopology::HexaID>& hexaGoal, const VecCoord& goals );
    inline void addContribution ( double& valueWrite, int& nbTest, const type::vector<double>& valueRead, const unsigned int& gridID, const int coeff );
    inline void addContribution ( double& valueWrite, int& nbTest, double*** valueRead, const int& x, const int& y, const int& z, const int coeff, const bool& useStiffnessMap );
};

#if !defined(SOFA_COMPONENT_ENGINE_DISTANCES_CPP)
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API Distances<defaulttype::Vec3Types>; 
#endif

} // namespace sofa::component::engine::analyze
