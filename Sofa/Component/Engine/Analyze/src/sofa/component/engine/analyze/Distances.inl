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

#include <sofa/component/engine/analyze/Distances.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.h>
#include <sofa/core/loader/VoxelLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/type/RGBAColor.h>
#include <algorithm>
#include <functional>
#include <queue>
#include <fstream>

namespace sofa::component::engine::analyze
{

using std::queue;
using sofa::core::loader::VoxelLoader;

template<class DataTypes>
Distances< DataTypes >::Distances ( sofa::component::topology::container::dynamic::DynamicSparseGridTopologyContainer* hexaTopoContainer, core::behavior::MechanicalState<DataTypes>* targetPointSet ) :
    d_showMapIndex (initData (&d_showMapIndex, (unsigned int)0, "showMapIndex", "Frame DOF index on which display values." ) ),
    d_showDistanceMap (initData (&d_showDistanceMap, false, "showDistancesMap", "show the distance for each point of the target point set." ) ),
    d_showGoalDistanceMap (initData (&d_showGoalDistanceMap, false, "showGoalDistancesMap", "show the distance for each point of the target point set." ) ),
    d_showTextScaleFactor (initData (&d_showTextScaleFactor, 0.001, "showTextScaleFactor", "Scale to apply on the text." ) ),
    d_showGradientMap (initData (&d_showGradientMap, false, "showGradients", "show gradients for each point of the target point set." ) ),
    d_showGradientsScaleFactor (initData (&d_showGradientsScaleFactor, 0.1, "showGradientsScaleFactor", "scale for the gradients displayed." ) ),
    d_offset (initData (&d_offset, Coord(), "offset", "translation offset between the topology and the point set." ) ),
    d_distanceType (initData (&d_distanceType, {"Geodesic", "Harmonic", "Stiffness Diffusion", "Voronoi", "Harmonic with Stiffness"}, "d_distanceType", "type of distance to compute for inserted frames." ) ),
    d_initTarget (initData (&d_initTarget, false, "initTarget", "initialize the target MechanicalObject from the grid." ) ),
    d_initTargetStep (initData (&d_initTargetStep, 1, "initTargetStep", "initialize the target MechanicalObject from the grid using this step." ) ),
    d_zonesFramePair (initData (&d_zonesFramePair, "zonesFramePair", "Correspondence between the segmented value and the frames." ) ),
    d_harmonicMaxValue (initData (&d_harmonicMaxValue, 100.0, "harmonicMaxValue", "Max value used to initialize the harmonic distance grid." ) ),
    d_fileDistance(initData(&d_fileDistance, "filename", "file containing the result of the computation of the distances")),
    d_targetPath(initData(&d_targetPath, "targetPath", "path to the goal point set topology")),
    target ( targetPointSet ) ,
    d_hexaContainerPath(initData(&d_hexaContainerPath, "hexaContainerPath", "path to the grid used to compute the distances")),
    hexaContainer ( hexaTopoContainer )
{
    this->addAlias(&d_fileDistance, "d_fileDistance");
    d_zonesFramePair.setDisplayed(false); // GUI can not display map.

    this->f_printLog.setValue(true);

    showMapIndex.setOriginalData(&d_showMapIndex);
    showDistanceMap.setOriginalData(&d_showDistanceMap);
    showGoalDistanceMap.setOriginalData(&d_showGoalDistanceMap);
    showTextScaleFactor.setOriginalData(&d_showTextScaleFactor);
    showGradientMap.setOriginalData(&d_showGradientMap);
    showGradientsScaleFactor.setOriginalData(&d_showGradientsScaleFactor);
    offset.setOriginalData(&d_offset);
    distanceType.setOriginalData(&d_distanceType);
    initTarget.setOriginalData(&d_initTarget);
    initTargetStep.setOriginalData(&d_initTargetStep);
    zonesFramePair.setOriginalData(&d_zonesFramePair);
    harmonicMaxValue.setOriginalData(&d_harmonicMaxValue);
    fileDistance.setOriginalData(&d_fileDistance);
    targetPath.setOriginalData(&d_targetPath);
    hexaContainerPath.setOriginalData(&d_hexaContainerPath);

}


template<class DataTypes>
void Distances< DataTypes >::init()
{
    if ( !hexaContainer ) return;
    hexaContainer->getContext()->get ( hexaGeoAlgo );
    if ( !hexaGeoAlgo )
    {
        msg_error() << "Can not find the hexahedron geometry algorithms component.";
        return;
    }

    VoxelLoader* voxelGridLoader;
    this->getContext()->get( voxelGridLoader);
    if ( !voxelGridLoader )
    {
        msg_error() << "Can not find the Voxel Grid Loader component.";
        return;
    }
    densityValues = voxelGridLoader->getData();
    segmentIDData = voxelGridLoader->getSegmentID();


    // Init the DOFs at each voxel center according to the step
    if ( d_initTarget.getValue())
    {
        unsigned int size = hexaContainer->getNumberOfHexahedra();
        unsigned int realSize = 0;
        const unsigned int step = d_initTargetStep.getValue();
        target->resize( size);
        helper::WriteAccessor< Data< VecCoord > > xto = *target->write(core::vec_id::write_access::position);
        helper::WriteAccessor< Data< VecCoord > > xto0 = *target->write(core::vec_id::write_access::restPosition);
        const Coord& offSet = d_offset.getValue();
        const type::Vec3& voxelSize = hexaContainer->d_voxelSize.getValue();
        for ( unsigned int i = 0; i < size; i++)
        {
            Coord pos = hexaGeoAlgo->computeHexahedronRestCenter ( i );
            const int x = int ( pos[0] / voxelSize[0] );
            const int y = int ( pos[1] / voxelSize[1] );
            const int z = int ( pos[2] / voxelSize[2] );
            if ( !(x%step) && !(y%step) && !(z%step))
            {
                Coord center = hexaGeoAlgo->computeHexahedronCenter( i) + offSet;
                xto[realSize] = center;
                xto0[realSize] = center;
                realSize++;
            }
        }
        target->resize( realSize); // Resize to the real size.
    }
}


template<class DataTypes>
void Distances< DataTypes >::reinit()
{
    update();
}

template<class DataTypes>
void Distances< DataTypes >::doUpdate()
{


    /*
    if( true)
      computeDistanceMap


      if (true)
    getDistances
    */



}



template<class DataTypes>
void Distances< DataTypes >::computeDistanceMap ( VecCoord beginElts, const double& distMax )
{
    type::vector<core::topology::BaseMeshTopology::HexaID> hfrom;
    findCorrespondingHexas ( hfrom, beginElts );

    std::string filename=d_fileDistance.getValue();
    if (d_distanceType.getValue().getSelectedId() == TYPE_GEODESIC ) filename+="Geodesical";
    else if (d_distanceType.getValue().getSelectedId() == TYPE_HARMONIC ) filename+="Harmonic";
    else if (d_distanceType.getValue().getSelectedId() == TYPE_STIFFNESS_DIFFUSION ) filename+="StiffnessPropagation";
    else if (d_distanceType.getValue().getSelectedId() == TYPE_VORONOI ) filename+="Voronoi";
    else if (d_distanceType.getValue().getSelectedId() == TYPE_HARMONIC_STIFFNESS ) filename+="StiffnessHarmonic";
    else
    {
        msg_error() << "distance Type unknown when adding an element.";
        return;
    }

    if (sofa::helper::system::DataRepository.findFile(filename)) // If the file is existing
    {
        // Load the distance map from the file
        filename=sofa::helper::system::DataRepository.getFile(filename);
        msg_info() << "Using filename:" << filename << " to load the distances";
        std::ifstream distanceFile(filename.c_str());
        distanceMap.read(distanceFile);
        distanceFile.close();
    }
    else // Else, compute it and write it into the file
    {
        distanceMap.clear();
        distanceMap.resize ( beginElts.size() );

        if (d_distanceType.getValue().getSelectedId() == TYPE_GEODESIC )
        {
            for ( unsigned int i = 0; i < beginElts.size(); i++ )
            {
                VecCoord tmpBeginElts;
                tmpBeginElts.push_back( beginElts[i]);
                computeGeodesicalDistance ( i, tmpBeginElts, false, distMax );
            }
        }
        else if (d_distanceType.getValue().getSelectedId() == TYPE_HARMONIC )
        {
            for ( int i = 0; i < (int)beginElts.size(); i++ )
                computeHarmonicCoords ( i, hfrom, false );
        }
        else if (d_distanceType.getValue().getSelectedId() == TYPE_STIFFNESS_DIFFUSION )
        {
            for ( unsigned int i = 0; i < beginElts.size(); i++ )
            {
                VecCoord tmpBeginElts;
                tmpBeginElts.push_back( beginElts[i]);
                computeGeodesicalDistance ( i, tmpBeginElts, true, distMax );
            }
        }
        else if (d_distanceType.getValue().getSelectedId() == TYPE_VORONOI )
        {
            for ( unsigned int i = 0; i < beginElts.size(); i++ )
            {
                VecCoord tmpBeginElts;
                tmpBeginElts.push_back( beginElts[i]);
                computeVoronoiDistances ( i, tmpBeginElts, distMax );
            }
        }
        else if (d_distanceType.getValue().getSelectedId() == TYPE_HARMONIC_STIFFNESS )
        {
            for ( int i = 0; i < (int)beginElts.size(); i++ )
                computeHarmonicCoords ( i, hfrom, true );
        }

        msg_info() << "Writing filename:" << filename << " to save the distances";
        std::ofstream distanceFile(filename.c_str());
        distanceMap.write(distanceFile);
        distanceFile.close();
    }
}


template<class DataTypes>
void Distances< DataTypes >::addElt ( const Coord& elt, VecCoord beginElts, const double& distMax )
{
    const unsigned int mapIndex = distanceMap.size();
    distanceMap.resize ( mapIndex+1 );
    VecCoord tmpvcoord = beginElts;
    tmpvcoord.push_back( elt);

    if (d_distanceType.getValue().getSelectedId() == TYPE_GEODESIC )
    {
        computeGeodesicalDistance ( mapIndex, tmpvcoord, false, distMax );
    }
    else if (d_distanceType.getValue().getSelectedId() == TYPE_HARMONIC )
    {
        type::vector<core::topology::BaseMeshTopology::HexaID> hfrom;
        findCorrespondingHexas ( hfrom, tmpvcoord );
        computeHarmonicCoords ( mapIndex, hfrom, false );
    }
    else if (d_distanceType.getValue().getSelectedId() == TYPE_STIFFNESS_DIFFUSION )
    {
        computeGeodesicalDistance ( mapIndex, tmpvcoord, true, distMax );
    }
    else if (d_distanceType.getValue().getSelectedId() == TYPE_VORONOI )
    {
        computeVoronoiDistances ( mapIndex, tmpvcoord, distMax );
    }
    else if (d_distanceType.getValue().getSelectedId() == TYPE_HARMONIC_STIFFNESS )
    {
        type::vector<core::topology::BaseMeshTopology::HexaID> hfrom;
        findCorrespondingHexas ( hfrom, tmpvcoord );
        computeHarmonicCoords ( mapIndex, hfrom, true );
    }
    else
    {
        msg_error() << "distance Type unknown when adding an element.";
    }
}


template<class DataTypes>
void Distances< DataTypes >::computeGeodesicalDistance ( const unsigned int& mapIndex, const VecCoord& beginElts, const bool& diffuseAccordingToStiffness, const double& distMax )
{
    distanceMap[mapIndex].clear();
    distanceMap[mapIndex].resize ( hexaContainer->getNumberOfHexahedra() );

    for ( unsigned int i = 0; i < hexaContainer->getNumberOfHexahedra(); ++i)
        distanceMap[mapIndex][i] = -1.0;

    queue<Distance> hexasBeingParsed;
    std::set<core::topology::BaseMeshTopology::HexaID> hexasParsed;
    Distance hexaCoord;
    type::Vec3 baryC;
    const Coord& offSet = d_offset.getValue();

    // Get the hexas corresponding to the position 'point'
    Real dist;
    for ( typename VecCoord::const_iterator it = beginElts.begin(); it != beginElts.end(); it++)
    {
        hexaCoord.first = hexaGeoAlgo->findNearestElementInRestPos ( *it - offSet, baryC, dist );
        hexaCoord.second = 0.0;//( point - offSet - hexaGeoAlgo->computeHexahedronRestCenter ( hexaCoord.first ) ).norm();
        hexasBeingParsed.push ( hexaCoord );
    }

    // Propagate
    while ( !hexasBeingParsed.empty() )
    {
        hexaCoord = hexasBeingParsed.front(); // Get the front element of the queue.
        hexasBeingParsed.pop();             // Remove it from the queue.
        const core::topology::BaseMeshTopology::HexaID& hexaID = hexaCoord.first;
        const double& distance = hexaCoord.second;

        if ( hexasParsed.contains ( hexaID )) continue;
        hexasParsed.insert ( hexaID ); // This hexa has been parsed

        // Continue if the distance max is reached
        const Coord hexaIDpos = hexaGeoAlgo->computeHexahedronRestCenter ( hexaID );

        // Propagate
        std::set<core::topology::BaseMeshTopology::HexaID> neighbors;
        getNeighbors ( hexaID, neighbors );

        unsigned int hexaID1;
        find1DCoord(hexaID1, hexaGeoAlgo->computeHexahedronRestCenter(hexaID));
        const double densityValue1 = densityValues[hexaID1];

        for ( std::set<core::topology::BaseMeshTopology::HexaID>::iterator it = neighbors.begin(); it != neighbors.end(); it++ )
        {
            Distance newDist;
            double stiffCoeff = 1.0;
            if (diffuseAccordingToStiffness)
            {
                //unsigned int hexaID2;
                //find1DCoord(hexaID2, hexaGeoAlgo->computeHexahedronRestCenter(*it));
                //double densityValue2 = densityValues[hexaID2];
                stiffCoeff = 1.0 + (densityValue1/* - densityValue2*/) / 255.0 * 5.0; // From 1 to 10
            }
            newDist.first = *it;
            newDist.second = distance + (( hexaGeoAlgo->computeHexahedronRestCenter ( *it ) - hexaIDpos ).norm() * stiffCoeff);
            if ( distMax != 0 && newDist.second > distMax ) continue; // End on distMax
            if ( distanceMap[mapIndex][*it] == -1.0 || newDist.second < distanceMap[mapIndex][*it] ) distanceMap[mapIndex][*it] = newDist.second;
            hexasBeingParsed.push ( newDist );
        }
    }
}


template<class DataTypes>
void Distances< DataTypes >::computeHarmonicCoords ( const unsigned int& mapIndex, const type::vector<core::topology::BaseMeshTopology::HexaID>& hfrom, const bool& useStiffnessMap )
{
    // Init the distance Map. TODO: init the distance map between each elt before diffusing
    type::vector<double>& dMIndex = distanceMap[mapIndex];
    dMIndex.clear();
    dMIndex.resize ( hexaContainer->getNumberOfHexahedra() );

    const sofa::type::vector<sofa::core::topology::BaseMeshTopology::HexaID>& iirg = hexaContainer->d_idxInRegularGrid.getValue();
    const std::map<unsigned int, unsigned int>& zones = d_zonesFramePair.getValue();

    for ( unsigned int j = 0; j < hexaContainer->getNumberOfHexahedra(); j++ )
    {
        std::map<unsigned int, unsigned int>::const_iterator it = zones.find( (unsigned int)segmentIDData[iirg[j]]);
        if ( it != zones.end())
        {
            if ( it->second == mapIndex)
                dMIndex[j] = 0.0;
            else
                dMIndex[j] = d_harmonicMaxValue.getValue();
        }
        else
            dMIndex[j] = d_harmonicMaxValue.getValue() / 2.0;
    }

    for ( type::vector<core::topology::BaseMeshTopology::HexaID>::const_iterator it = hfrom.begin(); it != hfrom.end(); it++ )
        dMIndex[*it] = d_harmonicMaxValue.getValue();

    dMIndex[hfrom[mapIndex]] = 0.0;

    msg_info() << "Compute distance map.";

    sofa::type::Vec3i res = hexaContainer->d_resolution.getValue();
    bool convergence = false;

    double*** distMap = new double** [res[0]];
    double*** dMCpy = new double** [res[0]];
    for ( int x = 0; x < res[0]; x++ )
    {
        distMap[x] = new double* [res[1]];
        dMCpy  [x] = new double* [res[1]];
        for ( int y = 0; y < res[1]; y++ )
        {
            distMap[x][y] = new double[res[2]];
            dMCpy  [x][y] = new double[res[2]];
            for ( int z = 0; z < res[2]; z++ )
                distMap[x][y][z] = -1.0;
        }
    }

    const type::Vec3& voxelSize = hexaContainer->d_voxelSize.getValue();
    for ( unsigned int i = 0; i < dMIndex.size(); i++ )
    {
        Coord pos = hexaGeoAlgo->computeHexahedronRestCenter ( i );
        const int x = int ( pos[0] / voxelSize[0] );
        const int y = int ( pos[1] / voxelSize[1] );
        const int z = int ( pos[2] / voxelSize[2] );
        distMap[x][y][z] = dMIndex[i];
    }

    while ( !convergence )
    {
        // Copy the current values.
        for ( int x = 0; x < res[0]; x++ )
            for ( int y = 0; y < res[1]; y++ )
                for ( int z = 0; z < res[2]; z++ )
                    dMCpy[x][y][z] = distMap[x][y][z];

        // Apply the filter
        //10 iterations per thread
        int x,y,z;
        for (  x = 0; x < res[0]; x++ )
            for (  y = 0; y < res[1]; y++ )
            {
                for (  z = 0; z < res[2]; z++ )
                {
                    // Avoid to compute the value for the 'from' hexa and the unexisting hexas.
                    if ( dMCpy[x][y][z] == -1.0 || dMCpy[x][y][z] == 0.0 || dMCpy[x][y][z] == d_harmonicMaxValue.getValue() ) continue;


                    double value = 0;
                    int nbTest = 4; // Contribution of the current case 'i'

                    if ( z > 0 )
                    {
                        addContribution ( value, nbTest, dMCpy, x, y, z-1, 2, useStiffnessMap );// gridID-res[0]*res[1]
                        if ( y > 0 )
                            addContribution ( value, nbTest, dMCpy, x, y-1, z-1, 1, useStiffnessMap );// gridID-res[0]*res[1]-res[0]
                        if ( y < res[1]-1 )
                            addContribution ( value, nbTest, dMCpy, x, y+1, z-1, 1, useStiffnessMap );// gridID-res[0]*res[1]+res[0]
                        if ( x > 0 )
                            addContribution ( value, nbTest, dMCpy, x-1, y, z-1, 1, useStiffnessMap );// gridID-res[0]*res[1]-1
                        if ( x < res[0]-1 )
                            addContribution ( value, nbTest, dMCpy, x+1, y, z-1, 1, useStiffnessMap );// gridID-res[0]*res[1]+1
                    }
                    if ( z < res[2]-1 )
                    {
                        addContribution ( value, nbTest, dMCpy, x, y, z+1, 2, useStiffnessMap );// gridID+res[0]*res[1]
                        if ( y > 0 )
                            addContribution ( value, nbTest, dMCpy, x, y-1, z+1, 1, useStiffnessMap );// gridID+res[0]*res[1]-res[0]
                        if ( y < res[1]-1 )
                            addContribution ( value, nbTest, dMCpy, x, y+1, z+1, 1, useStiffnessMap );// gridID+res[0]*res[1]+res[0]
                        if ( x > 0 )
                            addContribution ( value, nbTest, dMCpy, x-1, y, z+1, 1, useStiffnessMap );// gridID+res[0]*res[1]-1
                        if ( x < res[0]-1 )
                            addContribution ( value, nbTest, dMCpy, x+1, y, z+1, 1, useStiffnessMap );// gridID+res[0]*res[1]+1
                    }
                    if ( y > 0 )
                    {
                        addContribution ( value, nbTest, dMCpy, x, y-1, z, 2, useStiffnessMap );// gridID-res[0]
                        if ( x > 0 )
                            addContribution ( value, nbTest, dMCpy, x-1, y-1, z, 1, useStiffnessMap );// gridID-res[0]-1
                        if ( x < res[0]-1 )
                            addContribution ( value, nbTest, dMCpy, x+1, y-1, z, 1, useStiffnessMap );// gridID-res[0]+1
                    }
                    if ( y < res[1]-1 )
                    {
                        addContribution ( value, nbTest, dMCpy, x, y+1, z, 2, useStiffnessMap );// gridID+res[0]
                        if ( x > 0 )
                            addContribution ( value, nbTest, dMCpy, x-1, y+1, z, 1, useStiffnessMap );// gridID+res[0]-1
                        if ( x < res[0]-1 )
                            addContribution ( value, nbTest, dMCpy, x+1, y+1, z, 1, useStiffnessMap );// gridID+res[0]+1
                    }
                    if ( x > 0 )
                        addContribution ( value, nbTest, dMCpy, x-1, y, z, 2, useStiffnessMap );// gridID-1
                    if ( x < res[0]-1 )
                        addContribution ( value, nbTest, dMCpy, x+1, y, z, 2, useStiffnessMap );// gridID+1

                    // And store the result
                    distMap[x][y][z] = ( ( 4*dMCpy[x][y][z] + value ) / ( double ) nbTest );
                }
            }

        // Convergence test
        convergence = true;
        for ( int x = 0; x < res[0]; x++ )
            for ( int y = 0; y < res[1]; y++ )
                for ( int z = 0; z < res[2]; z++ )
                    if ( fabs(dMCpy[x][y][z] - distMap[x][y][z]) > (d_harmonicMaxValue.getValue() / 1000000.0) )
                        convergence = false;
    }

    // Copy the result in distanceMap
    for ( unsigned int i = 0; i < dMIndex.size(); i++ )
    {
        Coord pos = hexaGeoAlgo->computeHexahedronRestCenter ( i );
        const int x = int ( pos[0] / voxelSize[0] );
        const int y = int ( pos[1] / voxelSize[1] );
        const int z = int ( pos[2] / voxelSize[2] );
        dMIndex[i] = distMap[x][y][z];
    }

    for ( int x = 0; x < res[0]; x++ )
    {
        for ( int y = 0; y < res[1]; y++ )
        {
            delete [] dMCpy[x][y];
            delete [] distMap[x][y];
        }
        delete [] dMCpy[x];
        delete [] distMap[x];
    }
    delete [] dMCpy;
    delete [] distMap;

    msg_info() << "Distance map computed.";
}


template<class DataTypes>
void Distances< DataTypes >::computeVoronoiDistances( const unsigned int& /*mapIndex*/, const VecCoord& /*beginElts*/, const double& /*distMax*/ )
{
    // Propager a partir des begin pour obtenir le voronoi
    // stocker les frontieres
    // Sur chaque ligne, determiner l'elt min(max)
    // Remonter jusqu'aux reperes a partir de ces elts pour obtenir les lignes verte de (2)
    // initialiser les frontieres a 0 et propager
    // On obtient ligne rouge de (2)


}


template<class DataTypes>
void Distances< DataTypes >::getDistances ( VVD& distances, VecVecCoord& gradients, const VecCoord& goals )
{
    type::vector<core::topology::BaseMeshTopology::HexaID> hgoal;
    findCorrespondingHexas ( hgoal, goals );

    distances.clear();
    distances.resize ( distanceMap.size() );
    gradients.clear();
    gradients.resize ( distanceMap.size() );

    for ( unsigned int i = 0; i < distanceMap.size(); i++ )
    {
        // Compute Harmonic Coords
        type::vector<double>& dists = distances[i];
        VecCoord& grads = gradients[i];
        computeGradients ( i, dists, grads, hgoal, goals );
    }
}


template<class DataTypes>
void Distances< DataTypes >::addContribution ( double& valueWrite, int& nbTest, const type::vector<double>& valueRead, const unsigned int& gridID, const int coeff )
{
    bool existing;
    const core::topology::BaseMeshTopology::HexaID hexaID = hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( gridID, existing );
    if ( existing )
    {
        valueWrite += coeff * valueRead[hexaID];
        nbTest+=coeff;
    }
}


template<class DataTypes>
void Distances< DataTypes >::addContribution ( double& valueWrite, int& nbTest, double*** valueRead, const int& x, const int& y, const int& z, const int coeff, const bool& useStiffnessMap )
{
    if ( valueRead[x][y][z] != -1.0 )
    {
        int stiffnessCoeff = 1;
        if (useStiffnessMap)
        {
            unsigned int hexaID1;
            const Coord hexaIDpos ((Real)x, (Real)y, (Real)z);
            find1DCoord(hexaID1, hexaIDpos);
            stiffnessCoeff = densityValues[hexaID1]; //TODO use max if this is a frame
        }

        valueWrite += coeff * valueRead[x][y][z] * stiffnessCoeff;
        nbTest+=coeff;
    }
}


template<class DataTypes>
void Distances< DataTypes >::computeGradients ( const unsigned int mapIndex, type::vector<double>& distances, VecCoord& gradients, const type::vector<core::topology::BaseMeshTopology::HexaID>& hexaGoal, const VecCoord& goals )
{
    // Store the distance and compute gradient for each goal.
    sofa::type::Vec3i res = hexaContainer->d_resolution.getValue();
    const type::Vec3& voxelSize = hexaContainer->d_voxelSize.getValue();
    for ( unsigned int i = 0; i < hexaGoal.size(); i++ )
    {
        const core::topology::BaseMeshTopology::HexaID& hID = hexaGoal[i];
        const Coord& point = goals[i];

        // Distance value for the center of the voxel
        double distance = distanceMap[mapIndex][hID];
        if ( distance == -1.0)
        {
            Coord grad;
            gradients.push_back ( grad );
            distances.push_back ( distance );
            continue;
        }

        Coord grad;
        std::set<core::topology::BaseMeshTopology::HexaID> neighbors;
        getNeighbors ( hID, neighbors );

        const unsigned int gridID = hexaGeoAlgo->getRegularGridIndexFromTopoIndex ( hID );
        bool existing;
        core::topology::BaseMeshTopology::HexaID hexaID;

        // Test on X
        int nbTest = 0;
        hexaID = hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( gridID+1, existing );
        if ( existing && distanceMap[mapIndex][hexaID]!=-1.0 )
        {
            grad[0] += (Real)( distanceMap[mapIndex][hexaID] - distance );
            nbTest++;
        }
        hexaID = hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( gridID-1, existing );
        if ( existing && distanceMap[mapIndex][hexaID]!=-1.0 )
        {
            grad[0] += (Real)( distance - distanceMap[mapIndex][hexaID] );
            nbTest++;
        }
        if ( nbTest != 0 ) grad[0] /= (Real)(nbTest * voxelSize[0]);

        // Test on Y
        nbTest = 0;
        hexaID = hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( gridID+res[0], existing );
        if ( existing && distanceMap[mapIndex][hexaID]!=-1.0 )
        {
            grad[1] += (Real)( distanceMap[mapIndex][hexaID] - distance );
            nbTest++;
        }
        hexaID = hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( gridID-res[0], existing );
        if ( existing && distanceMap[mapIndex][hexaID]!=-1.0 )
        {
            grad[1] += (Real)( distance - distanceMap[mapIndex][hexaID] );
            nbTest++;
        }
        if ( nbTest != 0 ) grad[1] /= (Real)(nbTest * voxelSize[1]);

        // Test on Z
        nbTest = 0;
        hexaID = hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( gridID+res[0]*res[1], existing );
        if ( existing && distanceMap[mapIndex][hexaID]!=-1.0 )
        {
            grad[2] += (Real)( distanceMap[mapIndex][hexaID] - distance );
            nbTest++;
        }
        hexaID = hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( gridID-res[0]*res[1], existing );
        if ( existing && distanceMap[mapIndex][hexaID]!=-1.0 )
        {
            grad[2] += (Real)( distance - distanceMap[mapIndex][hexaID] );
            nbTest++;
        }
        if ( nbTest != 0 ) grad[2] /= (Real)(nbTest * voxelSize[2]);
        //*/

        gradients.push_back ( grad );

        // compute the distance value for the point from the distance on the voxel center
        double pointDist = (point - d_offset.getValue() - hexaGeoAlgo->computeHexahedronCenter(hID)) * grad + distance;
        if ( pointDist < 0.0) pointDist = 0.0;
        distances.push_back ( pointDist );
    }
}

template<class DataTypes>
void Distances< DataTypes >::findCorrespondingHexas ( type::vector<core::topology::BaseMeshTopology::HexaID>& hexas, const VecCoord& pointSet )
{
    for ( unsigned int i = 0; i < pointSet.size(); i++ )
    {
        bool existing;
        unsigned int hexa1DCoord;
        find1DCoord( hexa1DCoord, pointSet[i]);
        hexas.push_back ( hexaGeoAlgo->getTopoIndexFromRegularGridIndex ( hexa1DCoord, existing ) );
        if ( !existing ) msg_error() << "hexa not found. Point (" << pointSet[i] << ") may be outside of the grid.";
    }
}

template<class DataTypes>
void Distances< DataTypes >::find1DCoord ( unsigned int& hexaID, const Coord& point )
{
    const sofa::type::Vec3i& res = hexaContainer->d_resolution.getValue();
    const type::Vec3& voxelSize = hexaContainer->d_voxelSize.getValue();

    const int x = int ((point[0] - d_offset.getValue()[0]) / voxelSize[0]);
    const int y = int ((point[1] - d_offset.getValue()[1]) / voxelSize[1]);
    const int z = int ((point[2] - d_offset.getValue()[2]) / voxelSize[2]);
    hexaID = z*res[0]*res[1] + y*res[0] + x;
}

template<class DataTypes>
void Distances< DataTypes >::getNeighbors ( const core::topology::BaseMeshTopology::HexaID& hexaID, std::set<core::topology::BaseMeshTopology::HexaID>& neighbors ) const
{
    const core::topology::BaseMeshTopology::EdgesInHexahedron& edgeSet = hexaContainer->getEdgesInHexahedron ( hexaID );
    for ( unsigned int i = 0; i < edgeSet.size(); i++ )
    {
        const core::topology::BaseMeshTopology::HexahedraAroundEdge& hexaSet = hexaContainer->getHexahedraAroundEdge ( edgeSet[i] );
        for ( unsigned int j = 0; j < hexaSet.size(); j++ )
            if ( hexaSet[j] != hexaID )
                neighbors.insert ( hexaSet[j] );
    }
}

template<class DataTypes>
void Distances< DataTypes >::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    // Display the distance on each hexa of the grid
    if ( d_showDistanceMap.getValue() )
    {
        constexpr sofa::type::RGBAColor color(1.0f, 0.0f, 0.3f, 1.0f);

        const type::vector<double>& distMap = distanceMap[d_showMapIndex.getValue() % distanceMap.size()];
        for ( unsigned int j = 0; j < distMap.size(); j++ )
        {
            Coord point = hexaGeoAlgo->computeHexahedronRestCenter ( j );
            sofa::type::Vec3 tmpPt = sofa::type::Vec3 ( point[0], point[1], point[2] );
            std::ostringstream oss;
            oss << (distMap[j]);
            vparams->drawTool()->draw3DText(tmpPt, d_showTextScaleFactor.getValue(), color, oss.str().c_str());
        }
    }

}


} // namespace sofa::component::engine::analyze
