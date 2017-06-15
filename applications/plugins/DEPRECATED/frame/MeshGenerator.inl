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
#ifndef SOFA_FRAME_MESHGENERATOR_INL
#define SOFA_FRAME_MESHGENERATOR_INL

#include "MeshGenerator.h"

#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <SofaBaseTopology/PointSetGeometryAlgorithms.h>

#include <SofaBaseTopology/TopologyData.inl>

#include <SofaBaseMechanics/MechanicalObject.inl>
#include <SofaLoader/VoxelGridLoader.h>
#include <SofaOpenglVisual/OglAttribute.inl>

#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Visitor.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/glText.inl>
#include <algorithm>
#include <sofa/simulation/Node.h>

#include "GridMaterial.inl"


#define PRECISION 16384.0  // See Marching Cubes PRECISION


namespace sofa
{

namespace component
{

namespace engine
{
using namespace sofa::component::topology;
using namespace sofa::core::topology;
using namespace sofa::core;
using namespace sofa::helper::gl;
using namespace sofa::simulation;

template <class DataTypes>
MeshGenerator<DataTypes>::MeshGenerator ()
    : core::DataEngine (),
      roi ( initData ( &roi, Vec6i ( 0,0,0, 0xFFFF, 0xFFFF, 0xFFFF ), "ROI", "Region of interest (xmin, ymin, zmin, xmax, ymax, zmax)" ) ),
      mIsoValue ( initData ( &mIsoValue, 128.0f, "isoValue", "Iso-value to be used by marching cubes." ) ),
      mCubeSeeds ( initData ( &mCubeSeeds, vector<Vec<3, int> >(), "seeds", "Seeds to be used by marching cubes." ) ),
      smoothIterations ( initData ( &smoothIterations, (unsigned int)0, "smoothIterations", "Number of iterations used to smooth the generated surface." ) ),
      smoothedMesh0 ( initData ( &smoothedMesh0, "smoothedMesh0", "copy of X0 once smoothed." ) ),
      showHexas2Tri ( initData ( &showHexas2Tri, false, "showHexas2Tri", "Show hexas to tri relations." ) ),
      showTri2Hexas ( initData ( &showTri2Hexas, false, "showTri2Hexas", "Show tri to hexas relations." ) ),
      showRegularGridIndices ( initData ( &showRegularGridIndices, false, "showRegularGridIndices", "Show regular grid indices." ) ),
      showTextScaleFactor ( initData ( &showTextScaleFactor, 0.001, "showTextScaleFactor","Scale to apply on the text." ) ),
      gridMat(NULL),
      voxelSize ( initData ( &voxelSize, SCoord(), "voxelSize","Size of the voxels." ) ),
      voxelOrigin ( initData ( &voxelOrigin, SCoord(), "voxelOrigin","Origin of the voxels array." ) ),
      voxelDimension ( initData ( &voxelDimension, GCoord(), "voxelDimension","Dimension of the voxels array." ) )
{
    smoothedMesh0.setDisplayed( false);
}


template <class DataTypes>
MeshGenerator<DataTypes>::~MeshGenerator()
{
}


template <class DataTypes>
void MeshGenerator<DataTypes>::init()
{
    addInput(&roi);
    addInput(&mIsoValue);
    addInput(&mCubeSeeds);

    sout << "Hexa2TriangleTopologicalMapping::init(): Begin." << sendl;

    this->getContext()->get(_to_topo);
    if ( !_to_topo )
    {
        serr << "Hexa2TriangleTopologicalMapping::init(): Error. You must use a TriangleSetTopologyContainer as target in the Hexa2TriangleTopologicalMapping." <<sendl;
        return;
    }

    this->getContext()->get ( _to_tstm );
    if ( !_to_tstm )
    {
        serr << "Hexa2TriangleTopologicalMapping::init(): Error. Can not find the TriangleSetTopologyModifier." << sendl;
        return;
    }

    this->getContext()->get ( _to_geomAlgo );
    if ( !_to_geomAlgo )
    {
        serr << "Hexa2TriangleTopologicalMapping::init(). Error: can not find the TriangleSetGeometryAlgorithms." << sendl;
        return;
    }

    this->_to_DOFs = _to_geomAlgo->getDOF();
    if ( !_to_DOFs )
    {
        serr << "Hexa2TriangleTopologicalMapping::init(). Error: can not find the DOFs of the triangular topology." << sendl;
        return;
    }

    this->getContext()->get ( gridMat, core::objectmodel::BaseContext::SearchRoot );
    if ( !gridMat )
    {
        serr << "GridMaterial not found." << sendl;
        return;
    }

    // Create specific handler for PointData
    smoothedMesh0.createTopologicalEngine(_to_topo);
    smoothedMesh0.registerTopologicalData();

    voxelSize.setParent ( &gridMat->voxelSize);
    voxelOrigin.setParent ( &gridMat->origin);
    voxelDimension.setParent ( &gridMat->dimension);

    roi.setValue ( Vec6i ( -1,-1,-1,0xFF,0xFF,0xFF ) );
    const Vec3i& res = voxelDimension.getValue();

    initVoxels();

    initOglAttributes();

    const defaulttype::Vector3& vSize = voxelSize.getValue();
    const Vec6i &roiValue=roi.getValue();

    // Configuration of the Marching Cubes algorithm
    marchingCubes.setDataResolution ( Vec<3, int> ( res[0], res[1], res[2] ) );
    if ( roiValue != Vec6i() )
    {
        marchingCubes.setROI ( Vector3 ( roiValue[0],roiValue[1],roiValue[2] ), Vector3 ( roiValue[3],roiValue[4],roiValue[5] ) );
        marchingCubes.setBoundingBox ( roiValue );
    }
    marchingCubes.setDataVoxelSize ( Vector3 ( vSize[0], vSize[1], vSize[2] ) );
    marchingCubes.setStep ( 1 );
    marchingCubes.setConvolutionSize ( 0 ); //apply Smoothing on data if convolutionSize > 0
    //sofa::component::container::MechanicalObject<DataTypes>* inputDOFs = static_cast<sofa::component::container::MechanicalObject<DataTypes>*>(_from_DOFs);
    marchingCubes.setVerticesTranslation (voxelOrigin.getValue() - vSize / 2.0);

    if ( triangleIndexInRegularGrid.getValue().empty() )
    {
        helper::WriteAccessor<Data<typename DataTypes::VecCoord> > xto = * _to_DOFs->write(core::VecCoordId::position());
        helper::WriteAccessor<Data<typename DataTypes::VecCoord> > xtoInRestedPos = * _to_DOFs->write(core::VecCoordId::restPosition());

        sofa::helper::vector<Vector3> vertices;
        sofa::helper::vector<unsigned int> triangles;
        helper::vector< helper::vector<unsigned int> > triangleIndexInRegularGrid;

        if ( mCubeSeeds.getValue().empty() )
        {
            sout << "Init of the mCube seeds." << sendl;
            vector<Vec<3, int> >& seeds = *mCubeSeeds.beginEdit();
            marchingCubes.findSeeds ( seeds, mIsoValue.getValue(), &valueData[0] );
            sout << "You can add the following seeds to the scene file by adding to the MeshGenerator component the option 'seeds=\"" << seeds << "\"' to obtain a faster initialization." << sendl;
            mCubeSeeds.endEdit();
        }

        /*/ Test de l'isovalue a fournir. => Prendre une valeur entre le min et le max
        float min = 0xFFFF, max = 0;
        const vector<unsigned char>& minMaxValues = _from_topo->valuesIndexedInTopology.getValue();
        for( unsigned int i = 0; i < minMaxValues.size(); i++)
        {
        if( minMaxValues[i] < min) min = minMaxValues[i];
        if( minMaxValues[i] > max) max = minMaxValues[i];
        }
        sout << "nbValues: " << minMaxValues.size() << " minValue: " << min << ", maxValue: " << max << sendl;
        sout << "nbHexahedra: " << _from_topo->getNbHexahedra() << sendl;
        //*/

        // Run marching cubes => Get the triangular mesh
        marchingCubes.run ( &valueData[0], mCubeSeeds.getValue(), mIsoValue.getValue(), triangles, vertices, &triangleIndexInRegularGrid );

        // Init the triangular model
        _to_topo->clear();
        _to_topo->setNbPoints ( vertices.size() );

        // Init the DOFs (= Insert vertices)
        _to_DOFs->resize ( vertices.size() );

        for ( unsigned int i = 0; i < vertices.size(); ++i )
        {
            xto[i] = vertices[i];
            xtoInRestedPos[i] = vertices[i];
        }

        smoothedMesh0.setValue( xtoInRestedPos.ref());

        // Init the triangular topology (= Insert faces)
        for ( unsigned int i = 0; i < triangles.size() / 3; ++i )
        {
            _to_tstm->addTriangleProcess ( Triangle ( triangles[3*i], triangles[3*i+1], triangles[3*i+2] ) );
        }

        smoothMesh();

        updateOglAttributes();

        updateTrianglesInfos( triangleIndexInRegularGrid);

        _to_tstm->notifyEndingEvent();
    }
    sout << "End." << sendl;

    setDirtyValue();
}


template <class DataTypes>
void MeshGenerator<DataTypes>::initVoxels()
{
    const Vec3i& res = voxelDimension.getValue();
    valueData.resize(res[0]*res[1]*res[2]);
    segmentIDData.resize(res[0]*res[1]*res[2]);
    CImg<VoxelType>& grid = gridMat->getGrid();
    unsigned int i = 0;
    for (int z = 0; z < res[2]; ++z)
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x)
            {
                (grid(x,y,z)==0)?valueData[i]=0:valueData[i]=255;
                segmentIDData[i] = grid(x,y,z);
                ++i;
            }
}


template <class DataTypes>
void MeshGenerator<DataTypes>::initOglAttributes()
{
    sofa::helper::vector< OglFloatAttribute* > vecFloatAttribute;
    segmentationID = NULL;
    static_cast<simulation::Node*>(this->getContext())->get<OglFloatAttribute> ( &vecFloatAttribute, core::objectmodel::BaseContext::SearchDown );
    for ( unsigned int i=0; i<vecFloatAttribute.size(); ++i )
    {
        if ( vecFloatAttribute[i]->getId() == "segmentation" )
        {
            segmentationID=vecFloatAttribute[i];
            segmentationID->setIndexShader ( 0 );
            segmentationID->setUsage( GL_STREAM_DRAW);
        }
    }
    if ( !segmentationID )
    {
        sout << "Can't find the Ogl Attribute component 'segmentation'. Don't use it" << sendl;
    }

    sofa::helper::vector< OglFloat3Attribute* > vecFloat3Attribute;
    restPosition = NULL;
    restNormal = NULL;
    static_cast<simulation::Node*>(this->getContext())->get<OglFloat3Attribute> ( &vecFloat3Attribute, core::objectmodel::BaseContext::SearchDown );
    for ( unsigned int i=0; i<vecFloat3Attribute.size(); ++i )
    {
        if ( vecFloat3Attribute[i]->getId() == "restPosition" )
        {
            restPosition=vecFloat3Attribute[i];
            restPosition->setIndexShader ( 0 );
            restPosition->setUsage( GL_STREAM_DRAW);
        }
        else if ( vecFloat3Attribute[i]->getId() == "restNormal" )
        {
            restNormal=vecFloat3Attribute[i];
            restNormal->setIndexShader ( 0 );
            restNormal->setUsage( GL_STREAM_DRAW);
        }
    }
    if ( !restPosition )
    {
        sout << "Can't find the Ogl Attribute component 'restPosition'. Don't use it" << sendl;
    }
    if ( !restNormal )
    {
        sout << "Can't find the Ogl Attribute component 'restNormal'. Don't use it" << sendl;
    }
}


template <class DataTypes>
void MeshGenerator<DataTypes>::getFromIndex ( vector<unsigned int>& fromIndices, const unsigned int toIndex ) const
{
    fromIndices.clear();
    const vector< vector< HexaIDInRegularGrid > >& triIRG = triangleIndexInRegularGrid.getValue();
    fromIndices = triIRG[toIndex];
}


template <class DataTypes>
void MeshGenerator<DataTypes>::getToIndex ( vector<unsigned int>& toIndices, const unsigned int fromIndex ) const
{
    toIndices.clear();
    const map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > >& triIDirg2iit = triangleIDInRegularGrid2IndexInTopo.getValue();
    typename map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > >::const_iterator itTri = triIDirg2iit.find( fromIndex);
    if( itTri != triIDirg2iit.end())
        for( typename ElementSet< BaseMeshTopology::TriangleID >::const_iterator it = itTri->second.begin(); it != itTri->second.end(); ++it)
            toIndices.push_back( *it);
}


/*
template <class DataTypes>
void MeshGenerator<DataTypes>::updateTopologicalMappingTopDown()
{
        // Handle topological changes on the hexa topology
        std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
        std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

        while ( itBegin != itEnd )
        {
                TopologyChangeType changeType = ( *itBegin )->getChangeType();

                switch ( changeType )
                {

                case core::topology::ENDING_EVENT:
                        {
                                // sout << "INFO_print : MeshGenerator - ENDING_EVENT" << sendl;
                                _to_tstm->propagateTopologicalChanges();
                                _to_tstm->notifyEndingEvent();
                                _to_tstm->propagateTopologicalChanges();
                                break;
                        }

                case core::topology::TRIANGLESREMOVED:
                        {
                                //sout << "INFO_print : MeshGenerator - TRIANGLESREMOVED" << sendl;
                                // Nothing to do.
                                break;
                        }

                case core::topology::HEXAHEDRAREMOVED:
                        {
                                const sofa::helper::vector<BaseMeshTopology::PointID> &tab = ( static_cast< const HexahedraRemoved *> ( *itBegin ) )->getArray();
                                vector<BaseMeshTopology::PointID> removedHexahedraID;

                                removeOldMesh( removedHexahedraID, tab);
                                localyRemesh ( removedHexahedraID );
                                break;
                        }

                case core::topology::POINTSREMOVED:
                        {
                                // sout << "INFO_print : MeshGenerator - POINTSREMOVED" << sendl;
                                // do nothing
                                break;
                        }

                case core::topology::POINTSRENUMBERING:
                        {
                                // sout << "INFO_print : MeshGenerator - POINTSRENUMBERING" << sendl;
                                // do nothing
                                break;
                        }

                default:
                        // Ignore events that are not Triangle related.
                        break;
                };

                ++itBegin;
        }
        _to_tstm->propagateTopologicalChanges();
}
*/

template <class DataTypes>
void MeshGenerator<DataTypes>::removeVoxels ( const sofa::helper::vector<unsigned int>& removedHexahedraID )
{
    // Delete voxels
    gridMat->removeVoxels( removedHexahedraID);
    for (sofa::helper::vector<unsigned int>::const_iterator it = removedHexahedraID.begin(); it != removedHexahedraID.end(); ++it)
    {
        valueData[*it] = 0;
        segmentIDData[*it] = 0;
    }

    // Update mesh
    removeOldMesh( removedHexahedraID );
    localyRemesh ( removedHexahedraID );

    _to_tstm->propagateTopologicalChanges();
    _to_tstm->notifyEndingEvent();
    _to_tstm->propagateTopologicalChanges();
}


template <class DataTypes>
void MeshGenerator<DataTypes>::update()
{
    // Not implemented. As the voxel array is not an input data, the method 'removeVoxels' must be explicit called to update the ouput topology.
}


template <class DataTypes>
void MeshGenerator<DataTypes>::removeOldMesh ( const sofa::helper::vector<unsigned int>& removedHexahedraID )
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Remove_old_mesh");
    simulation::Visitor::printNode("Remove_Triangles_From_Topology");
#endif
    // List all the triangles corresponding to the removed hexahedra
    set< BaseMeshTopology::TriangleID > trianglesToRemoveSet;
    for ( vector<unsigned int>::const_iterator itHexaRemoved = removedHexahedraID.begin(); itHexaRemoved != removedHexahedraID.end(); ++itHexaRemoved )
    {
        vector<unsigned int> toIndices;
        getToIndex ( toIndices, *itHexaRemoved );
        for( vector<unsigned int>::iterator itTriangleToRemove = toIndices.begin(); itTriangleToRemove != toIndices.end(); ++itTriangleToRemove)
        {
            trianglesToRemoveSet.insert ( *itTriangleToRemove );
        }
    }

    //  Remove them from the topology
    vector< BaseMeshTopology::TriangleID > trianglesToRemove;
    for ( set< BaseMeshTopology::TriangleID >::const_reverse_iterator it = set< BaseMeshTopology::TriangleID >::const_reverse_iterator ( trianglesToRemoveSet.end() ); it != set< BaseMeshTopology::TriangleID >::const_reverse_iterator ( trianglesToRemoveSet.begin() ); ++it )
        trianglesToRemove.push_back ( *it );
    _to_tstm->removeTriangles ( trianglesToRemove, true, true );

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Remove_Triangles_From_Topology");
    simulation::Visitor::printNode("Update_Struct_Related_To_Triangle");
#endif
    // Renumber the triangles indices in regular grid.
    vector< vector< HexaIDInRegularGrid > >& triIDirg = *triangleIndexInRegularGrid.beginEdit();
    map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > >& triIDirg2iit = *triangleIDInRegularGrid2IndexInTopo.beginEdit();

    unsigned int lastElt = triIDirg.size();
    for ( set< BaseMeshTopology::TriangleID >::const_reverse_iterator itTriRemoved = set< BaseMeshTopology::TriangleID >::const_reverse_iterator ( trianglesToRemoveSet.end() ); itTriRemoved != set< BaseMeshTopology::TriangleID >::const_reverse_iterator ( trianglesToRemoveSet.begin() ); ++itTriRemoved )
    {
        lastElt--;
        vector< HexaIDInRegularGrid >& idirgSet = triIDirg[*itTriRemoved];
        for( vector< HexaIDInRegularGrid >::iterator itIDirg =  idirgSet.begin(); itIDirg != idirgSet.end(); ++itIDirg)
        {
            typename map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > >::iterator itMap = triIDirg2iit.find( *itIDirg);
            if( itMap != triIDirg2iit.end())
            {
                itMap->second.erase( *itTriRemoved);
                if( itMap->second.empty()) triIDirg2iit.erase( itMap);
            }
        }

        // Then, we change the id of the last elt moved in the topology.
        if( *itTriRemoved != lastElt)
        {
            idirgSet = triIDirg[lastElt];
            for( vector< HexaIDInRegularGrid >::iterator itIDirg =  idirgSet.begin(); itIDirg != idirgSet.end(); ++itIDirg)
            {
                typename map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > >::iterator itMap = triIDirg2iit.find( *itIDirg);
                if( itMap != triIDirg2iit.end())
                {
                    itMap->second.erase( lastElt);
                    itMap->second.insert( *itTriRemoved);
                }
            }
        }

        triIDirg[*itTriRemoved] = triIDirg[lastElt];
    }
    triIDirg.resize( lastElt);

    triangleIndexInRegularGrid.endEdit();
    triangleIDInRegularGrid2IndexInTopo.endEdit();


    if (_to_topo->getNumberOfTriangles() != triangleIndexInRegularGrid.getValue().size())
        serr << "Error when removing !! triangleIndexInRegularGrid has a wrong size. tri in topo: " << _to_topo->getNumberOfTriangles() << ", TidInRG: " << triangleIndexInRegularGrid.getValue().size() << sendl;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Update_Struct_Related_To_Triangle");
    simulation::Visitor::printCloseNode("Remove_old_mesh");
#endif
}


template <class DataTypes>
void MeshGenerator<DataTypes>::localyRemesh ( const sofa::helper::vector<BaseMeshTopology::PointID> &removedHexaID )
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Localy_remesh");
#endif
    vector< Vector3 > vertices;
    vector< unsigned int> triangles;
    unsigned int oldVertSize = _to_DOFs->getX0()->size();
    unsigned int oldTriSize = _to_topo->getNumberOfTriangles();

    computeNewMesh ( vertices, triangles, removedHexaID );

    addNewEltsInTopology (vertices, triangles );

    smoothMesh ( oldVertSize, oldTriSize);

    updateOglAttributes ( oldVertSize, oldTriSize);

    _to_tstm->notifyEndingEvent();
    _to_tstm->propagateTopologicalChanges();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Localy_remesh");
#endif
}


template <class DataTypes>
void MeshGenerator<DataTypes>::computeNewMesh ( vector< Vector3 >& vertices, vector< unsigned int>& triangles, const sofa::helper::vector<BaseMeshTopology::PointID> &removedHexaID )
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_New_Mesh");
    simulation::Visitor::printNode("Vertices_Init");
#endif
    //Copy data ever existing: vertices and triangles data.
    //container::MechanicalObject<MechanicalBarycentricMapping::OutDataTypes>::VecCoord& xtoInRestedPos = *_to_DOFs->getX0();
    Vec3i resolution = voxelDimension.getValue();

    // Find the 1 neighbor ring of removed hexahedra.
    set<unsigned int> hexaNeighbors;
    unsigned int yOffset = resolution[0];
    unsigned int zOffset = resolution[0]*resolution[1];
    unsigned int maxX = resolution[0]-1, maxY = resolution[1]-1, maxZ = resolution[2]-1;

    for( sofa::helper::vector<BaseMeshTopology::PointID>::const_iterator it = removedHexaID.begin(); it != removedHexaID.end(); ++it)
    {
        const unsigned int& hexaID = *it;
        unsigned int z = (unsigned int)( hexaID / (resolution[0]*resolution[1]));
        unsigned int remain = ( hexaID % (resolution[0]*resolution[1]));
        unsigned int y = (unsigned int)( remain / resolution[0]);
        unsigned int x = remain % resolution[0]; //(unsigned int)( *it - y*resolution[0] - z*resolution[0]*resolution[1]);

        if( z > 0)
        {
            if( y > 0)
            {
                if( x > 0) hexaNeighbors.insert( *it - 1 - yOffset - zOffset);
                if( x < maxX) hexaNeighbors.insert( *it + 1 - yOffset - zOffset);
                hexaNeighbors.insert( *it - yOffset - zOffset);
            }
            if( y < maxY)
            {
                if( x > 0) hexaNeighbors.insert( *it - 1 + yOffset - zOffset);
                if( x < maxX) hexaNeighbors.insert( *it + 1 + yOffset - zOffset);
                hexaNeighbors.insert( *it + yOffset - zOffset);
            }
            if( x > 0) hexaNeighbors.insert( *it - 1 - zOffset);
            if( x < maxX) hexaNeighbors.insert( *it + 1 - zOffset);
            hexaNeighbors.insert( *it - zOffset);
        }
        if( z  < maxZ)
        {
            if( y > 0)
            {
                if( x > 0) hexaNeighbors.insert( *it - 1 - yOffset + zOffset);
                if( x < maxX) hexaNeighbors.insert( *it + 1 - yOffset + zOffset);
                hexaNeighbors.insert( *it - yOffset + zOffset);
            }
            if( y < maxY)
            {
                if( x > 0) hexaNeighbors.insert( *it - 1 + yOffset + zOffset);
                if( x < maxX) hexaNeighbors.insert( *it + 1 + yOffset + zOffset);
                hexaNeighbors.insert( *it + yOffset + zOffset);
            }
            if( x > 0) hexaNeighbors.insert( *it - 1 + zOffset);
            if( x < maxX) hexaNeighbors.insert( *it + 1 + zOffset);
            hexaNeighbors.insert( *it + zOffset);
        }
        if( y > 0)
        {
            if( x > 0) hexaNeighbors.insert( *it - 1 - yOffset);
            if( x < maxX) hexaNeighbors.insert( *it + 1 - yOffset);
            hexaNeighbors.insert( *it - yOffset);
        }
        if( y < maxY)
        {
            if( x > 0) hexaNeighbors.insert( *it - 1 + yOffset);
            if( x < maxX) hexaNeighbors.insert( *it + 1 + yOffset);
            hexaNeighbors.insert( *it     + yOffset);
        }
        if( x > 0) hexaNeighbors.insert( *it - 1);
        if( x < maxX) hexaNeighbors.insert( *it + 1);
    }

    // Delete the redundant elements.
    for( sofa::helper::vector<BaseMeshTopology::PointID>::const_iterator it = removedHexaID.begin(); it != removedHexaID.end(); ++it)
        hexaNeighbors.erase( *it);

    // Get the local vertices
    const typename DataTypes::VecCoord& xto0 = *_to_DOFs->getX0();
    vector< Vector3 > tmpVertices;
    std::map< Vector3, PointID> map_vertices;
    const SeqTriangles& seqTriangles = _to_topo->getTriangles();
    for( set<unsigned int>::iterator itHexa = hexaNeighbors.begin(); itHexa != hexaNeighbors.end(); ++itHexa)
    {
        vector<unsigned int> triangleSet;
        getToIndex( triangleSet, *itHexa);
        for( vector<unsigned int>::const_iterator itTri = triangleSet.begin(); itTri != triangleSet.end(); ++itTri)
        {
            const Triangle& pointSet = seqTriangles[*itTri];
            for( Triangle::const_iterator itPoint = pointSet.begin(); itPoint != pointSet.end(); ++itPoint)
            {
                Vec3d tmpCoord = Vec3d ( int ( xto0[*itPoint][0] * PRECISION ) / PRECISION, int ( xto0[*itPoint][1] * PRECISION ) / PRECISION, int ( xto0[*itPoint][2] * PRECISION ) / PRECISION );
                if( map_vertices.find( tmpCoord) == map_vertices.end())
                {
                    tmpVertices.push_back( tmpCoord);
                    map_vertices.insert ( std::make_pair ( tmpCoord, *itPoint ) );
                }
            }
        }
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Vertices_Init");
    simulation::Visitor::printNode("Find_Seeds");
#endif
    // get all the mCube indices
    set< Vec3i > cubes;
    for( sofa::helper::vector<BaseMeshTopology::PointID>::const_iterator it = removedHexaID.begin(); it != removedHexaID.end(); ++it)
    {
        const unsigned int& hexaID = *it;
        unsigned int z = (unsigned int)( hexaID / (resolution[0]*resolution[1]));
        unsigned int remain = ( hexaID % (resolution[0]*resolution[1]));
        unsigned int y = (unsigned int)( remain / resolution[0]);
        unsigned int x = remain % resolution[0];

        cubes.insert( Vec3i( x  , y  , z  ));
        cubes.insert( Vec3i( x-1, y  , z  ));
        cubes.insert( Vec3i( x  , y-1, z  ));
        cubes.insert( Vec3i( x  , y  , z-1));
        cubes.insert( Vec3i( x-1, y-1, z  ));
        cubes.insert( Vec3i( x  , y-1, z-1));
        cubes.insert( Vec3i( x-1, y  , z-1));
        cubes.insert( Vec3i( x-1, y-1, z-1));
    }

    vector<Vec<3, int> >& seeds = *mCubeSeeds.beginEdit();
    seeds.clear();
    for( set< Vec3i >::iterator it = cubes.begin(); it != cubes.end(); ++it)
    {seeds.push_back( *it);}
    mCubeSeeds.endEdit();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Find_Seeds");
    simulation::Visitor::printNode("MCube_Algo");
#endif
    // run sans propagate.
    unsigned int oldVerticesSize = tmpVertices.size();
    vector< vector<unsigned int> > triangleIndexInRegularGrid;
    marchingCubes.setVerticesIndexOffset( xto0.size() - oldVerticesSize);
    marchingCubes.run ( &valueData[0], mCubeSeeds.getValue(), mIsoValue.getValue(), triangles, tmpVertices, map_vertices, &triangleIndexInRegularGrid, false );

    // Copy all the new vertices
    for( unsigned int i = oldVerticesSize; i < tmpVertices.size(); ++i)
        vertices.push_back( tmpVertices[i]);

    serr << "trInRG size: " <<  this->triangleIndexInRegularGrid.getValue().size() << sendl;
    serr << "triangles to insert: " <<  triangles.size() << sendl;
    serr << "trInRG to insert: " <<  triangleIndexInRegularGrid.size() << sendl;

    updateTrianglesInfos( triangleIndexInRegularGrid);

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("MCube_Algo");
    simulation::Visitor::printCloseNode("Compute_New_Mesh");
#endif
}


template <class DataTypes>
void MeshGenerator<DataTypes>::addNewEltsInTopology ( const sofa::helper::vector< Vector3 >& vertices, const sofa::helper::vector< unsigned int>& triangles )
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Add_New_Elements_In_Topology");
    simulation::Visitor::printNode("Add_points_In_Topology");
#endif

    helper::WriteAccessor<Data<typename DataTypes::VecCoord> > xto = * _to_DOFs->write(core::VecCoordId::position());
    helper::WriteAccessor<Data<typename DataTypes::VecCoord> > xtoInRestedPos = * _to_DOFs->write(core::VecCoordId::restPosition());
    unsigned int oldVertSize = xto.size();

    _to_tstm->addPointsProcess ( vertices.size() );
    _to_tstm->addPointsWarning ( vertices.size() );

    for ( unsigned int i = 0; i < vertices.size(); ++i )
    {
        xtoInRestedPos[i+oldVertSize] = vertices[i];
        xto[i+oldVertSize] = vertices[i];
    }

    _to_tstm->propagateTopologicalChanges();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Add_points_In_Topology");
    simulation::Visitor::printNode("Add_triangles_In_Topology");
#endif

    serr << "Nb tri in topo: " <<  _to_topo->getNumberOfTriangles() << sendl;

    // Init the triangular topology (= Insert faces)
    int nb_elems = _to_topo->getNumberOfTriangles();
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< TriangleID > trianglesIndexList;

    for ( unsigned int i = 0; i < triangles.size(); i+=3 )
    {
        Triangle tri ( triangles[i], triangles[i+1], triangles[i+2] );
        triangles_to_create.push_back ( tri );
        trianglesIndexList.push_back ( nb_elems );
        ++nb_elems;
    }

    _to_tstm->addTrianglesProcess ( triangles_to_create );
    _to_tstm->addTrianglesWarning ( triangles_to_create.size(), triangles_to_create, trianglesIndexList );
    _to_tstm->propagateTopologicalChanges();

    if (_to_topo->getNumberOfTriangles() != triangleIndexInRegularGrid.getValue().size())
        serr << "Error when inserting !! triangleIndexInRegularGrid has a wrong size. tri in topo: " << _to_topo->getNumberOfTriangles() << ", TidInRG: " << triangleIndexInRegularGrid.getValue().size() << sendl;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Add_triangles_In_Topology");
    simulation::Visitor::printCloseNode("Add_New_Elements_In_Topology");
#endif
}


template <class DataTypes>
void MeshGenerator<DataTypes>::smoothMesh ( const unsigned int oldVertSize, const unsigned int oldTriSize)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("smooth_Mesh");
#endif
    // Get points around the new mesh
    helper::WriteAccessor<Data<typename DataTypes::VecCoord> > xto = * _to_DOFs->write(core::VecCoordId::position());
    //helper::WriteAccessor<Data<MState::VecCoord> > xtoInRestedPos = * _to_DOFs->write(core::VecCoordId::restPosition());
    const typename DataTypes::VecCoord& xto0 = *_to_DOFs->getX0();
    const SeqTriangles& triSeq = _to_topo->getTriangles();

    helper::vector<unsigned int> border;

    if( oldVertSize != 0)
    {
        helper::vector<unsigned int> newPatchVertices;
        for ( unsigned int i = oldTriSize; i < triSeq.size(); ++i )
            for ( unsigned int j = 0; j < 3; ++j )
                newPatchVertices.push_back ( triSeq[i][j] );
        std::sort( newPatchVertices.begin(), newPatchVertices.end());
        newPatchVertices.erase(std::unique( newPatchVertices.begin(), newPatchVertices.end()),newPatchVertices.end() );

        helper::vector<unsigned int> newVertices;
        for( unsigned int i = oldVertSize; i < xto0.size(); ++i)
            newVertices.push_back( i);

        std::set_difference( newPatchVertices.begin(), newPatchVertices.end(), newVertices.begin(), newVertices.end(), std::back_inserter(border));
    }//*/

    const typename DataTypes::VecCoord& smoothedxto0 = smoothedMesh0.getValue();
    typename DataTypes::VecCoord vrestposCpy;
    vrestposCpy.resize( _to_topo->getNumberOfTriangles());
    for ( helper::vector<unsigned int>::iterator itPoints = border.begin(); itPoints != border.end(); ++itPoints )
        vrestposCpy[*itPoints] = smoothedxto0[*itPoints];

    // Smooth
    for( unsigned int j = 0; j < smoothIterations.getValue(); ++j)
    {
        for ( unsigned int i = oldVertSize; i < xto.size(); ++i )
            vrestposCpy[i] = xto[i];

        for ( unsigned int i = oldVertSize; i < xto.size(); ++i )
        {
            unsigned int nbPointAround = 1;
            //sofa::helper::set<unsigned int> pointSet;
            const TrianglesAroundVertex& triSet = _to_topo->getTrianglesAroundVertex ( i );
            for ( TrianglesAroundVertex::const_iterator itTriSet = triSet.begin(); itTriSet != triSet.end(); ++itTriSet )
            {
                xto[i] +=  vrestposCpy[triSeq[*itTriSet][0]] + vrestposCpy[triSeq[*itTriSet][1]] + vrestposCpy[triSeq[*itTriSet][2]];
                nbPointAround += 3;
            }
            xto[i] /= nbPointAround;
        }
    }

#ifndef SOFA_SMP
    // Update smoothedMeshxto0 with new smoothed mesh at rest
    typename DataTypes::VecCoord& smoothedMeshxto0 = *(smoothedMesh0.beginEdit());
    smoothedMeshxto0.resize ( xto.size() );
    for ( unsigned int i = oldVertSize; i < xto.size(); ++i )
        smoothedMeshxto0[i] = xto[i];
    smoothedMesh0.endEdit();
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("smooth_Mesh");
#endif
}


template <class DataTypes>
void MeshGenerator<DataTypes>::updateOglAttributes ( const unsigned int oldVertSize, const unsigned int oldTriSize)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("UpdateOglAttributes");
#endif
    //const MState::VecCoord& xto0 = *_to_DOFs->getX0();
    const typename DataTypes::VecCoord& xto = *_to_DOFs->getX();
    const SeqTriangles& triSeq = _to_topo->getTriangles();
    if ( segmentationID )
    {

#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode("Update_segmentationID");
#endif
        ResizableExtVector<float>& segmentID = * ( segmentationID->beginEdit() );

        segmentID.resize ( xto.size() );

        typename DataTypes::Coord coord;
        unsigned int element;
        const typename GridMat::GCoord& gridDim = voxelDimension.getValue();
        const typename GridMat::SCoord& vSize = voxelSize.getValue();
        for ( unsigned int i = oldVertSize; i < xto.size(); ++i )
        {
            coord = _to_geomAlgo->getPointRestPosition ( i );
            for (unsigned int j = 0; j < 3; ++j)
                coord[j] = (coord[j] - voxelOrigin.getValue()[j]) / vSize[j];
            element = coord[0] + coord[1] * gridDim[0] + coord[2] * gridDim[0] * gridDim[1];
            segmentID[i] = segmentIDData[element];
        }
        segmentationID->endEdit();
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printCloseNode("Update_segmentationID");
#endif
    }

    if ( restPosition && restNormal )
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode("Update_restPosition");
#endif
        ResizableExtVector<ExtVec3fTypes::Coord>& vrestpos = * ( restPosition->beginEdit() );
        vrestpos.resize ( xto.size() );

        for ( unsigned int i = oldVertSize; i < xto.size(); ++i )
        {
            vrestpos[i] = xto[i];
        }

        ResizableExtVector<ExtVec3fTypes::Coord>& vrestnormals = * ( restNormal->beginEdit() );
        vrestnormals.resize ( xto.size() );

        sofa::helper::set<unsigned int> points;
        for ( unsigned int i = oldTriSize; i < _to_topo->getNumberOfTriangles() ; ++i )
        {
            points.insert ( triSeq[i][0] );
            points.insert ( triSeq[i][1] );
            points.insert ( triSeq[i][2] );
        }
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printCloseNode("Update_restPosition");
        simulation::Visitor::printNode("Update_restNormal");
#endif
        for ( sofa::helper::set<unsigned int>::iterator itPoints = points.begin(); itPoints != points.end(); ++itPoints )
        {
            vrestnormals[*itPoints].clear();
            const TrianglesAroundVertex& triSet = _to_topo->getTrianglesAroundVertex ( *itPoints );
            for ( TrianglesAroundVertex::const_iterator itTriSet = triSet.begin(); itTriSet != triSet.end(); ++itTriSet )
            {
                const ExtVec3fTypes::Coord  v1 = xto[triSeq[*itTriSet][0]];
                const ExtVec3fTypes::Coord  v2 = xto[triSeq[*itTriSet][1]];
                const ExtVec3fTypes::Coord  v3 = xto[triSeq[*itTriSet][2]];
                ExtVec3fTypes::Coord n = cross ( v2-v1, v3-v1 );

                n.normalize();
                vrestnormals[*itPoints] += n;
            }
            vrestnormals[*itPoints].normalize();
        }
        restPosition->endEdit();
        restNormal->endEdit();
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printCloseNode("Update_restNormal");
#endif
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("UpdateOglAttributes");
#endif
}


template <class DataTypes>
void MeshGenerator<DataTypes>::updateTrianglesInfos( const vector< vector<unsigned int> >& triIDirg)
{
    // Add all the regular grid indices of the new triangles to the 'triangleIndexInRegularGrid' struct
    vector< vector< HexaIDInRegularGrid > >& triangleIDirg = *triangleIndexInRegularGrid.beginEdit();
    map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > >& triIDirg2IDit = *triangleIDInRegularGrid2IndexInTopo.beginEdit();
    unsigned int nbTriangle = triangleIDirg.size();
    for( vector< vector<unsigned int> >::const_iterator it = triIDirg.begin(); it != triIDirg.end(); ++it)
    {
        triangleIDirg.push_back( *it);
        for( vector<unsigned int>::const_iterator itTri = it->begin(); itTri != it->end(); ++itTri)
            triIDirg2IDit[*itTri].insert( nbTriangle);
        ++nbTriangle;
    }
    triangleIDInRegularGrid2IndexInTopo.endEdit();
    triangleIndexInRegularGrid.endEdit();
}


template <class DataTypes>
void MeshGenerator<DataTypes>::draw()
{
    //if ( !getContext()->getShowMappings() ) return;

    const double& scaleFactor = showTextScaleFactor.getValue();

    if( showHexas2Tri.getValue())
    {
        // Display the connecting map H2T.
        glDisable ( GL_LIGHTING );
        glPointSize ( 10 );
        glBegin ( GL_LINES );
        unsigned int nbVoxels = voxelDimension.getValue()[0] * voxelDimension.getValue()[1] * voxelDimension.getValue()[2];
        for ( unsigned int i = 0; i < nbVoxels; ++i)
        {
            Vec3d hexaCoord;
            if( !gridMat->getCoord( i, hexaCoord)) continue;
            vector<unsigned int> triID;
            getToIndex( triID, i);
            for ( vector<unsigned int>::const_iterator itTri = triID.begin(); itTri != triID.end(); ++itTri)
            {
                Vec3d triCoord = _to_geomAlgo->computeTriangleCenter ( *itTri );

                glColor3f ( 1,1,1 );
                helper::gl::glVertexT ( hexaCoord );
                glColor3f ( 0,0,1 );
                helper::gl::glVertexT ( triCoord );
            }
        }
        glEnd();
        glPointSize ( 1 );
    }

    if( showTri2Hexas.getValue())
    {
        // Display the connecting vector T2H.
        glDisable ( GL_LIGHTING );
        glPointSize ( 10 );
        glBegin ( GL_LINES );
        for ( int i = 0; i < _to_topo->getNbTriangles(); ++i)
        {
            Vec3d triCoord = _to_geomAlgo->computeTriangleCenter ( i );
            vector<unsigned int> hexaID;
            getFromIndex( hexaID, i);
            for ( vector<unsigned int>::const_iterator itHex = hexaID.begin(); itHex != hexaID.end(); ++itHex )
            {
                Vec3d hexaCoord;
                if( !gridMat->getCoord( *itHex, hexaCoord)) continue;

                glColor3f ( 1,1,1 );
                helper::gl::glVertexT ( hexaCoord );
                glColor3f ( 0,0,1 );
                helper::gl::glVertexT ( triCoord );
            }
        }
        glEnd();
        glPointSize ( 1 );
    }

    if( showRegularGridIndices.getValue())
    {
        // Display the regular grid indices
        glColor4f ( 1,1,0,1 );
        unsigned int nbVoxels = voxelDimension.getValue()[0] * voxelDimension.getValue()[1] * voxelDimension.getValue()[2];
        for ( unsigned int i = 0; i < nbVoxels; ++i)
        {
            Vec3i gCoord;
            Vec3d hexaCoord;
            gridMat->getiCoord( i, gCoord);
            int label = gridMat->grid( gCoord[0], gCoord[1], gCoord[2]);
            if( !gridMat->getCoord( i, hexaCoord) || (label==0)) continue;
            GlText::draw ( i, hexaCoord, scaleFactor );
        }
    }
}


template <class DataTypes>
void MeshGenerator<DataTypes>::handleEvent ( core::objectmodel::Event * /*event*/ )
{
    std::list<const TopologyChange *>::const_iterator itBegin= _to_topo->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd= _to_topo->endChange();
    while( itBegin != itEnd)
    {
        TopologyChangeType changeType = ( *itBegin )->getChangeType();
        if( changeType == core::topology::TRIANGLESREMOVED)
            serr << " ################ Triangles removed" << sendl;
        if( changeType == core::topology::POINTSREMOVED || changeType == core::topology::POINTSRENUMBERING)
            serr << " ################ Points removed" << sendl;
        ++itBegin;
    }
}


template <class DataTypes>
void MeshGenerator<DataTypes>::dispMaps() const
{
    serr << "triangleIndexInRegularGrid:" << sendl;
    unsigned int cpt = 0;
    for( vector< vector< HexaIDInRegularGrid > >::const_iterator it = triangleIndexInRegularGrid.getValue().begin(); it != triangleIndexInRegularGrid.getValue().end(); ++it)
    {
        serr << cpt << " - ";
        for( vector< HexaIDInRegularGrid >::const_iterator itVec = it->begin(); itVec != it->end(); ++itVec)
        {
            serr << *itVec << " ";
        }
        serr << sendl;
        ++cpt;
    }

    serr << "triangleIDInRegularGrid2IndexInTopo:" << sendl;
    for( typename map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > >::const_iterator it = triangleIDInRegularGrid2IndexInTopo.getValue().begin(); it != triangleIDInRegularGrid2IndexInTopo.getValue().end(); ++it)
    {
        const ElementSet< BaseMeshTopology::TriangleID >& tmp = it->second;
        serr << it->first << " - ";
        for( typename ElementSet< BaseMeshTopology::TriangleID >::const_iterator itTmp = tmp.begin(); itTmp != tmp.end(); ++itTmp)
        {
            serr << *itTmp << " ";
        }
        serr << sendl;
    }

}


template <class DataTypes>
void MeshGenerator<DataTypes>::getHexaCoord( Coord& coord, const unsigned int hexaID) const
{
    SCoord sCoord;
    if( !gridMat->getCoord( hexaID, sCoord)) serr << "Warning: unexisting coord ID." << sendl;

    for (unsigned int i = 0; i < 3; ++i) coord[i] = sCoord[i];
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_FRAME_MESHGENERATOR_H
