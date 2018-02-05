/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_FRAME_HEXA_REMOVER_INL
#define SOFA_FRAME_HEXA_REMOVER_INL


#include "HexaRemover.h"
#include "MeshGenerator.inl"
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.inl>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/CollisionEndEvent.h>


namespace sofa
{

namespace component
{

namespace topology
{

#define X  (current_orientation+1)%3
#define Y  (current_orientation+2)%3

using namespace core::objectmodel;

template <class DataTypes>
HexaRemover<DataTypes>::HexaRemover():
    showElements(core::objectmodel::Base::initData(&showElements, false, "showElements", "Display parsed elements.\nBlue: triangles parsed.\nGreen: parsed hexas out of intersection volumes.\nRed: parsed hexas inside intersection volumes to remove."))
    , showVolumes(core::objectmodel::Base::initData(&showVolumes, false, "showVolumes", "Display intersection volumes"))
{
}




template <class DataTypes>
HexaRemover<DataTypes>::~HexaRemover()
{
}




template <class DataTypes>
void HexaRemover<DataTypes>::init ()
{
    *f_listening.beginEdit() = true;
    f_listening.endEdit();

    std::vector< MeshGen* > v;
    ((simulation::Node*)this->getContext())->get<MeshGen> ( &v,core::objectmodel::BaseContext::SearchDown );

    if ( v.empty() )
    {
        serr << "Init(): No Hexa2TriangleTopologicalMapping in the scene graph." << sendl;
        return;
    }
    meshGen = v[0];
    meshGen->getContext()->get ( sData, core::objectmodel::BaseContext::Local );
    if ( !sData ) {serr << "Init(): Can't find the mechanical mapping" << sendl; return;}

    meshGen->getContext()->get ( triGeoAlgo );
    if ( !triGeoAlgo ) {serr << "Init(): Can't find the triangle geoAlgo component" << sendl; return;}

    getContext()->get ( rasterizer,core::objectmodel::BaseContext::SearchRoot );
    if ( !rasterizer ) serr << "Init(). Unable to find the scene graph component rasterizer" << sendl;


    // List all the cutting models.
    ((simulation::Node*)getContext())->get<MTopology > ( &cuttingModels, Tag ( "Cutting" ), core::objectmodel::BaseContext::SearchDown );

    if ( !cuttingModels.empty() )
    {
        sout << "Cutting Models found:" << sendl;
        for ( helper::vector<MTopology*>::iterator it = cuttingModels.begin(); it != cuttingModels.end(); it++ )
            sout << " " << ( *it )->getName() << sendl;
    }
    else
    {
        sout << "No Cutting Model found." << sendl;
    }
}



template <class DataTypes>
bool HexaRemover<DataTypes>::isTheModelInteresting ( MTopology* model ) const
{
    return ( meshGen->getTo() == model );
}



template <class DataTypes>
void HexaRemover<DataTypes>::findTheCollidedVoxels ( unsigned int triangleID)
{
    sofa::helper::vector< unsigned int > hexasID;
    meshGen->getFromIndex ( hexasID, triangleID );
    if (showElements.getValue())
        collisionTrianglesCoords.insert ( triGeoAlgo->computeTriangleCenter ( triangleID ) );
    for ( sofa::helper::vector< unsigned int >::const_iterator itHexaID = hexasID.begin(); itHexaID != hexasID.end(); ++itHexaID )
    {
        // Propagate to find the hexas to remove
        propagateFrom ( *itHexaID);
    }
}



template <class DataTypes>
void HexaRemover<DataTypes>::propagateFrom ( const unsigned int hexa)
{
    helper::set<unsigned int>& hexaToRemove = this->contactInfos[meshGen].getHexaToRemove();
    helper::set<unsigned int> parsedHexas;
    if ( hexaToRemove.find ( hexa ) != hexaToRemove.end() ) return;
    unsigned int hexaID;
    std::stack<unsigned int> hexasToParse; // Stack of hexas to parse.
    hexasToParse.push ( hexa ); // Adds the first given hexa.
    //const SCoord& halfVSize = meshGen->voxelSize.getValue() / 2.0;
    //const float sphereRadius = sqrt( halfVSize[0] * halfVSize[0] + halfVSize[1] * halfVSize[1] + halfVSize[2] * halfVSize[2]);

    // While the propagation is not finished
    while ( !hexasToParse.empty() )
    {
        hexaID = hexasToParse.top(); // Get the last cube on the stack.
        hexasToParse.pop();          // Remove it from the stack.

        if ( parsedHexas.find ( hexaID ) != parsedHexas.end() )   continue; // if this hexa is ever parsed, continue
        if ( hexaToRemove.find ( hexaID ) != hexaToRemove.end() ) continue; // if this hexa has ever been removed, continue

        // Compute the current voxel position with the mechanical mapping.
        Coord hexaCoord;
        std::map<unsigned int, Vec3d>::iterator itMappedCoord = voxelMappedCoord.find( hexaID);
        if( itMappedCoord == voxelMappedCoord.end())
        {
            Coord hexaRestCoord;
            meshGen->getHexaCoord( hexaRestCoord, hexaID );
            sData->apply(hexaCoord, hexaRestCoord);
            voxelMappedCoord.insert( std::make_pair(hexaID,hexaCoord));
        }
        else
        {
            hexaCoord = itMappedCoord->second;
        }

        parsedHexas.insert ( hexaID );
        if (showElements.getValue())
            parsedHexasCoords.insert ( hexaCoord );

        if ( !isPointInside ( hexaCoord ) )
            continue;

        hexaToRemove.insert ( hexaID );
        if (showElements.getValue())
            removedHexasCoords.insert ( hexaCoord );

        // Propagate to the neighbors
        helper::set<unsigned int> neighbors;
        getNeighbors ( hexaID, neighbors );
        for ( helper::set<unsigned int>::const_iterator it = neighbors.begin(); it != neighbors.end(); it++ )
        {
            hexasToParse.push ( *it );
        }
    }
}


template <class DataTypes>
void HexaRemover<DataTypes>::getNeighbors ( const unsigned int hexaID, helper::set<unsigned int>& neighbors ) const
{
    const GCoord& dim = meshGen->voxelDimension.getValue();
    GCoord gCoord;
    meshGen->gridMat->getiCoord( hexaID, gCoord);

    const unsigned int& x = gCoord[0];
    const unsigned int& y = gCoord[1];
    const unsigned int& z = gCoord[2];

    if (x > (unsigned int) 0) neighbors.insert ( hexaID-1 );
    if (x < (unsigned int) dim[0]-1) neighbors.insert ( hexaID+1 );
    if (y > (unsigned int) 0) neighbors.insert ( hexaID-dim[0] );
    if (y < (unsigned int) dim[1]-1) neighbors.insert ( hexaID+dim[0] );
    if (z > (unsigned int) 0) neighbors.insert ( hexaID-dim[1]*dim[0] );
    if (z < (unsigned int) dim[2]-1) neighbors.insert ( hexaID+dim[1]*dim[0] );
}


template <class DataTypes>
bool HexaRemover<DataTypes>::removeVoxels()
{
    ContactInfos &infos=this->contactInfos[meshGen];
    std::set< unsigned int > items = infos.getHexaToRemove();
    if ( items.empty() )
        return false;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode ( "removeVoxels" );
#endif
    sofa::helper::vector<unsigned int> vitems;
    vitems.reserve ( items.size() );
    vitems.insert ( vitems.end(), items.rbegin(), items.rend() );

    meshGen->removeVoxels( vitems);

    infos.clear();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode ( "removeVoxels" );
    return true;
#endif
}


template <class DataTypes>
void HexaRemover<DataTypes>::detectVoxels()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode ( "detectVoxels" );
#endif
    buildCollisionVolumes();
    for ( std::set<unsigned int>::const_iterator it = trianglesToParse.begin(); it != trianglesToParse.end(); ++it)
        findTheCollidedVoxels ( *it );

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode ( "detectVoxels" );
#endif
}

template <class DataTypes>
void HexaRemover<DataTypes>::buildCollisionVolumes()
{
    MTopology *current_model;
    const gpu::cuda::CudaVector<gpu::cuda::Object>& ldiObjects = rasterizer->ldiObjects;
    const LDI* ldiDir = rasterizer->ldiDir;
    const int nbo = rasterizer->vmtopology.size();

    const int tgsize = gpu::cuda::CudaLDI_get_triangle_group_size();
    helper::vector<int> tgobj;
    if ((int)ldiObjects.size() >= nbo)
        for (int obj=0; obj<nbo; obj++)
        {
            int ntg = (ldiObjects[obj].nbt + tgsize-1)/tgsize;
            while (ntg--)
                tgobj.push_back(obj);
        }

    const Real psize = rasterizer->pixelSize.getValue();

#ifdef DRAW_ONE_LDI
    {
        int axis=0;
#else
    for ( int axis=0; axis<3; ++axis )
    {
#endif
        const LDI& ldi = ldiDir[axis];
        /*
        const int iX = ( axis+1 ) %3;
        const int iY = ( axis+2 ) %3;
        const int iZ =  axis     ;
        */
        if ( ldi.nbcu==0 ) continue;

        for ( int bx=0; bx < ldi.nbcu; ++bx )
        {
            int ci = ldi.haveTriangleStart[bx];
            int cx = ci%ldi.cnx;
            int cy = ci/ldi.cnx;

            const Cell& cell = ldi.cells[bx];
            if ( cell.nbLayers == 0 ) continue;
            static helper::vector<int> layers;
            static helper::vector<int> inobjs;
            //                                                const CellCountLayer& counts = ldi.cellCounts[bx];
            layers.resize ( cell.nbLayers );
            inobjs.resize ( cell.nbLayers );

            for ( int i=0, l = cell.firstLayer; i<cell.nbLayers; ++i )
            {
                layers[i] = l;
                if ( l == -1 )
                {
                    serr << "Invalid2 layer " << i << sendl;
                    layers.resize ( i );
                    break;
                }
                l = ldi.nextLayer[l];
            }

            int cl0 = ( ( ldi.cy0+cy ) << Rasterizer::CELL_YSHIFT );
            int cc0 = ( ( ldi.cx0+cx ) << Rasterizer::CELL_XSHIFT );

            const CellCountLayer& counts = ldi.cellCounts[bx];
            for ( int l=0; l < Rasterizer::CELL_NY; ++l )
            {
                //                                                    Real y = (Real)(cl0 + l) * psize;
                //                                                    Real y0 = y - (psize*0.5);
                //                                                    Real y1 = y + (psize*0.5);
                for ( int c=0; c < Rasterizer::CELL_NX; ++c )
                {
                    //                                                            Real x = (Real)(cc0 + c) * psize;
                    //                                                            Real x0 = x - (psize*0.5);
                    //                                                            Real x1 = x + (psize*0.5);
                    int incount = 0;
                    int nl = counts[l][c];
                    if (nl > (int)layers.size()) nl = layers.size();
                    bool first_front = false;
                    int first_layer = 0;
                    int first_obj = 0;
                    int first_tid = -1;
                    for ( int li=0; li<nl  ; ++li )
                    {
                        int layer = layers[li];

                        int tid = ldi.cellLayers[layer][l][c].tid;

                        //if (tid == -1) break;
                        int tg = (tid >> Rasterizer::SHIFT_TID) / tgsize; // tid >> (SHIFT_TRIANGLE_GROUP_SIZE+SHIFT_TID);
                        int obj = (tg < (int)tgobj.size()) ? tgobj[tg] : -1;

                        if (obj == -1)
                        {
                            serr << "ERROR: tid " << (tid>>Rasterizer::SHIFT_TID) << " object invalid exit draw......." << sendl;
                            glEnd();
                            return;
                        }

                        bool front = ((tid&1) != 0);
                        tid = (tid >> (Rasterizer::SHIFT_TID))- ldiObjects[obj].t0;

                        if ( front )
                        {
                            if ( incount >= 0 ) inobjs[incount] = obj;
                            ++incount;
                        }
                        else
                        {
                            --incount;
                            if ( first_front &&
                                    ( ( obj != first_obj ) || // simple collision
                                            ( obj == first_obj && incount > 0 && obj != inobjs[incount-1]) // collision inside another object
                                    ))
                            {
                                MTopology* first_model = (obj != first_obj) ? rasterizer->vmtopology[first_obj]: rasterizer->vmtopology[inobjs[incount-1]]; // Allow to correctly add the trianglesToParse in the case 'collision inside another object'
                                current_model = rasterizer->vmtopology[obj];
                                if ( ! ( ( isTheModelInteresting ( current_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), first_model ) != cuttingModels.end() ) ||
                                        ( isTheModelInteresting ( first_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), current_model ) != cuttingModels.end() ) ) )
                                    continue;

                                Real y = ( Real ) ( cl0 + l ) * psize;
                                Real x = ( Real ) ( cc0 + c ) * psize;
                                Real z0 = ldi.cellLayers[first_layer][l][c].z;
                                Real z1 = ldi.cellLayers[layer][l][c].z;

                                SReal minDepth, maxDepth;
                                if ( z0 < z1 )
                                {
                                    minDepth = z0;
                                    maxDepth = z1;
                                }
                                else
                                {
                                    minDepth = z1;
                                    maxDepth = z0;
                                }

                                // Store the collision volume and the triangle ID
                                addVolume( collisionVolumes[axis], x, y, minDepth, maxDepth);
                                if ( isTheModelInteresting ( current_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), first_model ) != cuttingModels.end() )
                                    trianglesToParse.insert( tid);
                                else if ( isTheModelInteresting ( first_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), current_model ) != cuttingModels.end() )
                                    trianglesToParse.insert( first_tid);
                            }
                        }
                        first_front = front;
                        first_layer = layer;
                        first_obj = obj;
                        first_tid = tid;
                    }
                }
            }
            glEnd();
        }
    }
}



template <class DataTypes>
void HexaRemover<DataTypes>::addVolume( RasterizedVol& rasterizedVolume, double x, double y, double zMin, double zMax)
{
    RasterizedVol::iterator it = rasterizedVolume.find( x);
    if (it != rasterizedVolume.end())
    {
        it->second.insert( std::pair<double, std::pair<double, double> >( y, std::make_pair<double, double>(zMin, zMax)));
    }
    else
    {
        std::multimap<double, std::pair< double, double> > temp;
        temp.insert( std::pair<double, std::pair<double, double> >( y, std::make_pair<double, double>(zMin, zMax)));
        rasterizedVolume.insert( std::make_pair<double, std::multimap<double, std::pair<double, double> > >( x, temp));
    }
}


template <class DataTypes>
bool HexaRemover<DataTypes>::isCrossingCube( const Vector3& point, const float& radius ) const
{
    for ( unsigned int axis=0; axis<3; ++axis )
    {
        bool crossing = false;
        const int iX = ( axis+1 ) %3;
        const int iY = ( axis+2 ) %3;
        const int iZ = axis;

        const Real psize = rasterizer->pixelSize.getValue();
        const double& xMin = (point[iX]-radius) - psize*0.5;
        const double& xMax = (point[iX]+radius) + psize*0.5;
        const double& yMin = (point[iY]-radius) - psize*0.5;
        const double& yMax = (point[iY]+radius) + psize*0.5;
        const double& zMin = (point[iZ]-radius);
        const double& zMax = (point[iZ]+radius);
        RasterizedVol::const_iterator itMax = collisionVolumes[axis].upper_bound(xMax);
        for (RasterizedVol::const_iterator it = collisionVolumes[axis].lower_bound( xMin); it != itMax; ++it)
        {
            if (crossing) break;
            std::multimap<double, std::pair< double, double> >::const_iterator it2Max = it->second.upper_bound(yMax);
            for (std::multimap<double, std::pair< double, double> >::const_iterator it2 = it->second.lower_bound( yMin); it2 != it2Max; ++it2)
                if (zMin > it2->second.first && zMax < it2->second.second)
                {
                    crossing = true;
                    break;
                }
        }
        if (!crossing) return false;
    }
    return true;
}



template <class DataTypes>
bool HexaRemover<DataTypes>::isPointInside( const Vector3& point ) const
{
    for ( unsigned int axis=0; axis<3; ++axis )
    {
        bool crossing = false;
        const int iX = ( axis+1 ) %3;
        const int iY = ( axis+2 ) %3;
        const int iZ = axis;

        const Real psize = rasterizer->pixelSize.getValue();
        const double& xMin = point[iX] - psize*0.5;
        const double& xMax = point[iX] + psize*0.5;
        const double& yMin = point[iY] - psize*0.5;
        const double& yMax = point[iY] + psize*0.5;
        const double& z = point[iZ];
        RasterizedVol::const_iterator itMax = collisionVolumes[axis].upper_bound(xMax);
        for (RasterizedVol::const_iterator it = collisionVolumes[axis].lower_bound( xMin); it != itMax; ++it)
        {
            if (crossing) break;
            std::multimap<double, std::pair< double, double> >::const_iterator it2Max = it->second.upper_bound(yMax);
            for (std::multimap<double, std::pair< double, double> >::const_iterator it2 = it->second.lower_bound( yMin); it2 != it2Max; ++it2)
                if (z > it2->second.first && z < it2->second.second)
                {
                    crossing = true;
                    break;
                }
        }
        if (!crossing) return false;
    }
    return true;
}


template <class DataTypes>
void HexaRemover<DataTypes>::clear()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode ( "clear" );
#endif
    for( unsigned int i = 0; i < 3; ++i) collisionVolumes[i].clear();
    trianglesToParse.clear();
    voxelMappedCoord.clear();
    contactInfos[meshGen].clear();
    clearDebugVectors();
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode ( "clear" );
#endif
}


template <class DataTypes>
void HexaRemover<DataTypes>::clearDebugVectors()
{
    parsedHexasCoords.clear();
    removedHexasCoords.clear();
    collisionTrianglesCoords.clear();
}


template <class DataTypes>
void HexaRemover<DataTypes>::drawParsedHexas()
{
    glPushAttrib ( GL_ENABLE_BIT );
    glDisable ( GL_LIGHTING );
    glDisable ( GL_COLOR_MATERIAL );
    glDisable ( GL_BLEND );
    glDisable ( GL_TEXTURE_2D );
    glEnable ( GL_DEPTH_TEST );
    glColor3f ( 0,1,0 );
    glPointSize ( 10 );
    glBegin ( GL_POINTS );
    for ( typename helper::set<Coord>::iterator it = parsedHexasCoords.begin(); it != parsedHexasCoords.end(); it++ )
        helper::gl::glVertexT ( *it );
    glEnd();
    glPointSize ( 1 );
    glPopAttrib();
}

template <class DataTypes>
void HexaRemover<DataTypes>::drawRemovedHexas()
{
    glPushAttrib ( GL_ENABLE_BIT );
    glDisable ( GL_LIGHTING );
    glDisable ( GL_COLOR_MATERIAL );
    glDisable ( GL_BLEND );
    glDisable ( GL_TEXTURE_2D );
    glEnable ( GL_DEPTH_TEST );
    glColor3f ( 1,0,0 );
    glPointSize ( 10 );
    glBegin ( GL_POINTS );
    for ( typename helper::set<Coord>::iterator it = removedHexasCoords.begin(); it != removedHexasCoords.end(); it++ )
        helper::gl::glVertexT ( *it );
    glEnd();
    glPointSize ( 1 );
    glPopAttrib();
}


template <class DataTypes>
void HexaRemover<DataTypes>::drawCollisionTriangles()
{
    glPushAttrib ( GL_ENABLE_BIT );
    glDisable ( GL_LIGHTING );
    glDisable ( GL_COLOR_MATERIAL );
    glDisable ( GL_BLEND );
    glDisable ( GL_TEXTURE_2D );
    glEnable ( GL_DEPTH_TEST );
    glColor3f ( 0,0,1 );
    glPointSize ( 10 );
    glBegin ( GL_POINTS );
    for ( typename helper::set<Coord>::iterator it = collisionTrianglesCoords.begin(); it != collisionTrianglesCoords.end(); it++ )
        helper::gl::glVertexT ( *it );
    glEnd();
    glPointSize ( 1 );
    glPopAttrib();
}



template <class DataTypes>
void HexaRemover<DataTypes>::drawCollisionVolumes()
{
    const double& psize = rasterizer->psize;
    for ( unsigned int axis=0; axis<3; ++axis )
    {
        //if ((showWhichAxis.getValue() != 0) && showWhichAxis.getValue() != axis+1) continue;
        const int iX = ( axis+1 ) %3;
        const int iY = ( axis+2 ) %3;
        const int iZ =  axis;

        if ( axis == 0)
            glColor3f( 1, 0, 0);
        else if ( axis == 1)
            glColor3f( 0, 1, 0);
        else if ( axis == 2)
            glColor3f( 0, 0, 1);

        RasterizedVol& rasterizedVolume = collisionVolumes[axis];
        for (RasterizedVol::const_iterator it = rasterizedVolume.begin(); it != rasterizedVolume.end(); ++it)
        {
            double x = it->first;
            const std::multimap<double, std::pair< double, double> >& multiMap = it->second;
            for (std::multimap<double, std::pair< double, double> >::const_iterator it2 = multiMap.begin(); it2 != multiMap.end(); ++it2)
            {
                double y = it2->first;
                double zMin = it2->second.first;
                double zMax = it2->second.second;

                BBox box;
                box[0][iX] = x - ( psize*0.5 );
                box[0][iY] = y - ( psize*0.5 );
                box[0][iZ] = zMin;
                box[1][iX] = x + ( psize*0.5 );
                box[1][iY] = y + ( psize*0.5 );
                box[1][iZ] = zMax;
                if ( axis == 0)
                {
                    glBegin( GL_QUADS);
                    // z min face
                    glVertex3f( box[0][0], box[0][1], box[0][2]);
                    glVertex3f( box[0][0], box[0][1], box[1][2]);
                    glVertex3f( box[0][0], box[1][1], box[1][2]);
                    glVertex3f( box[0][0], box[1][1], box[0][2]);

                    // z max face
                    glVertex3f( box[1][0], box[0][1], box[0][2]);
                    glVertex3f( box[1][0], box[0][1], box[1][2]);
                    glVertex3f( box[1][0], box[1][1], box[1][2]);
                    glVertex3f( box[1][0], box[1][1], box[0][2]);
                    glEnd();
                }
                else if ( axis == 1)
                {
                    glBegin( GL_QUADS);
                    // z min face
                    glVertex3f( box[0][0], box[0][1], box[0][2]);
                    glVertex3f( box[1][0], box[0][1], box[0][2]);
                    glVertex3f( box[1][0], box[0][1], box[1][2]);
                    glVertex3f( box[0][0], box[0][1], box[1][2]);

                    // z max face
                    glVertex3f( box[0][0], box[1][1], box[0][2]);
                    glVertex3f( box[1][0], box[1][1], box[0][2]);
                    glVertex3f( box[1][0], box[1][1], box[1][2]);
                    glVertex3f( box[0][0], box[1][1], box[1][2]);
                    glEnd();
                }
                else if ( axis == 2)
                {
                    glBegin( GL_QUADS);
                    // z min face
                    glVertex3f( box[0][0], box[0][1], box[0][2]);
                    glVertex3f( box[0][0], box[1][1], box[0][2]);
                    glVertex3f( box[1][0], box[1][1], box[0][2]);
                    glVertex3f( box[1][0], box[0][1], box[0][2]);

                    // z max face
                    glVertex3f( box[0][0], box[0][1], box[1][2]);
                    glVertex3f( box[0][0], box[1][1], box[1][2]);
                    glVertex3f( box[1][0], box[1][1], box[1][2]);
                    glVertex3f( box[1][0], box[0][1], box[1][2]);
                    glEnd();
                }
            }
        }
    }
}



/*
                        template <class DataTypes>
                        void HexaRemover<DataTypes>::drawCollisionVolumes()
			{
				glPushAttrib ( GL_ENABLE_BIT );
				glDisable ( GL_LIGHTING );
				glDisable ( GL_COLOR_MATERIAL );
				glDisable ( GL_TEXTURE_2D );
				glEnable ( GL_BLEND );
				glEnable ( GL_DEPTH_TEST );
				glColor3f ( 0,0,1 );
				glPointSize ( 10 );
				for ( unsigned int i = 0; i < 3; i++ )
                                    for ( std::set<BoundingBox>::iterator it = collisionVolumesCoords[i].begin(); it != collisionVolumesCoords[i].end(); it++ )
					{
						if ( i==0 ) glColor4f ( 0.82, 0.04, 0.07, 1.0 );
						else if ( i==1 ) glColor4f ( 0.88, 0.78, 0.17, 1.0 );
						else if ( i==2 ) glColor4f ( 0.23, 0.18, 0.83, 1.0 );
						drawBoundingBox ( *it );
					}
                                glPointSize ( 1 );
                                glPopAttrib();
			}
*/

template <class DataTypes>
void HexaRemover<DataTypes>::drawBoundingBox ( const BoundingBox& bbox )
{
    Real x1 = bbox.first[0];
    Real y1 = bbox.first[1];
    Real z1 = bbox.first[2];
    Real x2 = bbox.second[0];
    Real y2 = bbox.second[1];
    Real z2 = bbox.second[2];

    Coord p0 = Coord ( x1, y1, z1 );
    Coord p1 = Coord ( x2, y1, z1 );
    Coord p2 = Coord ( x2, y2, z1 );
    Coord p3 = Coord ( x1, y2, z1 );
    Coord p4 = Coord ( x1, y1, z2 );
    Coord p5 = Coord ( x2, y1, z2 );
    Coord p6 = Coord ( x2, y2, z2 );
    Coord p7 = Coord ( x1, y2, z2 );

    glBegin ( GL_QUADS );
    // Left
    //glColor4f( 0.82, 0.04, 0.07, 1.0);
    helper::gl::glVertexT ( p0 );
    helper::gl::glVertexT ( p3 );
    helper::gl::glVertexT ( p7 );
    helper::gl::glVertexT ( p4 );

    // Right
    helper::gl::glVertexT ( p1 );
    helper::gl::glVertexT ( p5 );
    helper::gl::glVertexT ( p6 );
    helper::gl::glVertexT ( p2 );

    // Up
    //glColor4f( 0.88, 0.78, 0.17, 1.0);
    helper::gl::glVertexT ( p2 );
    helper::gl::glVertexT ( p6 );
    helper::gl::glVertexT ( p7 );
    helper::gl::glVertexT ( p3 );

    // Bottom
    helper::gl::glVertexT ( p0 );
    helper::gl::glVertexT ( p4 );
    helper::gl::glVertexT ( p5 );
    helper::gl::glVertexT ( p1 );

    // Front
    //glColor4f( 0.23, 0.18, 0.83, 1.0);
    helper::gl::glVertexT ( p0 );
    helper::gl::glVertexT ( p1 );
    helper::gl::glVertexT ( p2 );
    helper::gl::glVertexT ( p3 );

    // Back
    helper::gl::glVertexT ( p4 );
    helper::gl::glVertexT ( p7 );
    helper::gl::glVertexT ( p6 );
    helper::gl::glVertexT ( p5 );

    glEnd();
}


template <class DataTypes>
void HexaRemover<DataTypes>::draw()
{
    if (showElements.getValue())
    {
        drawParsedHexas();
        drawRemovedHexas();
        drawCollisionTriangles();
    }
    if ( showVolumes.getValue())
        drawCollisionVolumes();
}


template <class DataTypes>
void HexaRemover<DataTypes>::handleEvent ( core::objectmodel::Event* ev )
{
    if ( dynamic_cast<sofa::simulation::CollisionEndEvent *> ( ev ) )
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode ( "HexaRemover" );
#endif
        if (rasterizer->getNbPairs()) //Only try to remove voxels when it is necessary
        {
            clear();
            detectVoxels();
            if (removeVoxels()) // Launch a new Collision Detection using the new Meshs
            {
                sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
                ((simulation::Node*)rasterizer->getContext())->execute<simulation::CollisionVisitor>(params);
            }


        }
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printCloseNode ( "HexaRemover" );
#endif
    }
}

}

}

}

#endif
