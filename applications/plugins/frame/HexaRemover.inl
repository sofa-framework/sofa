#ifndef SOFA_FRAME_HEXA_REMOVER_INL
#define SOFA_FRAME_HEXA_REMOVER_INL


#include "HexaRemover.h"
#include "MeshGenerater.inl"
#include <sofa/component/topology/DynamicSparseGridTopologyModifier.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.inl>

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/CollisionEndEvent.h>


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
void HexaRemover<DataTypes>::findTheCollidedVoxels ( unsigned int triangleID, const Vector3& minBBVolume, const Vector3& maxBBVolume )
{
    sofa::helper::vector< unsigned int > hexasID;
    meshGen->getFromIndex ( hexasID, triangleID );
    if (showElements.getValue())
        collisionTrianglesCoords.insert ( triGeoAlgo->computeTriangleCenter ( triangleID ) );
    for ( sofa::helper::vector< unsigned int >::const_iterator itHexaID = hexasID.begin(); itHexaID != hexasID.end(); ++itHexaID )
    {
        // Propagate to find the hexas to remove
        propagateFrom ( *itHexaID, minBBVolume,maxBBVolume );
    }
}



template <class DataTypes>
void HexaRemover<DataTypes>::propagateFrom ( const unsigned int hexa, const Vector3& minBBVolume, const Vector3& maxBBVolume )
{
    helper::set<unsigned int>& hexaToRemove = this->contactInfos[meshGen].getHexaToRemove();
    helper::set<unsigned int> parsedHexas;
    if ( hexaToRemove.find ( hexa ) != hexaToRemove.end() ) return;
    unsigned int hexaID;
    std::stack<unsigned int> hexasToParse; // Stack of hexas to parse.
    hexasToParse.push ( hexa ); // Adds the first given hexa.
    const SCoord& halfVSize = meshGen->voxelSize.getValue() / 2.0;
    const float sphereRadius = sqrt( halfVSize[0] * halfVSize[0] + halfVSize[1] * halfVSize[1] + halfVSize[2] * halfVSize[2]);


    // While the propagation is not finished
    while ( !hexasToParse.empty() )
    {
        hexaID = hexasToParse.top(); // Get the last cube on the stack.
        hexasToParse.pop();          // Remove it from the stack.

        if ( parsedHexas.find ( hexaID ) != parsedHexas.end() )   continue; // if this hexa is ever parsed, continue
        if ( hexaToRemove.find ( hexaID ) != hexaToRemove.end() ) continue; // if this hexa has ever been removed, continue

        // TODO // Update the hexas position (if using sleeping bary mapping.
        // Update the mechanical Coords before computing AABB box.
        //VecCoord& out = *fromDOFs->getX();
        //VecCoord& in = *toDOFs->getX();
        //mapper->applyOnePoint( hexaID, out, in);

        Coord hexaCoord;
        meshGen->getHexaCoord( hexaCoord, hexaID );
        parsedHexas.insert ( hexaID );
        if (showElements.getValue())
            parsedHexasCoords.insert ( hexaCoord );

        // Compute and compare bounding volumes of the hexas and of the intersection volume.
#if 1
        if ( !isCrossingSphere ( minBBVolume, maxBBVolume, hexaCoord, sphereRadius ) )
            continue;
#else
        if ( !isInsideAABB ( minBBVolume, maxBBVolume, hexaCoord ) )
            continue;
#endif
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

    unsigned int z = ((unsigned int)( hexaID / (dim[0]*dim[1])));
    unsigned int remain = ( hexaID % (dim[0]*dim[1]));
    unsigned int y = ((unsigned int)( remain / dim[0]));
    unsigned int x = (remain % dim[0]);

    if (x > (unsigned int) 0) neighbors.insert ( hexaID-1 );
    if (x < (unsigned int) dim[0]) neighbors.insert ( hexaID+1 );
    if (y > (unsigned int) 0) neighbors.insert ( hexaID-dim[1] );
    if (y < (unsigned int) dim[1]) neighbors.insert ( hexaID+dim[1] );
    if (z > (unsigned int) 0) neighbors.insert ( hexaID-dim[1]*dim[0] );
    if (z < (unsigned int) dim[2]) neighbors.insert ( hexaID+dim[1]*dim[0] );
}


template <class DataTypes>
bool HexaRemover<DataTypes>::removeVoxels()
{
    //TODO clean this method !!
    ContactInfos &infos=this->contactInfos[meshGen];
    std::set< unsigned int > items = infos.getHexaToRemove();
    if ( items.empty() )
        return false;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode ( "removeVoxels" );
#endif
    simulation::Node *node_curr = dynamic_cast<simulation::Node*> ( meshGen->getContext() );

    bool is_topoMap = true;

    while ( is_topoMap )
    {
        is_topoMap = false;
        std::vector< core::objectmodel::BaseObject * > listObject;
        node_curr->get<core::objectmodel::BaseObject> ( &listObject, core::objectmodel::BaseContext::Local );
        for ( unsigned int i=0; i<listObject.size(); ++i )
        {
            MeshGen *meshGen = dynamic_cast<MeshGen *> ( listObject[i] );
            if ( meshGen != NULL )
            {
                is_topoMap = true;
                std::set< unsigned int > loc_items = items;
                items.clear();
                for ( std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it )
                {
                    vector<unsigned int> indices;
                    meshGen->getFromIndex ( indices, *it );
                    for ( vector<unsigned int>::const_iterator itIndices = indices.begin(); itIndices != indices.end(); itIndices++ )
                    {
                        //std::cout << *it << " -> " << *itIndices << std::endl;
                        items.insert ( *itIndices );
                    }
                }
                break;
            }
        }
    }

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

    int nbcoll = 0;
    const Real psize = rasterizer->pixelSize.getValue();

#ifdef DRAW_ONE_LDI
    {
        int axis=0;
#else
    for ( int axis=0; axis<3; ++axis )
    {
#endif
        const LDI& ldi = ldiDir[axis];
        const int iX = ( axis+1 ) %3;
        const int iY = ( axis+2 ) %3;
        const int iZ =  axis     ;
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
            bool first_front = false;
            int first_layer = 0;
            int first_obj = 0;
            int first_tid = -1;
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
                            if ( incount >= 0 )
                                inobjs[incount] = obj;
                            ++incount;
                        }
                        else
                        {
                            --incount;
                            if ( first_front &&
                                    ( //( obj == first_obj && incount > 0 && obj == inobjs[incount-1] ) || // self-collision
                                            //( obj == first_obj && incount && obj != inobjs[incount-1]) || // collision inside another object
                                            ( obj != first_obj ) ) ) // collision
                            {
                                MTopology* first_model = rasterizer->vmtopology[first_obj];
                                current_model = rasterizer->vmtopology[obj];
                                if ( ! ( ( isTheModelInteresting ( current_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), first_model ) != cuttingModels.end() ) ||
                                        ( isTheModelInteresting ( first_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), current_model ) != cuttingModels.end() ) ) )
                                    continue;
                                ++nbcoll;
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

                                BoundingBox collisionVol;
                                collisionVol.first[iX] = x - ( psize*0.5 );
                                collisionVol.first[iY] = y - ( psize*0.5 );
                                collisionVol.first[iZ] = minDepth;
                                collisionVol.second[iX] = x + ( psize*0.5 );
                                collisionVol.second[iY] = y + ( psize*0.5 );
                                collisionVol.second[iZ] = maxDepth;
                                if (showVolumes.getValue())
                                    collisionVolumesCoords[axis].insert ( collisionVol );
                                if ( isTheModelInteresting ( current_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), first_model ) != cuttingModels.end() )
                                    findTheCollidedVoxels ( tid, collisionVol.first, collisionVol.second );
                                else if ( isTheModelInteresting ( first_model ) && std::find ( cuttingModels.begin(), cuttingModels.end(), current_model ) != cuttingModels.end() )
                                    findTheCollidedVoxels ( first_tid, collisionVol.first, collisionVol.second );
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

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode ( "detectVoxels" );
#endif
}

template <class DataTypes>
bool HexaRemover<DataTypes>::isCrossingAABB ( const Vector3& min1, const Vector3& max1, const Vector3& min2, const Vector3& max2 ) const
{
    if ( min1[0] < max2[0] && min1[1] < max2[1] && min1[2] < max2[2] &&
            min2[0] < max1[0] && min2[1] < max1[1] && min2[2] < max1[2] ) return true;
    return false;
}

template <class DataTypes>
bool HexaRemover<DataTypes>::isInsideAABB ( const Vector3& min1, const Vector3& max1, const Vector3& point ) const
{
    if ( min1[0] < point[0] && min1[1] < point[1] && min1[2] < point[2] &&
            point[0] < max1[0] && point[1] < max1[1] && point[2] < max1[2] ) return true;
    return false;
}

template <class DataTypes>
bool HexaRemover<DataTypes>::isCrossingSphere(const Vector3& min1, const Vector3& max1, const Vector3& sCenter, const float& radius ) const
{
    float dist_squared = radius * radius;
    if (sCenter[0] < min1[0]) dist_squared -= squared(sCenter[0] - min1[0]);
    else if (sCenter[0] > max1[0]) dist_squared -= squared(sCenter[0] - max1[0]);
    if (sCenter[1] < min1[1]) dist_squared -= squared(sCenter[1] - min1[1]);
    else if (sCenter[1] > max1[1]) dist_squared -= squared(sCenter[1] - max1[1]);
    if (sCenter[2] < min1[2]) dist_squared -= squared(sCenter[2] - min1[2]);
    else if (sCenter[2] > max1[2]) dist_squared -= squared(sCenter[2] - max1[2]);
    return dist_squared > 0;
}


template <class DataTypes>
void HexaRemover<DataTypes>::clear()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode ( "clear" );
#endif
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
    for ( unsigned int i = 0; i < 3; i++ ) collisionVolumesCoords[i].clear();
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
    glPopAttrib();
}


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
    glPopAttrib();
}


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
