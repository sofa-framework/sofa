/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LOADER_VOXELIZER_INL
#define SOFA_COMPONENT_LOADER_VOXELIZER_INL

#include "Voxelizer.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/io/ImageRAW.h>
#include <sofa/helper/system/FileRepository.h>
#include <iostream>
#include <fstream>


namespace sofa
{

namespace component
{

namespace loader
{

template <class DataTypes>
Voxelizer<DataTypes>::Voxelizer()
    : useROI (initData(&useROI, false, "useROI", "Use the given regions of interest."))
    , vROICenter (initData(&vROICenter, "vROICenter", "position of the center of the ROI"))
    , vROIRadius (initData(&vROIRadius, "vROIRadius", "radius of the ROI"))
    , boundingBox (initData(&boundingBox, "boundingBox", "Define a bounding bos for the rasterization."))
    , triangularModelPath (initData(&triangularModelPath, "triangularModelPath", "path to the triangular models"))
    , voxelSize (initData(&voxelSize, "voxelSize", "voxels size"))
    , generateRAWFiles (initData(&generateRAWFiles, true, "generateRAWFiles", "generate RAW files"))
    , valueFileName (initData(&valueFileName, "valueFileName", "RAW file name"))
    , segmentationFileName (initData(&segmentationFileName, "segmentationFileName", "segmentation file name"))
    , infoFileName (initData(&infoFileName, "infoFileName", "information about the RAW file name"))
    , rawOrigin (initData(&rawOrigin, "rawOrigin", "origin of the RAW array. Mechanical object Data named \"position\" can connect to this data to obtain the correct offset."))
    , resolution (initData(&resolution, "resolution", "resolution of the RAW array. RAW loader can connect to this data."))
    , showRasterizedVolumes (initData(&showRasterizedVolumes, false, "showRasterizedVolumes", "Show rasterized volumes."))
    , showWireFrameMode (initData(&showWireFrameMode, false, "showWireFrameMode", "Show in wire frame."))
    , showWhichAxis (initData(&showWhichAxis, (unsigned int)0, "showWhichAxis", "0 - all Axis\n1 - X Axis\n2 - Y Axis\n3 - Z Axis."))
{
    this->addAlias(&valueFileName, "filename");
    rasterizer = NULL;
    valueImg = NULL;
    segmentationImg = NULL;
}


template <class DataTypes>
Voxelizer<DataTypes>::~Voxelizer()
{
    if (valueImg) delete valueImg;
    if (segmentationImg) delete segmentationImg;

    // TODO Properly delete this structure. This code leads to a seg fault
    //if ( rasterizedVolumes)
    //{
    //    for (unsigned int i = 0; i < 3; ++i)
    //        delete [] rasterizedVolumes[i];
    //    delete [] rasterizedVolumes;
    //}
}


template <class DataTypes>
void Voxelizer<DataTypes>::init()
{
    if (generateRAWFiles.getValue())
    {
        // Check if the RAW file has ever been generated
        std::string rawFileName = valueFileName.getValue();
        std::string segFileName = segmentationFileName.getValue();
        if (sofa::helper::system::DataRepository.findFile(rawFileName) &&
            sofa::helper::system::DataRepository.findFile(segFileName))
        {
            loadInfos();
            sout << "RAW and segmentation files are ever generated. Using existing ones." << sendl;
            std::cout << "Voxelizer(" << getName() << "): RAW and segmentation files are ever generated. Using existing ones." << std::endl;
            return;
        }
    }

    // Get the rasterizer
    ((simulation::Node*)simulation::getSimulation()->getContext())->get(rasterizer);
    if ( ! rasterizer)
    {
        serr << "Rasterizer not found" << sendl;
        return;
    }

    // Get the triangular model paths
    const string& targetPaths = triangularModelPath.getValue();
    std::list<std::string> allPaths;
    size_t pos1 = 0;
    while ( true)
    {
        size_t pos2 = std::min(targetPaths.find(",", pos1), targetPaths.find("\n", pos1));
        if ( pos2 == string::npos)
        {
            allPaths.push_back( targetPaths.substr( pos1));
            break;
        }
        allPaths.push_back( targetPaths.substr( pos1, pos2-pos1));
        pos1 = pos2+1;
    }

    // Get the triangular models
    //simulation::Node* currentNode = static_cast<simulation::Node*>(this->getContext());
    simulation::Node* rootNode = static_cast<simulation::Node*>(simulation::getSimulation()->getContext());
    for (std::list<std::string>::const_iterator it = allPaths.begin(); it != allPaths.end(); ++it)
    {
        if ( it->compare( "NULL") == 0)
        {
            vTriangularModel.push_back(NULL);
        }
        else
        {
            string name (*it);
            // If the path is relative, remove first level.
            //if( name.substr(0, 3).compare("../") == 0)
            //    name = name.substr( 3);
            MTopology* model = findObject<MTopology>(name, rootNode);
            if ( !model) serr << "impossible to cast :" << name << "." << sendl;
            vTriangularModel.push_back(model);
        }
    }

    // Check everybody has the same size.
    if ( useROI.getValue() &&
            (vTriangularModel.size() != vROIRadius.getValue().size() ||
                    vTriangularModel.size() != vROICenter.getValue().size() ))
    {
        serr << "vROICenter.getValue().size() " << vROICenter.getValue().size() << sendl;
        serr << "vROIRadius.getValue().size() " << vROIRadius.getValue().size() << sendl;
        serr << "vTriangularModel.size() " << vTriangularModel.size() << sendl;
        serr << "All the vectors defining the ROI must have the same size." << sendl;
        return;
    }

    // Prepare the scene to rasterize only interesting objects
    changeRasterizerSettings();

    //Launch the visitor CollisionDetectionVisitor from the root to construct the LDI struct.
    rootNode->execute<simulation::CollisionDetectionVisitor>(sofa::core::ExecParams::defaultInstance());

    unsigned int nbModels = vTriangularModel.size();
    rasterizedVolumes = new RasterizedVol*[3]; // 3 for each axis
    for (unsigned int i = 0; i < 3; ++i)
        rasterizedVolumes[i] = new RasterizedVol[nbModels];
    generateFullVolumes(rasterizedVolumes);

    if (generateRAWFiles.getValue())
        createImages(rasterizedVolumes);

    // Clean changes
    reloadRasterizerSettings();
}


template <class DataTypes>
bool Voxelizer<DataTypes>::canLoad()
{
    return true;
}


template <class DataTypes>
bool Voxelizer<DataTypes>::load()
{
    return true;
}


template <class DataTypes>
bool Voxelizer<DataTypes>::createImages(RasterizedVol** rasterizedVolume)
{
    // If files are existing, do nothing.
    std::string valuefilename (valueFileName.getValue());
    std::string segmentationfilename (segmentationFileName.getValue());
    if (sofa::helper::system::DataRepository.findFile(valuefilename) &&
        sofa::helper::system::DataRepository.findFile(segmentationfilename))
    {
        sout << "RAW files exist." << sendl;
        return true;
    }

    if ( valueImg) delete valueImg;
    if ( segmentationImg) delete segmentationImg;
    valueImg = new helper::io::ImageRAW;
    segmentationImg = new helper::io::ImageRAW;

    const Vec3d& voxelsSize = voxelSize.getValue();
    Vec3d& origin = *rawOrigin.beginEdit();
    origin = rasterizer->bbox[0] + voxelsSize/2.0;
    sout << "RAW origin is: " << rawOrigin << sendl;

    //std::cout << "bbox: " << rasterizer->bbox[0] << ", " << rasterizer->bbox[1] << std::endl;

    // Init images
    Vec3d imgSize = (rasterizer->bbox[1] - rasterizer->bbox[0]);
    Vec3d& dimension = *resolution.beginEdit();
    unsigned int width = dimension[0] = (unsigned int)(imgSize[0] / voxelsSize[0]);
    unsigned int height = dimension[1] = (unsigned int)(imgSize[1] / voxelsSize[1]);
    unsigned int depth = dimension[2] = (unsigned int)(imgSize[2] / voxelsSize[2]);
    sout << "writing RAW of size " << width << " x " << height << " x " << depth << sendl;
    valueImg->init( width, height, depth, 1, sofa::helper::io::Image::UNORM8, sofa::helper::io::Image::L);
    valueImg->initHeader(0);
    segmentationImg->init( width, height, depth, 1, sofa::helper::io::Image::UNORM8, sofa::helper::io::Image::L);
    segmentationImg->initHeader(0);
    unsigned char* valData = valueImg->getPixels();
    unsigned char* segData = segmentationImg->getPixels();
    const bool& checkROI = useROI.getValue();

    //* // Parse volume
    for (unsigned int z = 0; z < depth; ++z)
    {
        for (unsigned int y = 0; y < height; ++y)
        {
            for (unsigned int x = 0; x < width; ++x)
            {
                unsigned int indexPixel = z*height*width + y*width + x;
                Vec3d pos (origin[0] + x*voxelsSize[0], origin[1] + y*voxelsSize[1], origin[2] + z*voxelsSize[2]);

                Vec3d vecHalfVoxelSize (voxelsSize[0]/2.0, voxelsSize[1]/2.0, voxelsSize[2]/2.0);
                BBox box( pos - vecHalfVoxelSize, pos + vecHalfVoxelSize);
                unsigned int segID = isCoordInside( box, rasterizedVolume[0], 0);
                if ( segID)
                {
                    if (checkROI)
                    {
                        const Vec3d& roiCenter = vROICenter.getValue()[segID-1];
                        const double& radius = vROIRadius.getValue()[segID-1];
                        if ((roiCenter - pos).norm() < radius)
                            segData[indexPixel] = segID;
                        else
                            segData[indexPixel] = 0;
                    }
                    else
                    {
                        segData[indexPixel] = segID;
                    }
                }
                else
                {
                    segData[indexPixel] = 0;
                    for (unsigned int i = 0; i < vTriangularModel.size(); ++i)
                    {
                        if ( ! vTriangularModel[i])
                        {
                            if (checkROI)
                            {
                                const Vec3d& roiCenter = vROICenter.getValue()[i];
                                const double& radius = vROIRadius.getValue()[i];
                                if ((roiCenter - pos).norm() < radius)
                                {
                                    segData[indexPixel] = i+1;
                                    break;
                                }
                            }
                            else
                            {
                                segData[indexPixel] = i+1;
                            }
                        }
                    }
                }
                if (segData[indexPixel])
                    valData[indexPixel] = 255;
                else
                    valData[indexPixel] = 0;
            }
        }
    }
    rawOrigin.endEdit();
    resolution.endEdit();

    // Save images
    if (!valueImg->save(valueFileName.getValue())) serr << "value RAW save failed." << sendl;
    if (!segmentationImg->save(segmentationFileName.getValue())) serr << "segmentation RAW save failed." << sendl;
    saveInfos();

    return true;
}


template <class DataTypes>
template <class T>
bool Voxelizer<DataTypes>::canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
{
    // Split paths
    std::list<std::string> allPaths;
    const string& targetPaths = string(arg->getAttribute("triangularModelPath", ""));
    size_t pos1 = 0;
    while ( true)
    {
        size_t pos2 = std::min(targetPaths.find(",", pos1), targetPaths.find("\n", pos1));
        if ( pos2 == string::npos)
        {
            allPaths.push_back( targetPaths.substr( pos1));
            break;
        }
        allPaths.push_back( targetPaths.substr( pos1, pos2-pos1));
        pos1 = pos2+1;
    }

    for (std::list<std::string>::iterator it= allPaths.begin(); it != allPaths.end(); ++it)
    {
        if (strcmp(it->c_str(), "NULL") == 0) continue; // NULL describes the volume outside all the meshes
        if (arg->findObject(it->c_str()) == NULL)
            context->serr << "ERROR[Voxelizer]: Cannot create "<<className(obj)<<" as "<< *it << " is missing."<<context->sendl;
        if (dynamic_cast<MTopology*>(arg->findObject(it->c_str())) == NULL)
        {
            context->serr << "ERROR[Voxelizer]: Cannot cast target collision DOFs." << *it <<context->sendl;
            return false;
        }
    }

    return true;
}


template <class DataTypes>
void Voxelizer<DataTypes>::addVolume( RasterizedVol& rasterizedVolume, double x, double y, double zMin, double zMax, int /*axis*/)
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
void Voxelizer<DataTypes>::generateFullVolumes( RasterizedVol** rasterizedVolume)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode ( "rasterizeObject" );
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

    const Real psize = rasterizer->pixelSize.getValue();

#ifdef DRAW_ONE_LDI
    {
        int axis=0;
#else
    for ( int axis=0; axis<3; ++axis )
    {
#endif
        const LDI& ldi = ldiDir[axis];
        //const int iX = ( axis+1 ) %3;
        //const int iY = ( axis+2 ) %3;
        //const int iZ =  axis     ;
        if ( ldi.nbcu==0 ) continue;

        for ( int bx=0; bx < ldi.nbcu; ++bx )
        {
            int ci = ldi.haveTriangleStart[bx];
            int cx = ci%ldi.cnx;
            int cy = ci/ldi.cnx;

            const Cell& cell = ldi.cells[bx];
            if ( cell.nbLayers == 0 ) continue;
            static helper::vector<int> layers;
            static helper::vector<int> inLayers;
            static helper::vector<int> inobjs;
            layers.resize ( cell.nbLayers );
            inobjs.resize ( cell.nbLayers );
            inLayers.resize ( cell.nbLayers );

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
            int first_obj = 0;
            int first_tid = -1;
            for ( int l=0; l < Rasterizer::CELL_NY; ++l )
            {
                for ( int c=0; c < Rasterizer::CELL_NX; ++c )
                {
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
                            serr << "ERROR: tid " << (tid>>Rasterizer::SHIFT_TID) << " object invalid......." << sendl;
                            return;
                        }

                        bool front = ((tid&1) != 0);
                        tid = (tid >> (Rasterizer::SHIFT_TID))- ldiObjects[obj].t0;

                        if ( front )
                        {
                            if ( incount >= 0 )
                            {
                                inobjs[incount] = obj;
                                inLayers[incount] = layer;
                            }
                            ++incount;
                        }
                        else
                        {
                            --incount;
                            Real y = ( Real ) ( cl0 + l ) * psize;
                            Real x = ( Real ) ( cc0 + c ) * psize;
                            // Find the first layer corresponding to this object
                            int indexFirstLayer = -1;
                            for (int incpt = 0; incpt <= incount; ++incpt)
                            {
                                if ( inobjs[incpt] == obj) indexFirstLayer = inLayers[incpt];
                            }

                            if ( indexFirstLayer == -1)
                            {
                                serr << "Returned indexFirstLayer == -1. Object not stored ?" << sendl;
                                continue;
                            }

                            Real z0 = ldi.cellLayers[indexFirstLayer][l][c].z;
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

                            // Find the corresponding mesh
                            int indexMesh = -1;
                            current_model = rasterizer->vmtopology[obj];
                            for (unsigned int i = 0; i < vTriangularModel.size(); ++i)
                                if ( current_model == vTriangularModel[i])
                                    indexMesh = i;
                            if ( indexMesh == -1)
                            {
                                serr << "indexMesh is NULL. It should'nt be ! Fixed tags are wrong." << sendl;
                                continue;
                            }

                            // Add this volume
                            addVolume( rasterizedVolume[axis][indexMesh], x, y, minDepth, maxDepth, axis);
                        }
                        first_front = front;
                        first_obj = obj;
                        first_tid = tid;
                    }
                }
            }
            glEnd();
        }
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode ( "rasterizeObject" );
#endif

}


template <class DataTypes>
bool Voxelizer<DataTypes>::isCoordInside( const Vec3d& position, const RasterizedVol& rasterizedVolume, const unsigned int /*axis*/)
{
    // TODO: this can be optimized with lower and upper bound access. We have to use iX, iY and iZ for vector access.
    const double& x = position[0];
    const double& y = position[1];
    const double& z = position[2];
    const Real psize = rasterizer->pixelSize.getValue();
    for (RasterizedVol::const_iterator it = rasterizedVolume.begin(); it != rasterizedVolume.end(); ++it)
    {
        if (it->first > x - psize || it->first < x + psize)
        {
            for (std::multimap<double, std::pair< double, double> >::const_iterator it2 = it->second.begin();
                    it2 != it->second.end(); ++it2)
            {
                if ( it2->first > y - psize || it2->first < y + psize)
                {
                    const std::pair< double, double>& depth = it2->second;
                    if (z > depth.first && z < depth.second)
                        return true;
                }
            }
        }
    }
    return false;
}


template <class DataTypes>
unsigned int Voxelizer<DataTypes>::isCoordInside( const Vec3d& position, const RasterizedVol* rasterizedVolume, const unsigned int axis)
{
    for (unsigned int i = 0; i < vTriangularModel.size(); ++i)
    {
        if ( isCoordInside( position, rasterizedVolume[i], axis))
            return i+1;
    }
    return 0;
}


template <class DataTypes>
bool Voxelizer<DataTypes>::isCoordInside( const BBox& bbox, const RasterizedVol& rasterizedVolume, const unsigned int axis)
{
    const int iX = ( axis+1 ) %3;
    const int iY = ( axis+2 ) %3;
    const int iZ = axis;

    const Real psize = rasterizer->pixelSize.getValue();
    const double& xMin = bbox[0][iX] - psize*0.5;
    const double& xMax = bbox[1][iX] + psize*0.5;
    const double& yMin = bbox[0][iY] - psize*0.5;
    const double& yMax = bbox[1][iY] + psize*0.5;
    RasterizedVol::const_iterator itMax = rasterizedVolume.upper_bound(xMax);
    for (RasterizedVol::const_iterator it = rasterizedVolume.lower_bound( xMin); it != itMax; ++it)
    {
        std::multimap<double, std::pair< double, double> >::const_iterator it2Max = it->second.upper_bound(yMax);
        for (std::multimap<double, std::pair< double, double> >::const_iterator it2 = it->second.lower_bound( yMin); it2 != it2Max; ++it2)
            if (bbox[0][iZ] > it2->second.first && bbox[1][iZ] < it2->second.second)
                return true;
    }
    return false;
}


template <class DataTypes>
unsigned int Voxelizer<DataTypes>::isCoordInside( const BBox& bbox, const RasterizedVol* rasterizedVolume, const unsigned int axis)
{
    for (unsigned int i = 0; i < vTriangularModel.size(); ++i)
    {
        if ( isCoordInside( bbox, rasterizedVolume[i], axis))
            return i+1;
    }
    return 0;
}


template <class DataTypes>
unsigned int Voxelizer<DataTypes>::isCoordInside( const BBox& /*bbox*/, const RasterizedVol** /*rasterizedVolume*/)
{
    /*
         bool inside = true;
         for (unsigned int axis = 0; axis < 3; ++axis)
         {
             if( !isCoordInside( bbox, rasterizedVolume[axis], axis))
                 inside = false;
         }
           return inside;*/
    serr << "not yet implemented !" << sendl;
    return 0;
}


template <class DataTypes>
bool Voxelizer<DataTypes>::isCoordIntersecting( const BBox& bbox, const RasterizedVol& rasterizedVolume, const unsigned int axis)
{
    const int iX = ( axis+1 ) %3;
    const int iY = ( axis+2 ) %3;
    const int iZ = axis;

    const Real psize = rasterizer->pixelSize.getValue();
    const double& xMin = bbox[0][iX] - psize*1.5;
    const double& xMax = bbox[1][iX] + psize*1.5;
    const double& yMin = bbox[0][iY] - psize*1.5;
    const double& yMax = bbox[1][iY] + psize*1.5;
    RasterizedVol::const_iterator itMax = rasterizedVolume.upper_bound(xMax);
    for (RasterizedVol::const_iterator it = rasterizedVolume.lower_bound( xMin); it != itMax; ++it)
    {
        std::multimap<double, std::pair< double, double> >::const_iterator it2Max = it->second.upper_bound(yMax);
        for (std::multimap<double, std::pair< double, double> >::const_iterator it2 = it->second.lower_bound( yMin); it2 != it2Max; ++it2)
            if (bbox[0][iZ] - psize < it2->second.second && bbox[1][iZ] + psize > it2->second.first)
                return true;
    }
    return false;
}


template <class DataTypes>
unsigned int Voxelizer<DataTypes>::isCoordIntersecting( const BBox& bbox, const RasterizedVol* rasterizedVolume, const unsigned int axis)
{
    for (unsigned int i = 0; i < vTriangularModel.size(); ++i)
    {
        if ( isCoordIntersecting( bbox, rasterizedVolume[i], axis))
            return i+1;
    }
    return 0;
}


template <class DataTypes>
unsigned int Voxelizer<DataTypes>::isCoordIntersecting( const BBox& /*bbox*/, const RasterizedVol** /*rasterizedVolume*/)
{
    /*
         bool inside = true;
         for (unsigned int axis = 0; axis < 3; ++axis)
         {
             if( !isCoordIntersecting( bbox, rasterizedVolume[axis], axis))
                 inside = false;
         }
           return inside;*/
    serr << "not yet implemented !" << sendl;
    return 0;
}


template <class DataTypes>
void Voxelizer<DataTypes>::temporaryChangeTags()
{
    hadSelfCollisionTag.clear();

    // Temporary Replace rasterizer tags by "RasterizeOnLoad"
    rasterizerTags = rasterizer->getTags();
    for (TagSet::const_iterator it = rasterizerTags.begin(); it != rasterizerTags.end(); ++it)
        rasterizer->removeTag( *it);
    rasterizer->addTag( Tag("RasterizeOnLoad"));

    //for (TagSet::const_iterator it = rasterizer->getTags().begin(); it != rasterizer->getTags().end(); ++it)
    //    serr << "pour le rasterizer: " << *it << sendl;

    // Add tag "RasterizeOnLoad" on objects
    for (vector<MTopology*>::const_iterator it = vTriangularModel.begin(); it != vTriangularModel.end(); ++it)
        if ( *it != NULL)
        {
            hadSelfCollisionTag.push_back( (*it)->hasTag (Tag("SelfCollision")));
            (*it)->addTag( Tag("RasterizeOnLoad"));
            if ( !(*it)->hasTag (Tag("SelfCollision"))) (*it)->addTag( Tag("SelfCollision"));

            //for (TagSet::const_iterator it2 = (*it)->getTags().begin(); it2 != (*it)->getTags().end(); ++it2)
            //    serr << "Pour l'obj " << (*it)->getName() << ": " << *it2 << sendl;
        }

}


template <class DataTypes>
void Voxelizer<DataTypes>::restoreTags()
{
    // Restore rasterizer tags
    rasterizer->removeTag( Tag("RasterizeOnLoad"));
    for (TagSet::const_iterator it = rasterizerTags.begin(); it != rasterizerTags.end(); ++it)
        rasterizer->addTag( *it);

    // Remove tag "RasterizeOnLoad" on objects
    unsigned int i = 0;
    for (vector<MTopology*>::const_iterator it = vTriangularModel.begin(); it != vTriangularModel.end(); ++it)
        if ( *it != NULL)
        {
            (*it)->removeTag( Tag("RasterizeOnLoad"));
            if (!hadSelfCollisionTag[i]) (*it)->removeTag( Tag("SelfCollision"));
            ++i;
        }
}


template <class DataTypes>
template<class T>
T* Voxelizer<DataTypes>::findObject( string path, const BaseContext* context)
{
    std::string::size_type pos_slash = path.find("/");

    const sofa::core::objectmodel::BaseNode* currentNode = dynamic_cast< const sofa::core::objectmodel::BaseNode *>(context);
    if (pos_slash == std::string::npos)
    {
        if (path.empty())
        {
            return NULL;
        }

        T* result;
        context->get(result, sofa::core::objectmodel::BaseContext::Local);
        return result;
    }
    else
    {
        if (path.substr(0,3).compare("../") == 0)
        {
            const sofa::simulation::tree::GNode* currentGNode = dynamic_cast< const sofa::simulation::tree::GNode *>(context);
            if ( !currentGNode)
            {
                serr << "can not cast the current node into GNode" << sendl;
                return NULL;
            }
            return findObject<T>(path.substr(3), currentGNode->getParent()->getContext());
        }

        std::string name_expected = path.substr(0,pos_slash);
        path = path.substr(pos_slash+1);
        sofa::helper::vector< sofa::core::objectmodel::BaseNode* > list_child = currentNode->getChildren();

        for (unsigned int i=0; i< list_child.size(); ++i)
        {
            if (list_child[i]->getName() == name_expected)
                return findObject<T>(path, list_child[i]->getContext());
        }
    }
    serr << "invalid path: " << path << sendl;
    return NULL;
}


template <class DataTypes>
bool Voxelizer<DataTypes>::saveRAW( const std::string filename, const unsigned char* data, const unsigned int size) const
{
    FILE *file;
    std::cout << "Writing RAW file " << filename << std::endl;
    if ((file = fopen(filename.c_str(), "wb")) == NULL)
    {
        std::cerr << "File write access failed : " << filename << std::endl;
        return false;
    }

    bool isWriteOk = fwrite(data, size, 1, file) == size;
    fclose(file);
    return isWriteOk;
}


template <class DataTypes>
void Voxelizer<DataTypes>::saveInfos()
{
    std::ofstream fileStream (infoFileName.getValue().c_str());
    if (!fileStream.is_open())
    {
        serr << "Can not open " << infoFileName.getValue() << sendl;
    }
    std::cout << "Writing info file " << infoFileName.getValue() << std::endl;
    fileStream << "voxelType: 1" << std::endl;// << CImg<voxelType>::pixel_type() << endl;
    fileStream << "dimensions: " << resolution.getValue() << std::endl;
    fileStream << "origin: " << rawOrigin.getValue() << std::endl;
    fileStream << "voxelSize: " << voxelSize.getValue() << std::endl;
    fileStream.close();
}


template <class DataTypes>
void Voxelizer<DataTypes>::loadInfos()
{
    std::ifstream fileStream (infoFileName.getValue().c_str(), std::ifstream::in);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << infoFileName.getValue() << sendl;
    }
    std::string str;

    // Voxel Type
    fileStream >> str;
    char vtype[32];
    fileStream.getline(vtype,32); // voxeltype not used yet

    // Resolution
    fileStream >> str;
    Vec3d& res = *resolution.beginEdit();
    fileStream >> res;
    resolution.endEdit();

    // Origin
    fileStream >> str;
    Vec3d& origin = *rawOrigin.beginEdit();
    fileStream >> origin;
    rawOrigin.endEdit();

    // Voxel Size
    fileStream >> str;
    Vec3d voxelsize;
    fileStream >> voxelsize;
    if ( voxelsize != voxelSize.getValue())
        serr << "The RAW file loaded deos not match with the wanted resolution" << sendl;

    fileStream.close();
}


template <class DataTypes>
void Voxelizer<DataTypes>::changeRasterizerSettings()
{
    rasterizerPixelSize = rasterizer->pixelSize.getValue();
    double newPixelSize = std::min( std::min(voxelSize.getValue()[0],voxelSize.getValue()[1]),voxelSize.getValue()[2]);
    rasterizer->pixelSize.setValue( newPixelSize);
    rasterizerBBox = rasterizer->sceneBBox.getValue();
    bool emptyBBox = true;
    for (unsigned int j = 0; j < 2; ++j)
        for (unsigned int i = 0; i < 3; ++i)
        {
            if (boundingBox.getValue()[j][i] != 0)
                emptyBBox = false;
        }
    if (!emptyBBox)
        rasterizer->sceneBBox.setValue( boundingBox.getValue());

    temporaryChangeTags();
    rasterizer->reinit();
}


template <class DataTypes>
void Voxelizer<DataTypes>::reloadRasterizerSettings()
{
    rasterizer->pixelSize.setValue( rasterizerPixelSize);
    rasterizer->sceneBBox.setValue( rasterizerBBox);
    restoreTags();
    rasterizer->reinit();
}


template <class DataTypes>
typename Voxelizer<DataTypes>::RasterizedVol** Voxelizer<DataTypes>::getRasterizedVolumes() const
{
    return rasterizedVolumes;
}


template <class DataTypes>
void Voxelizer<DataTypes>::draw()
{
    const Real& psize = std::min( std::min(voxelSize.getValue()[0],voxelSize.getValue()[1]),voxelSize.getValue()[2]);

    // Display volumes retrieved from depth peeling textures
    if ( showRasterizedVolumes.getValue())
    {
        if (this->getContext()->getShowWireFrame() || showWireFrameMode.getValue()) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable( GL_LIGHTING);
        for ( unsigned int axis=0; axis<3; ++axis )
        {
            if ((showWhichAxis.getValue() != 0) && showWhichAxis.getValue() != axis+1) continue;
            const int iX = ( axis+1 ) %3;
            const int iY = ( axis+2 ) %3;
            const int iZ =  axis;

            const unsigned int nbModel = vTriangularModel.size();
            for ( unsigned int indexModel = 0; indexModel < nbModel; ++indexModel)
            {
                if ( axis == 0)
                    glColor3f( (indexModel+1)/(double)nbModel, 0, 0);
                else if ( axis == 1)
                    glColor3f( 0, (indexModel+1)/(double)nbModel, 0);
                else if ( axis == 2)
                    glColor3f( 0, 0, (indexModel+1)/(double)nbModel);

                RasterizedVol& rasterizedVolume = rasterizedVolumes[axis][indexModel];
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
        if (showWireFrameMode.getValue() && !this->getContext()->getShowWireFrame()) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glPopAttrib();
    }

}



}

}

}

#endif
