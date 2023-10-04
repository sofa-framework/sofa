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
#include <sofa/component/topology/container/grid/SparseGridTopology.h>

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/type/fixed_array.h>
#include <sofa/component/topology/container/grid/polygon_cube_intersection/polygon_cube_intersection.h>
#include <sofa/core/loader/VoxelLoader.h>

#include <fstream>
#include <string>
#include <cmath>

using std::pair;
using sofa::core::loader::VoxelLoader;
using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::helper;

namespace sofa::component::topology::container::grid
{

int SparseGridTopologyClass = core::RegisterObject("Sparse grid in 3D")
        .addAlias("SparseGrid")
        .add< SparseGridTopology >()
        ;


const float SparseGridTopology::WEIGHT27[8][27] =
{
    // each weight of the jth fine vertex to the ith coarse vertex
    {1.0, 0.5, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.25, 0.125, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.25, 0.125, 0.0,  0.0, 0.0,  0.0, 1.0, 0.5, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0,  0.0, 0.25, 0.125, 0.0,  0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.25, 0.125, 0.0,  0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.5, 0.0},
    {0.0, 0.5, 1.0, 0.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0,  0.125, 0.25, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0,  0.125, 0.25, 0.0, 0.0,  0.0, 0.0, 0.5, 1.0, 0.0, 0.25, 0.5, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0,  0.0, 0.0,  0.125, 0.25, 0.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0,  0.125, 0.25, 0.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 0.5, 1.0}
};

const SparseGridTopology::Index SparseGridTopology::cornerIndicesFromFineToCoarse[8][8]=
{
    // fine vertices forming the ith coarse cube (with XYZ order)
    {  0,  9,  3, 12,  1, 10,  4, 13},
    {  9, 18, 12, 21, 10, 19, 13, 22},
    {  3, 12,  6, 15,  4, 13,  7, 16},
    { 12, 21, 15, 24, 13, 22, 16, 25},
    {  1, 10,  4, 13,  2, 11,  5, 14},
    { 10, 19, 13, 22, 11, 20, 14, 23},
    {  4, 13,  7, 16,  5, 14,  8, 17},
    { 13, 22, 16, 25, 14, 23, 17, 26}
};



SparseGridTopology::SparseGridTopology(bool _isVirtual)
    : _fillWeighted(initData(&_fillWeighted, true, "fillWeighted", "Is quantity of matter inside a cell taken into account? (.5 for boundary, 1 for inside)"))
    , d_bOnlyInsideCells(initData(&d_bOnlyInsideCells, false, "onlyInsideCells", "Select only inside cells (exclude boundary cells)"))
    , n(initData(&n, type::Vec3i(2,2,2), "n", "grid resolution"))
    , _min(initData(&_min, type::Vec3(0_sreal,0_sreal,0_sreal), "min","Min"))
    , _max(initData(&_max, type::Vec3(0_sreal,0_sreal,0_sreal), "max","Max"))
    , _cellWidth(initData(&_cellWidth, (SReal)0.0, "cellWidth","if > 0 : dimension of each cell in the created grid"))
    , _nbVirtualFinerLevels( initData(&_nbVirtualFinerLevels, 0, "nbVirtualFinerLevels", "create virtual (not in the animation tree) finer sparse grids in order to dispose of finest information (usefull to compute better mechanical properties for example)"))
    , dataResolution(initData(&dataResolution, type::Vec3i(0,0,0), "dataResolution", "Dimension of the voxel File"))
    , voxelSize(initData(&voxelSize, type::Vec3(1_sreal,1_sreal,1_sreal), "voxelSize", "Dimension of one voxel"))
    , marchingCubeStep(initData(&marchingCubeStep, (unsigned int) 1, "marchingCubeStep", "Step of the Marching Cube algorithm"))
    , convolutionSize(initData(&convolutionSize, (unsigned int) 0, "convolutionSize", "Dimension of the convolution kernel to smooth the voxels. 0 if no smoothing is required."))
    , facets(initData(&facets, "facets", "Input mesh facets"))
{
    isVirtual = _isVirtual;
    _alreadyInit = false;
    _finerSparseGrid = nullptr;
    _coarserSparseGrid = nullptr;
    _usingMC = false;

    _regularGrid = sofa::core::objectmodel::New<RegularGridTopology>();
}

SparseGridTopology::SparseGridTopology(type::Vec3i numVertices, type::BoundingBox box, bool _isVirtual)
    : SparseGridTopology(_isVirtual)
{
    setN(numVertices);
    setMin(box.minBBox());
    setMax(box.maxBBox());
}


void SparseGridTopology::init()
{
    if(_alreadyInit) return;
    _alreadyInit = true;

    type::Vec3i grid = n.getValue();

    if(grid[0] < 2) grid[0]= 2;
    if(grid[1] < 2) grid[1]= 2;
    if(grid[2] < 2) grid[2]= 2;

    n.setValue(grid);

    if( _nbVirtualFinerLevels.getValue() )
        buildVirtualFinerLevels();


    if( _finerSparseGrid != nullptr )
        buildFromFiner();
    else
        buildAsFinest();

    _nodeAdjacency.resize(seqPoints.getValue().size() );
    for(unsigned i=0; i<seqPoints.getValue().size(); ++i)
        _nodeAdjacency[i].assign(InvalidID);

    if (_nodeAdjacency.empty())
    {
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const auto& hexahedra = seqHexahedra.getValue();

    for(unsigned i=0; i<hexahedra.size(); ++i)
    {
        _nodeAdjacency[ hexahedra[i][0] ][RIGHT] = hexahedra[i][1];
        _nodeAdjacency[ hexahedra[i][0] ][UP] = hexahedra[i][2];
        _nodeAdjacency[ hexahedra[i][0] ][BEHIND] = hexahedra[i][4];

        _nodeAdjacency[ hexahedra[i][1] ][LEFT] = hexahedra[i][0];
        _nodeAdjacency[ hexahedra[i][1] ][UP] = hexahedra[i][3];
        _nodeAdjacency[ hexahedra[i][1] ][BEHIND] = hexahedra[i][5];

        _nodeAdjacency[ hexahedra[i][2] ][RIGHT] = hexahedra[i][3];
        _nodeAdjacency[ hexahedra[i][2] ][DOWN] = hexahedra[i][0];
        _nodeAdjacency[ hexahedra[i][2] ][BEHIND] = hexahedra[i][6];

        _nodeAdjacency[ hexahedra[i][3] ][LEFT] = hexahedra[i][2];
        _nodeAdjacency[ hexahedra[i][3] ][DOWN] = hexahedra[i][1];
        _nodeAdjacency[ hexahedra[i][3] ][BEHIND] = hexahedra[i][7];

        _nodeAdjacency[ hexahedra[i][4] ][RIGHT] = hexahedra[i][5];
        _nodeAdjacency[ hexahedra[i][4] ][UP] = hexahedra[i][6];
        _nodeAdjacency[ hexahedra[i][4] ][BEFORE] = hexahedra[i][0];

        _nodeAdjacency[ hexahedra[i][5] ][LEFT] = hexahedra[i][4];
        _nodeAdjacency[ hexahedra[i][5] ][UP] = hexahedra[i][7];
        _nodeAdjacency[ hexahedra[i][5] ][BEFORE] = hexahedra[i][1];

        _nodeAdjacency[ hexahedra[i][6] ][RIGHT] = hexahedra[i][7];
        _nodeAdjacency[ hexahedra[i][6] ][DOWN] = hexahedra[i][4];
        _nodeAdjacency[ hexahedra[i][6] ][BEFORE] = hexahedra[i][2];

        _nodeAdjacency[ hexahedra[i][7] ][LEFT] = hexahedra[i][6];
        _nodeAdjacency[ hexahedra[i][7] ][DOWN] = hexahedra[i][5];
        _nodeAdjacency[ hexahedra[i][7] ][BEFORE] = hexahedra[i][3];
    }


    //	_nodeCubesAdjacency.clear();
    _nodeCubesAdjacency.resize(seqPoints.getValue().size() );

    for(unsigned i=0; i<hexahedra.size(); ++i)
    {
        for(int j=0; j<8; ++j)
        {
            _nodeCubesAdjacency[ hexahedra[i][j] ].push_back( i );
        }
    }

    m_upperElementType = geometry::ElementType::HEXAHEDRON;

}


void SparseGridTopology::buildAsFinest(  )
{
    VoxelLoader *loader;
    getContext()->get(loader);
    if( loader )
    {
        buildFromVoxelLoader(loader);
    }
    else
    {
        const std::string& _filename = fileTopology.getFullPath();

        if (_filename.length() > 4 && _filename.compare(_filename.length() - 4, 4, ".raw") == 0)
        {
            _usingMC = true;

            buildFromRawVoxelFile(_filename);
        }
        else if (_filename.length() > 6 && _filename.compare(_filename.length() - 6, 6, ".voxel") == 0)
        {
            buildFromVoxelFile(_filename);
        }
        else // given surface
        {
            helper::io::Mesh* mesh = nullptr;
            if (seqPoints.getValue().size() == 0 && _filename.empty())
            {
                msg_warning() << "no filename specified nor vertices given as parameters.";
                return;
            }
            else
            {
                if (_filename.empty())
                {
                    mesh = new sofa::helper::io::Mesh();
                    for (unsigned int i = 0; i < seqPoints.getValue().size(); ++i)
                        mesh->getVertices().push_back(seqPoints.getValue()[i]);
                    const auto& facets = this->facets.getValue();
                    const SeqTriangles& triangles = this->seqTriangles.getValue();
                    const SeqQuads& quads = this->seqQuads.getValue();
                    mesh->getFacets().resize(facets.size() + triangles.size() + quads.size());
                    for (size_t i = 0; i < facets.size(); ++i)
                        mesh->getFacets()[i].push_back(facets[i]);
                    for (size_t i0 = facets.size(), i = 0; i < triangles.size(); ++i)
                    {
                        mesh->getFacets()[i0 + i].resize(1);
                        mesh->getFacets()[i0 + i][0].resize(3);
                        mesh->getFacets()[i0 + i][0][0] = triangles[i][0];
                        mesh->getFacets()[i0 + i][0][1] = triangles[i][1];
                        mesh->getFacets()[i0 + i][0][2] = triangles[i][2];
                    }
                    for (size_t i0 = facets.size() + triangles.size(), i = 0; i < quads.size(); ++i)
                    {
                        mesh->getFacets()[i0 + i].resize(1);
                        mesh->getFacets()[i0 + i][0].resize(4);
                        mesh->getFacets()[i0 + i][0][0] = quads[i][0];
                        mesh->getFacets()[i0 + i][0][1] = quads[i][1];
                        mesh->getFacets()[i0 + i][0][2] = quads[i][2];
                        mesh->getFacets()[i0 + i][0][3] = quads[i][3];
                    }
                }
                else
                {
                    mesh = helper::io::Mesh::Create(_filename.c_str());
                }

                if (mesh == nullptr || mesh->getVertices().empty())
                {
                    msg_error() << "Loading mesh " << _filename << " failed.";
                    return;
                }
                else
                {
                    buildFromTriangleMesh(mesh);
                    dmsg_info() << "Mesh Loaded";
                    delete mesh;
                }
            }
        }

        // default stiffness coefficient : BOUNDARY=.5, INSIDE=1
        _stiffnessCoefs.resize( this->getNbHexahedra());
        _massCoefs.resize( this->getNbHexahedra());

        for(Size i=0; i<this->getNbHexahedra(); ++i)
        {
            if( getType(i)==BOUNDARY && _fillWeighted.getValue() )
            {
                _stiffnessCoefs[i] = .5;
                _massCoefs[i] = .5;
            }
            else
            {
                _stiffnessCoefs[i] = 1.0;
                _massCoefs[i] = 1.0;
            }
        }
    }
}


void SparseGridTopology::buildFromVoxelFile(const std::string& filename)
{
    if (dataVoxels.getValue().size() == 0)
    {
        std::ifstream file( filename.c_str() );
        if ( file.fail() )
        {
            msg_error() << "SparseGridTopology: failed to open file " << filename;
            return;
        }

        int fileNx, fileNy, fileNz;
        int nx, ny,nz;
        int xmin, ymin, zmin, xmax, ymax, zmax;
        float dx, dy, dz;

        file >> fileNx >> fileNy >> fileNz;
        file >> dx >> dy >> dz;

        // note that nx, ny, nz are the numbers of vertices in the regular grid
        // whereas fileNx, fileNy, fileNz are the numbers of voxels in each direction,
        // corresponding to the number of cubes in the regular grid and thus nx = fileNx + 1
        nx = fileNx+1;
        ny = fileNy+1;
        nz = fileNz+1;

        xmin = 0;
        ymin = 0;
        zmin = 0;

        xmax =(int)( dx * fileNx);
        ymax =(int)( dy * fileNy);
        zmax =(int)( dz * fileNz);

        n.setValue(type::Vec3i(nx,ny,nz));
        _min.setValue(type::Vec3((SReal)xmin, (SReal)ymin, (SReal)zmin));
        _max.setValue(type::Vec3((SReal)xmax, (SReal)ymax, (SReal)zmax));

        int value;
        dataVoxels.beginEdit()->resize(fileNx * fileNy * fileNz, (unsigned char) 0);

        for(int z=0; z<fileNz; ++z)
        {
            for(int y=0; y<fileNy; ++y)
            {
                for(int x=0; x<fileNx; ++x)
                {
                    file >> value;
                    if (value != 0)
                    {
                        setVoxel(x + fileNx * y + fileNx * fileNy * z,1);
                    }
                }
            }
        }

        file.close();
    }

    _regularGrid->setSize(getNx(), getNy(), getNz());
    _regularGrid->setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());
    const size_t nbCubesRG = _regularGrid->getNbHexahedra();
    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, InvalidID); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    // fill the regularGridTypes vector
    // at the moment, no BOUNDARY type voxels are generated at the finest level
    for(size_t i=0; i<nbCubesRG; ++i)
    {
        if(dataVoxels.getValue()[i] != 0.0f)
        {
            regularGridTypes[i] = INSIDE;
        }
    }

    buildFromRegularGridTypes(_regularGrid, regularGridTypes );
}

/** Create the data structure based on resolution, size and filling.
  \param numPoints  Number of points in the x,y,and z directions
  \param box  Volume occupied by the grid
  \param filling Voxel filling: true if the cell is defined, false if the cell is empty. Voxel order is: for(each z){ for(each y){ for(each x) }}}
  */
void SparseGridTopology::buildFromData( type::Vec3i numPoints, type::BoundingBox box, const vector<bool>& filling )
{
    n.setValue(numPoints);
    _min.setValue(box.minBBox());
    _max.setValue(box.maxBBox());
    type::Vec3i numVoxels(numPoints[0]-1,numPoints[1]-1,numPoints[2]-1);

    dataVoxels.beginEdit()->resize((unsigned int)(numVoxels[2] * numVoxels[1] * numVoxels[0]), (unsigned char) 0);
    dataVoxels.endEdit();

    assert( filling.size()== (unsigned) numVoxels[2]*numVoxels[1]*numVoxels[0]);
    vector<bool>::const_iterator f=filling.begin();
    for(int z=0; z<numVoxels[2]; ++z)
    {
        for(int y=0; y<numVoxels[1]; ++y)
        {
            for(int x=0; x<numVoxels[0]; ++x)
            {
                if ( *f )
                {
                    setVoxel(x + numVoxels[0] * (y + numVoxels[1] * z),1);
                }
                f++;
            }
        }
    }


    _regularGrid->setSize(getNx(), getNy(), getNz());
    _regularGrid->setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());
    const size_t nbCubesRG = _regularGrid->getNbHexahedra();
    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, InvalidID); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    // fill the regularGridTypes vector
    // at the moment, no BOUNDARY type voxels are generated at the finest level
    for(size_t i=0; i<nbCubesRG; ++i)
    {
        if(dataVoxels.getValue()[i] != 0.0f)
        {
            regularGridTypes[i] = INSIDE;
        }
    }

    buildFromRegularGridTypes(_regularGrid, regularGridTypes );

    _stiffnessCoefs.resize( this->getNbHexahedra());
    _massCoefs.resize( this->getNbHexahedra());
    for(size_t i=0; i<this->getNbHexahedra(); ++i)
    {
        _stiffnessCoefs[i] = _massCoefs[i] = 1.0;
    }

    std::stringstream tmp;
    tmp << "regularGrid  has " << _regularGrid->getNbHexahedra() << " hexahedra, "
        << "hexahedra = " << this->seqHexahedra.getValue() ;
    msg_info() << tmp.str() ;
}


//Building from a RAW file
void SparseGridTopology::buildFromRawVoxelFile(const std::string& filename)
{
    _regularGrid->setSize(getNx(),getNy(),getNz());
    const size_t nbCubesRG = _regularGrid->getNbHexahedra();

    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, InvalidID); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    //Loading from file
    if (dataVoxels.getValue().size() == 0)
    {
        FILE *file = fopen( filename.c_str(), "r" );
        if (!file) { msg_error()<< "FILE " << filename << " not found"; return;}
        const auto& dataResolutionValue = dataResolution.getValue();
        //Get the voxels from the file
        dataVoxels.beginEdit()->resize(dataResolutionValue[0]*dataResolutionValue[1]*dataResolutionValue[2], (unsigned char)0);

        const type::Vec3 transform(
                (getNx()-1)/static_cast<SReal>(dataResolutionValue[0]),
                (getNy()-1)/static_cast<SReal>(dataResolutionValue[1]),
                (getNz()-1)/static_cast<SReal>(dataResolutionValue[2]));

        for (unsigned int i=0; i<dataVoxels.beginEdit()->size(); i++)
        {
            const unsigned char value=getc(file);
            if ((int)value != 0)
            {
                setVoxel(i,1);

                const int z = i/(dataResolution.getValue()[0]*dataResolution.getValue()[1]);
                const int y = (  (i%(dataResolution.getValue()[0]*dataResolution.getValue()[1]))
                        /(dataResolution.getValue()[0])                  	    );
                const int x = i%dataResolution.getValue()[0];

                const unsigned int indexGrid =
                    (unsigned int)(z*transform[2])*(getNy()-1)*(getNx()-1)
                    + (unsigned int)(y*transform[1])*(getNx()-1)
                    + (unsigned int)(x*transform[0]);

                regularGridTypes[indexGrid] = INSIDE;
            }
        }

        fclose(file);
    }

    _min.setValue( type::Vec3( 0_sreal, 0_sreal, 0_sreal));
    _max.setValue( voxelSize.getValue().linearProduct(dataResolution.getValue())*(1));

    _regularGrid->setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());
    buildFromRegularGridTypes(_regularGrid, regularGridTypes);

    if (!isVirtual)
        updateMesh();
}


void SparseGridTopology::buildFromVoxelLoader(VoxelLoader * loader)
{
    msg_info()<<"SparseGridTopology::buildFromVoxelLoader(VoxelLoader * loader)";

    unsigned char *textureData;
    int width,height,depth;
    loader->createSegmentation3DTexture( &textureData, width, height, depth );

    _regularGrid->setSize(getNx(),getNy(),getNz());

    type::Vec3 vsize = loader->getVoxelSize(  );

    _regularGrid->setPos(0,width*vsize[0],0,height*vsize[1],0,depth*vsize[2]);

    _min.setValue( type::Vec3(0_sreal,0_sreal,0_sreal) );
    _max.setValue( type::Vec3(width*vsize[0],height*vsize[1],depth*vsize[2]) );

    const auto nbCubesRG = _regularGrid->getNbHexahedra();

    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, InvalidID); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    vector<Type> regularGridTypes(typename vector<Type>::Size(nbCubesRG), OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    vector<float> regularstiffnessCoef(typename vector<Type>::Size(nbCubesRG), 0.0);

    for(Index i=0; i<nbCubesRG; ++i)
    {
        const Vec3 hexacoord = _regularGrid->getCubeCoordinate(i);
        const RegularGridTopology::Hexa& hexa = _regularGrid->getHexahedron( hexacoord[0],hexacoord[1], hexacoord[2] );

        const SReal p0x = _regularGrid->getPX( hexa[0] ) / vsize[0];
        const SReal p0y = _regularGrid->getPY( hexa[0] ) / vsize[1];
        const SReal p0z = _regularGrid->getPZ( hexa[0] ) / vsize[2];
        const SReal p6x = _regularGrid->getPX( hexa[6] ) / vsize[0];
        const SReal p6y = _regularGrid->getPY( hexa[6] ) / vsize[1];
        const SReal p6z = _regularGrid->getPZ( hexa[6] ) / vsize[2];


        for(int x=(int)p0x; x<(int)p6x; ++x)
        {
            for(int y=(int)p0y; y<(int)p6y; ++y)
            {
                for(int z=(int)p0z; z<(int)p6z; ++z)
                {
                    const unsigned int idx = x + y * width + z * width * height;

                    regularstiffnessCoef[i] += textureData[idx];
                }
            }
        }

        regularstiffnessCoef[i] /= (float)((p6x-p0x)*(p6y-p0y)*(p6z-p0z));
        if( regularstiffnessCoef[i] !=0.0 )
            regularGridTypes[i] = INSIDE;
    }

    buildFromRegularGridTypes(_regularGrid, regularGridTypes);

    if (!isVirtual)
        updateMesh();

    _stiffnessCoefs.resize( this->getNbHexahedra());
    _massCoefs.resize( this->getNbHexahedra());
    for(size_t i=0; i<this->getNbHexahedra(); ++i)
    {
        if( _fillWeighted.getValue() )
        {
            _stiffnessCoefs[i] = regularstiffnessCoef[ _indicesOfCubeinRegularGrid[i] ];
            _massCoefs[i] = 1.0;//regularstiffnessCoef[ _indicesOfCubeinRegularGrid[i] ];
        }
        else
        {
            _stiffnessCoefs[i] = _massCoefs[i] = 1.0;
        }
    }
}


void SparseGridTopology::updateMesh()
{
    if (!_usingMC || dataVoxels.beginEdit()->size() == 0) return;

    _min.setValue( type::Vec3( 0_sreal, 0_sreal, 0_sreal));
    _max.setValue( voxelSize.getValue().linearProduct(dataResolution.getValue())*(1));

    //Creating if needed collision models and visual models
    sofa::type::vector< sofa::core::topology::BaseMeshTopology * > list_mesh;
    sofa::type::vector< Data< sofa::defaulttype::Vec3Types::VecCoord >* > list_X;

    //Get Collision Model
    sofa::type::vector< sofa::core::topology::BaseMeshTopology* > m_temp;
    this->getContext()->get< sofa::core::topology::BaseMeshTopology >(&m_temp, sofa::core::objectmodel::BaseContext::SearchDown);

    sofa::core::topology::BaseMeshTopology* collisionTopology=nullptr;
    for (unsigned int i=0; i<m_temp.size(); ++i)
    {
        if (m_temp[i] != this) {collisionTopology = m_temp[i]; break;}
    }

    if ( collisionTopology != nullptr && collisionTopology->getNbTriangles() == 0)
    {
        core::behavior::MechanicalState< sofa::defaulttype::Vec3Types > *mecha_temp =
                collisionTopology->getContext()->get< core::behavior::MechanicalState< sofa::defaulttype::Vec3Types > >();
        if (mecha_temp != nullptr && mecha_temp->getSize() < 2) //a triangle mesh has minimum 3elements
        {
            list_mesh.push_back(collisionTopology);
            list_X.push_back(mecha_temp->write(core::VecCoordId::position()));
        }
    }

    if (list_mesh.empty())
        return;				 //No Marching Cube to run

    //Configuration of the Marching Cubes algorithm

    marchingCubes.setDataResolution(type::Vec3i(dataResolution.getValue()[0],
            dataResolution.getValue()[1],
            dataResolution.getValue()[2]));
    marchingCubes.setDataVoxelSize(voxelSize.getValue());
    marchingCubes.setStep(marchingCubeStep.getValue());
    marchingCubes.setConvolutionSize(convolutionSize.getValue()); //apply Smoothing if convolutionSize > 0

    constructCollisionModels(list_mesh, list_X);
}


void SparseGridTopology::getMesh(sofa::helper::io::Mesh &m)
{
    if (!dataVoxels.getValue().empty())
    {
        type::vector<unsigned char> * datas = dataVoxels.beginEdit();
        marchingCubes.run(&(*datas)[0], 0.5f, m);
        dataVoxels.endEdit();
    }
}


template< class T >
void SparseGridTopology::constructCollisionModels(const sofa::type::vector< sofa::core::topology::BaseMeshTopology * > &list_mesh,
        const type::vector< Data< type::vector< Vec<3,T> > >* > &list_X)
{
    sofa::type::vector< PointID>	triangles;
    vector< type::Vec3 >		vertices;

    type::vector<unsigned char> * datas = dataVoxels.beginEdit();
    marchingCubes.run(&(*datas)[0], 0.5f, triangles, vertices);
    dataVoxels.endEdit();

    for (unsigned int i=0; i<list_mesh.size(); ++i)
    {
        list_mesh[i]->clear();
        helper::WriteAccessor<Data< type::vector< Vec<3,T> > > > x(list_X[i]);
        x.resize(vertices.size());
    }

    for (unsigned int j=0; j<list_mesh.size(); ++j)
    {
        helper::WriteAccessor<Data< type::vector< Vec<3,T> > > > x(list_X[j]);
        for (unsigned int id_point=0; id_point<vertices.size(); ++id_point)
        {
            x[id_point] = vertices[id_point];
        }
    }

    //Fill the facets
    for (unsigned int id_vertex=0; id_vertex<triangles.size(); id_vertex+=3)
    {
        for (unsigned int j=0; j<list_mesh.size(); ++j)
        {
            list_mesh[j]->addTriangle(triangles[id_vertex],triangles[id_vertex+1], triangles[id_vertex+2]);
        }
    }
}


void SparseGridTopology::buildFromTriangleMesh(sofa::helper::io::Mesh* mesh)
{
    // if not given sizes -> bounding box
    if( _min.getValue()== type::Vec3() && _max.getValue()== type::Vec3())
    {

        SReal xMin, xMax, yMin, yMax, zMin, zMax;
        computeBoundingBox(mesh->getVertices(), xMin, xMax, yMin, yMax, zMin, zMax);

        // increase the box a little
        type::Vec3 diff ( xMax-xMin, yMax - yMin, zMax - zMin );
        diff /= 100.0;

        _min.setValue(type::Vec3( xMin - diff[0], yMin - diff[1], zMin - diff[2] ));
        _max.setValue(type::Vec3( xMax + diff[0], yMax + diff[1], zMax + diff[2] ));
        dmsg_info() << "BBox size: " << (_max.getValue() - _min.getValue());
    }

    // if cellWidth is given, update n
    if (_cellWidth.getValue())
    {
        const SReal w = _cellWidth.getValue();
        type::Vec3 diff = _max.getValue() - _min.getValue();
        setN(type::Vec3i((int)ceil(diff[0] / w)+1, (int)ceil(diff[1] / w)+1, (int)ceil(diff[2] / w)+1));
        dmsg_info() << "Grid size: " << n.getValue();
    }

    _regularGrid->setSize(getNx(),getNy(),getNz());
    _regularGrid->setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());

    vector<Type> regularGridTypes; // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
    voxelizeTriangleMesh(mesh, _regularGrid, regularGridTypes);

    buildFromRegularGridTypes(_regularGrid, regularGridTypes);
}


void SparseGridTopology::voxelizeTriangleMesh(helper::io::Mesh* mesh,
        sofa::core::sptr<RegularGridTopology> regularGrid,
        vector<Type>& regularGridTypes) const
{
    regularGridTypes.resize(regularGrid->getNbHexahedra(), INSIDE);

    const type::vector< type::Vec3 >& vertices = mesh->getVertices();
    const size_t vertexSize = vertices.size();
    type::vector< Index > verticesHexa(vertexSize);
    const type::Vec3 delta = (regularGrid->getDx() + regularGrid->getDy() + regularGrid->getDz()) / 2;

    // Compute the grid element for each mesh vertex
    for (size_t i = 0; i < vertexSize; ++i)
    {
        const type::Vec3& vertex = vertices[i];
        Index index = regularGrid->findHexa(vertex);

        if (index != InvalidID)
            regularGridTypes[index] = BOUNDARY;

        // Case where 'findHexa' did not find the right hexa
        // Here we test the case where the point is close the surface (delta /2)
        // Useful when the point is on the boundary
        if (index == InvalidID)
        {
            type::Vec3 vertex2 = vertex;
            const type::Vec3 gmin = regularGrid->getMin();
            const type::Vec3 gmax = regularGrid->getMax();

            if ( (vertex2[0] - std::numeric_limits<float>::epsilon()) < gmin[0] )
                vertex2[0] = gmin[0] + delta[0];
            else if ( (vertex2[0] + std::numeric_limits<float>::epsilon()) > gmax[0] )
                vertex2[0] = gmax[0] - delta[0];

            if ( (vertex2[1] - std::numeric_limits<float>::epsilon()) < gmin[1] )
                vertex2[1] = gmin[1] + delta[1];
            else if ( (vertex2[1] + std::numeric_limits<float>::epsilon()) > gmax[1] )
                vertex2[1] = gmax[1] - delta[1];

            if ( (vertex2[2] - std::numeric_limits<float>::epsilon()) < gmin[2] )
                vertex2[2] = gmin[2] + delta[2];
            else if ( (vertex2[2] + std::numeric_limits<float>::epsilon()) > gmax[2])
                vertex2[2] = gmax[2] - delta[2];

            index = regularGrid->findHexa(vertex2);

            if(index == InvalidID)
                msg_error() << "vertex "<<i<<" not found in hexahedral topology";
        }

        verticesHexa[i] = index;
    }

    // For each triangle, compute BBox and test each element in bb if needed
    const auto& facets = mesh->getFacets();

    for (unsigned int f=0; f<facets.size(); f++)
    {
        const auto& facet = facets[f][0];
        for (unsigned int j=2; j<facet.size(); j++) // Triangularize
        {
            const Index c0 = verticesHexa[facet[0]];
            const Index c1 = verticesHexa[facet[j-1]];
            const Index c2 = verticesHexa[facet[j]];
            if((c0==c1)&&(c0==c2)&&(c0!=InvalidID)) // All vertices in same box discard now if possible
            {
                if(regularGridTypes[c0]==BOUNDARY)
                    continue;
            }
            // Compute box
            const type::Vec3 i0 = regularGrid->getCubeCoordinate(c0);
            const type::Vec3 i1 = regularGrid->getCubeCoordinate(c1);
            const type::Vec3 i2 = regularGrid->getCubeCoordinate(c2);

            const type::Vec3 iMin(	std::min(i0[0],std::min(i1[0],i2[0])),
                    std::min(i0[1],std::min(i1[1],i2[1])),
                    std::min(i0[2],std::min(i1[2],i2[2])));
            const type::Vec3 iMax(	std::max(i0[0],std::max(i1[0],i2[0])),
                    std::max(i0[1],std::max(i1[1],i2[1])),
                    std::max(i0[2],std::max(i1[2],i2[2])));

            for(unsigned int x=(unsigned int)iMin[0]; x<=(unsigned int)iMax[0]; ++x)
            {
                for(unsigned int y=(unsigned int)iMin[1]; y<=(unsigned int)iMax[1]; ++y)
                {
                    for(unsigned int z=(unsigned int)iMin[2]; z<=(unsigned int)iMax[2]; ++z)
                    {
                        // if already inserted discard
                        const Index index = regularGrid->getCubeIndex(x,y,z);

                        if(regularGridTypes[index]==BOUNDARY)
                            continue;

                        Hexa c = regularGrid->getHexaCopy(index);

                        CubeCorners corners;
                        for(int k=0; k<8; ++k)
                            corners[k] = regularGrid->getPoint( c[k] );

                        type::Vec3 cubeDiagonal = corners[6] - corners[0];

                        type::Vec3 cubeCenter = corners[0] + cubeDiagonal*.5;

                        const type::Vec3& A = vertices[facet[0]];
                        const type::Vec3& B = vertices[facet[j-1]];
                        const type::Vec3& C = vertices[facet[j]];

                        // Scale the triangle to the unit cube matching
                        float points[3][3];

                        for (unsigned short w=0; w<3; ++w)
                        {
                            points[0][w] = (float) ((A[w]-cubeCenter[w])/cubeDiagonal[w]);
                            points[1][w] = (float) ((B[w]-cubeCenter[w])/cubeDiagonal[w]);
                            points[2][w] = (float) ((C[w]-cubeCenter[w])/cubeDiagonal[w]);
                        }

                        float normal[3];
                        helper::polygon_cube_intersection::get_polygon_normal(normal,3,points);

                        if (helper::polygon_cube_intersection::fast_polygon_intersects_cube(3,points,normal,0,0))
                        {
                            regularGridTypes[index]=BOUNDARY;
                        }
                    }
                }
            }
        }
    }


    // TODO: regarder les cellules pleines, et les ajouter
    vector<bool> alreadyTested(typename vector<bool>::Size(regularGrid->getNbHexahedra()),false);
    std::stack< type::Vec3i > seed;
    const auto regN = regularGrid->d_n.getValue();
    // x==0 and x=nx-2
    for(int y=0; y< regN[1]-1; ++y)
    {
        for(int z=0; z<regN[2]-1; ++z)
        {
            launchPropagationFromSeed(type::Vec3i(0,y,z), regularGrid, regularGridTypes, alreadyTested,seed );
            launchPropagationFromSeed(type::Vec3i(regN[0]-2,y,z), regularGrid, regularGridTypes, alreadyTested,seed );
        }

        // y==0 and y=ny-2
        for(int x=0; x<regN[0]-1; ++x)
        {
            for(int z=0; z<regN[2]-1; ++z)
            {
                launchPropagationFromSeed(type::Vec3i(x,0,z), regularGrid, regularGridTypes, alreadyTested,seed );
                launchPropagationFromSeed(type::Vec3i(x,regN[1]-2,z), regularGrid, regularGridTypes, alreadyTested,seed );
            }

            // z==0 and z==Nz-2
            for(int y=0; y<regN[1]-1; ++y)
            {
                for(int x=0; x<regN[0]-1; ++x)
                {
                    launchPropagationFromSeed(type::Vec3i(x,y,0), regularGrid, regularGridTypes, alreadyTested,seed );
                    launchPropagationFromSeed(type::Vec3i(x,y,regN[2]-2), regularGrid, regularGridTypes, alreadyTested,seed );
                }
            }
        }
    }
}


void SparseGridTopology::launchPropagationFromSeed(const type::Vec3i &point,
        sofa::core::sptr<RegularGridTopology> regularGrid,
        vector<Type>& regularGridTypes,
        vector<bool>& alreadyTested,
        std::stack<type::Vec3i> &seed) const
{
    seed.push(point);
    while (!seed.empty())
    {
        const type::Vec3i &s=seed.top();
        seed.pop();
        propagateFrom(s,regularGrid,regularGridTypes,alreadyTested,seed);
    }
}

void SparseGridTopology::buildFromRegularGridTypes(sofa::core::sptr<RegularGridTopology> regularGrid, const vector<Type>& regularGridTypes)
{
    vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values

    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid->getNbHexahedra(), InvalidID); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    int cubeCntr = 0;

    // add BOUNDARY cubes to valid cells
    for(Index w=0; w<regularGrid->getNbHexahedra(); ++w)
    {
        if( regularGridTypes[w] == BOUNDARY && !d_bOnlyInsideCells.getValue())
        {
            _types.push_back(BOUNDARY);
            _indicesOfRegularCubeInSparseGrid[w] = cubeCntr++;
            _indicesOfCubeinRegularGrid.push_back( w );


            Hexa c = regularGrid->getHexaCopy(w);
            CubeCorners corners;
            for(int j=0; j<8; ++j)
            {
                corners[j] = regularGrid->getPoint( c[j] );
                cubeCornerPositionIndiceMap[corners[j]] = 0;
            }
            cubeCorners.push_back(corners);
        }
    }

    // add INSIDE cubes to valid cells
    for(Index w=0; w<regularGrid->getNbHexahedra(); ++w)
    {
        if( regularGridTypes[w] == INSIDE )
        {
            _types.push_back(INSIDE);
            _indicesOfRegularCubeInSparseGrid[w] = cubeCntr++;
            _indicesOfCubeinRegularGrid.push_back( w );

            Hexa c = regularGrid->getHexaCopy(w);
            CubeCorners corners;
            for(int j=0; j<8; ++j)
            {
                corners[j] = regularGrid->getPoint( c[j] );
                cubeCornerPositionIndiceMap[corners[j]] = 0;
            }
            cubeCorners.push_back(corners);
        }
    }

    type::vector<type::Vec3 >& seqPoints = *this->seqPoints.beginEdit(); seqPoints.clear();
    // compute corner indices
    int cornerCounter=0;

    for(MapBetweenCornerPositionAndIndice::iterator it = cubeCornerPositionIndiceMap.begin();
        it!=cubeCornerPositionIndiceMap.end(); ++it, ++cornerCounter)
    {
        (*it).second = cornerCounter;
        seqPoints.push_back( (*it).first );
    }

    this->seqPoints.beginEdit();
    nbPoints = (int)cubeCornerPositionIndiceMap.size();

    SeqHexahedra& hexahedra = *seqHexahedra.beginEdit();

    for( unsigned w=0; w<cubeCorners.size(); ++w)
    {
        Hexa c;
        for(int j=0; j<8; ++j)
            c[j] = cubeCornerPositionIndiceMap[cubeCorners[w][j]];

        hexahedra.push_back(c);
    }

    seqHexahedra.endEdit();
}


void SparseGridTopology::computeBoundingBox(const type::vector<type::Vec3>& vertices,
        SReal& xmin, SReal& xmax,
        SReal& ymin, SReal& ymax,
        SReal& zmin, SReal& zmax) const
{
    // bounding box computation
    xmin = vertices[0][0];
    ymin = vertices[0][1];
    zmin = vertices[0][2];
    xmax = vertices[0][0];
    ymax = vertices[0][1];
    zmax = vertices[0][2];

    for(unsigned w=1; w<vertices.size(); ++w)
    {
        if( vertices[w][0] > xmax )
            xmax = vertices[w][0];
        else if( vertices[w][0] < xmin)
            xmin = vertices[w][0];

        if( vertices[w][1] > ymax )
            ymax = vertices[w][1];
        else if( vertices[w][1] < ymin )
            ymin = vertices[w][1];

        if( vertices[w][2] > zmax )
            zmax = vertices[w][2];
        else if( vertices[w][2] < zmin )
            zmin = vertices[w][2];
    }
}


void SparseGridTopology::buildFromFiner()
{
    if (_finerSparseGrid->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    setNx( _finerSparseGrid->getNx()/2+1 );
    setNy( _finerSparseGrid->getNy()/2+1 );
    setNz( _finerSparseGrid->getNz()/2+1 );

    _regularGrid->setSize(getNx(),getNy(),getNz());

    setMin(_finerSparseGrid->getMin());
    setMax(_finerSparseGrid->getMax());

    // the cube size of the coarser mesh is twice the cube size of the finer mesh
    // if the finer mesh contains an odd number of cubes in any direction,
    // the coarser mesh will be a half cube size larger in that direction
    type::Vec3 dx = _finerSparseGrid->_regularGrid->getDx();
    type::Vec3 dy = _finerSparseGrid->_regularGrid->getDy();
    type::Vec3 dz = _finerSparseGrid->_regularGrid->getDz();
    setXmax(getXmin() + (getNx()-1) * (SReal)2.0 * dx[0]);
    setYmax(getYmin() + (getNy()-1) * (SReal)2.0 * dy[1]);
    setZmax(getZmin() + (getNz()-1) * (SReal)2.0 * dz[2]);

    _regularGrid->setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());

    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid->getNbHexahedra(), InvalidID ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values

    for(int i=0; i<getNx()-1; i++)
    {
        for(int j=0; j<getNy()-1; j++)
        {
            for(int k=0; k<getNz()-1; k++)
            {
                const int x = 2*i;
                const int y = 2*j;
                const int z = 2*k;

                fixed_array<Index,8> fineIndices;
                for(int idx=0; idx<8; ++idx)
                {
                    const int idxX = x + (idx & 1);
                    const int idxY = y + (idx & 2)/2;
                    const int idxZ = z + (idx & 4)/4;
                    if(idxX < _finerSparseGrid->getNx()-1 && idxY < _finerSparseGrid->getNy()-1 && idxZ < _finerSparseGrid->getNz()-1)
                        fineIndices[idx] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid->cube(idxX,idxY,idxZ) ];
                    else
                        fineIndices[idx] = InvalidID;
                }

                bool inside = true;
                bool outside = true;
                for( int w=0; w<8 && (inside || outside); ++w)
                {
                    if( fineIndices[w] == InvalidID ) inside=false;
                    else
                    {
                        if( _finerSparseGrid->getType( fineIndices[w] ) == BOUNDARY ) { inside=false; outside=false; }
                        else if( _finerSparseGrid->getType( fineIndices[w] ) == INSIDE ) {outside=false;}
                    }
                }

                if(outside) continue;
                if( inside )
                {
                    _types.push_back(INSIDE);
                }
                else
                {
                    _types.push_back(BOUNDARY);
                }


                Index coarseRegularIndice = _regularGrid->cube( i,j,k );
                Hexa c = _regularGrid->getHexaCopy( coarseRegularIndice );

                CubeCorners corners;
                for(int w=0; w<8; ++w)
                {
                    corners[w] = _regularGrid->getPoint( c[w] );
                    cubeCornerPositionIndiceMap[corners[w]] = 0;
                }

                cubeCorners.push_back(corners);

                _indicesOfRegularCubeInSparseGrid[coarseRegularIndice] = (int)cubeCorners.size()-1;
                _indicesOfCubeinRegularGrid.push_back( coarseRegularIndice );

                // 			_hierarchicalCubeMap[cubeCorners.size()-1]=fineIndices;
                _hierarchicalCubeMap.push_back( fineIndices );
            }
        }
    }

    // compute corner indices
    int cornerCounter=0;
    type::vector<type::Vec3 >& seqPoints = *this->seqPoints.beginEdit(); seqPoints.clear();
    for(MapBetweenCornerPositionAndIndice::iterator it=cubeCornerPositionIndiceMap.begin(); it!=cubeCornerPositionIndiceMap.end(); ++it,++cornerCounter)
    {
        (*it).second = cornerCounter;
        seqPoints.push_back( (*it).first );
    }
    this->seqPoints.endEdit();
    nbPoints = (int)cubeCornerPositionIndiceMap.size();

    SeqHexahedra& hexahedra = *seqHexahedra.beginEdit();
    for( unsigned w=0; w<cubeCorners.size(); ++w)
    {
        Hexa c;
        for(int j=0; j<8; ++j)
            c[j] = cubeCornerPositionIndiceMap[cubeCorners[w][j]];

        hexahedra.push_back(c);
    }
    seqHexahedra.endEdit();


    // for interpolation and restriction
    _hierarchicalPointMap.resize(this->seqPoints.getValue().size());
    _finerSparseGrid->_inverseHierarchicalPointMap.resize(_finerSparseGrid->seqPoints.getValue().size());
    _finerSparseGrid->_inversePointMap.resize(_finerSparseGrid->seqPoints.getValue().size()); _finerSparseGrid->_inversePointMap.fill(InvalidID);
    _pointMap.resize(this->seqPoints.getValue().size()); _pointMap.fill(InvalidID);

    for( unsigned w=0; w<seqHexahedra.getValue().size(); ++w)
    {
        const fixed_array<Index, 8>& child = _hierarchicalCubeMap[w];

        type::vector<Index> fineCorners(27);
        fineCorners.fill(InvalidID);
        for(int fineCube=0; fineCube<8; ++fineCube)
        {
            if( child[fineCube] == InvalidID ) continue;

            const Hexa& cube = _finerSparseGrid->getHexahedron(child[fineCube]);

            for(int vertex=0; vertex<8; ++vertex)
            {
                // 					if( fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=-1 && fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=cube[vertex] )
                // 						msg_error()<<"couille fineCorners";
                fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]=cube[vertex];
            }

        }


        for(int coarseCornerLocalIndice=0; coarseCornerLocalIndice<8; ++coarseCornerLocalIndice)
        {
            for( int fineVertexLocalIndice=0; fineVertexLocalIndice<27; ++fineVertexLocalIndice)
            {
                if( fineCorners[fineVertexLocalIndice] == InvalidID ) continue; // this fine vertex is not in any fine cube

                Index coarseCornerGlobalIndice = seqHexahedra.getValue()[w][coarseCornerLocalIndice];
                Index fineVertexGlobalIndice = fineCorners[fineVertexLocalIndice];

                if( WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice] )
                {
                    _hierarchicalPointMap[coarseCornerGlobalIndice][fineVertexGlobalIndice] = WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice];
                    // 						_hierarchicalPointMap[coarseCornerGlobalIndice].push_back( std::pair<int,float>(fineVertexGlobalIndice, WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice]) );

                    _finerSparseGrid->_inverseHierarchicalPointMap[fineVertexGlobalIndice][coarseCornerGlobalIndice] = WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice];

                    if( WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice] == 1.0 )
                    {
                        _finerSparseGrid->_inversePointMap[fineVertexGlobalIndice] = coarseCornerGlobalIndice;
                        // 							msg_error()<<getPX(coarseCornerGlobalIndice)<<" "<<getPY(coarseCornerGlobalIndice)<<" "<<getPZ(coarseCornerGlobalIndice)<<" ----- ";
                        // 							msg_error()<<_finerSparseGrid->getPX(fineVertexGlobalIndice)<<" "<<_finerSparseGrid->getPY(fineVertexGlobalIndice)<<" "<<_finerSparseGrid->getPZ(fineVertexGlobalIndice);
                    }
                }
            }
        }
    }

    for( unsigned i=0; i<_finerSparseGrid->_inversePointMap.size(); ++i)
    {
        if(_finerSparseGrid->_inversePointMap[i] != InvalidID)
            _pointMap[ _finerSparseGrid->_inversePointMap[i] ] = i;
    }

    _finerSparseGrid->_coarserSparseGrid = this;
    _finerSparseGrid->_inverseHierarchicalCubeMap.resize( _finerSparseGrid->seqHexahedra.getValue().size(), InvalidID);
    for( unsigned i=0; i<_hierarchicalCubeMap.size(); ++i)
    {
        for(int w=0; w<8; ++w)
        {
            if(_hierarchicalCubeMap[i][w] != InvalidID)
                _finerSparseGrid->_inverseHierarchicalCubeMap[ _hierarchicalCubeMap[i][w] ] = i;
        }
    }

    // compute stiffness coefficient from children
    _stiffnessCoefs.resize( this->getNbHexahedra() );
    _massCoefs.resize( this->getNbHexahedra() );
    for(size_t i=0; i<this->getNbHexahedra(); ++i)
    {
        type::fixed_array<Index,8> finerChildren = this->_hierarchicalCubeMap[i];
        for(int w=0; w<8; ++w)
        {
            if( finerChildren[w] != InvalidID)
            {
                _stiffnessCoefs[i] += _finerSparseGrid->_stiffnessCoefs[finerChildren[w]];
                _massCoefs[i] += _finerSparseGrid->_massCoefs[finerChildren[w]];
            }
        }
        _stiffnessCoefs[i] /= 8.0;//(float)nbchildren;
        _massCoefs[i] /= 8.0;
    }
}


void SparseGridTopology::buildVirtualFinerLevels()
{
    const int nb = _nbVirtualFinerLevels.getValue();

    _virtualFinerLevels.resize(nb);

    int newnx=n.getValue()[0],newny=n.getValue()[1],newnz=n.getValue()[2];
    for( int i=0; i<nb; ++i)
    {
        newnx = (newnx-1)*2+1;
        newny = (newny-1)*2+1;
        newnz = (newnz-1)*2+1;
    }

    _virtualFinerLevels[0] = sofa::core::objectmodel::New< SparseGridTopology >(true);
    _virtualFinerLevels[0]->setNx( newnx );
    _virtualFinerLevels[0]->setNy( newny );
    _virtualFinerLevels[0]->setNz( newnz );
    _virtualFinerLevels[0]->setMin( _min.getValue() );
    _virtualFinerLevels[0]->setMax( _max.getValue() );
    this->addSlave(_virtualFinerLevels[0]); //->setContext( this->getContext( ) );
    const std::string& fileTopology = this->fileTopology.getValue();
    if (fileTopology.empty()) // If no file is defined, try to build from the input Datas
    {
        _virtualFinerLevels[0]->seqPoints.setParent(&this->seqPoints);
        _virtualFinerLevels[0]->facets.setParent(&this->facets);
        _virtualFinerLevels[0]->seqTriangles.setParent(&this->seqTriangles);
        _virtualFinerLevels[0]->seqQuads.setParent(&this->seqQuads);
    }
    else
        _virtualFinerLevels[0]->load(fileTopology.c_str());
    _virtualFinerLevels[0]->_fillWeighted.setValue( _fillWeighted.getValue() );
    _virtualFinerLevels[0]->init();

    dmsg_info()<<"SparseGridTopology "<<getName()<<" buildVirtualFinerLevels : ";
    dmsg_info()<<"("<<newnx<<"x"<<newny<<"x"<<newnz<<") -> "<< _virtualFinerLevels[0]->getNbHexahedra() <<" elements , ";

    for(int i=1; i<nb; ++i)
    {
        _virtualFinerLevels[i] = sofa::core::objectmodel::New< SparseGridTopology >(true);
        this->addSlave(_virtualFinerLevels[i]);

        _virtualFinerLevels[i]->setFinerSparseGrid(_virtualFinerLevels[i-1].get());

        _virtualFinerLevels[i]->init();

        dmsg_info()<<"("<<_virtualFinerLevels[i]->getNx()<<"x"<<_virtualFinerLevels[i]->getNy()<<"x"<<_virtualFinerLevels[i]->getNz()<<") -> "<< _virtualFinerLevels[i]->getNbHexahedra() <<" elements , ";
    }

    this->setFinerSparseGrid(_virtualFinerLevels[nb-1].get());
}


/// return the cube containing the given point (or InvalidID if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
SparseGridTopology::Index SparseGridTopology::findCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    const Index indiceInRegularGrid = _regularGrid->findCube( pos,fx,fy,fz);
    if( indiceInRegularGrid == InvalidID)
        return InvalidID;
    else
        return _indicesOfRegularCubeInSparseGrid[indiceInRegularGrid];
}


/// return the cube containing the given point (or InvalidID if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
SparseGridTopology::Index SparseGridTopology::findNearestCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    const auto& hexahedra = seqHexahedra.getValue();
    if (hexahedra.size() == 0) return InvalidID;
    Index indice = 0;
    float lgmin = 99999999.0f;

    const auto& points = seqPoints.getValue();
    for(unsigned w=0; w<hexahedra.size(); ++w)
    {
        if(!_usingMC && _types[w]!=BOUNDARY )continue;

        const Hexa& c = getHexahedron( w );
        const Index c0 = c[0];
        const Index c7 = c[6];

        const auto& p0 = points[c0];
        const auto& p7 = points[c7];

        type::Vec3 barycenter = (p0+p7) * .5;

        const float lg = (float)((pos-barycenter).norm());
        if( lg < lgmin )
        {
            lgmin = lg;
            indice = w;
        }
    }

    const Hexa& c = getHexahedron( indice );
    const Index c0 = c[0];

    const Index c7 = c[6];

    const type::Vec3 p0((SReal)getPX(c0), (SReal)getPY(c0), (SReal)getPZ(c0));
    const type::Vec3 p7((SReal)getPX(c7), (SReal)getPY(c7), (SReal)getPZ(c7));

    type::Vec3 relativePos = pos-p0;
    type::Vec3 diagonal = p7 - p0;

    fx = relativePos[0] / diagonal[0];
    fy = relativePos[1] / diagonal[1];
    fz = relativePos[2] / diagonal[2];

    return indice;
}


type::fixed_array<Index,6> SparseGridTopology::findneighboorCubes(Index indice )
{
    dmsg_info()<<"SparseGridTopology::findneighboorCubes : "<<indice<<" -> "<<_indicesOfCubeinRegularGrid[indice];
    dmsg_info()<<_indicesOfRegularCubeInSparseGrid[ _indicesOfCubeinRegularGrid[indice] ] ;
    type::fixed_array<Index,6> result;
    type::Vec3 c = _regularGrid->getCubeCoordinate( _indicesOfCubeinRegularGrid[indice] );
    dmsg_info()<<c;
    result[0] = c[0]<=0 ? InvalidID : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0]-1,(int)c[1],(int)c[2] )];
    result[1] = c[0]>=getNx()-2 ? InvalidID : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0]+1,(int)c[1],(int)c[2] )];
    result[2] = c[1]<=0 ? InvalidID : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1]-1,(int)c[2] )];
    result[3] = c[1]>=getNy()-2 ? InvalidID : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1]+1,(int)c[2] )];
    result[4] = c[2]<=0 ? InvalidID : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1],(int)c[2]-1 )];
    result[5] = c[2]>=getNz()-2 ? InvalidID : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1],(int)c[2]+1 )];
    return result;
}


SparseGridTopology::Type SparseGridTopology::getType(Index i )
{
    return _types[i];
}


float SparseGridTopology::getStiffnessCoef(Index elementIdx)
{
    return _stiffnessCoefs[ elementIdx ];
}


float SparseGridTopology::getMassCoef(Index elementIdx)
{
    return _massCoefs[ elementIdx ];
}


void SparseGridTopology::updateEdges()
{
    using IndexPair = pair<Index, Index>;
    std::map<IndexPair,bool> edgesMap;
    for(unsigned i=0; i<seqHexahedra.getValue().size(); ++i)
    {
        Hexa c = seqHexahedra.getValue()[i];

        // horizontal
        edgesMap[IndexPair(c[0],c[1])]=0;
        edgesMap[IndexPair(c[3],c[2])]=0;
        edgesMap[IndexPair(c[4],c[5])]=0;
        edgesMap[IndexPair(c[7],c[6])]=0;
        // vertical
        edgesMap[IndexPair(c[0],c[3])]=0;
        edgesMap[IndexPair(c[1],c[2])]=0;
        edgesMap[IndexPair(c[4],c[7])]=0;
        edgesMap[IndexPair(c[5],c[6])]=0;
        // profondeur
        edgesMap[IndexPair(c[0],c[4])]=0;
        edgesMap[IndexPair(c[1],c[5])]=0;
        edgesMap[IndexPair(c[3],c[7])]=0;
        edgesMap[IndexPair(c[2],c[6])]=0;
    }

    SeqEdges& edges = *seqEdges.beginEdit();
    edges.clear();
    edges.reserve(edgesMap.size());
    for( auto it=edgesMap.cbegin(); it!=edgesMap.cend(); ++it)
        edges.push_back( Edge( (*it).first.first,  (*it).first.second ));
    seqEdges.endEdit();
}


void SparseGridTopology::updateQuads()
{
    std::map<fixed_array<Index,4>,bool> quadsMap;
    for(unsigned i=0; i<seqHexahedra.getValue().size(); ++i)
    {
        Hexa c = seqHexahedra.getValue()[i];
        fixed_array<Index,4> v;

        v[0]=c[0]; v[1]=c[1]; v[2]=c[2]; v[3]=c[3];
        quadsMap[v]=0;
        v[0]=c[4]; v[1]=c[5]; v[2]=c[6]; v[3]=c[7];
        quadsMap[v]=0;
        v[0]=c[0]; v[1]=c[1]; v[2]=c[5]; v[3]=c[4];
        quadsMap[v]=0;
        v[0]=c[3]; v[1]=c[2]; v[2]=c[6]; v[3]=c[7];
        quadsMap[v]=0;
        v[0]=c[0]; v[1]=c[4]; v[2]=c[7]; v[3]=c[3];
        quadsMap[v]=0;
        v[0]=c[1]; v[1]=c[5]; v[2]=c[6]; v[3]=c[2];
        quadsMap[v]=0;
    }
    SeqQuads& quads = *seqQuads.beginEdit();
    quads.clear();
    quads.reserve(quadsMap.size());
    for( auto it=quadsMap.cbegin(); it!=quadsMap.cend(); ++it)
        quads.push_back( Quad( (*it).first[0],  (*it).first[1],(*it).first[2],(*it).first[3] ));
    seqQuads.endEdit();
}


void SparseGridTopology::propagateFrom( const type::Vec3i &point,
        sofa::core::sptr<RegularGridTopology> regularGrid,
        vector<Type>& regularGridTypes,
        vector<bool>& alreadyTested,
        std::stack< Vec<3,int> > &seed) const
{
    const int x=point[0];
    const int y=point[1];
    const int z=point[2];
    assert( x>=0 && x<=regularGrid->getNx()-2 && y>=0 && y<=regularGrid->getNy()-2 && z>=0 && z<=regularGrid->getNz()-2 );

    const Index indice = regularGrid->cube( x, y, z );
    if( alreadyTested[indice] || regularGridTypes[indice] == BOUNDARY ) return;

    alreadyTested[indice] = true;
    regularGridTypes[indice] = OUTSIDE;

    if(x>0)                     seed.push(type::Vec3i(x-1,y,z));
    if(x<regularGrid->getNx()-2) seed.push(type::Vec3i(x+1,y,z));

    if(y>0)                     seed.push(type::Vec3i(x,y-1,z));
    if(y<regularGrid->getNy()-2) seed.push(type::Vec3i(x,y+1,z));

    if(z>0)                     seed.push(type::Vec3i(x,y,z-1));
    if(z<regularGrid->getNz()-2) seed.push(type::Vec3i(x,y,z+1));
}

} //namespace sofa::component::topology::container::grid
