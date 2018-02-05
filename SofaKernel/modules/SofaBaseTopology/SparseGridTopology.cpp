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
#include <sofa/simulation/Node.h>
#include <SofaBaseTopology/SparseGridTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/polygon_cube_intersection/polygon_cube_intersection.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/defaulttype/VecTypes.h>

#include <fstream>
#include <string>
#include <math.h>



using std::pair;
using sofa::core::loader::VoxelLoader;
using namespace sofa::defaulttype;
using namespace sofa::helper;
namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(SparseGridTopology)

int SparseGridTopologyClass = core::RegisterObject("Sparse grid in 3D")
        .addAlias("SparseGrid")
        .add< SparseGridTopology >()
        ;


// 	  const float SparseGridTopology::WEIGHT[8][8] =
// 	  {
// 		  { 1, .5, .5, .25,  .5,.25,.25, .125 }, // fine cube 0 from coarser corner 0 -> what weight for a vertex?
// 		  { .5,1,.25,.5,.25,.5,.125,.25 },
// 		  {.5,.25,1,.5,.25,.125,.5,.25},
// 		  {.25,.5,.5,1,.125,.25,.25,.5},
// 		  {.5,.25,.25,.125,1,.5,.5,.25},
// 		  {.25,.5,.125,.25,.5,1,.25,.5},
// 		  {.25,.125,.5,.25,.5,.25,1,.5},
// 		  {.125,.25,.25,.5,.25,.5,.5,1}
// 	  };


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

const int SparseGridTopology::cornerIndicesFromFineToCoarse[8][8]=
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
    :
    _fillWeighted(initData(&_fillWeighted, true, "fillWeighted", "Is quantity of matter inside a cell taken into account? (.5 for boundary, 1 for inside)")),
    d_bOnlyInsideCells(initData(&d_bOnlyInsideCells, false, "onlyInsideCells", "Select only inside cells (exclude boundary cells)")),
    n(initData(&n, Vec3i(2,2,2), "n", "grid resolution")),
    _min(initData(&_min, Vector3(0,0,0), "min","Min")),
    _max(initData(&_max, Vector3(0,0,0), "max","Max")),
    _cellWidth(initData(&_cellWidth, (SReal)0.0, "cellWidth","if > 0 : dimension of each cell in the created grid")),
    _nbVirtualFinerLevels( initData(&_nbVirtualFinerLevels, 0, "nbVirtualFinerLevels", "create virtual (not in the animation tree) finer sparse grids in order to dispose of finest information (usefull to compute better mechanical properties for example)")),
    dataResolution(initData(&dataResolution, Vec3i(0,0,0), "dataResolution", "Dimension of the voxel File")),
    voxelSize(initData(&voxelSize, Vector3(1.0f,1.0f,1.0f), "voxelSize", "Dimension of one voxel")),
    marchingCubeStep(initData(&marchingCubeStep, (unsigned int) 1, "marchingCubeStep", "Step of the Marching Cube algorithm")),
    convolutionSize(initData(&convolutionSize, (unsigned int) 0, "convolutionSize", "Dimension of the convolution kernel to smooth the voxels. 0 if no smoothing is required.")),
    vertices(initData(&vertices, "vertices", "Input mesh vertices")),
    facets(initData(&facets, "facets", "Input mesh facets")),
    input_triangles(initData(&input_triangles, "input_triangles", "Input mesh triangles")),
    input_quads(initData(&input_quads, "input_quads", "Input mesh quads"))
{
    isVirtual = _isVirtual;
    _alreadyInit = false;
    _finerSparseGrid = NULL;
    _coarserSparseGrid = NULL;
    _usingMC = false;

    _regularGrid = sofa::core::objectmodel::New<RegularGridTopology>();

    //Add alias to use MeshLoader
    addAlias(&vertices,"position");
    addAlias(&input_triangles,"triangles");
    addAlias(&input_quads,"quads");
}

SparseGridTopology::SparseGridTopology(Vec3i numVertices, BoundingBox box, bool _isVirtual)
    :
    _fillWeighted(initData(&_fillWeighted, true, "fillWeighted", "Is quantity of matter inside a cell taken into account? (.5 for boundary, 1 for inside)")),
    d_bOnlyInsideCells(initData(&d_bOnlyInsideCells, false, "onlyInsideCells", "Select only inside cells (exclude boundary cells)")),
    n(initData(&n, Vec3i(2,2,2), "n", "grid resolution")),
    _min(initData(&_min, Vector3(0,0,0), "min","Min")),
    _max(initData(&_max, Vector3(0,0,0), "max","Max")),
    _cellWidth(initData(&_cellWidth, (SReal)0.0, "cellWidth","if > 0 : dimension of each cell in the created grid")),
    _nbVirtualFinerLevels( initData(&_nbVirtualFinerLevels, 0, "nbVirtualFinerLevels", "create virtual (not in the animation tree) finer sparse grids in order to dispose of finest information (usefull to compute better mechanical properties for example)")),
    dataResolution(initData(&dataResolution, Vec3i(0,0,0), "dataResolution", "Dimension of the voxel File")),
    voxelSize(initData(&voxelSize, Vector3(1.0f,1.0f,1.0f), "voxelSize", "Dimension of one voxel")),
    marchingCubeStep(initData(&marchingCubeStep, (unsigned int) 1, "marchingCubeStep", "Step of the Marching Cube algorithm")),
    convolutionSize(initData(&convolutionSize, (unsigned int) 0, "convolutionSize", "Dimension of the convolution kernel to smooth the voxels. 0 if no smoothing is required.")),
    vertices(initData(&vertices, "vertices", "Input mesh vertices")),
    facets(initData(&facets, "facets", "Input mesh facets")),
    input_triangles(initData(&input_triangles, "input_triangles", "Input mesh triangles")),
    input_quads(initData(&input_quads, "input_quads", "Input mesh quads"))
{
    isVirtual = _isVirtual;
    _alreadyInit = false;
    _finerSparseGrid = NULL;
    _coarserSparseGrid = NULL;
    _usingMC = false;

    _regularGrid = sofa::core::objectmodel::New<RegularGridTopology>();

    //Add alias to use MeshLoader
    addAlias(&vertices,"position");
    addAlias(&input_triangles,"triangles");
    addAlias(&input_quads,"quads");

    setN(numVertices);
    setMin(box.minBBox());
    setMax(box.maxBBox());
}


void SparseGridTopology::init()
{
    if(_alreadyInit) return;
    _alreadyInit = true;

    Vec3i grid = n.getValue();

    if(grid[0] < 2) grid[0]= 2;
    if(grid[1] < 2) grid[1]= 2;
    if(grid[2] < 2) grid[2]= 2;

    n.setValue(grid);

    if( _nbVirtualFinerLevels.getValue() )
        buildVirtualFinerLevels();


    if( _finerSparseGrid != NULL )
        buildFromFiner();
    else
        buildAsFinest();

    _nodeAdjacency.resize(seqPoints.getValue().size() );
    for(unsigned i=0; i<seqPoints.getValue().size(); ++i)
        _nodeAdjacency[i].assign(-1);

    for(unsigned i=0; i<seqHexahedra.getValue().size(); ++i)
    {
        _nodeAdjacency[ seqHexahedra.getValue()[i][0] ][RIGHT] = seqHexahedra.getValue()[i][1];
        _nodeAdjacency[ seqHexahedra.getValue()[i][0] ][UP] = seqHexahedra.getValue()[i][2];
        _nodeAdjacency[ seqHexahedra.getValue()[i][0] ][BEHIND] = seqHexahedra.getValue()[i][4];

        _nodeAdjacency[ seqHexahedra.getValue()[i][1] ][LEFT] = seqHexahedra.getValue()[i][0];
        _nodeAdjacency[ seqHexahedra.getValue()[i][1] ][UP] = seqHexahedra.getValue()[i][3];
        _nodeAdjacency[ seqHexahedra.getValue()[i][1] ][BEHIND] = seqHexahedra.getValue()[i][5];

        _nodeAdjacency[ seqHexahedra.getValue()[i][2] ][RIGHT] = seqHexahedra.getValue()[i][3];
        _nodeAdjacency[ seqHexahedra.getValue()[i][2] ][DOWN] = seqHexahedra.getValue()[i][0];
        _nodeAdjacency[ seqHexahedra.getValue()[i][2] ][BEHIND] = seqHexahedra.getValue()[i][6];

        _nodeAdjacency[ seqHexahedra.getValue()[i][3] ][LEFT] = seqHexahedra.getValue()[i][2];
        _nodeAdjacency[ seqHexahedra.getValue()[i][3] ][DOWN] = seqHexahedra.getValue()[i][1];
        _nodeAdjacency[ seqHexahedra.getValue()[i][3] ][BEHIND] = seqHexahedra.getValue()[i][7];

        _nodeAdjacency[ seqHexahedra.getValue()[i][4] ][RIGHT] = seqHexahedra.getValue()[i][5];
        _nodeAdjacency[ seqHexahedra.getValue()[i][4] ][UP] = seqHexahedra.getValue()[i][6];
        _nodeAdjacency[ seqHexahedra.getValue()[i][4] ][BEFORE] = seqHexahedra.getValue()[i][0];

        _nodeAdjacency[ seqHexahedra.getValue()[i][5] ][LEFT] = seqHexahedra.getValue()[i][4];
        _nodeAdjacency[ seqHexahedra.getValue()[i][5] ][UP] = seqHexahedra.getValue()[i][7];
        _nodeAdjacency[ seqHexahedra.getValue()[i][5] ][BEFORE] = seqHexahedra.getValue()[i][1];

        _nodeAdjacency[ seqHexahedra.getValue()[i][6] ][RIGHT] = seqHexahedra.getValue()[i][7];
        _nodeAdjacency[ seqHexahedra.getValue()[i][6] ][DOWN] = seqHexahedra.getValue()[i][4];
        _nodeAdjacency[ seqHexahedra.getValue()[i][6] ][BEFORE] = seqHexahedra.getValue()[i][2];

        _nodeAdjacency[ seqHexahedra.getValue()[i][7] ][LEFT] = seqHexahedra.getValue()[i][6];
        _nodeAdjacency[ seqHexahedra.getValue()[i][7] ][DOWN] = seqHexahedra.getValue()[i][5];
        _nodeAdjacency[ seqHexahedra.getValue()[i][7] ][BEFORE] = seqHexahedra.getValue()[i][3];
    }


    //	_nodeCubesAdjacency.clear();
    _nodeCubesAdjacency.resize(seqPoints.getValue().size() );

    for(unsigned i=0; i<seqHexahedra.getValue().size(); ++i)
    {
        for(int j=0; j<8; ++j)
        {
            _nodeCubesAdjacency[ seqHexahedra.getValue()[i][j] ].push_back( i );
        }
    }
}


void SparseGridTopology::buildAsFinest(  )
{
    //	serr<<"SparseGridTopology::buildAsFinest(  )"<<sendl;

    VoxelLoader *loader;
    getContext()->get(loader);
    if( loader )
    {
        buildFromVoxelLoader(loader);
    }
    else
    {
        std::string _filename = fileTopology.getFullPath();

        if (vertices.getValue().empty() )
        {
            if (_filename.empty())
            {
                serr << "SparseGridTopology: no filename specified." << sendl;
                return;
            }

            if (! sofa::helper::system::DataRepository.findFile ( _filename ))
                return;

            // initialize the following datafields:
            // xmin, xmax, ymin, ymax, zmin, zmax, evtl. nx, ny, nz
            // _regularGrid, _indicesOfRegularCubeInSparseGrid, _types
            // seqPoints, seqHexahedra.getValue(), nbPoints
            if(_filename.length() > 4 && _filename.compare(_filename.length()-4, 4, ".obj")==0)
            {
                // std::cout << "SparseGridTopology: using mesh "<<_filename<<std::endl;
                buildFromTriangleMesh(_filename);
            }
            else if(_filename.length() > 6 && _filename.compare(_filename.length()-6, 6, ".trian")==0)
            {
                // std::cout << "SparseGridTopology: using mesh "<<_filename<<std::endl;
                buildFromTriangleMesh(_filename);
            }
            else if(_filename.length() > 4 && _filename.compare(_filename.length()-4, 4, ".stl")==0)
            {
                // std::cout << "SparseGridTopology: using mesh "<<_filename<<std::endl;
                buildFromTriangleMesh(_filename);
            }
            else if(_filename.length() > 4 && _filename.compare(_filename.length()-4, 4, ".raw")==0)
            {
                // std::cout << "SparseGridTopology: using mesh "<<_filename<<std::endl;
                _usingMC = true;

                buildFromRawVoxelFile(_filename);
            }
            else if(_filename.length() > 6 && _filename.compare(_filename.length()-6, 6, ".voxel")==0)
            {
                buildFromVoxelFile(_filename);
            }
            else
            {
                serr << "SparseGridTopology::buildAsFinest: extension unrecognized " << sendl;
                return;
            }
        }
        else
        {
            buildFromTriangleMesh(_filename);
        }

        // default stiffness coefficient : BOUNDARY=.5, INSIDE=1
        _stiffnessCoefs.resize( this->getNbHexahedra());
        _massCoefs.resize( this->getNbHexahedra());

        for(int i=0; i<this->getNbHexahedra(); ++i)
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
            serr << "SparseGridTopology: failed to open file " << filename << sendl;
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

        n.setValue(Vec3i(nx,ny,nz));
        _min.setValue(Vector3((SReal)xmin, (SReal)ymin, (SReal)zmin));
        _max.setValue(Vector3((SReal)xmax, (SReal)ymax, (SReal)zmax));

        int value;
        int numVoxels = 0;
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
                        numVoxels++;
                    }
                }
            }
        }

        file.close();
    }

    _regularGrid->setSize(getNx(), getNy(), getNz());
    _regularGrid->setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());
    int nbCubesRG = _regularGrid->getNbHexahedra();
    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, -1); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    // fill the regularGridTypes vector
    // at the moment, no BOUNDARY type voxels are generated at the finest level
    for(int i=0; i<nbCubesRG; ++i)
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
void SparseGridTopology::buildFromData( Vec3i numPoints, BoundingBox box, const vector<bool>& filling )
{
    n.setValue(numPoints);
    _min.setValue(box.minBBox());
    _max.setValue(box.maxBBox());
    Vector3 numVoxels(numPoints[0]-1,numPoints[1]-1,numPoints[2]-1);

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
                    setVoxel(x + (int)numVoxels[0] * (y + (int)numVoxels[1] * z),1);
                }
                f++;
            }
        }
    }


    _regularGrid->setSize(getNx(), getNy(), getNz());
    _regularGrid->setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());
    int nbCubesRG = _regularGrid->getNbHexahedra();
    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, -1); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    // fill the regularGridTypes vector
    // at the moment, no BOUNDARY type voxels are generated at the finest level
    for(int i=0; i<nbCubesRG; ++i)
    {
        if(dataVoxels.getValue()[i] != 0.0f)
        {
            regularGridTypes[i] = INSIDE;
        }
    }

    buildFromRegularGridTypes(_regularGrid, regularGridTypes );

    _stiffnessCoefs.resize( this->getNbHexahedra());
    _massCoefs.resize( this->getNbHexahedra());
    for(int i=0; i<this->getNbHexahedra(); ++i)
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
    const int nbCubesRG = _regularGrid->getNbHexahedra();

    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, -1); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    //Loading from file
    if (dataVoxels.getValue().size() == 0)
    {
        FILE *file = fopen( filename.c_str(), "r" );
        if (!file) { serr<< "FILE " << filename << " not found"<<sendl; return;}
        //Get the voxels from the file
        dataVoxels.beginEdit()->resize(dataResolution.getValue()[0]*dataResolution.getValue()[1]*dataResolution.getValue()[2], (unsigned char)0);

        const Vector3 transform(                (getNx()-1)/(float)dataResolution.getValue()[0],
                (getNy()-1)/(float)dataResolution.getValue()[1],
                (getNz()-1)/(float)dataResolution.getValue()[2]);

        for (unsigned int i=0; i<dataVoxels.beginEdit()->size(); i++)
        {
            unsigned char value=getc(file);
            if ((int)value != 0)
            {
                setVoxel(i,1);

                const int z = i/(dataResolution.getValue()[0]*dataResolution.getValue()[1]);
                const int y = (  (i%(dataResolution.getValue()[0]*dataResolution.getValue()[1]))
                        /(dataResolution.getValue()[0])                  	    );
                const int x = i%dataResolution.getValue()[0];

                unsigned int indexGrid =
                    (unsigned int)(z*transform[2])*(getNy()-1)*(getNx()-1)
                    + (unsigned int)(y*transform[1])*(getNx()-1)
                    + (unsigned int)(x*transform[0]);

                regularGridTypes[indexGrid] = INSIDE;
            }
        }

        fclose(file);
    }

    _min.setValue( Vector3( 0, 0, 0));
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

    Vector3 vsize = loader->getVoxelSize(  );

    _regularGrid->setPos(0,width*vsize[0],0,height*vsize[1],0,depth*vsize[2]);

    _min.setValue( Vector3(0,0,0) );
    _max.setValue( Vector3(width*vsize[0],height*vsize[1],depth*vsize[2]) );

    const int nbCubesRG = _regularGrid->getNbHexahedra();

    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, -1); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)

    vector<float> regularstiffnessCoef(nbCubesRG, 0.0);

    for(int i=0; i<nbCubesRG; ++i)
    {
        const Vec3i& hexacoord = _regularGrid->getCubeCoordinate(i);
        const RegularGridTopology::Hexa& hexa = _regularGrid->getHexahedron( hexacoord[0],hexacoord[1], hexacoord[2] );

        SReal p0x = _regularGrid->getPX( hexa[0] ) / vsize[0];
        SReal p0y = _regularGrid->getPY( hexa[0] ) / vsize[1];
        SReal p0z = _regularGrid->getPZ( hexa[0] ) / vsize[2];
        SReal p6x = _regularGrid->getPX( hexa[6] ) / vsize[0];
        SReal p6y = _regularGrid->getPY( hexa[6] ) / vsize[1];
        SReal p6z = _regularGrid->getPZ( hexa[6] ) / vsize[2];


        for(int x=(int)p0x; x<(int)p6x; ++x)
        {
            for(int y=(int)p0y; y<(int)p6y; ++y)
            {
                for(int z=(int)p0z; z<(int)p6z; ++z)
                {
                    unsigned int idx = x + y * width + z * width * height;

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
    for(int i=0; i<this->getNbHexahedra(); ++i)
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

    _min.setValue( Vector3( 0, 0, 0));
    _max.setValue( voxelSize.getValue().linearProduct(dataResolution.getValue())*(1));

    //Creating if needed collision models and visual models
    // 	    using sofa::simulation::Node;
    sofa::helper::vector< sofa::core::topology::BaseMeshTopology * > list_meshf;
    sofa::helper::vector< Data< sofa::defaulttype::Vec3fTypes::VecCoord >* > list_Xf;

#ifndef SOFA_FLOAT
    sofa::helper::vector< sofa::core::topology::BaseMeshTopology * > list_meshd;
    sofa::helper::vector< Data< sofa::defaulttype::Vec3dTypes::VecCoord >* > list_Xd;
#endif

    //Get Collision Model
    sofa::helper::vector< sofa::core::topology::BaseMeshTopology* > m_temp;
    this->getContext()->get< sofa::core::topology::BaseMeshTopology >(&m_temp, sofa::core::objectmodel::BaseContext::SearchDown);

    sofa::core::topology::BaseMeshTopology* collisionTopology=NULL;
    for (unsigned int i=0; i<m_temp.size(); ++i)
    {
        if (m_temp[i] != this) {collisionTopology = m_temp[i]; break;}
    }

    if ( collisionTopology != NULL && collisionTopology->getNbTriangles() == 0)
    {
#ifndef SOFA_FLOAT
        core::behavior::MechanicalState< sofa::defaulttype::Vec3dTypes > *mecha_tempd =
                collisionTopology->getContext()->get< core::behavior::MechanicalState< sofa::defaulttype::Vec3dTypes > >();
        if (mecha_tempd != NULL && mecha_tempd->getSize() < 2) //a triangle mesh has minimum 3elements
        {
            list_meshd.push_back(collisionTopology);
            list_Xd.push_back(mecha_tempd->write(core::VecCoordId::position()));
        }
#endif
#ifndef SOFA_DOUBLE
        core::behavior::MechanicalState< sofa::defaulttype::Vec3fTypes > *mecha_tempf =
                collisionTopology->getContext()->get< core::behavior::MechanicalState< sofa::defaulttype::Vec3fTypes > >();
        if (mecha_tempf != NULL && mecha_tempf->getSize() < 2) //a triangle mesh has minimum 3elements
        {

            list_meshf.push_back(collisionTopology);
            list_Xf.push_back(mecha_tempf->write(core::VecCoordId::position()));
        }
#endif
    }

    if (
        list_meshf.empty()
#ifndef SOFA_FLOAT
        && list_meshd.empty()
#endif
        )
        return;				 //No Marching Cube to run

    //Configuration of the Marching Cubes algorithm

    marchingCubes.setDataResolution(Vec3i(dataResolution.getValue()[0],
            dataResolution.getValue()[1],
            dataResolution.getValue()[2]));
    marchingCubes.setDataVoxelSize(voxelSize.getValue());
    marchingCubes.setStep(marchingCubeStep.getValue());
    marchingCubes.setConvolutionSize(convolutionSize.getValue()); //apply Smoothing if convolutionSize > 0

    if (! list_meshf.empty())
        constructCollisionModels(list_meshf, list_Xf);
#ifndef SOFA_FLOAT
    else
        constructCollisionModels(list_meshd, list_Xd);
#endif
}


void SparseGridTopology::getMesh(sofa::helper::io::Mesh &m)
{
    if (!dataVoxels.getValue().empty())
    {
        helper::vector<unsigned char> * datas = dataVoxels.beginEdit();
        marchingCubes.run(&(*datas)[0], 0.5f, m);
        dataVoxels.endEdit();
    }
}


template< class T >
void SparseGridTopology::constructCollisionModels(const sofa::helper::vector< sofa::core::topology::BaseMeshTopology * > &list_mesh,
        const helper::vector< Data< helper::vector< Vec<3,T> > >* > &list_X)
{
    sofa::helper::vector< unsigned int>	triangles;
    vector< Vector3 >		vertices;

    helper::vector<unsigned char> * datas = dataVoxels.beginEdit();
    marchingCubes.run(&(*datas)[0], 0.5f, triangles, vertices);
    dataVoxels.endEdit();

    for (unsigned int i=0; i<list_mesh.size(); ++i)
    {
        list_mesh[i]->clear();
        helper::WriteAccessor<Data< helper::vector< Vec<3,T> > > > x(list_X[i]);
        x.resize(vertices.size());
    }

    for (unsigned int j=0; j<list_mesh.size(); ++j)
    {
        helper::WriteAccessor<Data< helper::vector< Vec<3,T> > > > x(list_X[j]);
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


void SparseGridTopology::buildFromTriangleMesh(const std::string& filename)
{
    helper::io::Mesh* mesh = NULL;

    if (filename.empty())
    {
        mesh = new helper::io::Mesh();
        for (unsigned int i=0; i<vertices.getValue().size(); ++i)
            mesh->getVertices().push_back(vertices.getValue()[i]);
        const vector < vector <int> >& facets = this->facets.getValue();
        const SeqTriangles& triangles = this->input_triangles.getValue();
        const SeqQuads& quads = this->input_quads.getValue();
        mesh->getFacets().resize(facets.size() + triangles.size() + quads.size());
        for (size_t i=0; i<facets.size(); ++i)
            mesh->getFacets()[i].push_back(facets[i]);
        for (size_t i0 = facets.size(), i=0; i<triangles.size(); ++i)
        {
            mesh->getFacets()[i0+i].resize(1);
            mesh->getFacets()[i0+i][0].resize(3);
            mesh->getFacets()[i0+i][0][0] = triangles[i][0];
            mesh->getFacets()[i0+i][0][1] = triangles[i][1];
            mesh->getFacets()[i0+i][0][2] = triangles[i][2];
        }
        for (size_t i0 = facets.size()+triangles.size(), i=0; i<quads.size(); ++i)
        {
            mesh->getFacets()[i0+i].resize(1);
            mesh->getFacets()[i0+i][0].resize(4);
            mesh->getFacets()[i0+i][0][0] = quads[i][0];
            mesh->getFacets()[i0+i][0][1] = quads[i][1];
            mesh->getFacets()[i0+i][0][2] = quads[i][2];
            mesh->getFacets()[i0+i][0][3] = quads[i][3];
        }
    }
    else
    {
        mesh = helper::io::Mesh::Create(filename.c_str());
    }

    if(mesh == NULL)
    {
        serr << "SparseGridTopology: loading mesh " << filename << " failed." <<sendl;
        return;
    }

    // if not given sizes -> bounding box
    if( _min.getValue()== Vector3() && _max.getValue()== Vector3())
    {

        SReal xMin, xMax, yMin, yMax, zMin, zMax;
        computeBoundingBox(mesh->getVertices(), xMin, xMax, yMin, yMax, zMin, zMax);

        // increase the box a little
        Vector3 diff ( xMax-xMin, yMax - yMin, zMax - zMin );
        diff /= 100.0;

        _min.setValue(Vector3( xMin - diff[0], yMin - diff[1], zMin - diff[2] ));
        _max.setValue(Vector3( xMax + diff[0], yMax + diff[1], zMax + diff[2] ));
        sout << "BBox size: " << (_max.getValue() - _min.getValue()) << sendl;
    }

    // if cellWidth is given, update n
    if (_cellWidth.getValue())
    {
        SReal w = _cellWidth.getValue();
        Vector3 diff = _max.getValue() - _min.getValue();
        setN(Vec3i((int)ceil(diff[0] / w)+1, (int)ceil(diff[1] / w)+1, (int)ceil(diff[2] / w)+1));
        sout << "Grid size: " << n.getValue() << sendl;
    }

    _regularGrid->setSize(getNx(),getNy(),getNz());
    _regularGrid->setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());

    vector<Type> regularGridTypes; // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
    voxelizeTriangleMesh(mesh, _regularGrid, regularGridTypes);

    buildFromRegularGridTypes(_regularGrid, regularGridTypes);

    sout << "Mesh Loaded" << sendl;

    delete mesh;
}


void SparseGridTopology::voxelizeTriangleMesh(helper::io::Mesh* mesh,
        RegularGridTopology::SPtr regularGrid,
        vector<Type>& regularGridTypes) const
{
    regularGridTypes.resize(regularGrid->getNbHexahedra(), INSIDE);

    //// find all initial mesh edges to compute intersection with cubes
    //const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();
    //std::set< SegmentForIntersection,ltSegmentForIntersection > segmentsForIntersection;
    //for (unsigned int i=0;i<facets.size();i++)
    //{
    //	const helper::vector<int>& facet = facets[i][0];
    //	for (unsigned int j=2; j<facet.size(); j++) // Triangularize
    //	{
    //		segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j]] ) );
    //		segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j-1]] ) );
    //		segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[j]],vertices[facet[j-1]] ) );
    //	}
    //}

    const helper::vector< Vector3 >& vertices = mesh->getVertices();
    const size_t vertexSize = vertices.size();
    helper::vector< int > verticesHexa(vertexSize);
    helper::vector< bool > facetDone;
    const Vector3 delta = (regularGrid->getDx() + regularGrid->getDy() + regularGrid->getDz()) / 2;

    // Compute the grid element for each mesh vertex
    for (size_t i = 0; i < vertexSize; ++i)
    {
        const Vector3& vertex = vertices[i];
        int index = regularGrid->findHexa(vertex);
        if (index > 0)
            regularGridTypes[index] = BOUNDARY;
        if (index == -1)
        {
            Vector3 vertex2 = vertex;
            const Vector3 gmin = regularGrid->getMin();
            const Vector3 gmax = regularGrid->getMax();

            if (vertex2[0] < gmin[0])
                vertex2[0] = gmin[0];
            else if (vertex2[0] > gmax[0])
                vertex2[0] = gmax[0] - delta[0];

            if (vertex2[1] < gmin[1])
                vertex2[1] = gmin[1];
            else if (vertex2[1] > gmax[1])
                vertex2[1] = gmax[1] - delta[1];

            if (vertex2[2] < gmin[2])
                vertex2[2] = gmin[2];
            else if (vertex2[2] > gmax[2])
                vertex2[2] = gmax[2] - delta[2];

            index = regularGrid->findHexa(vertex2);
        }

        verticesHexa[i] = index;
    }

    // For each triangle, compute BBox and test each element in bb if needed
    const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();

    for (unsigned int f=0; f<facets.size(); f++)
    {
        const helper::vector<int>& facet = facets[f][0];
        for (unsigned int j=2; j<facet.size(); j++) // Triangularize
        {
            int c0 = verticesHexa[facet[0]];
            int c1 = verticesHexa[facet[j-1]];
            int c2 = verticesHexa[facet[j]];
            if((c0==c1)&&(c0==c2)&&(c0!=-1)) // All vertices in same box discard now if possible
            {
                if(regularGridTypes[c0]==BOUNDARY)
                    continue;
            }
            // Compute box
            const Vector3 i0 = regularGrid->getCubeCoordinate(c0);
            const Vector3 i1 = regularGrid->getCubeCoordinate(c1);
            const Vector3 i2 = regularGrid->getCubeCoordinate(c2);
            const Vector3 iMin(	std::min(i0[0],std::min(i1[0],i2[0])),
                    std::min(i0[1],std::min(i1[1],i2[1])),
                    std::min(i0[2],std::min(i1[2],i2[2])));
            const Vector3 iMax(	std::max(i0[0],std::max(i1[0],i2[0])),
                    std::max(i0[1],std::max(i1[1],i2[1])),
                    std::max(i0[2],std::max(i1[2],i2[2])));

            for(unsigned int x=(unsigned int)iMin[0]; x<=(unsigned int)iMax[0]; ++x)
            {
                for(unsigned int y=(unsigned int)iMin[1]; y<=(unsigned int)iMax[1]; ++y)
                {
                    for(unsigned int z=(unsigned int)iMin[2]; z<=(unsigned int)iMax[2]; ++z)
                    {
                        // if already inserted discard
                        unsigned int index = regularGrid->getCubeIndex(x,y,z);
                        if(regularGridTypes[index]==BOUNDARY)
                            continue;

                        Hexa c = regularGrid->getHexaCopy(index);
                        CubeCorners corners;
                        for(int k=0; k<8; ++k)
                            corners[k] = regularGrid->getPoint( c[k] );
#ifdef SOFA_NEW_HEXA
                        Vector3 cubeDiagonal = corners[6] - corners[0];
#else
                        Vector3 cubeDiagonal = corners[7] - corners[0];
#endif
                        Vector3 cubeCenter = corners[0] + cubeDiagonal*.5;

                        const Vector3& A = vertices[facet[0]];
                        const Vector3& B = vertices[facet[j-1]];
                        const Vector3& C = vertices[facet[j]];

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
    vector<bool> alreadyTested(regularGrid->getNbHexahedra(),false);
    std::stack< Vec3i > seed;
    // x==0 and x=nx-2
    for(int y=0; y<regularGrid->getNy()-1; ++y)
    {
        for(int z=0; z<regularGrid->getNz()-1; ++z)
        {
            launchPropagationFromSeed(Vec3i(0,y,z), regularGrid, regularGridTypes, alreadyTested,seed );
            launchPropagationFromSeed(Vec3i(regularGrid->getNx()-2,y,z), regularGrid, regularGridTypes, alreadyTested,seed );
        }
        // y==0 and y=ny-2
        for(int x=0; x<regularGrid->getNx()-1; ++x)
        {
            for(int z=0; z<regularGrid->getNz()-1; ++z)
            {
                launchPropagationFromSeed(Vec3i(x,0,z), regularGrid, regularGridTypes, alreadyTested,seed );
                launchPropagationFromSeed(Vec3i(x,regularGrid->getNy()-2,z), regularGrid, regularGridTypes, alreadyTested,seed );
            }
            // z==0 and z==Nz-2
            for(int y=0; y<regularGrid->getNy()-1; ++y)
            {
                for(int x=0; x<regularGrid->getNx()-1; ++x)
                {
                    launchPropagationFromSeed(Vec3i(x,y,0), regularGrid, regularGridTypes, alreadyTested,seed );
                    launchPropagationFromSeed(Vec3i(x,y,regularGrid->getNz()-2), regularGrid, regularGridTypes, alreadyTested,seed );
                }
            }
        }
    }
}


void SparseGridTopology::launchPropagationFromSeed(const Vec3i &point,
        RegularGridTopology::SPtr regularGrid,
        vector<Type>& regularGridTypes,
        vector<bool>& alreadyTested,
        std::stack<Vec3i> &seed) const
{
    seed.push(point);
    while (!seed.empty())
    {
        const Vec3i &s=seed.top();
        seed.pop();
        propagateFrom(s,regularGrid,regularGridTypes,alreadyTested,seed);
    }
}

void SparseGridTopology::buildFromRegularGridTypes(RegularGridTopology::SPtr regularGrid, const vector<Type>& regularGridTypes)
{
    vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values

    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid->getNbHexahedra(), -1 ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    int cubeCntr = 0;

    // add BOUNDARY cubes to valid cells
    for(int w=0; w<regularGrid->getNbHexahedra(); ++w)
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
    for(int w=0; w<regularGrid->getNbHexahedra(); ++w)
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

    helper::vector<defaulttype::Vec<3,SReal> >& seqPoints = *this->seqPoints.beginEdit(); seqPoints.clear();
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


void SparseGridTopology::computeBoundingBox(const helper::vector<Vector3>& vertices,
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


void SparseGridTopology::buildFromFiner(  )
{
    //	serr<<"SparseGridTopology::buildFromFiner(  )"<<sendl;

    setNx( _finerSparseGrid->getNx()/2+1 );
    setNy( _finerSparseGrid->getNy()/2+1 );
    setNz( _finerSparseGrid->getNz()/2+1 );

    _regularGrid->setSize(getNx(),getNy(),getNz());

    setMin(_finerSparseGrid->getMin());
    setMax(_finerSparseGrid->getMax());

    // the cube size of the coarser mesh is twice the cube size of the finer mesh
    // if the finer mesh contains an odd number of cubes in any direction,
    // the coarser mesh will be a half cube size larger in that direction
    Vector3 dx = _finerSparseGrid->_regularGrid->getDx();
    Vector3 dy = _finerSparseGrid->_regularGrid->getDy();
    Vector3 dz = _finerSparseGrid->_regularGrid->getDz();
    setXmax(getXmin() + (getNx()-1) * (SReal)2.0 * dx[0]);
    setYmax(getYmin() + (getNy()-1) * (SReal)2.0 * dy[1]);
    setZmax(getZmin() + (getNz()-1) * (SReal)2.0 * dz[2]);

    _regularGrid->setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());

    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid->getNbHexahedra(), -1 ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values

    for(int i=0; i<getNx()-1; i++)
    {
        for(int j=0; j<getNy()-1; j++)
        {
            for(int k=0; k<getNz()-1; k++)
            {
                int x = 2*i;
                int y = 2*j;
                int z = 2*k;

                fixed_array<int,8> fineIndices;
                for(int idx=0; idx<8; ++idx)
                {
                    const int idxX = x + (idx & 1);
                    const int idxY = y + (idx & 2)/2;
                    const int idxZ = z + (idx & 4)/4;
                    if(idxX < _finerSparseGrid->getNx()-1 && idxY < _finerSparseGrid->getNy()-1 && idxZ < _finerSparseGrid->getNz()-1)
                        fineIndices[idx] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid->cube(idxX,idxY,idxZ) ];
                    else
                        fineIndices[idx] = -1;
                }

                bool inside = true;
                bool outside = true;
                for( int w=0; w<8 && (inside || outside); ++w)
                {
                    if( fineIndices[w] == -1 ) inside=false;
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


                int coarseRegularIndice = _regularGrid->cube( i,j,k );
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
    helper::vector<defaulttype::Vec<3,SReal> >& seqPoints = *this->seqPoints.beginEdit(); seqPoints.clear();
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
    _finerSparseGrid->_inversePointMap.resize(_finerSparseGrid->seqPoints.getValue().size()); _finerSparseGrid->_inversePointMap.fill(-1);
    _pointMap.resize(this->seqPoints.getValue().size()); _pointMap.fill(-1);

    for( unsigned w=0; w<seqHexahedra.getValue().size(); ++w)
    {
        const fixed_array<int, 8>& child = _hierarchicalCubeMap[w];

        helper::vector<int> fineCorners(27);
        fineCorners.fill(-1);
        for(int fineCube=0; fineCube<8; ++fineCube)
        {
            if( child[fineCube] == -1 ) continue;

            const Hexa& cube = _finerSparseGrid->getHexahedron(child[fineCube]);

            for(int vertex=0; vertex<8; ++vertex)
            {
                // 					if( fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=-1 && fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=cube[vertex] )
                // 						serr<<"couille fineCorners"<<sendl;
                fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]=cube[vertex];
            }

        }


        for(int coarseCornerLocalIndice=0; coarseCornerLocalIndice<8; ++coarseCornerLocalIndice)
        {
            for( int fineVertexLocalIndice=0; fineVertexLocalIndice<27; ++fineVertexLocalIndice)
            {
                if( fineCorners[fineVertexLocalIndice] == -1 ) continue; // this fine vertex is not in any fine cube

                int coarseCornerGlobalIndice = seqHexahedra.getValue()[w][coarseCornerLocalIndice];
                int fineVertexGlobalIndice = fineCorners[fineVertexLocalIndice];

                if( WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice] )
                {
                    _hierarchicalPointMap[coarseCornerGlobalIndice][fineVertexGlobalIndice] = WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice];
                    // 						_hierarchicalPointMap[coarseCornerGlobalIndice].push_back( std::pair<int,float>(fineVertexGlobalIndice, WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice]) );

                    _finerSparseGrid->_inverseHierarchicalPointMap[fineVertexGlobalIndice][coarseCornerGlobalIndice] = WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice];

                    if( WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice] == 1.0 )
                    {
                        _finerSparseGrid->_inversePointMap[fineVertexGlobalIndice] = coarseCornerGlobalIndice;
                        // 							serr<<getPX(coarseCornerGlobalIndice)<<" "<<getPY(coarseCornerGlobalIndice)<<" "<<getPZ(coarseCornerGlobalIndice)<<" ----- ";
                        // 							serr<<_finerSparseGrid->getPX(fineVertexGlobalIndice)<<" "<<_finerSparseGrid->getPY(fineVertexGlobalIndice)<<" "<<_finerSparseGrid->getPZ(fineVertexGlobalIndice)<<sendl;
                    }
                }
            }
        }
    }

    for( unsigned i=0; i<_finerSparseGrid->_inversePointMap.size(); ++i)
    {
        if(_finerSparseGrid->_inversePointMap[i] != -1)
            _pointMap[ _finerSparseGrid->_inversePointMap[i] ] = i;
    }

    // 		for(unsigned i=0;i<_finerSparseGrid->seqPoints.size();++i)
    // 		{
    // 			serr<<i<<" : "<<_finerSparseGrid->seqPoints[i]<<sendl;
    // 		}
    //
    // 		for(unsigned i=0;i<_finerSparseGrid->seqHexahedra.getValue().size();++i)
    // 		{
    // 			serr<<i<<" : "<<_finerSparseGrid->seqHexahedra.getValue()[i]<<sendl;
    //
    // 		}




    // // 		afficher la _hierarchicalPointMap
    // 		for(unsigned i=0;i<_hierarchicalPointMap.size();++i)
    // 		{
    // 			serr<<"POINT "<<i<<" "<<seqPoints[i]<<" : "<<_hierarchicalPointMap[i].size()<<" : ";
    // 			for(std::map<int,float>::iterator it = _hierarchicalPointMap[i].begin();it != _hierarchicalPointMap[i].end() ; ++it )
    // 			{
    // 				serr<<(*it).first<<", "<<(*it).second<<" # ";
    // 			}
    // 			serr<<sendl;
    // 		}
    // //
    // // 		// // 		afficher la _inverseHierarchicalPointMap

    // 		serr<<"_inverseHierarchicalPointMap :"<<sendl;
    // 		for(unsigned i=0;i<_finerSparseGrid->_inverseHierarchicalPointMap.size();++i)
    // 		{
    // 			serr<<"POINT "<<i<<" "<<seqPoints[i]<<" : "<<_finerSparseGrid->_inverseHierarchicalPointMap[i].size()<<" : ";
    // 			for(std::map<int,float>::iterator it = _finerSparseGrid->_inverseHierarchicalPointMap[i].begin();it != _finerSparseGrid->_inverseHierarchicalPointMap[i].end() ; ++it )
    // 			{
    // 				serr<<(*it).first<<", "<<(*it).second<<" # ";
    // 			}
    // 			serr<<sendl;
    // 		}

    // 		serr<<"_inversePointMap :"<<sendl;
    // 		for(unsigned i=0;i<_finerSparseGrid->_inversePointMap.size();++i)
    // 		{
    // 			serr<<"POINT "<<i<<" -> "<<_finerSparseGrid->_inversePointMap[i]<<sendl;
    // 		}


    // 		for(int o=0;o<_hierarchicalPointMap.size();++o)
    // 		{
    // 			serr<<o<<" : ";
    // 			for(std::set<int>::iterator it=_hierarchicalPointMap[o].begin();it!=_hierarchicalPointMap[o].end();++it)
    // 				serr<<*it<<" ";
    // 			serr<<sendl;
    // 		}


    // 		serr<<"seqHexahedra.getValue() : "<<seqHexahedra.getValue()<<sendl;
    // 		serr<<"seqPoints : "<<seqPoints<<sendl;



    _finerSparseGrid->_coarserSparseGrid = this;
    _finerSparseGrid->_inverseHierarchicalCubeMap.resize( _finerSparseGrid->seqHexahedra.getValue().size(), -1);
    for( unsigned i=0; i<_hierarchicalCubeMap.size(); ++i)
    {
        for(int w=0; w<8; ++w)
        {
            if(_hierarchicalCubeMap[i][w] != -1)
                _finerSparseGrid->_inverseHierarchicalCubeMap[ _hierarchicalCubeMap[i][w] ] = i;
        }
    }

    // compute stiffness coefficient from children
    _stiffnessCoefs.resize( this->getNbHexahedra() );
    _massCoefs.resize( this->getNbHexahedra() );
    for(int i=0; i<this->getNbHexahedra(); ++i)
    {
        helper::fixed_array<int,8> finerChildren = this->_hierarchicalCubeMap[i];
        unsigned nbchildren = 0;
        for(int w=0; w<8; ++w)
        {
            if( finerChildren[w] != -1 )
            {
                _stiffnessCoefs[i] += _finerSparseGrid->_stiffnessCoefs[finerChildren[w]];
                _massCoefs[i] += _finerSparseGrid->_massCoefs[finerChildren[w]];
                ++nbchildren;
            }
        }
        _stiffnessCoefs[i] /= 8.0;//(float)nbchildren;
        _massCoefs[i] /= 8.0;
    }
}


void SparseGridTopology::buildVirtualFinerLevels()
{
    int nb = _nbVirtualFinerLevels.getValue();

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
        _virtualFinerLevels[0]->vertices.setParent(&this->vertices);
        _virtualFinerLevels[0]->facets.setParent(&this->facets);
        _virtualFinerLevels[0]->input_triangles.setParent(&this->input_triangles);
        _virtualFinerLevels[0]->input_quads.setParent(&this->input_quads);
    }
    else
        _virtualFinerLevels[0]->load(fileTopology.c_str());
    _virtualFinerLevels[0]->_fillWeighted.setValue( _fillWeighted.getValue() );
    _virtualFinerLevels[0]->init();

    sout<<"SparseGridTopology "<<getName()<<" buildVirtualFinerLevels : ";
    sout<<"("<<newnx<<"x"<<newny<<"x"<<newnz<<") -> "<< _virtualFinerLevels[0]->getNbHexahedra() <<" elements , ";

    for(int i=1; i<nb; ++i)
    {
        _virtualFinerLevels[i] = sofa::core::objectmodel::New< SparseGridTopology >(true);
        this->addSlave(_virtualFinerLevels[i]);

        _virtualFinerLevels[i]->setFinerSparseGrid(_virtualFinerLevels[i-1].get());

        _virtualFinerLevels[i]->init();

        sout<<"("<<_virtualFinerLevels[i]->getNx()<<"x"<<_virtualFinerLevels[i]->getNy()<<"x"<<_virtualFinerLevels[i]->getNz()<<") -> "<< _virtualFinerLevels[i]->getNbHexahedra() <<" elements , ";
    }

    sout<<sendl;

    this->setFinerSparseGrid(_virtualFinerLevels[nb-1].get());
}


/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int SparseGridTopology::findCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    int indiceInRegularGrid = _regularGrid->findCube( pos,fx,fy,fz);
    if( indiceInRegularGrid == -1 )
        return -1;
    else
        return _indicesOfRegularCubeInSparseGrid[indiceInRegularGrid];
}


/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int SparseGridTopology::findNearestCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz)
{
    if (seqHexahedra.getValue().size() == 0) return -1;
    int indice = 0;
    float lgmin = 99999999.0f;

    for(unsigned w=0; w<seqHexahedra.getValue().size(); ++w)
    {
        if(!_usingMC && _types[w]!=BOUNDARY )continue;

        const Hexa& c = getHexahedron( w );
        int c0 = c[0];
#ifdef SOFA_NEW_HEXA
        int c7 = c[6];
#else
        int c7 = c[7];
#endif
        Vector3 p0((SReal)getPX(c0), (SReal)getPY(c0), (SReal)getPZ(c0));
        Vector3 p7((SReal)getPX(c7), (SReal)getPY(c7), (SReal)getPZ(c7));

        Vector3 barycenter = (p0+p7) * .5;

        float lg = (float)((pos-barycenter).norm());
        if( lg < lgmin )
        {
            lgmin = lg;
            indice = w;
        }
    }

    const Hexa& c = getHexahedron( indice );
    int c0 = c[0];
#ifdef SOFA_NEW_HEXA
    int c7 = c[6];
#else
    int c7 = c[7];
#endif
    Vector3 p0((SReal)getPX(c0), (SReal)getPY(c0), (SReal)getPZ(c0));
    Vector3 p7((SReal)getPX(c7), (SReal)getPY(c7), (SReal)getPZ(c7));

    Vector3 relativePos = pos-p0;
    Vector3 diagonal = p7 - p0;

    fx = relativePos[0] / diagonal[0];
    fy = relativePos[1] / diagonal[1];
    fz = relativePos[2] / diagonal[2];

    return indice;
}


helper::fixed_array<int,6> SparseGridTopology::findneighboorCubes( int indice )
{
    sout<<"SparseGridTopology::findneighboorCubes : "<<indice<<" -> "<<_indicesOfCubeinRegularGrid[indice]<<sendl;
    sout<<_indicesOfRegularCubeInSparseGrid[ _indicesOfCubeinRegularGrid[indice] ] <<sendl;
    helper::fixed_array<int,6> result;
    Vector3 c = _regularGrid->getCubeCoordinate( _indicesOfCubeinRegularGrid[indice] );
    sout<<c<<sendl;
    result[0] = c[0]<=0 ? -1 : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0]-1,(int)c[1],(int)c[2] )];
    result[1] = c[0]>=getNx()-2 ? -1 : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0]+1,(int)c[1],(int)c[2] )];
    result[2] = c[1]<=0 ? -1 : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1]-1,(int)c[2] )];
    result[3] = c[1]>=getNy()-2 ? -1 : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1]+1,(int)c[2] )];
    result[4] = c[2]<=0 ? -1 : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1],(int)c[2]-1 )];
    result[5] = c[2]>=getNz()-2 ? -1 : _indicesOfRegularCubeInSparseGrid[ _regularGrid->getCubeIndex( (int)c[0],(int)c[1],(int)c[2]+1 )];
    return result;
}


SparseGridTopology::Type SparseGridTopology::getType( int i )
{
    return _types[i];
}


float SparseGridTopology::getStiffnessCoef(int elementIdx)
{
    return _stiffnessCoefs[ elementIdx ];
}


float SparseGridTopology::getMassCoef(int elementIdx)
{
    return _massCoefs[ elementIdx ];
}


void SparseGridTopology::updateEdges()
{
    std::map<pair<int,int>,bool> edgesMap;
    for(unsigned i=0; i<seqHexahedra.getValue().size(); ++i)
    {
        Hexa c = seqHexahedra.getValue()[i];

#ifdef SOFA_NEW_HEXA
        // horizontal
        edgesMap[pair<int,int>(c[0],c[1])]=0;
        edgesMap[pair<int,int>(c[3],c[2])]=0;
        edgesMap[pair<int,int>(c[4],c[5])]=0;
        edgesMap[pair<int,int>(c[7],c[6])]=0;
        // vertical
        edgesMap[pair<int,int>(c[0],c[3])]=0;
        edgesMap[pair<int,int>(c[1],c[2])]=0;
        edgesMap[pair<int,int>(c[4],c[7])]=0;
        edgesMap[pair<int,int>(c[5],c[6])]=0;
        // profondeur
        edgesMap[pair<int,int>(c[0],c[4])]=0;
        edgesMap[pair<int,int>(c[1],c[5])]=0;
        edgesMap[pair<int,int>(c[3],c[7])]=0;
        edgesMap[pair<int,int>(c[2],c[6])]=0;
#else
        // horizontal
        edgesMap[pair<int,int>(c[0],c[1])]=0;
        edgesMap[pair<int,int>(c[3],c[2])]=0;
        edgesMap[pair<int,int>(c[4],c[5])]=0;
        edgesMap[pair<int,int>(c[7],c[6])]=0;
        // vertical
        edgesMap[pair<int,int>(c[0],c[2])]=0;
        edgesMap[pair<int,int>(c[1],c[3])]=0;
        edgesMap[pair<int,int>(c[4],c[6])]=0;
        edgesMap[pair<int,int>(c[5],c[7])]=0;
        // profondeur
        edgesMap[pair<int,int>(c[0],c[4])]=0;
        edgesMap[pair<int,int>(c[1],c[5])]=0;
        edgesMap[pair<int,int>(c[2],c[6])]=0;
        edgesMap[pair<int,int>(c[3],c[7])]=0;
#endif
    }

    SeqEdges& edges = *seqEdges.beginEdit();
    edges.clear();
    edges.reserve(edgesMap.size());
    for( std::map<pair<int,int>,bool>::iterator it=edgesMap.begin(); it!=edgesMap.end(); ++it)
        edges.push_back( Edge( (*it).first.first,  (*it).first.second ));
    seqEdges.endEdit();
}


void SparseGridTopology::updateQuads()
{
    std::map<fixed_array<int,4>,bool> quadsMap;
    for(unsigned i=0; i<seqHexahedra.getValue().size(); ++i)
    {
        Hexa c = seqHexahedra.getValue()[i];
        fixed_array<int,4> v;

#ifdef SOFA_NEW_HEXA
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
#else

        v[0]=c[0]; v[1]=c[1]; v[2]=c[3]; v[3]=c[2];
        quadsMap[v]=0;
        v[0]=c[4]; v[1]=c[5]; v[2]=c[7]; v[3]=c[6];
        quadsMap[v]=0;
        v[0]=c[0]; v[1]=c[1]; v[2]=c[5]; v[3]=c[4];
        quadsMap[v]=0;
        v[0]=c[2]; v[1]=c[3]; v[2]=c[7]; v[3]=c[6];
        quadsMap[v]=0;
        v[0]=c[0]; v[1]=c[4]; v[2]=c[6]; v[3]=c[2];
        quadsMap[v]=0;
        v[0]=c[1]; v[1]=c[5]; v[2]=c[7]; v[3]=c[3];
        quadsMap[v]=0;
#endif
    }
    SeqQuads& quads = *seqQuads.beginEdit();
    quads.clear();
    quads.reserve(quadsMap.size());
    for( std::map<fixed_array<int,4>,bool>::iterator it=quadsMap.begin(); it!=quadsMap.end(); ++it)
        quads.push_back( Quad( (*it).first[0],  (*it).first[1],(*it).first[2],(*it).first[3] ));
    seqQuads.endEdit();
}


void SparseGridTopology::updateHexahedra()
{
    //	seqHexahedra.getValue().clear();
    //	seqHexahedra.getValue().reserve(_cubes.size());
    //	seqHexahedra.getValue().push_back(Hexa(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
    //		point(x  ,y+1,z  ),point(x+1,y+1,z  ),
    //		point(x  ,y  ,z+1),point(x+1,y  ,z+1),
    //		point(x  ,y+1,z+1),point(x+1,y+1,z+1)));
}


//bool SparseGridTopology::intersectionSegmentBox( const SegmentForIntersection& seg, const CubeForIntersection& cube  )
//{
//	Vector3 afAWdU, afADdU, afAWxDdU;
//	SReal fRhs;
//
//
//	Vector3 kDiff = seg.center - cube.center;
//
//	afAWdU[0] = fabs( seg.dir* cube.dir[0]);
//	afADdU[0] = fabs( kDiff * cube.dir[0] );
//	fRhs = cube.norm[0] + seg.norm*afAWdU[0];
//	if (afADdU[0] > fRhs)
//	{
//		return false;
//	}
//
//	afAWdU[1] = fabs(seg.dir*cube.dir[1]);
//	afADdU[1] = fabs(kDiff*cube.dir[1]);
//	fRhs = cube.norm[1] + seg.norm*afAWdU[1];
//	if (afADdU[1] > fRhs)
//	{
//		return false;
//	}
//
//	afAWdU[2] = fabs(seg.dir*cube.dir[2]);
//	afADdU[2] = fabs(kDiff*cube.dir[2]);
//	fRhs = cube.norm[2] + seg.norm*afAWdU[2];
//	if (afADdU[2] > fRhs)
//	{
//		return false;
//	}
//
//	Vector3 kWxD = seg.dir.cross(kDiff);
//
//	afAWxDdU[0] = fabs(kWxD*cube.dir[0]);
//	fRhs = cube.norm[1]*afAWdU[2] + cube.norm[2]*afAWdU[1];
//	if (afAWxDdU[0] > fRhs)
//	{
//		return false;
//	}
//
//	afAWxDdU[1] = fabs(kWxD*cube.dir[1]);
//	fRhs = cube.norm[0]*afAWdU[2] + cube.norm[2]*afAWdU[0];
//	if (afAWxDdU[1] > fRhs)
//	{
//		return false;
//	}
//
//	afAWxDdU[2] = fabs(kWxD*cube.dir[2]);
//	fRhs = cube.norm[0]*afAWdU[1] +cube.norm[1]*afAWdU[0];
//	if (afAWxDdU[2] > fRhs)
//	{
//		return false;
//	}
//
//	return true;
//}


void SparseGridTopology::propagateFrom( const Vec3i &point,
        RegularGridTopology::SPtr regularGrid,
        vector<Type>& regularGridTypes,
        vector<bool>& alreadyTested,
        std::stack< Vec<3,int> > &seed) const
{
    const int x=point[0];
    const int y=point[1];
    const int z=point[2];
    assert( x>=0 && x<=regularGrid->getNx()-2 && y>=0 && y<=regularGrid->getNy()-2 && z>=0 && z<=regularGrid->getNz()-2 );

    unsigned indice = regularGrid->cube( x, y, z );
    if( alreadyTested[indice] || regularGridTypes[indice] == BOUNDARY ) return;

    alreadyTested[indice] = true;
    regularGridTypes[indice] = OUTSIDE;

    if(x>0)                     seed.push(Vec3i(x-1,y,z));
    if(x<regularGrid->getNx()-2) seed.push(Vec3i(x+1,y,z));

    if(y>0)                     seed.push(Vec3i(x,y-1,z));
    if(y<regularGrid->getNy()-2) seed.push(Vec3i(x,y+1,z));

    if(z>0)                     seed.push(Vec3i(x,y,z-1));
    if(z<regularGrid->getNz()-2) seed.push(Vec3i(x,y,z+1));
}

} // namespace topology

} // namespace component

} // namespace sofa
