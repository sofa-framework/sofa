/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <fstream>
#include <string>

#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/topology/SparseGridTopology.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/polygon_cube_intersection/polygon_cube_intersection.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/defaulttype/VecTypes.h>


using std::cerr;
using std::endl;
using std::pair;

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


SparseGridTopology::SparseGridTopology()
    :
    n(initData(&n,Vec<3,int>(2,2,2),"n","grid resolution")),
    min(initData(&min,Vec3d(0,0,0),"min","Min")),
    max(initData(&max,Vec3d(0,0,0),"max","Max")),
    dim_voxels(initData(&dim_voxels,Vec<3,int>(512,512,246),"dim_voxels","Dimension of the voxel File")),
    size_voxel(initData(&size_voxel,Vec3f(1.0f,1.0f,1.0f),"size_voxel","Dimension of one voxel")),
    resolution(initData(&resolution, (unsigned int) 128, "resolution", "Resolution of the Marching Cube")),
    smoothData(initData(&smoothData, (unsigned int) 0, "smoothData", "Dimension of the convolution kernel to smooth the voxels. 0 if no smoothing is required."))
{
    _alreadyInit = false;
    _finerSparseGrid = NULL;
    _coarserSparseGrid = NULL;
    _usingMC = false;
}


bool SparseGridTopology::load(const char* filename)
{
    std::string f(filename);
    if ( sofa::helper::system::DataRepository.findFile ( f ) )
        f = sofa::helper::system::DataRepository.getFile ( f );

    this->filename.setValue( f );
// 		cerr<<"SparseGridTopology::load : "<<filename<<"    "<<this->filename.getValue()<<endl;
    return true;
}



void SparseGridTopology::init()
{
    if(_alreadyInit) return;
    _alreadyInit = true;

    this->MeshTopology::init();
    invalidate();

    Vec<3,int> grid = n.getValue();

    if(grid[0] < 2) grid[0]= 2;
    if(grid[1] < 2) grid[1]= 2;
    if(grid[2] < 2) grid[2]= 2;

    n.setValue(grid);

    if( _finerSparseGrid != NULL )
        buildFromFiner();
    else
        buildAsFinest();

    _nodeAdjacency.resize(seqPoints.size() );
    for(unsigned i=0; i<seqPoints.size(); ++i)
        _nodeAdjacency[i].assign(-1);

    for(unsigned i=0; i<seqHexas.size(); ++i)
    {
        _nodeAdjacency[ seqHexas[i][0] ][RIGHT] = seqHexas[i][1];
        _nodeAdjacency[ seqHexas[i][0] ][UP] = seqHexas[i][2];
        _nodeAdjacency[ seqHexas[i][0] ][BEHIND] = seqHexas[i][4];

        _nodeAdjacency[ seqHexas[i][1] ][LEFT] = seqHexas[i][0];
        _nodeAdjacency[ seqHexas[i][1] ][UP] = seqHexas[i][3];
        _nodeAdjacency[ seqHexas[i][1] ][BEHIND] = seqHexas[i][5];

        _nodeAdjacency[ seqHexas[i][2] ][RIGHT] = seqHexas[i][3];
        _nodeAdjacency[ seqHexas[i][2] ][DOWN] = seqHexas[i][0];
        _nodeAdjacency[ seqHexas[i][2] ][BEHIND] = seqHexas[i][6];

        _nodeAdjacency[ seqHexas[i][3] ][LEFT] = seqHexas[i][2];
        _nodeAdjacency[ seqHexas[i][3] ][DOWN] = seqHexas[i][1];
        _nodeAdjacency[ seqHexas[i][3] ][BEHIND] = seqHexas[i][7];

        _nodeAdjacency[ seqHexas[i][4] ][RIGHT] = seqHexas[i][5];
        _nodeAdjacency[ seqHexas[i][4] ][UP] = seqHexas[i][6];
        _nodeAdjacency[ seqHexas[i][4] ][BEFORE] = seqHexas[i][0];

        _nodeAdjacency[ seqHexas[i][5] ][LEFT] = seqHexas[i][4];
        _nodeAdjacency[ seqHexas[i][5] ][UP] = seqHexas[i][7];
        _nodeAdjacency[ seqHexas[i][5] ][BEFORE] = seqHexas[i][1];

        _nodeAdjacency[ seqHexas[i][6] ][RIGHT] = seqHexas[i][7];
        _nodeAdjacency[ seqHexas[i][6] ][DOWN] = seqHexas[i][4];
        _nodeAdjacency[ seqHexas[i][6] ][BEFORE] = seqHexas[i][2];

        _nodeAdjacency[ seqHexas[i][7] ][LEFT] = seqHexas[i][6];
        _nodeAdjacency[ seqHexas[i][7] ][DOWN] = seqHexas[i][5];
        _nodeAdjacency[ seqHexas[i][7] ][BEFORE] = seqHexas[i][3];
    }


// 		  _nodeCubesAdjacency.clear();
    _nodeCubesAdjacency.resize(seqPoints.size() );
    for(unsigned i=0; i<seqHexas.size(); ++i)
    {
        for(int j=0; j<8; ++j)
        {
            _nodeCubesAdjacency[ seqHexas[i][j] ].push_back( i );
        }
    }

// 		  cerr<<"_nodeCubesAdjacency :"<<_nodeCubesAdjacency<<endl;

// 		  cerr<<"SparseGridTopology::init() :   "<<this->getName()<<"    cubes size = ";
// 		  cerr<<seqHexas.size()<<"       ";
// 		  cerr<<_types.size()<<endl;

}



void SparseGridTopology::buildAsFinest(  )
{
    // 		  cerr<<"SparseGridTopology::buildAsFinest(  )\n";
    const  std::string&	_filename(filename.getValue());
    if (_filename.empty())
    {
        std::cerr << "SparseGridTopology: no filename specified." << std::endl;
        return;
    }

    // initialize the following datafields:
    // xmin, xmax, ymin, ymax, zmin, zmax, evtl. nx, ny, nz
    // _regularGrid, _indicesOfRegularCubeInSparseGrid, _types
    // seqPoints, seqHexas, nbPoints
    if(_filename.length() > 4 && _filename.compare(_filename.length()-4, 4, ".obj")==0)
    {
        //			std::cout << "SparseGridTopology: using mesh "<<_filename<<std::endl;
        buildFromTriangleMesh(_filename);
    }
    else if(_filename.length() > 6 && _filename.compare(_filename.length()-6, 6, ".trian")==0)
    {
        //			std::cout << "SparseGridTopology: using mesh "<<_filename<<std::endl;
        buildFromTriangleMesh(_filename);
    }
    else if(_filename.length() > 4 && _filename.compare(_filename.length()-4, 4, ".raw")==0)
    {
        //			std::cout << "SparseGridTopology: using mesh "<<_filename<<std::endl;
        _usingMC = true;
        buildFromRawVoxelFile(_filename);
    }
    else if(_filename.length() > 6 && _filename.compare(_filename.length()-6, 6, ".voxel")==0)
    {
        buildFromVoxelFile(_filename);
    }
}

void SparseGridTopology::buildFromVoxelFile(const std::string& filename)
{
    std::ifstream file( filename.c_str() );
    if ( file == NULL )
    {
        std::cerr << "SparseGridTopology: failed to open file " << filename << std::endl;
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

    n.setValue(Vec<3,int>(nx,ny,nz));
    min.setValue(Vec3d(xmin, ymin, zmin));
    max.setValue(Vec3d(xmax, ymax, zmax));

    _regularGrid.setSize(nx, ny, nz);
    _regularGrid.setPos(xmin, xmax, ymin, ymax, zmin, zmax);


    int value;
    int numVoxels = 0;
    dataVoxels.resize(fileNx * fileNy * fileNz, 0.0f);
    for(int z=0; z<fileNz; ++z)
        for(int y=0; y<fileNy; ++y)
            for(int x=0; x<fileNx; ++x)
            {
                file >> value;
                if (value != 0)
                {
                    dataVoxels[x + fileNx * y + fileNx * fileNy * z] = (float) value;
                    numVoxels++;
                }
            }

    file.close();

    int nbCubesRG = _regularGrid.getNbHexas();
    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, -1); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
    // fill the regularGridTypes vector
    // at the moment, no BOUNDARY type voxels are generated at the finest level
    for(int i=0; i<nbCubesRG; ++i)
    {
        if(dataVoxels[i] != 0.0f)
            regularGridTypes[i] = INSIDE;
    }

    buildFromRegularGridTypes(_regularGrid, regularGridTypes);
}



//Building from a RAW file
void SparseGridTopology::buildFromRawVoxelFile(const std::string& filename)
{
    FILE *file = fopen( filename.c_str(), "r" );
    unsigned char value;

    _regularGrid.setSize(getNx(),getNy(),getNz());
    const int nbCubesRG = _regularGrid.getNbHexas();

    _indicesOfRegularCubeInSparseGrid.resize(nbCubesRG, -1); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    vector<Type> regularGridTypes(nbCubesRG, OUTSIDE); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)


    Vec3d voxel_min, voxel_max;
    const Vec3d transform(                (getNx()-1)/(float)dim_voxels.getValue()[0],
            (getNy()-1)/(float)dim_voxels.getValue()[1],
            (getNz()-1)/(float)dim_voxels.getValue()[2]);

    //Get the voxels from the file
    dataVoxels.resize(dim_voxels.getValue()[0]*dim_voxels.getValue()[1]*dim_voxels.getValue()[2], 0.0f);

    for (unsigned int i=0; i<dataVoxels.size(); i++)
    {
        value=getc(file);
        if ((int)value != 0)
        {
            dataVoxels[i] = 1.0f;

            const int z = i/(dim_voxels.getValue()[0]*dim_voxels.getValue()[1]);
            const int y = (  (i%(dim_voxels.getValue()[0]*dim_voxels.getValue()[1]))
                    /(dim_voxels.getValue()[0])                  	    );
            const int x = i%dim_voxels.getValue()[0];

            unsigned int indexGrid =
                (unsigned int)(z*transform[2])*(getNy()-1)*(getNx()-1)
                + (unsigned int)(y*transform[1])*(getNx()-1)
                + (unsigned int)(x*transform[0]);

            regularGridTypes[indexGrid] = INSIDE;

        }
    }
    fclose(file);

    double s = std::max(dim_voxels.getValue()[0],std::max(dim_voxels.getValue()[1],dim_voxels.getValue()[2]));
    min.setValue(size_voxel.getValue()*s*(-0.5)); //Centered on 0
    max.setValue(size_voxel.getValue()*s*0.5);

    _regularGrid.setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());
    buildFromRegularGridTypes(_regularGrid, regularGridTypes);
    updateMesh();
}

void SparseGridTopology::updateMesh()
{
    if (!_usingMC || dataVoxels.size() == 0) return;


    double s = std::max(dim_voxels.getValue()[0],std::max(dim_voxels.getValue()[1],dim_voxels.getValue()[2]));
    min.setValue(size_voxel.getValue()*s*(-0.5));
    max.setValue(size_voxel.getValue()*s*0.5);

    //Creating if needed collision models and visual models
    using sofa::simulation::tree::GNode;

    sofa::helper::vector< sofa::component::topology::MeshTopology * > list_mesh;
    sofa::helper::vector< sofa::helper::vector< Vec3d >* > list_X;

    const GNode *context = static_cast< GNode* >(this->getContext());

    for (GNode::ChildIterator it= context->child.begin(); it != context->child.end(); ++it)
    {
        //Get Collision Model
        sofa::component::topology::MeshTopology  *m_temp = (*it)->get< sofa::component::topology::MeshTopology >();
        if (    m_temp != NULL
                && m_temp != this
                && m_temp->getFilename() == "")
        {
            MechanicalObject< Vec3Types > *mecha_temp = static_cast< GNode *>(m_temp->getContext())->get< MechanicalObject< Vec3Types > >();
            if (mecha_temp != NULL)
            {
                list_mesh.push_back(m_temp);
                list_X.push_back(mecha_temp->getX());
            }
        }
    }
    mesh_MC.clear();
    map_indices.clear();
    MC.setSize(Vec<3,int>(dim_voxels.getValue()[0],dim_voxels.getValue()[1],dim_voxels.getValue()[2]));

    //Cubic voxels
    MC.setGridSize(Vec<3,int>(      resolution.getValue()*dim_voxels.getValue()[0]/s,
            (int)(resolution.getValue()*dim_voxels.getValue()[1]/s),
            (int)(resolution.getValue()*dim_voxels.getValue()[2]/s)));

    MC.RenderMarchCube(&dataVoxels[0], 0.25f,mesh_MC, map_indices, size_voxel.getValue(), smoothData.getValue()); //Apply Smoothing

    constructCollisionModels(list_mesh, list_X, mesh_MC, map_indices);

}

void SparseGridTopology::getMesh(sofa::helper::io::Mesh &m)
{
    MC.createMesh(mesh_MC, map_indices, m);
}

void SparseGridTopology::constructCollisionModels(const sofa::helper::vector< sofa::component::topology::MeshTopology * > &list_mesh,
        const sofa::helper::vector< sofa::helper::vector< Vec3d >* >            &list_X,
        const sofa::helper::vector< unsigned int> mesh_MC,
        std::map< unsigned int, Vec3f >     map_indices) const
{
    for (unsigned int i=0; i<list_mesh.size(); ++i)
    {
        list_mesh[i]->clear();
        std::cout << list_mesh[i]->getName() << " Name Topology \n";
        list_X[i]->resize(map_indices.size());
    }
    //Fill the dofs : WARNING mesh from Marching Cube has indices starting from ID 1, a sofa mesh begins with ID 0
    for (unsigned int id_point=0; id_point<map_indices.size(); ++id_point)
    {
        const Vec3f point(map_indices[id_point+1]);
        for (unsigned int j=0; j<list_mesh.size(); ++j)
        {
            (*list_X[j])[id_point] = point;
        }
    }

    //Fill the facets
    for (unsigned int id_vertex=0; id_vertex<mesh_MC.size(); id_vertex+=3)
    {
        for (unsigned int j=0; j<list_mesh.size(); ++j)
        {
            list_mesh[j]->addTriangle(mesh_MC[id_vertex]-1,mesh_MC[id_vertex+1]-1, mesh_MC[id_vertex+2]-1);
        }
    }
}


void SparseGridTopology::buildFromTriangleMesh(const std::string& filename)
{
    helper::io::Mesh* mesh = helper::io::Mesh::Create(filename.c_str());

    if(mesh == NULL)
    {
        std::cerr << "SparseGridTopology: loading mesh " << filename << " failed." <<std::endl;
        return;
    }

    // if not given sizes -> bounding box
    if( min.getValue()== Vec3d() && max.getValue()== Vec3d())
    {
        double xMin, xMax, yMin, yMax, zMin, zMax;
        computeBoundingBox(mesh->getVertices(), xMin, xMax, yMin, yMax, zMin, zMax);

        // increase the box a little
        Vec3 diff ( xMax-xMin, yMax - yMin, zMax - zMin );
        diff /= 100.0;
        min.setValue(Vec3d( xMin - diff[0], yMin - diff[1], zMin - diff[2] ));
        max.setValue(Vec3d( xMax + diff[0], yMax + diff[1], zMax + diff[2] ));
    }

    _regularGrid.setSize(getNx(),getNy(),getNz());
    _regularGrid.setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());

    vector<Type> regularGridTypes; // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
    voxelizeTriangleMesh(mesh, _regularGrid, regularGridTypes);

    buildFromRegularGridTypes(_regularGrid, regularGridTypes);

    delete mesh;
}

void SparseGridTopology::voxelizeTriangleMesh(helper::io::Mesh* mesh,
        RegularGridTopology& regularGrid,
        vector<Type>& regularGridTypes) const
{
    regularGridTypes.resize(regularGrid.getNbHexas(), INSIDE);

    // 			// find all initial mesh edges to compute intersection with cubes
    //             const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();
    //             std::set< SegmentForIntersection,ltSegmentForIntersection > segmentsForIntersection;
    //             for (unsigned int i=0;i<facets.size();i++)
    //             {
    //               const helper::vector<int>& facet = facets[i][0];
    //               for (unsigned int j=2; j<facet.size(); j++) // Triangularize
    //               {
    //                 segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j]] ) );
    //                 segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j-1]] ) );
    //                 segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[j]],vertices[facet[j-1]] ) );
    //               }
    //             }


    for(int i=0; i<regularGrid.getNbHexas(); ++i) // all possible cubes (even empty)
    {
        Hexa c = regularGrid.getHexaCopy(i);
        CubeCorners corners;
        for(int j=0; j<8; ++j)
            corners[j] = regularGrid.getPoint( c[j] );

        //               CubeForIntersection cubeForIntersection( corners );
        //
        //               for(std::set< SegmentForIntersection,ltSegmentForIntersection >::iterator it=segmentsForIntersection.begin();it!=segmentsForIntersection.end();++it)
        //                 {
        //                   if(intersectionSegmentBox( *it, cubeForIntersection ))
        //                   {
        //                     _types.push_back(BOUNDARY);
        // 					regularGridTypes[i]=BOUNDARY;
        //
        //                     for(int k=0;k<8;++k)
        //                       cubeCornerPositionIndiceMap[corners[k]] = 0;
        //
        // 					cubeCorners.push_back(corners);
        // 					_indicesOfRegularCubeInSparseGrid[i] = cubeCorners.size()-1;
        //
        //                     break;
        //                   }
        //                 }

        Vec3 cubeDiagonal = corners[7] - corners[0];
        Vec3 cubeCenter = corners[0] + cubeDiagonal*.5;


        bool notAlreadyIntersected = true;

        const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();
        const helper::vector<Vec3>& vertices = mesh->getVertices();
        for (unsigned int f=0; f<facets.size() && notAlreadyIntersected; f++)
        {
            const helper::vector<int>& facet = facets[f][0];
            for (unsigned int j=2; j<facet.size()&& notAlreadyIntersected; j++) // Triangularize
            {
                const Vec3& A = vertices[facet[0]];
                const Vec3& B = vertices[facet[j-1]];
                const Vec3& C = vertices[facet[j]];

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
                    regularGridTypes[i]=BOUNDARY;
                    notAlreadyIntersected=false;
                }
            }
        }
    }


    // TODO: regarder les cellules pleines, et les ajouter

    vector<bool> alreadyTested(regularGrid.getNbHexas());
    for(int w=0; w<regularGrid.getNbHexas(); ++w)
        alreadyTested[w]=false;

    // x==0 and x=nx-2
    for(int y=0; y<regularGrid.getNy()-1; ++y)
        for(int z=0; z<regularGrid.getNz()-1; ++z)
        {
            propagateFrom( 0, y, z, regularGrid, regularGridTypes, alreadyTested );
            propagateFrom( regularGrid.getNx()-2, y, z, regularGrid, regularGridTypes, alreadyTested );
        }
    // y==0 and y=ny-2
    for(int x=0; x<regularGrid.getNx()-1; ++x)
        for(int z=0; z<regularGrid.getNz()-1; ++z)
        {
            propagateFrom( x, 0, z, regularGrid, regularGridTypes, alreadyTested );
            propagateFrom( x, regularGrid.getNy()-2, z, regularGrid, regularGridTypes, alreadyTested );
        }
    // z==0 and z==Nz-2
    for(int y=0; y<regularGrid.getNy()-1; ++y)
        for(int x=0; x<regularGrid.getNx()-1; ++x)
        {
            propagateFrom( x, y, 0, regularGrid, regularGridTypes, alreadyTested );
            propagateFrom( x, y, regularGrid.getNz()-2, regularGrid, regularGridTypes, alreadyTested );
        }
}

void SparseGridTopology::buildFromRegularGridTypes(RegularGridTopology& regularGrid, const vector<Type>& regularGridTypes)
{
    vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values

    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid.getNbHexas(), -1 ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    int cubeCntr = 0;
    // add BOUNDARY cubes to valid cells
    for(int w=0; w<regularGrid.getNbHexas(); ++w)
        if( regularGridTypes[w] == BOUNDARY )
        {
            _types.push_back(BOUNDARY);
            _indicesOfRegularCubeInSparseGrid[w] = cubeCntr++;

            Hexa c = regularGrid.getHexaCopy(w);
            CubeCorners corners;
            for(int j=0; j<8; ++j)
            {
                corners[j] = regularGrid.getPoint( c[j] );
                cubeCornerPositionIndiceMap[corners[j]] = 0;
            }
            cubeCorners.push_back(corners);
        }

    // add INSIDE cubes to valid cells
    for(int w=0; w<regularGrid.getNbHexas(); ++w)
        if( regularGridTypes[w] == INSIDE )
        {
            _types.push_back(INSIDE);
            _indicesOfRegularCubeInSparseGrid[w] = cubeCntr++;

            Hexa c = regularGrid.getHexaCopy(w);
            CubeCorners corners;
            for(int j=0; j<8; ++j)
            {
                corners[j] = regularGrid.getPoint( c[j] );
                cubeCornerPositionIndiceMap[corners[j]] = 0;
            }
            cubeCorners.push_back(corners);
        }

    // compute corner indices
    int cornerCounter=0;
    for(MapBetweenCornerPositionAndIndice::iterator it = cubeCornerPositionIndiceMap.begin();
        it!=cubeCornerPositionIndiceMap.end(); ++it, ++cornerCounter)
    {
        (*it).second = cornerCounter;
        seqPoints.push_back( (*it).first );
    }
    nbPoints = cubeCornerPositionIndiceMap.size();

    for( unsigned w=0; w<cubeCorners.size(); ++w)
    {
        Hexa c;
        for(int j=0; j<8; ++j)
            c[j] = cubeCornerPositionIndiceMap[cubeCorners[w][j]];

        seqHexas.push_back(c);
    }
}

void SparseGridTopology::computeBoundingBox(const helper::vector<Vec3>& vertices,
        double& xmin, double& xmax,
        double& ymin, double& ymax,
        double& zmin, double& zmax) const
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
// 		cerr<<"SparseGridTopology::buildFromFiner(  )\n";


    setNx( _finerSparseGrid->getNx()/2+1 );
    setNy( _finerSparseGrid->getNy()/2+1 );
    setNz( _finerSparseGrid->getNz()/2+1 );

    _regularGrid.setSize(getNx(),getNy(),getNz());

    setMin(_finerSparseGrid->getMin());

    // the cube size of the coarser mesh is twice the cube size of the finer mesh
    // if the finer mesh contains an odd number of cubes in any direction,
    // the coarser mesh will be a half cube size larger in that direction
    Vec3 dx = _finerSparseGrid->_regularGrid.getDx();
    Vec3 dy = _finerSparseGrid->_regularGrid.getDy();
    Vec3 dz = _finerSparseGrid->_regularGrid.getDz();
    setXmax(getXmin() + (getNx()-1) * 2.0 * dx[0]);
    setYmax(getYmin() + (getNy()-1) * 2.0 * dy[1]);
    setZmax(getZmin() + (getNz()-1) * 2.0 * dz[2]);


    _regularGrid.setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());

    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid.getNbHexas(), -1 ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values


    for(int i=0; i<getNx()-1; i++)
        for(int j=0; j<getNy()-1; j++)
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
                        fineIndices[idx] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube(idxX,idxY,idxZ) ];
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
                if( inside ) _types.push_back(INSIDE);
                else _types.push_back(BOUNDARY);


                int coarseRegularIndice = _regularGrid.cube( i,j,k );
                Hexa c = _regularGrid.getHexaCopy( coarseRegularIndice );

                CubeCorners corners;
                for(int w=0; w<8; ++w)
                {
                    corners[w] = _regularGrid.getPoint( c[w] );
                    cubeCornerPositionIndiceMap[corners[w]] = 0;
                }

                cubeCorners.push_back(corners);

                _indicesOfRegularCubeInSparseGrid[coarseRegularIndice] = cubeCorners.size()-1;

// 			_hierarchicalCubeMap[cubeCorners.size()-1]=fineIndices;
                _hierarchicalCubeMap.push_back( fineIndices );
            }


    // compute corner indices
    int cornerCounter=0;
    for(MapBetweenCornerPositionAndIndice::iterator it=cubeCornerPositionIndiceMap.begin(); it!=cubeCornerPositionIndiceMap.end(); ++it,++cornerCounter)
    {
        (*it).second = cornerCounter;
        seqPoints.push_back( (*it).first );
    }
    nbPoints = cubeCornerPositionIndiceMap.size();

    for( unsigned w=0; w<cubeCorners.size(); ++w)
    {
        Hexa c;
        for(int j=0; j<8; ++j)
            c[j] = cubeCornerPositionIndiceMap[cubeCorners[w][j]];

        seqHexas.push_back(c);
    }


    // for interpolation and restriction
    _hierarchicalPointMap.resize(seqPoints.size());
    _finerSparseGrid->_inverseHierarchicalPointMap.resize(_finerSparseGrid->seqPoints.size());
    _finerSparseGrid->_inversePointMap.resize(_finerSparseGrid->seqPoints.size()); _finerSparseGrid->_inversePointMap.fill(-1);
    _pointMap.resize(seqPoints.size()); _pointMap.fill(-1);
    for( unsigned w=0; w<seqHexas.size(); ++w)
    {
        const fixed_array<int, 8>& child = _hierarchicalCubeMap[w];



        helper::vector<int> fineCorners(27);
        fineCorners.fill(-1);
        for(int fineCube=0; fineCube<8; ++fineCube)
        {
            if( child[fineCube] == -1 ) continue;

            const Hexa& cube = _finerSparseGrid->getHexa(child[fineCube]);

            for(int vertex=0; vertex<8; ++vertex)
            {
// 					if( fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=-1 && fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=cube[vertex] )
// 						cerr<<"couille fineCorners\n";
                fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]=cube[vertex];
            }

        }


        for(int coarseCornerLocalIndice=0; coarseCornerLocalIndice<8; ++coarseCornerLocalIndice)
        {
            for( int fineVertexLocalIndice=0; fineVertexLocalIndice<27; ++fineVertexLocalIndice)
            {
                if( fineCorners[fineVertexLocalIndice] == -1 ) continue; // this fine vertex is not in any fine cube

                int coarseCornerGlobalIndice = seqHexas[w][coarseCornerLocalIndice];
                int fineVertexGlobalIndice = fineCorners[fineVertexLocalIndice];

                if( WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice] )
                {
                    _hierarchicalPointMap[coarseCornerGlobalIndice][fineVertexGlobalIndice] = WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice];
// 						_hierarchicalPointMap[coarseCornerGlobalIndice].push_back( std::pair<int,float>(fineVertexGlobalIndice, WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice]) );


                    _finerSparseGrid->_inverseHierarchicalPointMap[fineVertexGlobalIndice][coarseCornerGlobalIndice] = WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice];

                    if( WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice] == 1.0 )
                    {
                        _finerSparseGrid->_inversePointMap[fineVertexGlobalIndice] = coarseCornerGlobalIndice;
// 							cerr<<getPX(coarseCornerGlobalIndice)<<" "<<getPY(coarseCornerGlobalIndice)<<" "<<getPZ(coarseCornerGlobalIndice)<<" ----- ";
// 							cerr<<_finerSparseGrid->getPX(fineVertexGlobalIndice)<<" "<<_finerSparseGrid->getPY(fineVertexGlobalIndice)<<" "<<_finerSparseGrid->getPZ(fineVertexGlobalIndice)<<endl;
                    }
                }
            }
        }

    }


    for( unsigned i=0; i<_finerSparseGrid->_inversePointMap.size(); ++i)
        _pointMap[ _finerSparseGrid->_inversePointMap[i] ]=i;

// 		for(unsigned i=0;i<_finerSparseGrid->seqPoints.size();++i)
// 		{
// 			cerr<<i<<" : "<<_finerSparseGrid->seqPoints[i]<<endl;
// 		}
//
// 		for(unsigned i=0;i<_finerSparseGrid->seqHexas.size();++i)
// 		{
// 			cerr<<i<<" : "<<_finerSparseGrid->seqHexas[i]<<endl;
//
// 		}




// // 		afficher la _hierarchicalPointMap
// 		for(unsigned i=0;i<_hierarchicalPointMap.size();++i)
// 		{
// 			cerr<<"POINT "<<i<<" "<<seqPoints[i]<<" : "<<_hierarchicalPointMap[i].size()<<" : ";
// 			for(std::map<int,float>::iterator it = _hierarchicalPointMap[i].begin();it != _hierarchicalPointMap[i].end() ; ++it )
// 			{
// 				cerr<<(*it).first<<", "<<(*it).second<<" # ";
// 			}
// 			cerr<<endl;
// 		}
// //
// // 		// // 		afficher la _inverseHierarchicalPointMap

// 		cerr<<"_inverseHierarchicalPointMap :"<<endl;
// 		for(unsigned i=0;i<_finerSparseGrid->_inverseHierarchicalPointMap.size();++i)
// 		{
// 			cerr<<"POINT "<<i<<" "<<seqPoints[i]<<" : "<<_finerSparseGrid->_inverseHierarchicalPointMap[i].size()<<" : ";
// 			for(std::map<int,float>::iterator it = _finerSparseGrid->_inverseHierarchicalPointMap[i].begin();it != _finerSparseGrid->_inverseHierarchicalPointMap[i].end() ; ++it )
// 			{
// 				cerr<<(*it).first<<", "<<(*it).second<<" # ";
// 			}
// 			cerr<<endl;
// 		}

// 		cerr<<"_inversePointMap :"<<endl;
// 		for(unsigned i=0;i<_finerSparseGrid->_inversePointMap.size();++i)
// 		{
// 			cerr<<"POINT "<<i<<" -> "<<_finerSparseGrid->_inversePointMap[i]<<endl;
// 		}


// 		for(int o=0;o<_hierarchicalPointMap.size();++o)
// 		{
// 			cerr<<o<<" : ";
// 			for(std::set<int>::iterator it=_hierarchicalPointMap[o].begin();it!=_hierarchicalPointMap[o].end();++it)
// 				cerr<<*it<<" ";
// 			cerr<<endl;
// 		}


// 		cerr<<"seqHexas : "<<seqHexas<<endl;
// 		cerr<<"seqPoints : "<<seqPoints<<endl;



    _finerSparseGrid->_coarserSparseGrid = this;
    _finerSparseGrid->_inverseHierarchicalCubeMap.resize( _finerSparseGrid->seqHexas.size(), -1);
    for( unsigned i=0; i<_hierarchicalCubeMap.size(); ++i)
        for(int w=0; w<8; ++w)
        {
            if(_hierarchicalCubeMap[i][w] != -1)
                _finerSparseGrid->_inverseHierarchicalCubeMap[ _hierarchicalCubeMap[i][w] ] = i;
        }
}


///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////




/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int SparseGridTopology::findCube(const Vec3& pos, double& fx, double &fy, double &fz)
{
    int indiceInRegularGrid = _regularGrid.findCube( pos,fx,fy,fz);
    if( indiceInRegularGrid == -1 )
        return -1;
    else
        return _indicesOfRegularCubeInSparseGrid[indiceInRegularGrid];
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int SparseGridTopology::findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz)
{
    int indice = 0;
    float lgmin = 99999999.0f;

    for(unsigned w=0; w<seqHexas.size(); ++w)
    {
        if(!_usingMC && _types[w]!=BOUNDARY )continue;

        const Hexa& c = getHexa( w );
        int c0 = c[0];
        int c7 = c[7];
        Vec3 p0(getPX(c0),getPY(c0),getPZ(c0));
        Vec3 p7(getPX(c7),getPY(c7),getPZ(c7));

        Vec3 barycenter = (p0+p7) * .5;

        float lg = (float)((pos-barycenter).norm());
        if( lg < lgmin )
        {
            lgmin = lg;
            indice = w;
        }
    }

    const Hexa& c = getHexa( indice );
    int c0 = c[0];
    int c7 = c[7];
    Vec3 p0(getPX(c0),getPY(c0),getPZ(c0));
    Vec3 p7(getPX(c7),getPY(c7),getPZ(c7));

    Vec3 relativePos = pos-p0;
    Vec3 diagonal = p7 - p0;

    fx = relativePos[0] / diagonal[0];
    fy = relativePos[1] / diagonal[1];
    fz = relativePos[2] / diagonal[2];

    return indice;
}


SparseGridTopology::Type SparseGridTopology::getType( int i )
{
    return _types[i];
}



///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////



void SparseGridTopology::updateEdges()
{
    std::map<pair<int,int>,bool> edgesMap;
    for(unsigned i=0; i<seqHexas.size(); ++i)
    {
        Hexa c = seqHexas[i];
        // horizontal
        edgesMap[pair<int,int>(c[0],c[1])]=0;
        edgesMap[pair<int,int>(c[2],c[3])]=0;
        edgesMap[pair<int,int>(c[4],c[5])]=0;
        edgesMap[pair<int,int>(c[6],c[7])]=0;
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
    for(unsigned i=0; i<seqHexas.size(); ++i)
    {
        Hexa c = seqHexas[i];

        fixed_array<int,4> v;
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

    }

    seqQuads.clear();
    seqQuads.reserve(quadsMap.size());
    for( std::map<fixed_array<int,4>,bool>::iterator it=quadsMap.begin(); it!=quadsMap.end(); ++it)
        seqQuads.push_back( Quad( (*it).first[0],  (*it).first[1],(*it).first[2],(*it).first[3] ));
}

void SparseGridTopology::updateHexas()
{
    // 					seqHexas.clear();
    // 					seqHexas.reserve(_cubes.size());

    // 								seqHexas.push_back(Hexa(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
    // 										point(x  ,y+1,z  ),point(x+1,y+1,z  ),
    // 										point(x  ,y  ,z+1),point(x+1,y  ,z+1),
    // 										point(x  ,y+1,z+1),point(x+1,y+1,z+1)));
}


////////////////////////////////////////
////////////////////////////////////////

//     bool SparseGridTopology::intersectionSegmentBox( const SegmentForIntersection& seg, const CubeForIntersection& cube  )
//     {
//       Vec3 afAWdU, afADdU, afAWxDdU;
//       Real fRhs;
//
//
//       Vec3 kDiff = seg.center - cube.center;
//
//       afAWdU[0] = fabs( seg.dir* cube.dir[0]);
//       afADdU[0] = fabs( kDiff * cube.dir[0] );
//       fRhs = cube.norm[0] + seg.norm*afAWdU[0];
//       if (afADdU[0] > fRhs)
//       {
//         return false;
//       }
//
//       afAWdU[1] = fabs(seg.dir*cube.dir[1]);
//       afADdU[1] = fabs(kDiff*cube.dir[1]);
//       fRhs = cube.norm[1] + seg.norm*afAWdU[1];
//       if (afADdU[1] > fRhs)
//       {
//         return false;
//       }
//
//       afAWdU[2] = fabs(seg.dir*cube.dir[2]);
//       afADdU[2] = fabs(kDiff*cube.dir[2]);
//       fRhs = cube.norm[2] + seg.norm*afAWdU[2];
//       if (afADdU[2] > fRhs)
//       {
//         return false;
//       }
//
//       Vec3 kWxD = seg.dir.cross(kDiff);
//
//       afAWxDdU[0] = fabs(kWxD*cube.dir[0]);
//       fRhs = cube.norm[1]*afAWdU[2] + cube.norm[2]*afAWdU[1];
//       if (afAWxDdU[0] > fRhs)
//       {
//         return false;
//       }
//
//       afAWxDdU[1] = fabs(kWxD*cube.dir[1]);
//       fRhs = cube.norm[0]*afAWdU[2] + cube.norm[2]*afAWdU[0];
//       if (afAWxDdU[1] > fRhs)
//       {
//         return false;
//       }
//
//       afAWxDdU[2] = fabs(kWxD*cube.dir[2]);
//       fRhs = cube.norm[0]*afAWdU[1] +cube.norm[1]*afAWdU[0];
//       if (afAWxDdU[2] > fRhs)
//       {
//         return false;
//       }
//
//       return true;
//     }


void SparseGridTopology::propagateFrom( const int i, const int j, const int k,
        RegularGridTopology& regularGrid,
        vector<Type>& regularGridTypes,
        vector<bool>& alreadyTested  ) const
{
    assert( i>=0 && i<=regularGrid.getNx()-2 && j>=0 && j<=regularGrid.getNy()-2 && k>=0 && k<=regularGrid.getNz()-2 );

    unsigned indice = regularGrid.cube( i, j, k );

    if( alreadyTested[indice] || regularGridTypes[indice] == BOUNDARY )
        return;

    alreadyTested[indice] = true;
    regularGridTypes[indice] = OUTSIDE;

    if(i>0) propagateFrom( i-1, j, k, regularGrid, regularGridTypes, alreadyTested );
    if(i<regularGrid.getNx()-2) propagateFrom( i+1, j, k, regularGrid, regularGridTypes, alreadyTested );
    if(j>0) propagateFrom( i, j-1, k, regularGrid, regularGridTypes, alreadyTested );
    if(j<regularGrid.getNy()-2) propagateFrom( i, j+1, k, regularGrid, regularGridTypes, alreadyTested );
    if(k>0) propagateFrom( i, j, k-1, regularGrid, regularGridTypes, alreadyTested );
    if(k<regularGrid.getNz()-2) propagateFrom( i, j, k+1, regularGrid, regularGridTypes, alreadyTested );
}




/////////////////////


} // namespace topology

} // namespace component

} // namespace sofa
