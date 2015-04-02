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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/topology/MultiResSparseGridTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/xml/ObjectFactory.h>
#include <sofa/helper/system/gl.h>
#include <string>
#include <fstream>
using sofa::defaulttype::Vec3f;
using std::ifstream;
using std::string;

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using std::cerr;
using std::cout;
using std::endl;

void create(MultiResSparseGridTopology*& obj, simulation::xml::ObjectDescription* arg)
{
    const char* nx = arg->getAttribute("nx","1.0");
    const char* ny = arg->getAttribute("ny","1.0");
    const char* nz = arg->getAttribute("nz","1.0");
    const char* reso = arg->getAttribute("resolution");
    const char* filevoxel = arg->getAttribute("filevoxel");

    if (!filevoxel)
    {
        std::cerr << arg->getType() << " requires a voxel file attribute\n";
        obj = NULL;
    }
    if (!nx || !ny || !nz)
    {
        std::cerr << "GridTopology requires nx, ny and nz attributes\n";
    }
    else
    {
        obj = new MultiResSparseGridTopology(filevoxel,atoi(reso),
                (float) atof(arg->getAttribute("scale","1.0f")));
    }
}

SOFA_DECL_CLASS(MultiResSparseGridTopology)

helper::Creator<simulation::xml::ObjectFactory, MultiResSparseGridTopology> MultiResSparseGridTopologyClass("MultiResSparseGridTopology");

MultiResSparseGridTopology::MultiResSparseGridTopology()//:GridTopology(nx,ny,nz)
{}

MultiResSparseGridTopology::MultiResSparseGridTopology(const char* filevoxel, int resol, float scale)//:GridTopology(nx,ny,nz)
{
    int indiceRes = resol;
    resolution = resol;

    SparseGrid voxels1,hteDef,*voxel;

    hteDef = SparseGrid(scale);
    hteDef.initFromVOX((char *)filevoxel,255);

    int i = 0;
    vectorSparseGrid.push_back(hteDef);
    while((vectorSparseGrid[i].getSparseGridMap().size())>1)
    {
        voxels1 = SparseGrid(scale);
        voxels1.pasResolution(vectorSparseGrid[i]);
        vectorSparseGrid.push_back(voxels1);
        ++i;
    }

    indiceRes = vectorSparseGrid.size() - indiceRes - 1;
    if(indiceRes < 0)
    {
        indiceRes = 0;
    }
    voxel = &(vectorSparseGrid[indiceRes]);
    resolution = indiceRes;
    nbPoints =  (vectorSparseGrid[resolution].getNumVertices());

    ///Parametrer les dimensions et la position de la grille
    //setP0(Vec3(0.0,0.0,0.0));
    setDx(Vec3(voxel->getDimVoxX(),0,0));
    setDy(Vec3(0,voxel->getDimVoxY(),0));
    setDz(Vec3(0,0,voxel->getDimVoxZ()));

}

MultiResSparseGridTopology::Vec3 MultiResSparseGridTopology::getPoint(int i) const
{
    float x,y,z;
    x = ((vectorSparseGrid[resolution]).getVertices())[i][0];
    y = ((vectorSparseGrid[resolution]).getVertices())[i][1];
    z = ((vectorSparseGrid[resolution]).getVertices())[i][2];
    return (Vec3f((float)p0[0]+x,(float)p0[1]+y,(float)p0[2]+z));
}

// MultiResSparseGridTopology::Vec3 MultiResSparseGridTopology::getPoint(int i, int j, int k)
// {
//     MultiResSparseGridTopology::SparseGrid::Vertex *tmp =
// 		    &((vectorSparseGrid[resolution]).getVertexMap())[MultiResSparseGridTopology::SparseGrid::Index3D(i,j,k)];
//     float x,y,z;
//     x = tmp->vertexPosition[0];
//     y = tmp->vertexPosition[1];
//     z = tmp->vertexPosition[2];
//     return (Vec3f((float)p0[0]+x,(float)p0[1]+y,(float)p0[2]+z));
// }


double MultiResSparseGridTopology::getPX(int i) const
{
    return getPoint(i)[0];
}
double MultiResSparseGridTopology::getPY(int i) const
{
    return getPoint(i)[1];
}
double MultiResSparseGridTopology::getPZ(int i) const
{
    return getPoint(i)[2];
}

bool MultiResSparseGridTopology::hasPos() const
{
    return true;
}

int MultiResSparseGridTopology::getNbVoxels() const
{
    return (vectorSparseGrid[resolution].getSparseGridMap().size());
}

int MultiResSparseGridTopology::getNbPoints() const
{
    return (vectorSparseGrid[resolution].getNumVertices());
}

int MultiResSparseGridTopology::point(int x, int y, int z)
{
    MultiResSparseGridTopology::SparseGrid::Vertex tmp =
        ((vectorSparseGrid[resolution]).getVertexMap())[MultiResSparseGridTopology::SparseGrid::Index3D(x,y,z)];
    return tmp.index;
}


GridTopology::Cube MultiResSparseGridTopology::getCube(int i)
{
    int DimX = (vectorSparseGrid[resolution]).getDimX();
    int DimY = (vectorSparseGrid[resolution]).getDimY();

    int x = i%(DimX);
    i/=(DimX);
    int y = i%(DimY);
    i/=(DimY);
    int z = i;

    return Cube(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
            point(x  ,y+1,z  ),point(x+1,y+1,z  ),
            point(x  ,y  ,z+1),point(x+1,y  ,z+1),
            point(x  ,y+1,z+1),point(x+1,y+1,z+1));
}

int MultiResSparseGridTopology::findCube(const Vec3& pos, double& fx, double &fy, double &fz) const
{
    SparseGrid const * voxels = &vectorSparseGrid[resolution];
    ///se remettre dans le repere de l'objet
    Vec3 p = pos-p0;
    double x = p*dx*inv_dx2;
    double y = p*dy*inv_dy2;
    double z = p*dz*inv_dz2;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    fx = x-ix;
    fy = y-iy;
    fz = z-iz;
    int DimX = (vectorSparseGrid[resolution]).getDimX();
    int DimY = (vectorSparseGrid[resolution]).getDimY();

    if (voxels->getSparseGridMap().find(MultiResSparseGridTopology::SparseGrid::Index3D(ix,iy,iz)) ==
        voxels->getSparseGridMap().end())
        return -1;
    return ix + iy*DimX + iz*DimX*DimY;

}

int MultiResSparseGridTopology::findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz) const
{
    SparseGrid const *voxels = & vectorSparseGrid[resolution];
    bool cond = false;

    int DimX = (vectorSparseGrid[resolution]).getDimX();
    int DimY = (vectorSparseGrid[resolution]).getDimY();
    int DimZ = (vectorSparseGrid[resolution]).getDimZ();
    int i = 0,j,k,l;
    ///se remettre dans le repere de l'objet
    Vec3 p = pos-p0;
    double x = p*dx*inv_dx2;
    double y = p*dy*inv_dy2;
    double z = p*dz*inv_dz2;
    double dist,distX,distY,distZ, distRef = -1;
    int ix = int(x+1000000)-1000000; // Do not round toward 0...
    int iy = int(y+1000000)-1000000;
    int iz = int(z+1000000)-1000000;
    fx = x-ix;
    fy = y-iy;
    fz = z-iz;
    int ixTmp = ix, iyTmp = iy, izTmp = iz;
    while (cond == false)
    {
        ++i;
        for(j = -i; j<=i; ++j)
        {
            for(k = -(i-abs(j)); k <= (i-abs(j)) ; ++k)
            {
                for(l = -(i-abs(j)-abs(k)); l<= (i-abs(j)-abs(k)); ++l)
                {
                    if((abs(l)+abs(j)+abs(k))==i)
                    {
                        if (voxels->getSparseGridMap().find(MultiResSparseGridTopology::SparseGrid::Index3D(ix+j,iy+k,iz+l)) !=
                            voxels->getSparseGridMap().end())
                        {
                            distX = (j * DimX)- (fx*DimX - DimX/2);
                            distY = ((k * DimY)- (fy*DimY - DimY/2));
                            distZ = ((l * DimZ)- (fz*DimZ - DimZ/2));
                            dist = sqrt((distX*distX)+(distY*distY)+(distZ*distZ));
                            if (distRef==-1)
                            {
                                distRef = dist;
                                ixTmp = ix + j;
                                iyTmp = iy + k;
                                izTmp =  iz + l;
                            }
                            if (dist < distRef)
                            {
                                ixTmp = ix + j;
                                iyTmp = iy + k;
                                izTmp =  iz + l;
                                distRef = dist;
                            }
                        }
                    }
                }
            }
        }
        if (distRef != -1)
            cond=true;
    }

    fx = x - ixTmp;
    fy = y - iyTmp;
    fz = z - izTmp;
    return ixTmp + iyTmp*DimX + izTmp*DimX*DimY;

}
// int MultiResSparseGridTopology::getIndicesInSpace( sofa::helper::vector<int>& indices,
//         float xmin,float xmax,float ymin,float ymax,float zmin,float zmax )
// {
// 	float x,y,z;
// 	std::map<SparseGrid::Index3D,SparseGrid::Vertex>::iterator i;
// 	for(i = (vectorSparseGrid[resolution]).getVertexMap().begin();
// 		   i !=  (vectorSparseGrid[resolution]).getVertexMap().end();i++)
// 	{
//             x = (*i).second.vertexPosition[0];
//             y = (*i).second.vertexPosition[1];
//             z = (*i).second.vertexPosition[2];
// 		if ( xmin <= x && x <= xmax)
// 			if( ymin <= y && y <= ymax)
// 				if( zmin <= z && z <= zmax){
// 					indices.push_back((*i).second.index);
// 				}
// 	}
//     return 0;
// }



MultiResSparseGridTopology::SparseGrid::SparseGrid(void)
{
    FIN_DE_LISTE = -1;
    scale = 1;
}

MultiResSparseGridTopology::SparseGrid::SparseGrid(int x)
{
    FIN_DE_LISTE = x;
    scale = 1;
}

MultiResSparseGridTopology::SparseGrid::SparseGrid(float scale_)
{
    FIN_DE_LISTE = -1;
    scale = scale_;
}

MultiResSparseGridTopology::SparseGrid::~SparseGrid(void)
{}


void MultiResSparseGridTopology::SparseGrid::afficherSparseGridMap()
{
    cout << "VoxelMap" << endl;
    std::map<Index3D,Voxel>::iterator i;
    for(i = voxelsMap.begin(); i != voxelsMap.end(); ++i)
    {
        cout << (*i).first.i << " " << (*i).first.j << " " << (*i).first.k << " : ";
        for(int l=0; l<8; ++l)
            cout << (*i).second.vertices[l] << " ";
        cout << endl;
    }


    /*    for(i = voxelsMap.begin(); i != voxelsMap.end();i++)
        {
            //V1 (x0,y0,z0)
            for(int l=0; l<3;l++)
                cout << vertexMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertexPosition[l] <<" ";
            cout << "     ";
            //V1 (x0,y0,z0)
            for(int l=0; l<3;l++)
                cout << vertexMap[Index3D((*i).first.i+1,(*i).first.j,(*i).first.k)].vertexPosition[l] << " ";
            cout << "     ";
            //V1 (x0,y0,z0)
            for(int l=0; l<3;l++)
                cout << vertexMap[Index3D((*i).first.i,(*i).first.j+1,(*i).first.k)].vertexPosition[l] <<" " ;
            cout << "     ";
            //V1 (x0,y0,z0)
            for(int l=0; l<3;l++)
                cout << vertexMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k+1)].vertexPosition[l] <<" ";
            cout << "     ";

            cout << endl;
        }
        cout << "end voxelMap" << endl;

    }


    void MultiResSparseGridTopology::SparseGrid::afficherVertexMap()
    {

        cout << "VertexMap" << endl;
        std::map<Index3D,Vertex>::iterator i;
        for(i = vertexMap.begin(); i != vertexMap.end();i++)
        {
            cout << (*i).first.i << " " << (*i).first.j << " " << (*i).first.k << " : ";
            for(int l=0; l<3;l++)
                cout << (*i).second.vertexPosition[l] << " ";
            cout << endl;
        }*/
    cout << "end of the vertexMap" << endl;

}


void MultiResSparseGridTopology::SparseGrid::afficherVertices()
{

    cout << "Vertices" << endl;
    for(int l=0; l<(int) vertexMap.size(); ++l)
    {
        for(int k=0; k<3; ++k)
            cout<< vertices[l][k] << " ";
        cout << " ";
    }
    cout << endl;
    cout << "end of vertices" << endl;

}


void MultiResSparseGridTopology::SparseGrid::afficherIndices()
{
    cout << "Indices" << endl;

    for(int l=0; l<numIndices; ++l)
        cout << indices[l] << " ";
    cout << "end of indices" << endl;

}

int MultiResSparseGridTopology::SparseGrid::pasResolution(SparseGrid vg)
{

    float dvx,dvy,dvz;
    int dx,dy,dz;
    std::map<Index3D,Voxel>::iterator i;

    /// build the child grid wich his twice taller

    dvx = vg.getDimVoxX();
    dvy = vg.getDimVoxY();
    dvz = vg.getDimVoxZ();

    setDimVoxX(2.0f*dvx);
    setDimVoxY(2.0f*dvy);
    setDimVoxZ(2.0f*dvz);

    dx = vg.getDimX();
    dy = vg.getDimY();
    dz = vg.getDimZ();

    if(dx!=1) dx/=2; if(dy!=1) dy/=2; if(dz!=1) dz/=2;

    if (dx%2 && dx!=1)   setDimX(dx+1);
    else setDimX( dx );

    if (dy%2 && dy!=1) setDimY(dy+1);
    else setDimY( dy );

    if (dz%2 && dz!=1) setDimZ(dz+1);
    else setDimZ( dz );

    for(i = vg.voxelsMap.begin(); i != vg.voxelsMap.end(); ++i)
    {
        insertVoxel( ((*i).first.i)/2, ((*i).first.j)/2, ((*i).first.k)/2, (*i).second.density*0.125f );
    }

    buildFromSparseGridMap(voxelsMap.size());
    return 0;
}

void MultiResSparseGridTopology::SparseGrid::setIndicesMap()
{

    int num = 0;
    std::map<Index3D, Vertex>::iterator iter;
    ///for each pixel, we have to define the indices for the six corresponding faces
    std::map<Index3D,Voxel>::iterator i;
    /// to find the voxels of the surface, we check if the voxel has neighbours or not
    for(i = voxelsMap.begin(); i != voxelsMap.end(); ++i)
    {
        ///finding the order for each of the 8th vertices
        iter = vertexMap.find(Index3D((*i).first.i,(*i).first.j,(*i).first.k));
        int v1 = (*iter).second.index;
        //
        iter = vertexMap.find(Index3D((*i).first.i+1,(*i).first.j,(*i).first.k));
        int v2 = (*iter).second.index;
        //
        iter = vertexMap.find(Index3D((*i).first.i+1,(*i).first.j+1,(*i).first.k));
        int v3 = (*iter).second.index;
        //
        iter = vertexMap.find(Index3D((*i).first.i,(*i).first.j+1,(*i).first.k));
        int v4 = (*iter).second.index;
        //
        iter = vertexMap.find(Index3D((*i).first.i,(*i).first.j,(*i).first.k+1));
        int v5 = (*iter).second.index;
        //
        iter = vertexMap.find(Index3D((*i).first.i+1,(*i).first.j,(*i).first.k+1));
        int v6 = (*iter).second.index;
        //
        iter = vertexMap.find(Index3D((*i).first.i+1,(*i).first.j+1,(*i).first.k+1));
        int v7 = (*iter).second.index;
        //
        iter = vertexMap.find(Index3D((*i).first.i,(*i).first.j+1,(*i).first.k+1));
        int v8 = (*iter).second.index;

        ///Add to the map of voxels the vertices corresponding to a voxel
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[0]= v1;
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[1]= v2;
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[2]= v3;
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[3]= v4;
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[4]= v5;
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[5]= v6;
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[6]= v7;
        voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].vertices[7]= v8;

        //indices for face1
        indices[num+0]= v1;
        indices[num+1] = v2;
        indices[num+2] = v3;
        indices[num+3] = v4;
        indices[num+4] = FIN_DE_LISTE;
        //indices for face2
        indices[num+5]= v5;
        indices[num+8] = v6;
        indices[num+7] = v7;
        indices[num+6] = v8;
        indices[num+9] = FIN_DE_LISTE;
        //indices for face3
        indices[num+10]= v3;
        indices[num+13] = v4;
        indices[num+12] = v8;
        indices[num+11] = v7;
        indices[num+14] = FIN_DE_LISTE;
        //indices for face4
        indices[num+15]= v2;
        indices[num+16] = v1;
        indices[num+17] = v5;
        indices[num+18] = v6;
        indices[num+19] = FIN_DE_LISTE;
        //indices for face5
        indices[num+20]= v1;
        indices[num+21] = v4;
        indices[num+22] = v8;
        indices[num+23] = v5;
        indices[num+24] = FIN_DE_LISTE;
        //indices for face6
        indices[num+25]= v2;
        indices[num+28] = v3;
        indices[num+27] = v7;
        indices[num+26] = v6;
        indices[num+29] = FIN_DE_LISTE;
        num = num+30;
    }
}

void MultiResSparseGridTopology::SparseGrid::insertVoxel(int i, int j, int k, float d)
{

    voxelsMap[Index3D(i,j,k)].density += d;

}

int MultiResSparseGridTopology::SparseGrid::setPixels(const char *FileName, int color,float plane)
{
    char line[256];
    int j=0;
    string f;
    int width=0;
    int height=0;
    ifstream infile;
    int x=0;
    int y=0;
    infile.open(FileName);
    if (!infile)
        cerr << "error.." << endl;
    else
        for(int i=0; i < 4; ++i)
        {
            infile >> line;
            if(i==1)
                width = atoi(line);
            if(i==2)
                height = atoi(line);
        }
    cout<<"Width is "<<width<<endl;
    cout<<"Height is "<<height<<endl;

    setDimX( width );
    setDimY( height );

    for (y=0; y<height; ++y)
        for (x=0; x<width; ++x)
        {
            infile>>line;
            if (atoi(line)==color)
            {
                insertVoxel(x,y,(int)plane,1.0);
                ++j;
            }
        }

    cout<<"Number of black pixels in file "<<FileName<<" is "<<j<<endl;
    infile.close();
    return j;
}

void MultiResSparseGridTopology::SparseGrid::setVertexMap()
{
    std::map<Index3D,Voxel>::iterator i;
    float dx = getDimVoxX(), dy = getDimVoxY(), dz = getDimVoxZ();
    /// to find the voxels of the surface, we check if the voxel has neighbours or not
    for(i = voxelsMap.begin(); i != voxelsMap.end(); ++i)
    {
        //V1 (x0,y0,z0)
        vertexMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)];// = Vertex((*i).first.i*dx,(*i).first.j*dy,(*i).first.k*dz);
        //V2 (x1,y0,z0)
        vertexMap[Index3D((*i).first.i+1,(*i).first.j,(*i).first.k)];// Vertex(((*i).first.i+1)*dx,(*i).first.j*dy,(*i).first.k*dz);
        //V3 (x1,y1,z0)
        vertexMap[Index3D((*i).first.i+1,(*i).first.j+1,(*i).first.k)];// Vertex(((*i).first.i+1)*dx,((*i).first.j+1)*dy,(*i).first.k*dz);
        //V4 (x0,y1,z0)
        vertexMap[Index3D((*i).first.i,(*i).first.j+1,(*i).first.k)];// Vertex((*i).first.i*dx,((*i).first.j+1)*dy,((*i).first.k)*dz);
        //V1 (x0,y0,z1)
        vertexMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k+1)];// Vertex((*i).first.i*dx,(*i).first.j*dy,((*i).first.k+1)*dz);
        //V2 (x1,y0,z1)
        vertexMap[Index3D((*i).first.i+1,(*i).first.j,(*i).first.k+1)];// Vertex(((*i).first.i+1)*dx,(*i).first.j*dy,((*i).first.k+1)*dz);
        //V3 (x1,y1,z1)
        vertexMap[Index3D((*i).first.i+1,(*i).first.j+1,(*i).first.k+1)];// Vertex(((*i).first.i+1)*dx,((*i).first.j+1)*dy,((*i).first.k+1)*dz);
        //V4 (x0,y1,z1)
        vertexMap[Index3D((*i).first.i,(*i).first.j+1,(*i).first.k+1)];// Vertex((*i).first.i*dx,((*i).first.j+1)*dy,((*i).first.k+1)*dz);


    }
    /// Numerating the vertices, and making the vertices array
    int num = 0;
    std::map<Index3D, Vertex>::iterator iter;
    vertices.resize(vertexMap.size());
    for(iter= vertexMap.begin(); iter!= vertexMap.end(); ++iter)
    {
        (*iter).second.index = num;
        //vertices[num] = ((*iter).second.vertexPosition);
        vertices[num] = Vec3((*iter).first.i*dx, (*iter).first.j*dy, (*iter).first.k*dz);
        ++num;
    }

}

void MultiResSparseGridTopology::SparseGrid::setSurfaceSparseGrid()
{
    std::map<Index3D,Voxel>::iterator i;
    /// to find the voxels of the surface, we check if the voxel has neighbours or not
    for(i = voxelsMap.begin(); i != voxelsMap.end(); ++i)
    {
        if (voxelsMap.find(Index3D((*i).first.i+1,(*i).first.j,(*i).first.k))== voxelsMap.end())
            surfaceSparseGrid.push_back((*i).first);

        else if (voxelsMap.find(Index3D((*i).first.i-1,(*i).first.j,(*i).first.k))== voxelsMap.end())
            surfaceSparseGrid.push_back((*i).first);

        else if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j+1,(*i).first.k))== voxelsMap.end())
            surfaceSparseGrid.push_back((*i).first);

        else if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j-1,(*i).first.k))== voxelsMap.end())
            surfaceSparseGrid.push_back((*i).first);

        else if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j,(*i).first.k+1))== voxelsMap.end())
            surfaceSparseGrid.push_back((*i).first);

        else if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j,(*i).first.k-1))== voxelsMap.end())
            surfaceSparseGrid.push_back((*i).first);
    }
    surfaceIndices.resize(surfaceSparseGrid.size()*30);
    int counter = 0;
    for(i = voxelsMap.begin(); i != voxelsMap.end(); ++i)
    {
        if (voxelsMap.find(Index3D((*i).first.i+1,(*i).first.j,(*i).first.k))== voxelsMap.end())
        {
            surfaceIndices[counter] = (*i).second.vertices[1];
            surfaceIndices[counter+1] = (*i).second.vertices[5];
            surfaceIndices[counter+2] = (*i).second.vertices[6];
            surfaceIndices[counter+3] = (*i).second.vertices[2];
            surfaceIndices[counter+4] = FIN_DE_LISTE;
            counter = counter + 5;
        }
        if (voxelsMap.find(Index3D((*i).first.i-1,(*i).first.j,(*i).first.k))== voxelsMap.end())
        {
            surfaceIndices[counter] = (*i).second.vertices[0];
            surfaceIndices[counter+1] = (*i).second.vertices[3];
            surfaceIndices[counter+2] = (*i).second.vertices[7];
            surfaceIndices[counter+3] = (*i).second.vertices[4];
            surfaceIndices[counter+4] = FIN_DE_LISTE;
            counter = counter + 5;
        }
        if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j+1,(*i).first.k))== voxelsMap.end())
        {
            surfaceIndices[counter] = (*i).second.vertices[2];
            surfaceIndices[counter+1] = (*i).second.vertices[6];
            surfaceIndices[counter+2] = (*i).second.vertices[7];
            surfaceIndices[counter+3] = (*i).second.vertices[3];
            surfaceIndices[counter+4] = FIN_DE_LISTE;
            counter = counter + 5;
        }
        if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j-1,(*i).first.k))== voxelsMap.end())
        {
            surfaceIndices[counter] = (*i).second.vertices[1];
            surfaceIndices[counter+1] = (*i).second.vertices[0];
            surfaceIndices[counter+2] = (*i).second.vertices[4];
            surfaceIndices[counter+3] = (*i).second.vertices[5];
            surfaceIndices[counter+4] = FIN_DE_LISTE;
            counter = counter + 5;
        }
        if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j,(*i).first.k+1))== voxelsMap.end())
        {
            surfaceIndices[counter] = (*i).second.vertices[4];
            surfaceIndices[counter+1] = (*i).second.vertices[7];
            surfaceIndices[counter+2] = (*i).second.vertices[6];
            surfaceIndices[counter+3] = (*i).second.vertices[5];
            surfaceIndices[counter+4] = FIN_DE_LISTE;
            counter = counter + 5;
        }
        if (voxelsMap.find(Index3D((*i).first.i,(*i).first.j,(*i).first.k-1))== voxelsMap.end())
        {
            surfaceIndices[counter] = (*i).second.vertices[0];
            surfaceIndices[counter+1] = (*i).second.vertices[1];
            surfaceIndices[counter+2] = (*i).second.vertices[2];
            surfaceIndices[counter+3] = (*i).second.vertices[3];
            surfaceIndices[counter+4] = FIN_DE_LISTE;
            counter = counter + 5;
        }
    }
    numSurfaceIndices = counter;
}

void MultiResSparseGridTopology::SparseGrid::setDensity()
{

    std::map<Index3D, Voxel>::iterator i;

    voxelsDensity = new float [voxelsMap.size()];
    int j = 0;
    /// to find the voxels of the surface, we check if the voxel has neighbours or not
    for(i = voxelsMap.begin(); i != voxelsMap.end(); ++i)
    {
        voxelsDensity[j] = 1.0f-voxelsMap[Index3D((*i).first.i,(*i).first.j,(*i).first.k)].density;
        ++j;
    }
}


void MultiResSparseGridTopology::SparseGrid::buildFromSparseGridMap(int allPixels)
{

    setVertexMap();
    /// defining number of indices: (4indices for each face + 1 to indicate the end)*number of faces
    numIndices = allPixels*6*5;
    indices.resize(numIndices);
    /// finding the indices for each voxel
    setIndicesMap();
    /// finding voxels and faces of the surface
    setSurfaceSparseGrid();
    setDensity();
}

int MultiResSparseGridTopology::SparseGrid::readFilePNG( char* fileName,int color,int filesNo )
{
    /// 	pixels = new Index3D[512*512*filesNo];
    int pixel_no=0;
    char file_extension[64];
    string theFileName(fileName);
    for( int i=1; i<=filesNo; ++i )
    {
        setDimZ( filesNo );
        /// making the file name
        sprintf(file_extension,"%02d.pgm",i);
        string thisFileName = theFileName + string(file_extension);
        cout<<"Reading file "<<thisFileName<<endl;
        /// getting pixels of the given color from the file
        pixel_no += setPixels(thisFileName.c_str(),color,(float)i-1);
    }

    return pixel_no;
}

void MultiResSparseGridTopology::SparseGrid::initFromPNG( char* fileName,int color,int filesNo )
{

    int allPixels=0;

    allPixels = readFilePNG(fileName,color,filesNo);

    setDimVoxX(0.35f); // Dimension des voxels selon x.
    setDimVoxY(0.35f); // Dimension des voxels  selon y.
    setDimVoxZ(0.35f); // Dimension des voxels  selon z.

    buildFromSparseGridMap(allPixels);
}

int MultiResSparseGridTopology::SparseGrid::readFileVOX( char* fileName,int color )
{

    // le constructeur de ifstream permet d'ouvrir un fichier en lecture
    ifstream fichier( fileName );
    if ( fichier != NULL ) // ce test echoue si le fichier n'est pas ouvert
    {
        cout << "Ouverture du fichier " <<  fileName<< " reussit" << endl;
        int nb1, nb2, nb3,nb ;
        // Dimensions d'un voxels selon x, y, z
        float v1, v2, v3 ;
        // Recuperation des dimension de l'image. ( se trouvent dans l'entete du fichier VOXEL )
        fichier >> nb1 >> nb2 >> nb3;
        setDimX( nb1 );
        setDimY( nb2 );
        setDimZ( nb3 );

        //ces valeurs ne servent a rien pour l'instant
        fichier >> v1 >> v2 >> v3 ;

        setDimVoxX(v1*scale); // Dimension des voxels selon x.
        setDimVoxY(v2*scale); // Dimension des voxels  selon y.
        setDimVoxZ(v3*scale); // Dimension des voxels  selon z.

		int pixel_no = 0;
        for(int r=0; r<nb3; ++r)
        {
            for(int s = 0; s < nb2; ++s)
            {
                for(int t = 0; t < nb1; ++t)
                {
                    fichier >> nb ;
                    if (nb == color)
                    {
                        insertVoxel(t,s,r,1.0);
                        ++pixel_no;
                    }
                }
            }
        }
        fichier.close();
        return pixel_no;
    }
    else
    {
        cout << "Le fichier "<<fileName<<" n'existe pas !!\n";
        exit(0);
    }
}

void MultiResSparseGridTopology::SparseGrid::initFromVOX( char* fileName,int color)
{

    int allPixels=0;
    allPixels = readFileVOX(fileName,color);
    buildFromSparseGridMap(allPixels);
}

float MultiResSparseGridTopology::SparseGrid::getDensity(int i, int j, int k)
{

    return voxelsMap[Index3D(i,j,k)].density;
}



} // namespace topology

} // namespace component

} // namespace sofa

