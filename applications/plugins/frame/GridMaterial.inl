/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_INL
#define SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_INL

#include "GridMaterial.h"
#include <queue>

namespace sofa
{
namespace component
{
namespace material
{


template<class MaterialTypes>
GridMaterial<MaterialTypes>::GridMaterial()
{
}

template<class MaterialTypes>
void GridMaterial<MaterialTypes>::init()
{

    vector<VoxelGridLoader*> vg;
    sofa::core::objectmodel::BaseContext* context=  this->getContext();
    context->get<VoxelGridLoader>( &vg, core::objectmodel::BaseContext::Local);
    assert(vg.size()>0);
    this->voxelGridLoader = vg[0];

    if ( !this->voxelGridLoader )
    {
        serr << "VoxelGrid component not found" << sendl;
        this->nbVoxels = 0;
        this->Data = NULL;
        this->SegmentID = NULL;
    }
    else
    {
        this->voxelGridLoader->getVoxelSize(this->voxelSize);
        this->voxelGridLoader->getResolution(this->Dimension);
        // this->voxelGridLoader->getOrigin(this->Origin); // TO DO : update voxelGridLoader to add offsets
        this->nbVoxels = this->Dimension[0]*this->Dimension[1]*this->Dimension[2];
        this->Data = this->voxelGridLoader->getData();
        this->SegmentID = this->voxelGridLoader->getSegmentID();
    }

    Inherited::init();
}

// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
template<class MaterialTypes>
void GridMaterial<MaterialTypes>::computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& )
{

//                Real f = youngModulus.getValue()/((1 + poissonRatio.getValue())*(1 - 2 * poissonRatio.getValue()));
//                stressDiagonal = f * (1 - poissonRatio.getValue());
//                stressOffDiagonal = poissonRatio.getValue() * f;
//                shear = f * (1 - 2 * poissonRatio.getValue()) /2;
//
//
//                for(unsigned i=0; i<stress.size(); i++)
//                {
//                    stress[i][0] = stressDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][1] = stressOffDiagonal * strain[i][0] + stressDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][2] = stressOffDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressDiagonal * strain[i][2];
//                    stress[i][3] = shear * strain[i][3];
//                    stress[i][4] = shear * strain[i][4];
//                    stress[i][5] = shear * strain[i][5];
//                }
//                if( stressStrainMatrices != NULL ){
//                    VecStrStr&  m = *stressStrainMatrices;
//                    m.resize( stress.size() );
//                    m[0].fill(0);
//                    m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//                    m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//                    m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//                    for( unsigned i=1; i<m.size(); i++ ){
//                        m[i] = m[0];
//                    }
//                }
}

// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
template<class MaterialTypes>
void GridMaterial<MaterialTypes>::computeStress  ( VecElStr& stress, VecStrStr* stressStrainMatrices, const VecElStr& strain, const VecElStr& )
{
//                Real f = youngModulus.getValue()/((1 + poissonRatio.getValue())*(1 - 2 * poissonRatio.getValue()));
//                stressDiagonal = f * (1 - poissonRatio.getValue());
//                stressOffDiagonal = poissonRatio.getValue() * f;
//                shear = f * (1 - 2 * poissonRatio.getValue()) /2;
//
//
//                for(unsigned e=0; e<10; e++)
//                for(unsigned i=0; i<stress.size(); i++)
//                {
//                    stress[i][0][e] = stressDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//                    stress[i][1][e] = stressOffDiagonal * strain[i][0][e] + stressDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//                    stress[i][2][e] = stressOffDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressDiagonal * strain[i][2][e];
//                    stress[i][3][e] = shear * strain[i][3][e];
//                    stress[i][4][e] = shear * strain[i][4][e];
//                    stress[i][5][e] = shear * strain[i][5][e];
//                }
//                if( stressStrainMatrices != NULL ){
//                    VecStrStr&  m = *stressStrainMatrices;
//                    m.resize( stress.size() );
//                    m[0].fill(0);
//                    m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//                    m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//                    m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//                    for( unsigned i=1; i<m.size(); i++ ){
//                        m[i] = m[0];
//                    }
//                }
}







/*   Compute distances   */

template < class MaterialTypes >
double GridMaterial< MaterialTypes >::getDistance(const unsigned int& index1,const unsigned int& index2,const bool biasDistances)
{
    if(!nbVoxels) return -1;
    Vec3 coord1; if(!getCoord(index1,coord1)) return -1; // point1 not in grid
    Vec3 coord2; if(!getCoord(index2,coord2)) return -1; // point2 not in grid
    if(biasDistances)
    {
        double meanstiff=1;//=(getStiffness(index1)+getStiffness(index2))/2.; // TO DO: implement stiffness
        return (double)(coord2-coord1).norm()/meanstiff;
    }
    else return (double)(coord2-coord1).norm();
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::computeGeodesicalDistances ( const Vec3& point, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    int index=getIndex(point);
    return computeGeodesicalDistances (index, biasDistances, distMax );
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::computeGeodesicalDistances ( const int& index, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    unsigned int i,index1,index2;
    Distances.resize(this->nbVoxels);
    for(i=0; i<this->nbVoxels; i++) Distances[i]=distMax;

    if(index<0 || index>=nbVoxels) return false; // voxel out of grid
    if(!SegmentID[index]) return false;	// voxel out of object

    VUI neighbors;
    double d;

    std::queue<unsigned int> fifo;
    Distances[index]=0; fifo.push(index);
    while(!fifo.empty())
    {
        index1=fifo.front();
        get26Neighbors(index1, neighbors);
        for(i=0; i<neighbors.size(); i++)
        {
            index2=neighbors[i];
            if(SegmentID[index2]) // test if voxel is not void
            {
                d=Distances[index1]+getDistance(index1,index2,biasDistances);
                if(Distances[index2]>d)
                {
                    Distances[index2]=d;
                    fifo.push(index2);
                }
            }
        }
        fifo.pop();
    }
    return true;
}


template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::computeGeodesicalDistances ( const VecVec3& points, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    VI indices;
    for(unsigned int i=0; i<points.size(); i++) indices.push_back(getIndex(points[i]));
    return computeGeodesicalDistances ( indices, biasDistances, distMax );
}


template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::computeGeodesicalDistances ( const VI& indices, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    unsigned int i,nbi=indices.size(),index1,index2;
    Distances.resize(this->nbVoxels); Voronoi.resize(this->nbVoxels);
    for(i=0; i<this->nbVoxels; i++) {Distances[i]=distMax; Voronoi[i]=-1;}

    VUI neighbors;
    double d;

    std::queue<unsigned int> fifo;
    for(i=0; i<nbi; i++) if(indices[i]>=0 && indices[i]<nbVoxels) if(SegmentID[indices[i]]!=0) {Distances[indices[i]]=0; Voronoi[indices[i]]=i; fifo.push(indices[i]);}
    if(fifo.empty()) return false; // all input voxels out of grid
    while(!fifo.empty())
    {
        index1=fifo.front();
        get26Neighbors(index1, neighbors);
        for(i=0; i<neighbors.size(); i++)
        {
            index2=neighbors[i];
            if(SegmentID[index2]) // test if voxel is not void
            {
                d=Distances[index1]+getDistance(index1,index2,biasDistances);
                if(Distances[index2]>d)
                {
                    Distances[index2]=d; Voronoi[index2]=Voronoi[index1];
                    fifo.push(index2);
                }
            }
        }
        fifo.pop();
    }
    return true;
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::computeUniformSampling ( VecVec3& points, const bool biasDistances, unsigned int num_points, unsigned int max_iterations )
{
    if(!nbVoxels) return false;
    unsigned int i,k,initial_num_points=points.size();
    VI indices(num_points,-1);
    for(i=0; i<initial_num_points; i++) indices.push_back(getIndex(points[i]));
    points.resize(num_points);

// initialization: farthest point sampling (see [adams08])
    double dmax; int indexmax;

    if(initial_num_points==0) {indices[0]=0; while(SegmentID[indices[0]]==0) {indices[0]++; if(indices[0]==nbVoxels) return false;} } // take the first not empty voxel as a random point
    for(i=initial_num_points; i<num_points; i++)
    {
        if(i==0) i=1; // a random point has been inserted
        // get farthest point from all inserted points
        computeGeodesicalDistances(indices,biasDistances);
        dmax=-1; indexmax=-1; for(k=0; k<num_points; k++) {if(Distances[k]>dmax) {dmax=Distances[k]; indexmax=k;}}
        if(indexmax==-1) return false; // unable to add point
        indices[i]=indexmax;
    }

// Lloyd relaxation
    Vec3 pos,u,pos_point,pos_voxel;
    int count,count2=0;
    bool ok=false,ok2;
    double d,dmin; int indexmin;

    while(!ok && count2<max_iterations)
    {
        ok2=true;
        computeGeodesicalDistances(indices,biasDistances); // Voronoi
        VB flag(nbVoxels,false);
        for(i=initial_num_points; i<num_points; i++) 	// move to centroid of Voronoi cells
        {
            // estimate centroid given the measured distances = p + 1/N sum d(p,pi)*voxelsize*(p-pi)/|p-pi|
            getCoord(indices[i],pos_point);
            pos.fill(0); count=0;
            for(k=0; k<nbVoxels; k++)
                if(Voronoi[k]==i)
                {
                    getCoord(k,pos_voxel);
                    u=pos_point-pos_voxel; u.normalize();
                    u[0]*=(Real)voxelSize[0]; u[1]*=(Real)voxelSize[1]; u[2]*=(Real)voxelSize[2];
                    pos+=u*(Real)Distances[k];
                    count++;
                }
            pos/=(Real)count; 		pos+=pos_point;
            // get closest unoccupied point in object
            dmin=1E100; indexmin=-1;
            for(k=0; k<nbVoxels; k++) if(!flag[k]) if(SegmentID[k]!=0) {getCoord(k,pos_voxel); d=(pos-pos_voxel).norm2(); if(d<dmin) {flag[k]=true; dmin=d; indexmin=k;}}
            if(indices[i]!=indexmin) {ok2=false; indices[i]=indexmin;}
        }
        ok=ok2; count2++;
    }

// get points from indices
    for(i=initial_num_points; i<num_points; i++)
    {
        getCoord(indices[i],points[i]) ;
    }

    if(count2==max_iterations) return false; // reached max iterations
    return true;
}





/*         Utils         */
template < class MaterialTypes >
int GridMaterial< MaterialTypes >::getIndex(const Vec3i& icoord)
{
    if(!nbVoxels) return -1;
    for(int i=0; i<3; i++) if(icoord[i]<0 || icoord[i]>=this->Dimension[i]) return -1; // invalid icoord (out of grid)
    return icoord[0]+this->Dimension[0]*icoord[1]+this->Dimension[0]*this->Dimension[1]*icoord[2];
}

template < class MaterialTypes >
int GridMaterial< MaterialTypes >::getIndex(const Vec3& coord)
{
    if(!nbVoxels) return -1;
    Vec3i icoord;
    if(!getiCoord(coord,icoord)) return -1;
    return getIndex(icoord);
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::getiCoord(const Vec3& coord, Vec3i& icoord)
{
    if(!nbVoxels) return false;
    Real val;
    for(unsigned int i=0; i<3; i++)
    {
        val=(coord[i]-(Real)Origin[i])/(Real)voxelSize[i];
        val=((val-floor(val))<0.5)?floor(val):ceil(val); //round
        if(val<0 || val>=Dimension[i]) return false;
        icoord[i]=(int)val;
    }
    return true;
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::getiCoord(const unsigned int& index, Vec3i& icoord)
{
    if(!nbVoxels) return false;
    if(index<0 || index>=nbVoxels) return false;  // invalid index
    icoord[2]=index/this->Dimension[0]*this->Dimension[1];
    icoord[1]=(index-icoord[2]*this->Dimension[0]*this->Dimension[1])/this->Dimension[0];
    icoord[0]=index-icoord[2]*this->Dimension[0]*this->Dimension[1]-icoord[1]*this->Dimension[0];
    return true;
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::getCoord(const Vec3i& icoord, Vec3& coord)
{
    if(!nbVoxels) return false;
    for(unsigned int i=0; i<3; i++) if(icoord[i]<0 || icoord[i]>=this->Dimension[i]) return false; // invalid icoord (out of grid)
    coord=this->Origin;
    for(unsigned int i=0; i<3; i++) coord[i]+=(Real)this->voxelSize[i]*(Real)icoord[i];
    return true;
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::getCoord(const unsigned int& index, Vec3& coord)
{
    if(!nbVoxels) return false;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    else return getCoord(icoord,coord);
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::get6Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if(!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    for(unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
        icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
    }
    return true;
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::get18Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if(!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    for(unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
        icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
    }

    for(unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]-=2;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    return true;
}

template < class MaterialTypes >
bool GridMaterial< MaterialTypes >::get26Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if(!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    for(unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
        icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
    }
    for(unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]-=2;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    icoord[0]+=1;
    icoord[1]+=1;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]-=2;
    icoord[1]+=1;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]+=1;
    return true;
}


}

} // namespace component

} // namespace sofa

#endif


