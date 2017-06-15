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
#ifndef SOFA_COMPONENT_ENGINE_MESHBARYCENTRICMAPPERENGINE_INL
#define SOFA_COMPONENT_ENGINE_MESHBARYCENTRICMAPPERENGINE_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/MeshBarycentricMapperEngine.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
MeshBarycentricMapperEngine<DataTypes>::MeshBarycentricMapperEngine()
    : initialized(false)
    , InputMeshName( initData (&InputMeshName, "InputMeshName", "Name and path of Input mesh Topology") )
    , InputPositions( initData (&InputPositions, "InputPositions", "Initial positions of the master points"))
    , MappedPointPositions( initData (&MappedPointPositions, "MappedPointPositions", "Initial positions of the mapped points"))
    , BarycentricPositions(initData (&BarycentricPositions, "BarycentricPositions", "Output : Barycentric positions of the mapped points"))
    , TableElements(initData (&TableElements, "TableElements", "Output : Table that provides the element index to which each input point belongs"))
    , computeLinearInterpolation(initData(&computeLinearInterpolation, false, "computeLinearInterpolation", "if true, computes a linear interpolation (debug)"))
    , f_interpolationIndices(initData(&f_interpolationIndices, "LinearInterpolationIndices", "Indices of a linear interpolation"))
    , f_interpolationValues(initData(&f_interpolationValues, "LinearInterpolationValues", "Values of a linear interpolation"))
{
}


template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::init()
{

    this->update();

    if (TopoInput==NULL)
    {
        msg_error() <<"Can not work with no input topology.";
        return;
    }

    addInput(&InputMeshName);
    addInput(&InputPositions);

    addOutput(&BarycentricPositions);
    addOutput(&TableElements);
    setDirtyValue();


}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::update()
{

    using sofa::defaulttype::Vector3;
    using sofa::defaulttype::Matrix3;
    using sofa::defaulttype::Mat3x3d;
    using sofa::defaulttype::Vec3d;

    const std::string path = InputMeshName.getValue();


    if (path.size()>0)
    {
        this->getContext()->get(TopoInput ,path  );

    }
    else
        TopoInput = NULL;

    if(TopoInput==NULL)
    {
        serr<<"no TopoInput found !!"<<sendl;
        return;
    }
    /*
    else
         std::cout<< "topology named "<<TopoInput->getName()<<" found !! "<<path<<std::endl;

             */
    std::cout<<"size of InputPositions="<<InputPositions.getValue().size()<<std::endl;

    std::cout<<"size of InputPositions="<<InputPositions.getValue()<<std::endl;


    const VecCoord* in = &InputPositions.getValue();
    const VecCoord* out = &MappedPointPositions.getValue();


    cleanDirty();


    baryPos =  BarycentricPositions.beginWriteOnly();
    tableElts= TableElements.beginWriteOnly();
    baryPos->resize(out->size());
    tableElts->resize(out->size());

    if(computeLinearInterpolation.getValue())
    {
        linearInterpolIndices = f_interpolationIndices.beginEdit();
        linearInterpolValues = f_interpolationValues.beginEdit();

        linearInterpolIndices->clear();
        linearInterpolValues->clear();

        linearInterpolIndices->resize(out->size());
        linearInterpolValues->resize(out->size());
    }


    int outside = 0;

    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = TopoInput->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = TopoInput->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = TopoInput->getCubes();
#endif
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = TopoInput->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = TopoInput->getQuads();
    sofa::helper::vector<Matrix3> bases;
    sofa::helper::vector<Vector3> centers;

    if ( tetrahedra.empty() && cubes.empty() )
    {
        if ( triangles.empty() && quads.empty() )
        {
            //no 3D elements, nor 2D elements -> map on 1D elements

            const sofa::core::topology::BaseMeshTopology::SeqEdges& edges = TopoInput->getEdges();
            if ( edges.empty() ) return;

            clear1d ( (*out).size() );

            sofa::helper::vector< SReal >   lengthEdges;
            sofa::helper::vector< Vector3 > unitaryVectors;

            unsigned int e;
            for ( e=0; e<edges.size(); e++ )
            {
                lengthEdges.push_back ( ( (*in)[edges[e][1]]-(*in)[edges[e][0]] ).norm() );

                Vector3 V12 = ( (*in)[edges[e][1]]-(*in)[edges[e][0]] ); V12.normalize();
                unitaryVectors.push_back ( V12 );
            }

            for ( unsigned int i=0; i<(*out).size(); i++ )
            {
                SReal coef=0;
                for ( e=0; e<edges.size(); e++ )
                {
                    SReal lengthEdge = lengthEdges[e];
                    Vector3 V12 =unitaryVectors[e];

                    coef = ( V12 ) *Vector3 ((*out)[i]-(*in)[edges[e][0]] ) /lengthEdge;
                    if ( coef >= 0 && coef <= 1 )
                    {
                        addPointInLine ( e, &coef );
                        break;
                    }

                }
                //If no good coefficient has been found, we add to the last element
                if ( e == edges.size() ) addPointInLine ( edges.size()-1,&coef );

            }
        }
        else
        {
            // no 3D elements -> map on 2D elements
            clear2d ( (*out).size() ); // reserve space for 2D mapping
            int c0 = triangles.size();
            bases.resize ( triangles.size() +quads.size() );
            centers.resize ( triangles.size() +quads.size() );
            for ( unsigned int t = 0; t < triangles.size(); t++ )
            {
                Mat3x3d m,mt;
                m[0] = (*in)[triangles[t][1]]-(*in)[triangles[t][0]];
                m[1] = (*in)[triangles[t][2]]-(*in)[triangles[t][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[t].invert ( mt );
                centers[t] = ( (*in)[triangles[t][0]]+(*in)[triangles[t][1]]+(*in)[triangles[t][2]] ) /3;
            }
            for ( unsigned int c = 0; c < quads.size(); c++ )
            {
                Mat3x3d m,mt;
                m[0] = (*in)[quads[c][1]]-(*in)[quads[c][0]];
                m[1] = (*in)[quads[c][3]]-(*in)[quads[c][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[c0+c].invert ( mt );
                centers[c0+c] = ( (*in)[quads[c][0]]+(*in)[quads[c][1]]+(*in)[quads[c][2]]+(*in)[quads[c][3]] ) *0.25;
            }
            for ( unsigned int i=0; i<(*out).size(); i++ )
            {
                Vector3 pos = DataTypes::getCPos((*out)[i]);
                Vector3 coefs;
                int index = -1;
                double distance = 1e10;
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    Vec3d v = bases[t] * ( pos - (*in)[triangles[t][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
                    if ( d>0 ) d = ( pos-centers[t] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = t; }
                }
                for ( unsigned int c = 0; c < quads.size(); c++ )
                {
                    Vec3d v = bases[c0+c] * ( pos - (*in)[quads[c][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-1,v[0]-1 ),std::max ( v[2]-0.01,-v[2]-0.01 ) ) );
                    if ( d>0 ) d = ( pos-centers[c0+c] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = c0+c; }
                }
                if ( distance>0 )
                {
                    ++outside;
                }
                if ( index < c0 ){
                    std::cout<<"addPoint "<<i<<" in Triangle "<<index<<" coef bary :"<<coefs<<std::endl;
                    addPointInTriangle ( index, coefs.ptr(),i );
                }
                else
                    addPointInQuad ( index-c0, coefs.ptr() );
            }
        }
    }
    else
    {
        clear3d ( (*out).size() ); // reserve space for 3D mapping
        int c0 = tetrahedra.size();
        bases.resize ( tetrahedra.size() +cubes.size() );
        centers.resize ( tetrahedra.size() +cubes.size() );
        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            Mat3x3d m,mt;
            m[0] = (*in)[tetrahedra[t][1]]-(*in)[tetrahedra[t][0]];
            m[1] = (*in)[tetrahedra[t][2]]-(*in)[tetrahedra[t][0]];
            m[2] = (*in)[tetrahedra[t][3]]-(*in)[tetrahedra[t][0]];
            mt.transpose ( m );
            bases[t].invert ( mt );
            centers[t] = ( (*in)[tetrahedra[t][0]]+(*in)[tetrahedra[t][1]]+(*in)[tetrahedra[t][2]]+(*in)[tetrahedra[t][3]] ) *0.25;
            //sout << "Tetra "<<t<<" center="<<centers[t]<<" base="<<m<<sendl;
        }
        for ( unsigned int c = 0; c < cubes.size(); c++ )
        {
            Mat3x3d m,mt;
            m[0] = (*in)[cubes[c][1]]-(*in)[cubes[c][0]];
#ifdef SOFA_NEW_HEXA
            m[1] = (*in)[cubes[c][3]]-(*in)[cubes[c][0]];
#else
            m[1] = (*in)[cubes[c][2]]-(*in)[cubes[c][0]];
#endif
            m[2] = (*in)[cubes[c][4]]-(*in)[cubes[c][0]];
            mt.transpose ( m );
            bases[c0+c].invert ( mt );
            centers[c0+c] = ( (*in)[cubes[c][0]]+(*in)[cubes[c][1]]+(*in)[cubes[c][2]]+(*in)[cubes[c][3]]+(*in)[cubes[c][4]]+(*in)[cubes[c][5]]+(*in)[cubes[c][6]]+(*in)[cubes[c][7]] ) *0.125;
        }
        for ( unsigned int i=0; i<(*out).size(); i++ )
        {
            Vector3 pos = DataTypes::getCPos((*out)[i]);
            Vector3 coefs;
            int index = -1;
            double distance = 1e10;
            for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
            {
                Vector3 v = bases[t] * ( pos - (*in)[tetrahedra[t][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );
                if ( d>0 ) d = ( pos-centers[t] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = t; }
            }
            for ( unsigned int c = 0; c < cubes.size(); c++ )
            {
                Vector3 v = bases[c0+c] * ( pos - (*in)[cubes[c][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-1 ),std::max ( v[1]-1,v[2]-1 ) ) );
                if ( d>0 ) d = ( pos-centers[c0+c] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = c0+c; }
            }
            if ( distance>0 )
            {
                ++outside;
            }
            if ( index < c0 )
                addPointInTetra ( index, coefs.ptr() , i);
            else
                addPointInCube ( index-c0, coefs.ptr() );
        }
    }

    BarycentricPositions.endEdit();
    TableElements.endEdit();

    if(computeLinearInterpolation.getValue())
    {
        f_interpolationIndices.endEdit();
        f_interpolationValues.endEdit();
    }

}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::draw(const core::visual::VisualParams* )
{


}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInLine(const int /*lineIndex*/, const SReal* /*baryCoords*/)
{
    std::cout<<"addPointInLine not implemented"<<std::endl;

}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInTriangle(const int triangleIndex, const SReal* baryCoords,  const unsigned int pointIndex)
{
    if(tableElts==NULL|| baryPos==NULL)
        return;
    (*tableElts)[pointIndex] = triangleIndex;
    (*baryPos)[pointIndex][0] =( Real ) baryCoords[0];
    (*baryPos)[pointIndex][1] =( Real ) baryCoords[1];


    if(computeLinearInterpolation.getValue())
    {
        const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = TopoInput->getTriangles();

        if(linearInterpolIndices==NULL|| linearInterpolIndices==NULL || triangles.size()==0 )
            return;

        // node0
        (*linearInterpolIndices)[pointIndex].push_back(triangles[triangleIndex][0]);
        Real value = (Real)1.-(Real)(baryCoords[0]-baryCoords[1]);
        (*linearInterpolValues)[pointIndex].push_back(value);

        // node1
        (*linearInterpolIndices)[pointIndex].push_back(triangles[triangleIndex][1]);
        (*linearInterpolValues)[pointIndex].push_back((Real)baryCoords[0]);

        // node2
        (*linearInterpolIndices)[pointIndex].push_back(triangles[triangleIndex][2]);
        (*linearInterpolValues)[pointIndex].push_back((Real)baryCoords[1]);


    }



}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInQuad(const int /*quadIndex*/, const SReal* /*baryCoords*/)
{
    std::cout<<"addPointInQuad not implemented"<<std::endl;
}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInTetra(const int tetraIndex, const SReal* baryCoords, const unsigned int pointIndex)
{
    if(tableElts==NULL|| baryPos==NULL)
        return;
    (*tableElts)[pointIndex] = tetraIndex;
    (*baryPos)[pointIndex][0] =( Real ) baryCoords[0];
    (*baryPos)[pointIndex][1] =( Real ) baryCoords[1];
    (*baryPos)[pointIndex][2] =( Real ) baryCoords[2];

    if(computeLinearInterpolation.getValue())
    {
        const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = TopoInput->getTetrahedra();

        if(linearInterpolIndices==NULL|| linearInterpolIndices==NULL || tetrahedra.size()==0 )
            return;

        // node0
        (*linearInterpolIndices)[pointIndex].push_back(tetrahedra[tetraIndex][0]);
        Real value = (Real)1.-(Real)(baryCoords[0]-baryCoords[1]-baryCoords[2]);
        (*linearInterpolValues)[pointIndex].push_back(value);

        // node1
        (*linearInterpolIndices)[pointIndex].push_back(tetrahedra[tetraIndex][1]);
        (*linearInterpolValues)[pointIndex].push_back((Real)baryCoords[0]);

        // node2
        (*linearInterpolIndices)[pointIndex].push_back(tetrahedra[tetraIndex][2]);
        (*linearInterpolValues)[pointIndex].push_back((Real)baryCoords[1]);

        // node3
        (*linearInterpolIndices)[pointIndex].push_back(tetrahedra[tetraIndex][3]);
        (*linearInterpolValues)[pointIndex].push_back((Real)baryCoords[2]);

    }


}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInCube(const int /*cubeIndex*/, const SReal* /*baryCoords*/)
{
    std::cout<<"addPointInCube not implemented"<<std::endl;
}


template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::clear1d ( int /*reserve*/ )
{
// map1d.clear(); if ( reserve>0 ) map1d.reserve ( reserve );
}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::clear2d ( int /*reserve*/ )
{
// map2d.clear(); if ( reserve>0 ) map2d.reserve ( reserve );
}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::clear3d ( int /*reserve*/ )
{
// map3d.clear(); if ( reserve>0 ) map3d.reserve ( reserve );
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
