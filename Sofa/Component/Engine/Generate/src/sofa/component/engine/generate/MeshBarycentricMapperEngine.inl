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
#pragma once

#include <sofa/component/engine/generate/MeshBarycentricMapperEngine.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::engine::generate
{

template <class DataTypes>
MeshBarycentricMapperEngine<DataTypes>::MeshBarycentricMapperEngine()
    : d_inputPositions( initData (&d_inputPositions, "inputPositions", "Initial positions of the master points"))
    , d_mappedPointPositions( initData (&d_mappedPointPositions, "mappedPointPositions", "Initial positions of the points to be mapped"))
    , d_barycentricPositions(initData (&d_barycentricPositions, "barycentricPositions", "Output : Barycentric positions of the mapped points"))
    , d_tableElements(initData (&d_tableElements, "tableElements", "Output : Table that provides the index of the element to which each input point belongs"))
    , d_bComputeLinearInterpolation(initData(&d_bComputeLinearInterpolation, false, "computeLinearInterpolation", "if true, computes a linear interpolation (debug)"))
    , d_interpolationIndices(initData(&d_interpolationIndices, "linearInterpolationIndices", "Indices of a linear interpolation"))
    , d_interpolationValues(initData(&d_interpolationValues, "linearInterpolationValues", "Values of a linear interpolation"))
    , l_topology(initLink("topology", "Name of the master topology"))
{
    addInput(&d_inputPositions);
    addInput(&d_mappedPointPositions);

    addOutput(&d_barycentricPositions);
    addOutput(&d_tableElements);
}


template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::init()
{
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);

    if(l_topology.get() == nullptr)
    {
        sofa::core::topology::BaseMeshTopology::SPtr localTopology = nullptr;
        this->getContext()->get(localTopology);
        if (localTopology == nullptr)
        {
            msg_error() << "Can not work with no input topology.";
            return;
        }
        else
        {
            l_topology.set(localTopology);
            msg_warning() << "No topology given, will use the local one. (this automatic behavior is not recommended, consider explicit input of the \"topology\" field).";
        }
    }

    setDirtyValue();

    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::doUpdate()
{
    if (d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    using sofa::type::Vec3;
    using sofa::type::Matrix3;

    const VecCoord& in = d_inputPositions.getValue();
    const VecCoord& out = d_mappedPointPositions.getValue();

    {
        auto baryPos = sofa::helper::getWriteOnlyAccessor(d_barycentricPositions);
        auto tableElts = sofa::helper::getWriteOnlyAccessor(d_tableElements);

        baryPos.clear();
        tableElts.clear();

        baryPos.resize(out.size());
        tableElts.resize(out.size());
    }

    if (d_bComputeLinearInterpolation.getValue())
    {
        auto linearInterpolIndices = sofa::helper::getWriteOnlyAccessor(d_interpolationIndices);
        auto linearInterpolValues = sofa::helper::getWriteOnlyAccessor(d_interpolationValues);

        linearInterpolIndices.clear();
        linearInterpolValues.clear();

        linearInterpolIndices.resize(out.size());
        linearInterpolValues.resize(out.size());
    }

    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = l_topology->getTetrahedra();
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = l_topology->getHexahedra();

    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = l_topology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = l_topology->getQuads();
    sofa::type::vector<Matrix3> bases;
    sofa::type::vector<Vec3> centers;

    if ( tetrahedra.empty() && cubes.empty() )
    {
        if ( triangles.empty() && quads.empty() )
        {
            //no 3D elements, nor 2D elements -> map on 1D elements

            const sofa::core::topology::BaseMeshTopology::SeqEdges& edges = l_topology->getEdges();
            if ( edges.empty() ) return;

            sofa::type::vector< SReal >   lengthEdges;
            sofa::type::vector< Vec3 > unitaryVectors;

            unsigned int e;
            for ( e=0; e<edges.size(); e++ )
            {
                lengthEdges.push_back ( ( (in)[edges[e][1]]-(in)[edges[e][0]] ).norm() );

                Vec3 V12 = ( (in)[edges[e][1]]-(in)[edges[e][0]] ); V12.normalize();
                unitaryVectors.push_back ( V12 );
            }

            for ( unsigned int i=0; i<(out).size(); i++ )
            {
                SReal coef=0;
                for ( e=0; e<edges.size(); e++ )
                {
                    SReal lengthEdge = lengthEdges[e];
                    Vec3 V12 =unitaryVectors[e];

                    coef = ( V12 ) *Vec3 ((out)[i]-(in)[edges[e][0]] ) /lengthEdge;
                    if ( coef >= 0 && coef <= 1 )
                    {
                        addPointInLine ( e, &coef );
                        break;
                    }

                }
                //If no good coefficient has been found, we add to the last element
                if ( e == edges.size() ) addPointInLine ( int(edges.size() - 1),&coef );

            }
        }
        else
        {
            // no 3D elements -> map on 2D elements
            int c0 = int(triangles.size());
            bases.resize ( triangles.size() +quads.size() );
            centers.resize ( triangles.size() +quads.size() );
            for ( unsigned int t = 0; t < triangles.size(); t++ )
            {
                Mat3x3 m,mt;
                m[0] = (in)[triangles[t][1]]-(in)[triangles[t][0]];
                m[1] = (in)[triangles[t][2]]-(in)[triangles[t][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                const bool canInvert = bases[t].invert ( mt );
                assert(canInvert);
                SOFA_UNUSED(canInvert);
                centers[t] = ( (in)[triangles[t][0]]+(in)[triangles[t][1]]+(in)[triangles[t][2]] ) /3;
            }
            for ( unsigned int c = 0; c < quads.size(); c++ )
            {
                Mat3x3 m,mt;
                m[0] = (in)[quads[c][1]]-(in)[quads[c][0]];
                m[1] = (in)[quads[c][3]]-(in)[quads[c][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                const bool canInvert = bases[c0+c].invert ( mt );
                assert(canInvert);
                SOFA_UNUSED(canInvert);
                centers[c0+c] = ( (in)[quads[c][0]]+(in)[quads[c][1]]+(in)[quads[c][2]]+(in)[quads[c][3]] ) *0.25;
            }
            for ( unsigned int i=0; i<(out).size(); i++ )
            {
                auto pos = DataTypes::getCPos((out)[i]);
                type::Vec3 coefs;
                int index = -1;
                SReal distance = 1e10;
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    const auto v = bases[t] * ( pos - (in)[triangles[t][0]] );
                    SReal d = std::max ( std::max ( SReal(-v[0]), SReal(-v[1]) ),std::max (SReal(( v[2]<0?-v[2]:v[2] )-0.01), SReal(v[0]+v[1]-1) ) );
                    if ( d>0 ) d = ( pos-centers[t] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = t; }
                }
                for ( unsigned int c = 0; c < quads.size(); c++ )
                {
                    const auto v = bases[c0+c] * ( pos - (in)[quads[c][0]] );
                    SReal d = std::max ( std::max (SReal(-v[0]), SReal(-v[1]) ),std::max ( std::max (SReal(v[1]-1), SReal(v[0]-1) ),std::max (SReal(v[2]-0.01), SReal(-v[2]-0.01 )) ) );
                    if ( d>0 ) d = ( pos-centers[c0+c] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = c0+c; }
                }
                if ( index < c0 )
                {
                    addPointInTriangle ( index, coefs.ptr(),i );
                }
                else
                {
                    addPointInQuad(index - c0, coefs.ptr());
                }
            }
        }
    }
    else
    {
        int c0 = int(tetrahedra.size());
        bases.resize ( tetrahedra.size() +cubes.size() );
        centers.resize ( tetrahedra.size() +cubes.size() );
        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            Mat3x3 m,mt;
            m[0] = (in)[tetrahedra[t][1]]-(in)[tetrahedra[t][0]];
            m[1] = (in)[tetrahedra[t][2]]-(in)[tetrahedra[t][0]];
            m[2] = (in)[tetrahedra[t][3]]-(in)[tetrahedra[t][0]];
            mt.transpose ( m );
            const bool canInvert = bases[t].invert ( mt );
            assert(canInvert);
            SOFA_UNUSED(canInvert);
            centers[t] = ( (in)[tetrahedra[t][0]]+(in)[tetrahedra[t][1]]+(in)[tetrahedra[t][2]]+(in)[tetrahedra[t][3]] ) *0.25;
        }
        for ( unsigned int c = 0; c < cubes.size(); c++ )
        {
            Mat3x3 m,mt;
            m[0] = (in)[cubes[c][1]]-(in)[cubes[c][0]];
            m[1] = (in)[cubes[c][3]]-(in)[cubes[c][0]];
            m[2] = (in)[cubes[c][4]]-(in)[cubes[c][0]];
            mt.transpose ( m );
            const bool canInvert = bases[c0+c].invert ( mt );
            assert(canInvert);
            SOFA_UNUSED(canInvert);
            centers[c0+c] = ( (in)[cubes[c][0]]+(in)[cubes[c][1]]+(in)[cubes[c][2]]+(in)[cubes[c][3]]+(in)[cubes[c][4]]+(in)[cubes[c][5]]+(in)[cubes[c][6]]+(in)[cubes[c][7]] ) *0.125;
        }
        for ( unsigned int i=0; i<(out).size(); i++ )
        {
            auto pos = DataTypes::getCPos((out)[i]);
            sofa::type::Vec3 coefs;
            int index = -1;
            double distance = 1e10;
            for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
            {
                const auto v = bases[t] * ( pos - (in)[tetrahedra[t][0]] );
                SReal d = std::max ( std::max (SReal (-v[0]), SReal(-v[1]) ),std::max (SReal(-v[2]), SReal(v[0]+v[1]+v[2]-1 )) );
                if ( d>0 ) d = ( pos-centers[t] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = t; }
            }
            for ( unsigned int c = 0; c < cubes.size(); c++ )
            {
                const auto v = bases[c0+c] * ( pos - (in)[cubes[c][0]] );
                SReal d = std::max ( std::max (SReal(-v[0]), SReal(-v[1]) ),std::max ( std::max (SReal(-v[2]), SReal(v[0]-1) ),std::max (SReal(v[1]-1), SReal(v[2]-1 )) ) );
                if ( d>0 ) d = ( pos-centers[c0+c] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = c0+c; }
            }
            if ( index < c0 )
                addPointInTetra ( index, coefs.ptr() , i);
            else
                addPointInCube ( index-c0, coefs.ptr() );
        }
    }

    d_barycentricPositions.endEdit();
    d_tableElements.endEdit();

    if(d_bComputeLinearInterpolation.getValue())
    {
        d_interpolationIndices.endEdit();
        d_interpolationValues.endEdit();
    }

}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::draw(const core::visual::VisualParams* )
{


}


template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInLine(const sofa::Index /*lineIndex*/, const SReal* /*baryCoords*/)
{
    msg_error() << "addPointInLine not implemented";

}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInTriangle(const sofa::Index triangleIndex, const SReal* baryCoords,  const sofa::Index pointIndex)
{
    auto baryPos = sofa::helper::getWriteOnlyAccessor(d_barycentricPositions);
    auto tableElts = sofa::helper::getWriteOnlyAccessor(d_tableElements);

    tableElts[pointIndex] = triangleIndex;
    baryPos[pointIndex][0] = Real(baryCoords[0]);
    baryPos[pointIndex][1] = Real(baryCoords[1]);


    if(d_bComputeLinearInterpolation.getValue())
    {
        const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = l_topology->getTriangles();

        if (triangles.size() == 0)
        {
            return;
        }

        auto linearInterpolIndices = sofa::helper::getWriteOnlyAccessor(d_interpolationIndices);
        auto linearInterpolValues = sofa::helper::getWriteOnlyAccessor(d_interpolationValues);

        // node0
        linearInterpolIndices[pointIndex].push_back(triangles[triangleIndex][0]);
        Real value = Real(1.0) - Real(baryCoords[0]-baryCoords[1]);
        linearInterpolValues[pointIndex].push_back(value);

        // node1
        linearInterpolIndices[pointIndex].push_back(triangles[triangleIndex][1]);
        linearInterpolValues[pointIndex].push_back(Real(baryCoords[0]));

        // node2
        linearInterpolIndices[pointIndex].push_back(triangles[triangleIndex][2]);
        linearInterpolValues[pointIndex].push_back(Real(baryCoords[1]));


    }



}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInQuad(const sofa::Index /*quadIndex*/, const SReal* /*baryCoords*/)
{
    msg_error() << "addPointInQuad not implemented";
}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInTetra(const sofa::Index tetraIndex, const SReal* baryCoords, const sofa::Index pointIndex)
{
    auto baryPos = sofa::helper::getWriteOnlyAccessor(d_barycentricPositions);
    auto tableElts = sofa::helper::getWriteOnlyAccessor(d_tableElements);

    tableElts[pointIndex] = tetraIndex;
    baryPos[pointIndex][0] = Real(baryCoords[0]);
    baryPos[pointIndex][1] = Real(baryCoords[1]);
    baryPos[pointIndex][2] = Real(baryCoords[2]);

    if(d_bComputeLinearInterpolation.getValue())
    {
        const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = l_topology->getTetrahedra();

        if (tetrahedra.size() == 0)
        {
            return;
        }

        auto linearInterpolIndices = sofa::helper::getWriteOnlyAccessor(d_interpolationIndices);
        auto linearInterpolValues = sofa::helper::getWriteOnlyAccessor(d_interpolationValues);

        // node0
        linearInterpolIndices[pointIndex].push_back(tetrahedra[tetraIndex][0]);
        Real value = Real(1.0) - Real(baryCoords[0]-baryCoords[1]-baryCoords[2]);
        linearInterpolValues[pointIndex].push_back(value);

        // node1
        linearInterpolIndices[pointIndex].push_back(tetrahedra[tetraIndex][1]);
        linearInterpolValues[pointIndex].push_back(Real(baryCoords[0]));

        // node2
        linearInterpolIndices[pointIndex].push_back(tetrahedra[tetraIndex][2]);
        linearInterpolValues[pointIndex].push_back(Real(baryCoords[1]));

        // node3
        linearInterpolIndices[pointIndex].push_back(tetrahedra[tetraIndex][3]);
        linearInterpolValues[pointIndex].push_back(Real(baryCoords[2]));

    }


}

template <class DataTypes>
void MeshBarycentricMapperEngine<DataTypes>::addPointInCube(const sofa::Index /*cubeIndex*/, const SReal* /*baryCoords*/)
{
    msg_error() << "addPointInCube not implemented";
}

} //namespace sofa::component::engine::generate
