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

#include <sofa/component/mapping/linear/BarycentricMappingRigid.h>
#include <sofa/component/mapping/linear/BarycentricMapping.inl>

#include <sofa/helper/decompose.h>

namespace sofa::component::mapping::linear
{

template <class In, class Out>
BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::BarycentricMapperTetrahedronSetTopologyRigid(core::topology::BaseMeshTopology* fromTopology, core::topology::BaseMeshTopology* _toTopology)
    : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
      d_map(initData(&d_map, "map", "mapper data")),
      d_mapOrient(initData(&d_mapOrient, "mapOrient", "mapper data for mapped frames")),
      matrixJ(nullptr),
      updateJ(true)
{
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::clear (std::size_t reserve )
{

    type::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.clear();
    if ( reserve>0 )
        vectorData.reserve ( reserve );
    d_map.endEdit();

    type::vector<MappingOrientData>& vectorOrientData = *(d_mapOrient.beginEdit());
    vectorOrientData.clear();
    if ( reserve>0 )
        vectorOrientData.reserve ( reserve );

    d_mapOrient.endEdit();
}

template <class In, class Out>
typename BarycentricMapperTetrahedronSetTopologyRigid<In, Out>::Index
BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::addPointInTetra ( const Index tetraIndex, const SReal* baryCoords )
{
    type::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize (d_map.getValue().size() + 1 );
    MappingData& data = *vectorData.rbegin();
    d_map.endEdit();
    data.in_index = tetraIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return d_map.getValue().size() - 1;
}

template<class In, class Out>
typename BarycentricMapperTetrahedronSetTopologyRigid<In, Out>::Index
BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::addPointOrientationInTetra( const Index tetraIndex, const sofa::type::Matrix3 baryCoorsOrient )
{
    //storing the frame in 3 maps: one direction vector in one map  (3 coor. elements inside a map)
    // IPTR_BARCPP_ADDOR("addPointOrientation BEGIN" << endl);
    type::vector<MappingOrientData>& vectorData = *(d_mapOrient.beginEdit());
    vectorData.resize ( vectorData.size() +1 );
    MappingOrientData& data = *vectorData.rbegin();
    for (unsigned int dir = 0; dir < 3; dir++)
    {
        data[dir].in_index = tetraIndex;
        // IPTR_BARCPP_ADDOR("baryCoords of vector["<<dir<<"]: ");
        for (unsigned int coor = 0; coor < 3; coor++)
        {
            data[dir].baryCoords[coor] = ( Real ) baryCoorsOrient(coor,dir);
            //IPNTR_BARCPP_ADDOR(data[dir].baryCoords[coor] << " ");
        }
        //IPNTR_BARCPP_ADDOR(endl);

    }
    d_mapOrient.endEdit();
    // IPTR_BARCPP_ADDOR("addPointOrientation END" << endl);
    return d_map.getValue().size() - 1;
}


template<class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    const auto& tetrahedra = this->m_fromTopology->getTetrahedra();

    sofa::type::vector<sofa::type::Matrix3> bases;
    sofa::type::vector<sofa::type::Vec3> centers;

    clear ( out.size() );
    bases.resize ( tetrahedra.size() );
    centers.resize ( tetrahedra.size() );
    for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
    {
        sofa::type::Mat3x3d m,mt;
        m[0] = in[tetrahedra[t][1]]-in[tetrahedra[t][0]];
        m[1] = in[tetrahedra[t][2]]-in[tetrahedra[t][0]];
        m[2] = in[tetrahedra[t][3]]-in[tetrahedra[t][0]];
        mt.transpose ( m );
        const bool canInvert = bases[t].invert ( mt );
        assert(canInvert);
        SOFA_UNUSED(canInvert);
        centers[t] = ( in[tetrahedra[t][0]]+in[tetrahedra[t][1]]+in[tetrahedra[t][2]]+in[tetrahedra[t][3]] ) *0.25;
    }

    //find the closest tetra for given points of beam
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        const auto pos = out[i].getCenter(); // Rigid3dTypes::GetCPos(out[i]); // pos = defaulttype::Rigid3dType::getCPos(out[i]);

        //associate the point with the tetrahedron, point in Barycentric coors wrt. the closest tetra, store to an associated structure
        sofa::type::Vec3 coefs;
        int index = -1;
        SReal distance = 1e10;
        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            const auto v = bases[t] * ( pos - in[tetrahedra[t][0]] );
            SReal d = std::max(std::max(SReal(-v[0]), SReal(-v[1])), std::max(SReal(-v[2]), SReal(v[0] + v[1] + v[2] - 1)));
            if ( d>0 ) d = ( pos-centers[t] ).norm2();
            if ( d<distance )
            {
                coefs = v; distance = d; index = t;
            }
        }

        //convert the orientation to basis given by closest tetrahedron
        sofa::type::Quat<SReal> quatA = out[i].getOrientation();
        //initRigidOrientation[i]=quatA;
        sofa::type::Matrix3 orientationMatrix, orientationMatrixBary;
        quatA.toMatrix(orientationMatrix);    //direction vectors stored as columns
        orientationMatrix.transpose(); //to simplify the multiplication below
        for (unsigned c=0; c < 3; c++)
        {
            orientationMatrixBary[c]=bases[index]*orientationMatrix[c];
            orientationMatrixBary[c].normalize();
        }
        orientationMatrixBary.transpose();  //to get the directions as columns

        //store the point and orientation in local coordinates of tetrahedra frame
        addPointInTetra ( index, coefs.ptr() );
        addPointOrientationInTetra(index, orientationMatrixBary);
    }
}



template<class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::resize( core::State<Out>* toModel )
{
    toModel->resize(d_map.getValue().size());
}

template<class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    actualTetraPosition=in;
    //get number of point being mapped
    const sofa::type::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->m_fromTopology->getTetrahedra();
    const sofa::type::vector<MappingData >& map = this->d_map.getValue();
    const sofa::type::vector<MappingOrientData >& mapOrient = this->d_mapOrient.getValue();

//    typename In::VecCoord inCopy = in;

    for ( unsigned int i=0; i<map.size(); i++ )
    {
        //get barycentric coors for given point (node of a beam in this case)
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        const int index = map[i].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];

        sofa::type::Vec3 rotatedPosition= in[tetra[0]] * ( 1-fx-fy-fz ) + in[tetra[1]] * fx + in[tetra[2]] * fy + in[tetra[3]] * fz ;
        Out::setCPos(out[i] , rotatedPosition); // glPointPositions[i] );
    }

    sofa::type::vector< sofa::type::Mat<12,3> > rotJ;
    rotJ.resize(map.size());
    //point running over each DoF (assoc. with frame) in the out vector; get it from the d_mapOrient[0]
    for (unsigned int point = 0; point < mapOrient.size(); point++)
    {
        const int index = mapOrient[point][0].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];

        //compute the rotation of the rigid point using the "basis" approach
        sofa::type::Matrix3 orientationMatrix, polarMatrixQ; // orthogMatrix
        sofa::type::Matrix3 m,basis;
        m[0] = in[tetra[1]]-in[tetra[0]];
        m[1] = in[tetra[2]]-in[tetra[0]];
        m[2] = in[tetra[3]]-in[tetra[0]];
        basis.transpose ( m );

        for (unsigned int dir = 0; dir < 3; dir++)   //go through the three maps
        {
            sofa::type::Vec3 inGlobal;
            inGlobal[0] = mapOrient[point][dir].baryCoords[0];
            inGlobal[1] = mapOrient[point][dir].baryCoords[1];
            inGlobal[2] = mapOrient[point][dir].baryCoords[2];

            orientationMatrix[dir]= basis*inGlobal;
        }

        orientationMatrix.transpose();
        helper::Decompose<sofa::type::Matrix3::Real>::polarDecomposition(orientationMatrix, polarMatrixQ);
        sofa::type::Quat<SReal> quatA;
        quatA.fromMatrix(polarMatrixQ);
        Out::setCRot(out[point], quatA);
    }
} //apply





template<class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(d_map.getValue().size() );

    const sofa::type::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->m_fromTopology->getTetrahedra();
    const sofa::type::vector<MappingData >& map = this->d_map.getValue();
    // TODO: use d_mapOrient
    //const sofa::type::vector<MappingOrientData >& d_mapOrient = this->d_mapOrient.getValue();

    for( size_t i=0 ; i<map.size() ; ++i)
    {
        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];
        const int index = map[i].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];
        Out::setDPos(out[i] , in[tetra[0]] * ( 1-fx-fy-fz )
                + in[tetra[1]] * fx
                + in[tetra[2]] * fy
                + in[tetra[3]] * fz );

        sofa::type::Vec3 actualDRot(0,0,0);
        for (unsigned int vert = 0; vert < 4; vert++)
        {
            //if (in[tetra[vert]].norm() > 10e-6)
            actualDRot += cross(actualTetraPosition[tetra[vert]], in[tetra[vert]]);
        }

        Out::setDRot(out[i], actualDRot);
    }

} //applyJ

template<class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::type::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->m_fromTopology->getTetrahedra();
    const sofa::type::vector<MappingData >& map = this->d_map.getValue();
    typename core::behavior::MechanicalState<Out>* mechanicalObject;
    this->getContext()->get(mechanicalObject);

//    const typename  Out::VecCoord& pX =mechanicalObject->read(core::vec_id::read_access::position)->getValue();

    // TODO: use d_mapOrient
    //const sofa::type::vector<MappingOrientData >& d_mapOrient = this->d_mapOrient.getValue();

    actualPos.clear();
    actualPos.resize(map.size());

    for( size_t i=0 ; i<map.size() ; ++i)
    {
        const defaulttype::Rigid3Types::DPos v = defaulttype::Rigid3Types::getDPos(in[i]);
        const OutReal fx = ( OutReal ) map[i].baryCoords[0];
        const OutReal fy = ( OutReal ) map[i].baryCoords[1];
        const OutReal fz = ( OutReal ) map[i].baryCoords[2];
        const int index = map[i].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];
        out[tetra[0]] += v * ( 1-fx-fy-fz );
        out[tetra[1]] += v * fx;
        out[tetra[2]] += v * fy;
        out[tetra[3]] += v * fz;

        //compute the linear forces for each vertex from the torque, inspired by rigid mapping
        sofa::type::Vec3 torque = getVOrientation(in[i]);
        //if (torque.norm() > 10e-6) {
        for (unsigned int ti = 0; ti<4; ti++)
            out[tetra[ti]] -= cross(actualTetraPosition[tetra[ti]],torque);
    }


    actualOut.clear();
    actualOut.resize(out.size());
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = 0; j < out[i].size(); j++)
            actualOut[i][j] = 0.1f*out[i][j];

}  //applyJT

template<class In, class Out>
const sofa::linearalgebra::BaseMatrix* BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::getJ(int outSize, int inSize)
{
    if (outSize > 0 && d_map.getValue().size() == 0)
    {
        msg_error() << "Maps not created yet" ;
        return nullptr; // error: maps not yet created ?
    }
    if (!matrixJ)
    {
        matrixJ = new MatrixType;
    }

    if (matrixJ->rowBSize() != (MatrixTypeIndex)outSize || matrixJ->colBSize() != (MatrixTypeIndex)inSize)
    {
        matrixJ->resize(outSize*NOut, inSize*NIn);
    }
    else
        matrixJ->clear();

    const sofa::type::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->m_fromTopology->getTetrahedra();
    const sofa::type::vector<MappingData >& map = this->d_map.getValue();

    // TODO(dmarchal 2017-05-03) who do it & when it will be done. Otherwise I will delete that one day.
    // TODO: use d_mapOrient
    //const sofa::type::vector<MappingOrientData >& d_mapOrient = this->d_mapOrient.getValue();

    for (unsigned int beamNode = 0; beamNode < map.size(); beamNode++)
    {
        //linear forces
        const OutReal fx = ( OutReal ) map[beamNode].baryCoords[0];
        const OutReal fy = ( OutReal ) map[beamNode].baryCoords[1];
        const OutReal fz = ( OutReal ) map[beamNode].baryCoords[2];


        const int index = map[beamNode].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];

        for (int dim = 0; dim < 3; dim++)
        {
            matrixJ->add(beamNode*6+dim, 3*tetra[0]+dim, 1-fx-fy-fz);
            matrixJ->add(beamNode*6+dim, 3*tetra[1]+dim, fx);
            matrixJ->add(beamNode*6+dim, 3*tetra[2]+dim, fy);
            matrixJ->add(beamNode*6+dim, 3*tetra[3]+dim, fz);
        }

        for (int vert = 0; vert < 4; vert++)
        {
            sofa::type::Vec3 v;
            for (size_t dim = 0; dim < 3; dim++)
                v[dim] = actualTetraPosition[tetra[vert]][dim] - actualPos[beamNode][dim];
            matrixJ->add(beamNode*6+3, 3*tetra[vert]+1, -v[2]);
            matrixJ->add(beamNode*6+3, 3*tetra[vert]+2, +v[1]);
            matrixJ->add(beamNode*6+4, 3*tetra[vert]+0, +v[2]);
            matrixJ->add(beamNode*6+4, 3*tetra[vert]+2, -v[0]);
            matrixJ->add(beamNode*6+5, 3*tetra[vert]+0, -v[1]);
            matrixJ->add(beamNode*6+5, 3*tetra[vert]+1, +v[0]);
        }
    }

    matrixJ->compress();
    updateJ = false;

    return matrixJ;
} // getJ


template <class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const sofa::type::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->m_fromTopology->getTetrahedra();
    const sofa::type::vector<MappingData >& map = this->d_map.getValue();
    // TODO(dmarchal 2017-05-03) who do it & when it will be done. Otherwise I will delete that one day.
    // TODO: use d_mapOrient
    //const sofa::type::vector<MappingOrientData >& d_mapOrient = this->d_mapOrient.getValue();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const OutReal fx = ( OutReal ) map[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) map[indexIn].baryCoords[1];
                const OutReal fz = ( OutReal ) map[indexIn].baryCoords[2];
                const int index = map[indexIn].in_index;
                const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];

                o.addCol (tetra[0], data * ( 1-fx-fy-fz ) );
                o.addCol (tetra[1], data * fx );
                o.addCol (tetra[2], data * fy );
                o.addCol (tetra[3], data * fz );
            }
        }
    }
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopologyRigid<In,Out>::draw  (const core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::type::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->m_fromTopology->getTetrahedra();
    const sofa::type::vector<MappingData >& map = this->d_map.getValue();
    // TODO(dmarchal 2017-05-03) who do it & when it will be done. Otherwise I will delete that one day.
    // TODO: use d_mapOrient
    //const sofa::type::vector<MappingOrientData >& d_mapOrient = this->d_mapOrient.getValue();

    std::vector< sofa::type::Vec3 > points;
    {
        for ( unsigned int i=0; i<map.size(); i++ )
        {
            const Real fx = map[i].baryCoords[0];
            const Real fy = map[i].baryCoords[1];
            const Real fz = map[i].baryCoords[2];
            const int index = map[i].in_index;
            const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];
            Real f[4];
            f[0] = ( 1-fx-fy-fz );
            f[1] = fx;
            f[2] = fy;
            f[3] = fz;
            for ( int j=0; j<4; j++ )
            {
                if ( f[j]<=-0.0001 || f[j]>=0.0001 )
                {
                    //                     glColor3f((float)f[j],1,(float)f[j]);
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[tetra[j]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::type::RGBAColor ( 0,1,0,1 ) );

    points.clear();
    std::vector< sofa::type::Vec3 > tetraPoints;
    std::vector< sofa::type::Vec3 > tetraLines;

    for ( unsigned int i=0; i<map.size(); i++ )
    {
        //get velocity of the DoF
        //const defaulttype::Rigid3dTypes::DPos v = defaulttype::Rigid3dTypes::getDPos(in[i]);

        //get its coordinated wrt to the associated tetra with given index
        //const OutReal fx = ( OutReal ) map[i].baryCoords[0];
        //const OutReal fy = ( OutReal ) map[i].baryCoords[1];
        //const OutReal fz = ( OutReal ) map[i].baryCoords[2];
        const int index = map[i].in_index;
        const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetrahedra[index];

        for (size_t i = 0; i < actualPos.size(); i++)
            points.push_back(sofa::type::Vec3(actualPos[i][0],actualPos[i][1],actualPos[i][2]));

        for (unsigned int ti = 0; ti<4; ti++)
        {
            const typename In::Coord& tp0 = actualTetraPosition[tetra[ti]];
            typename In::Coord tp1 = actualTetraPosition[tetra[ti]]+actualOut[tetra[ti]];

            tetraPoints.push_back(sofa::type::Vec3(tp0[0],tp0[1],tp0[2]));

            tetraLines.push_back(sofa::type::Vec3(tp0[0],tp0[1],tp0[2]));
            tetraLines.push_back(sofa::type::Vec3(tp1[0],tp1[1],tp1[2]));
        }

        vparams->drawTool()->drawPoints ( points, 10, sofa::type::RGBAColor::red());
        vparams->drawTool()->drawPoints ( tetraPoints, 10, sofa::type::RGBAColor::magenta() );
        vparams->drawTool()->drawLines ( tetraLines, 3.0, sofa::type::RGBAColor::magenta() );
    }
}

} // namespace sofa::component::mapping::linear
