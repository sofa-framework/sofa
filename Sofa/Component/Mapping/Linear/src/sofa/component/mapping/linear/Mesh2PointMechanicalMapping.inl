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
#include <sofa/component/mapping/linear/Mesh2PointMechanicalMapping.h>

#include <sofa/component/mapping/linear/Mesh2PointTopologicalMapping.h>


namespace sofa::component::mapping::linear
{


template <class TIn, class TOut>
Mesh2PointMechanicalMapping<TIn, TOut>::Mesh2PointMechanicalMapping(core::State<In>* from, core::State<Out>* to)
    : Inherit(from, to)
    , l_topologicalMapping(initLink("topologicalMapping", "Link to a Mesh2PointTopologicalMapping"))
    , l_inputTopology(initLink("inputTopology", "Link to the input topology"))
    , l_outputTopology(initLink("outputTopology", "Link to the output topology"))
{
}

template <class TIn, class TOut>
Mesh2PointMechanicalMapping<TIn, TOut>::~Mesh2PointMechanicalMapping()
{
}

template <class TIn, class TOut>
void Mesh2PointMechanicalMapping<TIn, TOut>::init()
{
    this->Inherit::init();

    if (!l_topologicalMapping)
    {
        l_topologicalMapping.set(this->getContext()->template get<Mesh2PointTopologicalMapping>());
        if (!l_topologicalMapping)
        {
            msg_error() << "Cannot find a component " << l_topologicalMapping->getClassName() << " in the current context";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    if (!l_inputTopology)
    {
        l_inputTopology.set(this->fromModel->getContext()->getMeshTopology());
        if (!l_inputTopology)
        {
            msg_error() << "Cannot find a component " << l_inputTopology->getClassName() << " in the context of " << this->fromModel->getPathName();
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    if (!l_outputTopology)
    {
        l_outputTopology.set(this->toModel->getContext()->getMeshTopology());
        if (!l_outputTopology)
        {
            msg_error() << "Cannot find a component " << l_outputTopology->getClassName() << " in the context of " << this->toModel->getPathName();
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template <class TIn, class TOut>
void Mesh2PointMechanicalMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    if (!l_topologicalMapping) return;

    using sofa::InvalidID;

    helper::WriteAccessor< Data<OutVecCoord> > out = dOut;
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;

    const auto& pointMap = l_topologicalMapping->getPointsMappedFromPoint();
    const auto& edgeMap = l_topologicalMapping->getPointsMappedFromEdge();
    const auto& triangleMap = l_topologicalMapping->getPointsMappedFromTriangle();
    const auto& quadMap = l_topologicalMapping->getPointsMappedFromQuad();
    const auto& tetraMap = l_topologicalMapping->getPointsMappedFromTetra();
    const auto& hexaMap = l_topologicalMapping->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = l_inputTopology->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = l_inputTopology->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = l_inputTopology->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = l_inputTopology->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = l_inputTopology->getHexahedra();

    out.resize(l_outputTopology->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            if (pointMap[i][j] == InvalidID) continue;
            out[pointMap[i][j]] = in[i]+l_topologicalMapping->getPointBaryCoords()[j];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            if (edgeMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getEdgeBaryCoords()[j][0];
            out[edgeMap[i][j]] = in[ edges[i][0] ] * (1-fx)
                    +in[ edges[i][1] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
            if (triangleMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getTriangleBaryCoords()[j][0];
            double fy = l_topologicalMapping->getTriangleBaryCoords()[j][1];
            out[triangleMap[i][j]] = in[ triangles[i][0] ] * (1-fx-fy)
                    + in[ triangles[i][1] ] * fx
                    + in[ triangles[i][2] ] * fy;
        }
    }

    for(unsigned int i = 0; i < quadMap.size(); ++i)
    {
        for(unsigned int j = 0; j < quadMap[i].size(); ++j)
        {
            if (quadMap[i][j] == InvalidID) continue;
            const double fx = l_topologicalMapping->getQuadBaryCoords()[j][0];
            const double fy = l_topologicalMapping->getQuadBaryCoords()[j][1];
            out[quadMap[i][j]] = in[ quads[i][0] ] * ((1-fx) * (1-fy))
                    + in[ quads[i][1] ] * ((  fx) * (1-fy))
                    + in[ quads[i][2] ] * ((1-fx) * (  fy))
                    + in[ quads[i][3] ] * ((  fx) * (  fy));
        }
    }

    for(unsigned int i = 0; i < tetraMap.size(); ++i)
    {
        for(unsigned int j = 0; j < tetraMap[i].size(); ++j)
        {
            if (tetraMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getTetraBaryCoords()[j][0];
            double fy = l_topologicalMapping->getTetraBaryCoords()[j][1];
            double fz = l_topologicalMapping->getTetraBaryCoords()[j][2];
            out[tetraMap[i][j]] = in[ tetrahedra[i][0] ] * (1-fx-fy-fz)
                    + in[ tetrahedra[i][1] ] * fx
                    + in[ tetrahedra[i][2] ] * fy
                    + in[ tetrahedra[i][3] ] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            if (hexaMap[i][j] == InvalidID) continue;
            const double fx = l_topologicalMapping->getHexaBaryCoords()[j][0];
            const double fy = l_topologicalMapping->getHexaBaryCoords()[j][1];
            const double fz = l_topologicalMapping->getHexaBaryCoords()[j][2];
            out[hexaMap[i][j]] = in[ hexahedra[i][0] ] * ((1-fx) * (1-fy) * (1-fz))
                    + in[ hexahedra[i][1] ] * ((  fx) * (1-fy) * (1-fz))
					+ in[ hexahedra[i][3] ] * ((1-fx) * (  fy) * (1-fz))
					+ in[ hexahedra[i][2] ] * ((  fx) * (  fy) * (1-fz))
                    + in[ hexahedra[i][4] ] * ((1-fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][5] ] * ((  fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][6] ] * ((  fx) * (  fy) * (  fz))
                    + in[ hexahedra[i][7] ] * ((1-fx) * (  fy) * (  fz));
        }
    }
}

template <class TIn, class TOut>
void Mesh2PointMechanicalMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if (!l_topologicalMapping) return;

    using sofa::InvalidID;

    helper::WriteAccessor< Data<OutVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > in = dIn;

    const auto& pointMap = l_topologicalMapping->getPointsMappedFromPoint();
    const auto& edgeMap = l_topologicalMapping->getPointsMappedFromEdge();
    const auto& triangleMap = l_topologicalMapping->getPointsMappedFromTriangle();
    const auto& quadMap = l_topologicalMapping->getPointsMappedFromQuad();
    const auto& tetraMap = l_topologicalMapping->getPointsMappedFromTetra();
    const auto& hexaMap = l_topologicalMapping->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = l_inputTopology->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = l_inputTopology->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = l_inputTopology->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = l_inputTopology->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = l_inputTopology->getHexahedra();

    out.resize(l_outputTopology->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            if (pointMap[i][j] == InvalidID) continue;
            out[pointMap[i][j]] = in[i];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            if (edgeMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getEdgeBaryCoords()[j][0];
            out[edgeMap[i][j]] = in[ edges[i][0] ] * (1-fx)
                    +in[ edges[i][1] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
            if (triangleMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getTriangleBaryCoords()[j][0];
            double fy = l_topologicalMapping->getTriangleBaryCoords()[j][1];
            out[triangleMap[i][j]] = in[ triangles[i][0] ] * (1-fx-fy)
                    + in[ triangles[i][1] ] * fx
                    + in[ triangles[i][2] ] * fy;
        }
    }

    for(unsigned int i = 0; i < quadMap.size(); ++i)
    {
        for(unsigned int j = 0; j < quadMap[i].size(); ++j)
        {
            if (quadMap[i][j] == InvalidID) continue;
            const double fx = l_topologicalMapping->getQuadBaryCoords()[j][0];
            const double fy = l_topologicalMapping->getQuadBaryCoords()[j][1];
            out[quadMap[i][j]] = in[ quads[i][0] ] * ((1-fx) * (1-fy))
                    + in[ quads[i][1] ] * ((  fx) * (1-fy))
                    + in[ quads[i][2] ] * ((1-fx) * (  fy))
                    + in[ quads[i][3] ] * ((  fx) * (  fy));
        }
    }

    for(unsigned int i = 0; i < tetraMap.size(); ++i)
    {
        for(unsigned int j = 0; j < tetraMap[i].size(); ++j)
        {
            if (tetraMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getTetraBaryCoords()[j][0];
            double fy = l_topologicalMapping->getTetraBaryCoords()[j][1];
            double fz = l_topologicalMapping->getTetraBaryCoords()[j][2];
            out[tetraMap[i][j]] = in[ tetrahedra[i][0] ] * (1-fx-fy-fz)
                    + in[ tetrahedra[i][1] ] * fx
                    + in[ tetrahedra[i][2] ] * fy
                    + in[ tetrahedra[i][3] ] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            if (hexaMap[i][j] == InvalidID) continue;
            const double fx = l_topologicalMapping->getHexaBaryCoords()[j][0];
            const double fy = l_topologicalMapping->getHexaBaryCoords()[j][1];
            const double fz = l_topologicalMapping->getHexaBaryCoords()[j][2];
            out[hexaMap[i][j]] = in[ hexahedra[i][0] ] * ((1-fx) * (1-fy) * (1-fz))
                    + in[ hexahedra[i][1] ] * ((  fx) * (1-fy) * (1-fz))
					+ in[ hexahedra[i][3] ] * ((1-fx) * (  fy) * (1-fz))
					+ in[ hexahedra[i][2] ] * ((  fx) * (  fy) * (1-fz))
                    + in[ hexahedra[i][4] ] * ((1-fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][5] ] * ((  fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][6] ] * ((  fx) * (  fy) * (  fz))
                    + in[ hexahedra[i][7] ] * ((1-fx) * (  fy) * (  fz));
        }
    }
}

template <class TIn, class TOut>
void Mesh2PointMechanicalMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dOut, const Data<OutVecDeriv>& dIn)
{
    if (!l_topologicalMapping) return;

    using sofa::InvalidID;

    helper::WriteAccessor< Data<InVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<OutVecDeriv> > in = dIn;

    const auto& pointMap = l_topologicalMapping->getPointsMappedFromPoint();
    const auto& edgeMap = l_topologicalMapping->getPointsMappedFromEdge();
    const auto& triangleMap = l_topologicalMapping->getPointsMappedFromTriangle();
    const auto& quadMap = l_topologicalMapping->getPointsMappedFromQuad();
    const auto& tetraMap = l_topologicalMapping->getPointsMappedFromTetra();
    const auto& hexaMap = l_topologicalMapping->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = l_inputTopology->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = l_inputTopology->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = l_inputTopology->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = l_inputTopology->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = l_inputTopology->getHexahedra();

    out.resize(l_inputTopology->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            if (pointMap[i][j] == InvalidID) continue;
            out[i] += in[pointMap[i][j]];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            if (edgeMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getEdgeBaryCoords()[j][0];
            out[edges[i][0]] += in[ edgeMap[i][j] ] * (1-fx);
            out[edges[i][1]] += in[ edgeMap[i][j] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
            if (triangleMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getTriangleBaryCoords()[j][0];
            double fy = l_topologicalMapping->getTriangleBaryCoords()[j][1];
            out[ triangles[i][0] ] += in[triangleMap[i][j]] * (1-fx-fy);
            out[ triangles[i][1] ] += in[triangleMap[i][j]] * fx;
            out[ triangles[i][2] ] += in[triangleMap[i][j]] * fy;
        }
    }

    for(unsigned int i = 0; i < quadMap.size(); ++i)
    {
        for(unsigned int j = 0; j < quadMap[i].size(); ++j)
        {
            if (quadMap[i][j] == InvalidID) continue;
            const double fx = l_topologicalMapping->getQuadBaryCoords()[j][0];
            const double fy = l_topologicalMapping->getQuadBaryCoords()[j][1];
            out[ quads[i][0] ] += in[quadMap[i][j]] * ((1-fx) * (1-fy));
            out[ quads[i][1] ] += in[quadMap[i][j]] * ((  fx) * (1-fy));
            out[ quads[i][2] ] += in[quadMap[i][j]] * ((1-fx) * (  fy));
            out[ quads[i][3] ] += in[quadMap[i][j]] * ((  fx) * (  fy));
        }
    }

    for(unsigned int i = 0; i < tetraMap.size(); ++i)
    {
        for(unsigned int j = 0; j < tetraMap[i].size(); ++j)
        {
            if (tetraMap[i][j] == InvalidID) continue;
            double fx = l_topologicalMapping->getTetraBaryCoords()[j][0];
            double fy = l_topologicalMapping->getTetraBaryCoords()[j][1];
            double fz = l_topologicalMapping->getTetraBaryCoords()[j][2];
            out[ tetrahedra[i][0] ] += in[tetraMap[i][j]] * (1-fx-fy-fz);
            out[ tetrahedra[i][1] ] += in[tetraMap[i][j]] * fx;
            out[ tetrahedra[i][2] ] += in[tetraMap[i][j]] * fy;
            out[ tetrahedra[i][3] ] += in[tetraMap[i][j]] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            if (hexaMap[i][j] == InvalidID) continue;
            const double fx = l_topologicalMapping->getHexaBaryCoords()[j][0];
            const double fy = l_topologicalMapping->getHexaBaryCoords()[j][1];
            const double fz = l_topologicalMapping->getHexaBaryCoords()[j][2];
            out[ hexahedra[i][0] ] += in[hexaMap[i][j]] * ((1-fx) * (1-fy) * (1-fz));
            out[ hexahedra[i][1] ] += in[hexaMap[i][j]] * ((  fx) * (1-fy) * (1-fz));
			out[ hexahedra[i][3] ] += in[hexaMap[i][j]] * ((1-fx) * (  fy) * (1-fz));
			out[ hexahedra[i][2] ] += in[hexaMap[i][j]] * ((  fx) * (  fy) * (1-fz));
            out[ hexahedra[i][4] ] += in[hexaMap[i][j]] * ((1-fx) * (1-fy) * (  fz));
            out[ hexahedra[i][5] ] += in[hexaMap[i][j]] * ((  fx) * (1-fy) * (  fz));
            out[ hexahedra[i][6] ] += in[hexaMap[i][j]] * ((  fx) * (  fy) * (  fz));
            out[ hexahedra[i][7] ] += in[hexaMap[i][j]] * ((1-fx) * (  fy) * (  fz));
        }
    }
}


template <class TIn, class TOut>
void Mesh2PointMechanicalMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& dOut, const Data<OutMatrixDeriv>& dIn)
{
    if (!l_topologicalMapping)
        return;

    const sofa::type::vector< std::pair< Mesh2PointTopologicalMapping::Element, sofa::Index> >& pointSource = l_topologicalMapping->getPointSource();

    if (pointSource.empty())
        return;

    InMatrixDeriv& out = *dOut.beginEdit();
    const OutMatrixDeriv& in = dIn.getValue();

    const core::topology::BaseMeshTopology::SeqEdges& edges = l_inputTopology->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = l_inputTopology->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = l_inputTopology->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = l_inputTopology->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = l_inputTopology->getHexahedra();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                const Index indexUIn = colIt.index();
                const OutDeriv data = colIt.val();
                std::pair< Mesh2PointTopologicalMapping::Element, int> source = pointSource[indexUIn];
                const Index indexIn = indexUIn;

                switch (source.first)
                {
                case Mesh2PointTopologicalMapping::POINT:
                {
                    o.addCol(source.second, data);

                    break;
                }
                case Mesh2PointTopologicalMapping::EDGE:
                {
                    core::topology::BaseMeshTopology::Edge e = edges[source.second];
                    typename In::Deriv f = data;

                    double fx = 0;

                    if (l_topologicalMapping->getEdgeBaryCoords().size() == 1)
                    {
                        fx = l_topologicalMapping->getEdgeBaryCoords()[0][0];
                    }
                    else
                    {
                        const auto& edgeMap = l_topologicalMapping->getPointsMappedFromEdge();
                        bool err = true;
                        for(Size i = 0; i < edgeMap[source.second].size(); ++i)
                        {
                            if (edgeMap[source.second][i] == indexIn)
                            {
                                fx = l_topologicalMapping->getEdgeBaryCoords()[i][0];
                                err = false;
                                break;
                            }
                        }
                        if (err)
                        {
                            msg_error() <<" wrong source edge / destination point association" ;
                        }
                    }

                    o.addCol(e[0], f * (1 - fx));
                    o.addCol(e[1], f * fx);

                    break;
                }
                case Mesh2PointTopologicalMapping::TRIANGLE:
                {
                    core::topology::BaseMeshTopology::Triangle t = triangles[source.second];
                    typename In::Deriv f = data;

                    double fx = 0;
                    double fy = 0;

                    if (l_topologicalMapping->getTriangleBaryCoords().size() == 1)
                    {
                        fx = l_topologicalMapping->getTriangleBaryCoords()[0][0];
                        fy = l_topologicalMapping->getTriangleBaryCoords()[0][1];
                    }
                    else
                    {
                        const auto& triangleMap = l_topologicalMapping->getPointsMappedFromTriangle();
                        bool err = true;
                        for(Size i = 0; i < triangleMap[source.second].size(); ++i)
                        {
                            if (triangleMap[source.second][i] == indexIn)
                            {
                                fx = l_topologicalMapping->getTriangleBaryCoords()[i][0];
                                fy = l_topologicalMapping->getTriangleBaryCoords()[i][1];
                                err = false;
                                break;
                            }
                        }
                        if (err)
                        {
                            msg_error() << " wrong source triangle / destination point association" ;
                        }
                    }

                    o.addCol(t[0], f * (1 - fx - fy));
                    o.addCol(t[1], f * fx);
                    o.addCol(t[2], f * fy);

                    break;
                }
                case Mesh2PointTopologicalMapping::QUAD:
                {
                    core::topology::BaseMeshTopology::Quad q = quads[source.second];
                    typename In::Deriv f = data;

                    double fx = 0;
                    double fy = 0;

                    if (l_topologicalMapping->getQuadBaryCoords().size() == 1)
                    {
                        fx = l_topologicalMapping->getQuadBaryCoords()[0][0];
                        fy = l_topologicalMapping->getQuadBaryCoords()[0][1];
                    }
                    else
                    {
                        const auto& quadMap = l_topologicalMapping->getPointsMappedFromQuad();
                        bool err = true;
                        for(Size i = 0; i < quadMap[source.second].size(); ++i)
                        {
                            if (quadMap[source.second][i] == indexIn)
                            {
                                fx = l_topologicalMapping->getQuadBaryCoords()[i][0];
                                fy = l_topologicalMapping->getQuadBaryCoords()[i][1];
                                err = false;
                                break;
                            }
                        }
                        if (err)
                        {
                            msg_error() << " wrong source quad / destination point association" ;
                        }
                    }

                    o.addCol(q[0], f * ((1-fx) * (1-fy)));
                    o.addCol(q[1], f * (fx * (1-fy)));
					o.addCol(q[2], f * ((1-fx) * fy));
					o.addCol(q[3], f * (fx * fy));

                    break;
                }
                case Mesh2PointTopologicalMapping::TETRA:
                {
                    core::topology::BaseMeshTopology::Tetra t = tetrahedra[source.second];
                    typename In::Deriv f = data;

                    double fx = 0;
                    double fy = 0;
                    double fz = 0;

                    if (l_topologicalMapping->getTetraBaryCoords().size() == 1)
                    {
                        fx = l_topologicalMapping->getTetraBaryCoords()[0][0];
                        fy = l_topologicalMapping->getTetraBaryCoords()[0][1];
                        fz = l_topologicalMapping->getTetraBaryCoords()[0][2];
                    }
                    else
                    {
                        const auto& tetraMap = l_topologicalMapping->getPointsMappedFromTetra();
                        bool err = true;
                        for(Size i = 0; i < tetraMap[source.second].size(); ++i)
                        {
                            if (tetraMap[source.second][i] == indexIn)
                            {
                                fx = l_topologicalMapping->getTetraBaryCoords()[i][0];
                                fy = l_topologicalMapping->getTetraBaryCoords()[i][1];
                                fz = l_topologicalMapping->getTetraBaryCoords()[i][2];
                                err = false;
                                break;
                            }
                        }
                        if (err)
                        {
                            msg_error() << " wrong source tetra / destination point association" ;
                        }
                    }

                    o.addCol(t[0], f * (1-fx-fy-fz));
                    o.addCol(t[1], f * fx);
					o.addCol(t[2], f * fy);
					o.addCol(t[3], f * fz);

                    break;
                }
                case Mesh2PointTopologicalMapping::HEXA:
                {
                    core::topology::BaseMeshTopology::Hexa h = hexahedra[source.second];
                    typename In::Deriv f = data;

                    double fx = 0;
                    double fy = 0;
                    double fz = 0;
                    if (l_topologicalMapping->getHexaBaryCoords().size() == 1)
                    {
                        fx = l_topologicalMapping->getHexaBaryCoords()[0][0];
                        fy = l_topologicalMapping->getHexaBaryCoords()[0][1];
                        fz = l_topologicalMapping->getHexaBaryCoords()[0][2];
                    }
                    else
                    {
                        const auto& hexaMap = l_topologicalMapping->getPointsMappedFromHexa();
                        bool err = true;
                        for(Size i = 0; i < hexaMap[source.second].size(); ++i)
                        {
                            if (hexaMap[source.second][i] == indexIn)
                            {
                                fx = l_topologicalMapping->getHexaBaryCoords()[i][0];
                                fy = l_topologicalMapping->getHexaBaryCoords()[i][1];
                                fz = l_topologicalMapping->getHexaBaryCoords()[i][2];
                                err = false;
                                break;
                            }
                        }
                        if (err)
                        {
                            msg_error() << " wrong source hexa / destination point association" ;
                        }
                    }

                    const double oneMinFx = 1 - fx;
                    const double oneMinFy = 1 - fy;
                    const double oneMinFz = 1 - fz;

                    o.addCol(h[0] , f * oneMinFx * oneMinFy * oneMinFz);
                    o.addCol(h[1] , f * fx * oneMinFy * oneMinFz);
					o.addCol(h[3] , f * oneMinFx * fy * oneMinFz);
					o.addCol(h[2] , f * fx * fy * oneMinFz);
					o.addCol(h[4] , f * oneMinFx * oneMinFy * fz);
					o.addCol(h[5] , f * fx * oneMinFy * fz);
					o.addCol(h[6] , f * fx * fy * fz);
					o.addCol(h[7] , f * oneMinFx * fy * fz);

                    break;
                }
                default:

                    break;
                }

                ++colIt;
            }
        }
    }

    dOut.endEdit();
}

} //namespace sofa::component::mapping::linear
