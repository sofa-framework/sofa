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
#include <sofa/component/engine/transform/SmoothMeshEngine.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::engine::transform
{

template <class DataTypes>
SmoothMeshEngine<DataTypes>::SmoothMeshEngine()
    : input_position( initData (&input_position, "input_position", "Input position") )
    , input_indices( initData (&input_indices, "input_indices", "Position indices that need to be smoothed, leave empty for all positions") )
    , output_position( initData (&output_position, "output_position", "Output position") )
    , nb_iterations( initData (&nb_iterations, (unsigned int)1, "nb_iterations", "Number of iterations of laplacian smoothing") )
    , showInput( initData (&showInput, false, "showInput", "showInput") )
    , showOutput( initData (&showOutput, false, "showOutput", "showOutput") )
    , l_topology(initLink("topology", "link to the topology container"))
    , m_topology(nullptr)
{
    addInput(&input_position);
    addOutput(&output_position);
}

template <class DataTypes>
void SmoothMeshEngine<DataTypes>::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    setDirtyValue();
}

template <class DataTypes>
void SmoothMeshEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void SmoothMeshEngine<DataTypes>::doUpdate()
{
    using sofa::core::topology::BaseMeshTopology;

    if (!m_topology) return;

    helper::ReadAccessor< Data<VecCoord> > in(input_position);
    const helper::ReadAccessor< Data<type::vector<unsigned int > > > indices(input_indices);
    helper::WriteAccessor< Data<VecCoord> > out(output_position);

    out.resize(in.size());
    for (unsigned int i =0; i<in.size();i++) out[i] = in[i];
    
    for (unsigned int n=0; n < nb_iterations.getValue(); n++)
    {
        VecCoord t;
        t.resize(out.size());

        if(!indices.size())
        {
            for (unsigned int i = 0; i < out.size(); i++)
            {
                BaseMeshTopology::VerticesAroundVertex v = m_topology->getVerticesAroundVertex(i);
                if (v.size()>0) {
                    Coord p = Coord();
                    for (unsigned int j = 0; j < v.size(); j++)
                        p += out[v[j]];
                    t[i] = p / v.size();
                }
                else
                    t[i] = out[i];
            }
        }
        else
        {
            // init
            for (unsigned int i = 0; i < out.size(); i++)
            {
                
                t[i] = out[i];
            }            
            for(unsigned int i = 0; i < indices.size(); i++)
            {
                BaseMeshTopology::VerticesAroundVertex v = m_topology->getVerticesAroundVertex(indices[i]);
                if (v.size()>0) {
                    Coord p = Coord();
                    for (unsigned int j = 0; j < v.size(); j++)
                        p += out[v[j]];
                    t[indices[i]] = p / v.size();
                }
            }
        }
        for (unsigned int i = 0; i < out.size(); i++) out[i] = t[i];
    }

}

/*
 * @details Uses only the coordinates of the input mesh because the laplacian will approximate the
 * initial geometry. Thus, the output coordinates are "inside" the input one's
 */
template<class DataTypes>
void SmoothMeshEngine<DataTypes>::computeBBox(const core::ExecParams*, bool onlyVisible)
{
	if( !onlyVisible ) return;

	helper::ReadAccessor< Data<VecCoord> > x(input_position);

    type::BoundingBox bbox;
    for (const auto& p : x )
    {
        bbox.include(p);
    }

    this->f_bbox.setValue(bbox);
}

template <class DataTypes>
void SmoothMeshEngine<DataTypes>::draw(const core::visual::VisualParams* vparams)
{    
    if (!vparams->displayFlags().getShowVisualModels() || m_topology == nullptr) return;

    using sofa::type::Vec;
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const bool wireframe=vparams->displayFlags().getShowWireFrame();

    const sofa::core::topology::BaseMeshTopology::SeqTriangles& tri = m_topology->getTriangles();

    vparams->drawTool()->enableLighting();

    if(wireframe)
        vparams->drawTool()->setPolygonMode(0, true);

    if (this->showInput.getValue())
    {
        std::vector<sofa::type::Vec3> vertices;
        helper::ReadAccessor< Data<VecCoord> > in(input_position);

        constexpr sofa::type::RGBAColor color(1.0f, 0.76078431372f, 0.0f, 1.0f);
        vparams->drawTool()->setMaterial(color);

        for (unsigned int i=0; i<tri.size(); ++i)
        {
            const Vec<3,Real>& a = in[ tri[i][0] ];
            const Vec<3,Real>& b = in[ tri[i][1] ];
            const Vec<3,Real>& c = in[ tri[i][2] ];
            vertices.push_back(a);
            vertices.push_back(b);
            vertices.push_back(c);
        }
        vparams->drawTool()->drawTriangles(vertices,color);
    }

    if (this->showOutput.getValue())
    {
        std::vector<sofa::type::Vec3> vertices;
        helper::ReadAccessor< Data<VecCoord> > out(output_position);
        constexpr sofa::type::RGBAColor color(0.0f, 0.6f, 0.8f, 1.0f);

        for (unsigned int i=0; i<tri.size(); ++i)
        {
            const Vec<3,Real>& a = out[ tri[i][0] ];
            const Vec<3,Real>& b = out[ tri[i][1] ];
            const Vec<3,Real>& c = out[ tri[i][2] ];
            vertices.push_back(a);
            vertices.push_back(b);
            vertices.push_back(c);
        }
        vparams->drawTool()->drawTriangles(vertices, color);
    }

    if (wireframe)
        vparams->drawTool()->setPolygonMode(0, false);


}

} //namespace sofa::component::engine::transform
