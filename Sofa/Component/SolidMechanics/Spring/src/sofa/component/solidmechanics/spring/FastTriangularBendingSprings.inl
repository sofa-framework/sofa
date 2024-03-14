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

#include <sofa/component/solidmechanics/spring/FastTriangularBendingSprings.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::component::solidmechanics::spring
{

typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::applyEdgeCreation(Index /*edgeIndex*/, EdgeSpring &ei, const core::topology::BaseMeshTopology::Edge &, const sofa::type::vector<Index> &, const sofa::type::vector<SReal> &)
{
    ei.is_activated=false;
    ei.is_initialized=false;
}



template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::applyTriangleCreation(const sofa::type::vector<Index> &triangleAdded, const sofa::type::vector<core::topology::BaseMeshTopology::Triangle> &, const sofa::type::vector<sofa::type::vector<Index> > &, const sofa::type::vector<sofa::type::vector<SReal> > &)
{
    typename MechanicalState::ReadVecCoord restPosition = this->mstate->readRestPositions();

    helper::WriteOnlyAccessor< Data< type::vector<EdgeSpring > > > edgeData = d_edgeSprings;
        
    for (unsigned int i=0; i<triangleAdded.size(); ++i)
    {
        /// edges of the new triangle
        EdgesInTriangle te2 = this->m_topology->getEdgesInTriangle(triangleAdded[i]);
        /// vertices of the new triangle
        core::topology::BaseMeshTopology::Triangle t2 = this->m_topology->getTriangle(triangleAdded[i]);

		double epsilonSq = d_minDistValidity.getValue();
        const Real& bendingStiffness = d_bendingStiffness.getValue();
		epsilonSq *= epsilonSq;

        // for each edge in the new triangle
        for(unsigned int j=0; j<3; ++j)
        {
            EdgeSpring &ei = edgeData[te2[j]]; // edge spring
            unsigned int edgeIndex = te2[j];

            const auto& shell = this->m_topology->getTrianglesAroundEdge(edgeIndex);
                    
            if (shell.size()==2)   // there is another triangle attached to this edge, so a spring is needed
            {
                // the other triangle and its edges
                EdgesInTriangle te1;
                core::topology::BaseMeshTopology::Triangle t1;
                if(shell[0] == triangleAdded[i])
                {

                    te1 = this->m_topology->getEdgesInTriangle(shell[1]);
                    t1 = this->m_topology->getTriangle(shell[1]);

                }
                else
                {

                    te1 = this->m_topology->getEdgesInTriangle(shell[0]);
                    t1 = this->m_topology->getTriangle(shell[0]);
                }

                const int i1 = this->m_topology->getEdgeIndexInTriangle(te1, edgeIndex); // index of the vertex opposed to the current edge in the other triangle (?)
                const int i2 = this->m_topology->getEdgeIndexInTriangle(te2, edgeIndex); // index of the vertex opposed to the current edge in the new triangle (?)
                core::topology::BaseMeshTopology::Edge edge = this->m_topology->getEdge(edgeIndex);                  // indices of the vertices of the current edge

                const core::topology::BaseMeshTopology::PointID& v1 = t1[i1];
                const core::topology::BaseMeshTopology::PointID& v2 = t2[i2];
                const core::topology::BaseMeshTopology::PointID& e1 = edge[0];
                const core::topology::BaseMeshTopology::PointID& e2 = edge[1];

				Deriv vp = restPosition[v2]-restPosition[v1];
				Deriv ve = restPosition[e2]-restPosition[e1];

				if(vp.norm2()>epsilonSq && ve.norm2()>epsilonSq)
                    ei.setEdgeSpring( restPosition.ref(), v1, v2, e1, e2, bendingStiffness);
            }
            else
                ei.is_activated = ei.is_initialized = false;
        }
    }
}




template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::applyTriangleDestruction(const sofa::type::vector<Index> &triangleRemoved)
{
    typename MechanicalState::ReadVecCoord restPosition = this->mstate->readRestPositions();
    helper::WriteOnlyAccessor< Data< type::vector<EdgeSpring > > > edgeData = d_edgeSprings;
    for (unsigned int i=0; i<triangleRemoved.size(); ++i)
    {
        /// describe the jth edge index of triangle no i
        EdgesInTriangle te = this->m_topology->getEdgesInTriangle(triangleRemoved[i]);
        /// describe the jth vertex index of triangle no i
		double epsilonSq = d_minDistValidity.getValue();
        const Real& bendingStiffness = d_bendingStiffness.getValue();
		epsilonSq *= epsilonSq;

        for(unsigned int j=0; j<3; ++j)
        {
            EdgeSpring &ei = edgeData[te[j]];
            unsigned int edgeIndex = te[j];

            const auto& shell = this->m_topology->getTrianglesAroundEdge(edgeIndex);

            //check if there is going to be only 2 triangles after modification
            bool valid=false;
            std::vector<unsigned int> keepingTri;
            keepingTri.reserve(2);
            if(shell.size()>3)
            {
                unsigned int toSuppr=0;
                for(unsigned int k = 0 ; k < shell.size() ; ++k)
                    if(std::find(triangleRemoved.begin(),triangleRemoved.end(),shell[k])!=triangleRemoved.end())
                        toSuppr++;
                    else
                        keepingTri.push_back(shell[k]);

                if(shell.size()-toSuppr==2)
                    valid=true;
            }
            else if(shell.size()==3)
            {
                valid=true;
                if(shell[0]==triangleRemoved[i])
                {
                    keepingTri.push_back(shell[1]);
                    keepingTri.push_back(shell[2]);
                }
                else if(shell[1]==triangleRemoved[i])
                {
                    keepingTri.push_back(shell[0]);
                    keepingTri.push_back(shell[2]);
                }
                else
                {
                    keepingTri.push_back(shell[0]);
                    keepingTri.push_back(shell[1]);
                }
            }

            //in this case : set a bending spring
            if (valid)
            {
                EdgesInTriangle te1;
                core::topology::BaseMeshTopology::Triangle t1;
                EdgesInTriangle te2;
                core::topology::BaseMeshTopology::Triangle t2;

                te1 = this->m_topology->getEdgesInTriangle(keepingTri[0]);
                t1 = this->m_topology->getTriangle(keepingTri[0]);
                te2 = this->m_topology->getEdgesInTriangle(keepingTri[1]);
                t2 = this->m_topology->getTriangle(keepingTri[1]);

                const int i1 = this->m_topology->getEdgeIndexInTriangle(te1, edgeIndex);
                const int i2 = this->m_topology->getEdgeIndexInTriangle(te2, edgeIndex);

                core::topology::BaseMeshTopology::Edge edge = this->m_topology->getEdge(edgeIndex);

                const core::topology::BaseMeshTopology::PointID& v1 = t1[i1];
                const core::topology::BaseMeshTopology::PointID& v2 = t2[i2];
                const core::topology::BaseMeshTopology::PointID& e1 = edge[0];
                const core::topology::BaseMeshTopology::PointID& e2 = edge[1];

				Deriv vp = restPosition[v2]-restPosition[v1];
				Deriv ve = restPosition[e2]-restPosition[e1];

				if(vp.norm2()>epsilonSq && ve.norm2()>epsilonSq)
                {
                    ei.setEdgeSpring(restPosition.ref(), v1, v2, e1, e2, bendingStiffness);
                }
				else
					ei.is_activated = ei.is_initialized = false;
            }
            else
                ei.is_activated = ei.is_initialized = false;

        }

    }
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::applyPointDestruction(const sofa::type::vector<Index> &tab)
{
    const bool debug_mode = false;

    unsigned int last = this->m_topology->getNbPoints() -1;
    unsigned int i,j;

    helper::WriteOnlyAccessor < Data< type::vector<EdgeSpring > > > edgeInf = d_edgeSprings;

    //make a reverse copy of tab
    sofa::type::vector<unsigned int> lastIndexVec;
    lastIndexVec.reserve(tab.size());
    for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
        lastIndexVec.push_back(last - i_init);

    for ( i = 0; i < tab.size(); ++i)
    {
        unsigned int i_next = i;
        bool is_reached = false;
        while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
        {
            ++i_next;
            is_reached = (lastIndexVec[i_next] == tab[i]);
        }

        if(is_reached)
            lastIndexVec[i_next] = lastIndexVec[i];

        const auto &shell= this->m_topology->getTrianglesAroundVertex(lastIndexVec[i]);
        for (j=0; j<shell.size(); ++j)
        {
            core::topology::BaseMeshTopology::EdgesInTriangle tej = this->m_topology->getEdgesInTriangle(shell[j]);
            for(unsigned int k=0; k < 3 ; ++k)
            {
                unsigned int ind_j = tej[k];
                edgeInf[ind_j].replaceIndex( last, tab[i]);
            }
        }

        if(debug_mode)
        {
            for (unsigned int j_loc=0; j_loc<edgeInf.size(); ++j_loc)
            {
                edgeInf[j_loc].replaceIndex( last, tab[i]);
            }
        }

        --last;
    }
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::applyPointRenumbering(const sofa::type::vector<Index> &newIndices)
{
    helper::WriteOnlyAccessor < Data< type::vector<EdgeSpring> > > edgeInf = d_edgeSprings;
    for (unsigned int i = 0; i < this->m_topology->getNbEdges(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            edgeInf[i].replaceIndices(newIndices);
        }
    }
}


template<class DataTypes>
FastTriangularBendingSprings<DataTypes>::FastTriangularBendingSprings(/*double _ks, double _kd*/)
    : d_bendingStiffness(initData(&d_bendingStiffness,(SReal) 1.0,"bendingStiffness","bending stiffness of the material"))
    , d_minDistValidity(initData(&d_minDistValidity,(SReal) 0.000001,"minDistValidity","Distance under which a spring is not valid"))
    , l_topology(initLink("topology", "link to the topology container"))
    , d_edgeSprings(initData(&d_edgeSprings, "edgeInfo", "Internal edge data"))
{

}

template<class DataTypes>
FastTriangularBendingSprings<DataTypes>::~FastTriangularBendingSprings()
{

}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::init()
{
    this->Inherited::init();

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


    if (m_topology->getNbTriangles()==0)
    {
        msg_warning() << "No triangles found in linked Topology.";
    }
    d_edgeSprings.createTopologyHandler(m_topology);
    d_edgeSprings.linkToPointDataArray();
    d_edgeSprings.linkToTriangleDataArray();

    d_edgeSprings.setCreationCallback([this](Index edgeIndex, EdgeSpring& ei,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyEdgeCreation(edgeIndex, ei, edge, ancestors, coefs);
    });

    d_edgeSprings.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESADDED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::TrianglesAdded* triAdd = static_cast<const core::topology::TrianglesAdded*>(eventTopo);
        applyTriangleCreation(triAdd->getIndexArray(), triAdd->getElementArray(), triAdd->ancestorsList, triAdd->coefs);
    });

    d_edgeSprings.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::TrianglesRemoved* triRemove = static_cast<const core::topology::TrianglesRemoved*>(eventTopo);
        applyTriangleDestruction(triRemove->getArray());
    });

    d_edgeSprings.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRemoved* pRemove = static_cast<const core::topology::PointsRemoved*>(eventTopo);
        applyPointDestruction(pRemove->getArray());
    });

    d_edgeSprings.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSRENUMBERING, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRenumbering* pRenum = static_cast<const core::topology::PointsRenumbering*>(eventTopo);
        applyPointRenumbering(pRenum->getIndexArray());
    });
    
    this->reinit();
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::reinit()
{
    /// prepare to store info in the edge array
    helper::WriteOnlyAccessor< Data< type::vector<EdgeSpring> > > edgeInf = d_edgeSprings;
    edgeInf.resize(m_topology->getNbEdges());

    // set edge tensor to 0
    for (Index i=0; i<m_topology->getNbEdges(); ++i)
    {
        applyEdgeCreation(i, edgeInf[i],
            m_topology->getEdge(i), (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0);
    }

    // create edge tensor by calling the triangle creation function
    sofa::type::vector<Index> triangleAdded;
    for (unsigned int i=0; i<m_topology->getNbTriangles(); ++i)
        triangleAdded.push_back(i);

    applyTriangleCreation(triangleAdded,
        (const sofa::type::vector<core::topology::BaseMeshTopology::Triangle>)0,
        (const sofa::type::vector<sofa::type::vector<Index> >)0,
        (const sofa::type::vector<sofa::type::vector<SReal> >)0);
}

template <class DataTypes>
SReal FastTriangularBendingSprings<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    return m_potentialEnergy;
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();
    typename MechanicalState::WriteVecDeriv f(d_f);
    const type::vector<EdgeSpring>& edgeInf = d_edgeSprings.getValue();
    f.resize(x.size());

    m_potentialEnergy = 0;
    for(unsigned i=0; i<edgeInf.size(); i++ )
    {
        m_potentialEnergy += edgeInf[i].addForce(f.wref(),x,v);
    }
}

template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::addDForce(const core::MechanicalParams* mparams , DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    const VecDeriv& dx = d_dx.getValue();
    typename MechanicalState::WriteVecDeriv df(d_df);
    const type::vector<EdgeSpring>& edgeInf = d_edgeSprings.getValue();
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    df.resize(dx.size());
    for(unsigned i=0; i<edgeInf.size(); i++ )
    {
        edgeInf[i].addDForce(df.wref(),dx,kFactor);
    }
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal scale, unsigned int &offset)
{
    const type::vector<EdgeSpring>& springs = d_edgeSprings.getValue();
    for(unsigned i=0; i< springs.size() ; i++)
    {
        springs[i].addStiffness( mat, offset, scale, this);
    }
}

template <class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    static constexpr auto blockSize = DataTypes::deriv_total_size;
    static constexpr auto spatialDimension = DataTypes::spatial_dimensions;
    sofa::type::Mat<spatialDimension, spatialDimension, Real> localMatrix(type::NOINIT);

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const auto& spring : d_edgeSprings.getValue())
    {
        typename EdgeSpring::StiffnessMatrix K;
        spring.getStiffness(K);
        for (sofa::Index n1 = 0; n1 < spatialDimension; n1++)
        {
            for (sofa::Index n2 = 0; n2 < spatialDimension; n2++)
            {
                K.getsub(spatialDimension * n1, spatialDimension * n2, localMatrix);
                dfdx(blockSize * spring.vid[n1], blockSize * spring.vid[n2]) += localMatrix;
            }
        }
    }
}

template <class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    unsigned int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();

    const type::vector<EdgeSpring>& edgeInf = d_edgeSprings.getValue();
    constexpr sofa::type::RGBAColor color = sofa::type::RGBAColor::green();
    std::vector<sofa::type::Vec3> vertices;

    for(i=0; i<edgeInf.size(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            vertices.push_back(x[edgeInf[i].vid[EdgeSpring::A]]);
            vertices.push_back(x[edgeInf[i].vid[EdgeSpring::B]]);
        }
    }
    vparams->drawTool()->drawLines(vertices, 1.0, color);

}
template<class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::setEdgeSpring( const VecCoord& p, unsigned iA, unsigned iB, unsigned iC, unsigned iD, Real materialBendingStiffness )
{
    is_activated = is_initialized = true;

    vid[A]=iA;
    vid[B]=iB;
    vid[C]=iC;
    vid[D]=iD;

    Deriv NA = cross( p[vid[A]]-p[vid[C]], p[vid[A]]-p[vid[D]] );
    Deriv NB = cross( p[vid[B]]-p[vid[D]], p[vid[B]]-p[vid[C]] );
    Deriv NC = cross( p[vid[C]]-p[vid[B]], p[vid[C]]-p[vid[A]] );
    Deriv ND = cross( p[vid[D]]-p[vid[A]], p[vid[D]]-p[vid[B]] );

    alpha[A] =  NB.norm() / (NA.norm() + NB.norm());
    alpha[B] =  NA.norm() / (NA.norm() + NB.norm());
    alpha[C] = -ND.norm() / (NC.norm() + ND.norm());
    alpha[D] = -NC.norm() / (NC.norm() + ND.norm());

    // stiffness
    Deriv edgeDir = p[vid[C]]-p[vid[D]];
    edgeDir.normalize();
    Deriv AC = p[vid[C]]-p[vid[A]];
    Deriv BC = p[vid[C]]-p[vid[B]];
    Real ha = (AC - edgeDir * (AC*edgeDir)).norm(); // distance from A to CD
    Real hb = (BC - edgeDir * (BC*edgeDir)).norm(); // distance from B to CD
    Real l = (p[vid[C]]-p[vid[D]]).norm();          // distance from C to D
    lambda = (Real)(2./3) * (ha+hb)/(ha*ha*hb*hb) * l * materialBendingStiffness;
}

template<class _DataTypes>
typename FastTriangularBendingSprings<_DataTypes>::Real  FastTriangularBendingSprings<_DataTypes>::EdgeSpring::addForce( VecDeriv& f, const VecCoord& p, const VecDeriv& /*v*/) const
{
    if( !is_activated ) return 0;
    Deriv R = p[vid[A]]*alpha[A] +  p[vid[B]]*alpha[B] +  p[vid[C]]*alpha[C] +  p[vid[D]]*alpha[D];
    f[vid[A]] -= R * lambda * alpha[A];
    f[vid[B]] -= R * lambda * alpha[B];
    f[vid[C]] -= R * lambda * alpha[C];
    f[vid[D]] -= R * lambda * alpha[D];
    return R * R * lambda * (Real)0.5;
}

template<class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::addStiffness( sofa::linearalgebra::BaseMatrix *bm, unsigned int offset, SReal scale, core::behavior::ForceField< _DataTypes>* ff ) const
{
    StiffnessMatrix K;
    getStiffness( K );
    ff->addToMatrix(bm,offset,vid,K,scale);
}

template<class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::getStiffness( StiffnessMatrix &K ) const
{
    for( unsigned j=0; j<4; j++ )
        for( unsigned k=0; k<4; k++ )
        {
            K[j*3][k*3] = K[j*3+1][k*3+1] = K[j*3+2][k*3+2] = -lambda * alpha[j] * alpha[k];
        }
}

template<class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::replaceIndex(Index oldIndex, Index newIndex )
{
    for(unsigned i=0; i<4; i++)
        if( vid[i] == oldIndex )
            vid[i] = newIndex;
}

template<class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::replaceIndices( const type::vector<Index> &newIndices )
{
    for(unsigned i=0; i<4; i++)
        vid[i] = newIndices[vid[i]];
}


} // namespace sofa::component::solidmechanics::spring
