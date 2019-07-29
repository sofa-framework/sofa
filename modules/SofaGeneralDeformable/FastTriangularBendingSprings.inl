/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
//
// C++ Implementation: FastTriangularBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_FORCEFIELD_FastTriangularBendingSprings_INL
#define SOFA_COMPONENT_FORCEFIELD_FastTriangularBendingSprings_INL

#include <SofaGeneralDeformable/FastTriangularBendingSprings.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/defaulttype/RGBAColor.h>
#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{
typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyCreateFunction(unsigned int /*edgeIndex*/, EdgeSpring &ei, const core::topology::BaseMeshTopology::Edge &, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        ei.is_activated=false;
        ei.is_initialized=false;
    }
}



template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyTriangleCreation(const sofa::helper::vector<unsigned int> &triangleAdded, const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &, const sofa::helper::vector<sofa::helper::vector<double> > &)
{
    using namespace sofa::component::topology;
    if (ff)
    {
        typename MechanicalState::ReadVecCoord restPosition = ff->mstate->readRestPositions();

        helper::WriteAccessor<Data<helper::vector<EdgeSpring> > > edgeData(ff->d_edgeSprings);
        
        for (unsigned int i=0; i<triangleAdded.size(); ++i)
        {
            /// edges of the new triangle
            EdgesInTriangle te2 = ff->_topology->getEdgesInTriangle(triangleAdded[i]);
            /// vertices of the new triangle
            core::topology::BaseMeshTopology::Triangle t2 = ff->_topology->getTriangle(triangleAdded[i]);

			double epsilonSq = ff->d_minDistValidity.getValue();
			epsilonSq *= epsilonSq;

            // for each edge in the new triangle
            for(unsigned int j=0; j<3; ++j)
            {
                EdgeSpring &ei = edgeData[te2[j]]; // edge spring
                unsigned int edgeIndex = te2[j];

                const sofa::helper::vector< unsigned int > shell = ff->_topology->getTrianglesAroundEdge(edgeIndex);
                    
                if (shell.size()==2)   // there is another triangle attached to this edge, so a spring is needed
                {
                    // the other triangle and its edges
                    EdgesInTriangle te1;
                    core::topology::BaseMeshTopology::Triangle t1;
                    if(shell[0] == triangleAdded[i])
                    {

                        te1 = ff->_topology->getEdgesInTriangle(shell[1]);
                        t1 = ff->_topology->getTriangle(shell[1]);

                    }
                    else
                    {

                        te1 = ff->_topology->getEdgesInTriangle(shell[0]);
                        t1 = ff->_topology->getTriangle(shell[0]);
                    }

                    int i1 = ff->_topology->getEdgeIndexInTriangle(te1, edgeIndex); // index of the vertex opposed to the current edge in the other triangle (?)
                    int i2 = ff->_topology->getEdgeIndexInTriangle(te2, edgeIndex); // index of the vertex opposed to the current edge in the new triangle (?)
                    core::topology::BaseMeshTopology::Edge edge = ff->_topology->getEdge(edgeIndex);                  // indices of the vertices of the current edge

                    const core::topology::BaseMeshTopology::PointID& v1 = t1[i1];
                    const core::topology::BaseMeshTopology::PointID& v2 = t2[i2];
                    const core::topology::BaseMeshTopology::PointID& e1 = edge[0];
                    const core::topology::BaseMeshTopology::PointID& e2 = edge[1];

					Deriv vp = restPosition[v2]-restPosition[v1];
					Deriv ve = restPosition[e2]-restPosition[e1];

					if(vp.norm2()>epsilonSq && ve.norm2()>epsilonSq)
                        ei.setEdgeSpring( restPosition.ref(), v1, v2, e1, e2, (Real)ff->d_bendingStiffness.getValue() );
                }
                else
                    ei.is_activated = ei.is_initialized = false;
            }
        }
    }
}




template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyTriangleDestruction(const sofa::helper::vector<unsigned int> &triangleRemoved)
{
    using namespace sofa::component::topology;
    if (ff)
    {
        typename MechanicalState::ReadVecCoord restPosition = ff->mstate->readRestPositions();
        helper::vector<EdgeSpring>& edgeData = *(ff->d_edgeSprings.beginEdit());
        for (unsigned int i=0; i<triangleRemoved.size(); ++i)
        {
            /// describe the jth edge index of triangle no i
            EdgesInTriangle te = ff->_topology->getEdgesInTriangle(triangleRemoved[i]);
            /// describe the jth vertex index of triangle no i
			double epsilonSq = ff->d_minDistValidity.getValue();
			epsilonSq *= epsilonSq;

            for(unsigned int j=0; j<3; ++j)
            {
                EdgeSpring &ei = edgeData[te[j]];
                unsigned int edgeIndex = te[j];

                const sofa::helper::vector< unsigned int > shell = ff->_topology->getTrianglesAroundEdge(edgeIndex);

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

                    te1 = ff->_topology->getEdgesInTriangle(keepingTri[0]);
                    t1 = ff->_topology->getTriangle(keepingTri[0]);
                    te2 = ff->_topology->getEdgesInTriangle(keepingTri[1]);
                    t2 = ff->_topology->getTriangle(keepingTri[1]);

                    int i1 = ff->_topology->getEdgeIndexInTriangle(te1, edgeIndex);
                    int i2 = ff->_topology->getEdgeIndexInTriangle(te2, edgeIndex);

                    core::topology::BaseMeshTopology::Edge edge = ff->_topology->getEdge(edgeIndex);

                    const core::topology::BaseMeshTopology::PointID& v1 = t1[i1];
                    const core::topology::BaseMeshTopology::PointID& v2 = t2[i2];
                    const core::topology::BaseMeshTopology::PointID& e1 = edge[0];
                    const core::topology::BaseMeshTopology::PointID& e2 = edge[1];

					Deriv vp = restPosition[v2]-restPosition[v1];
					Deriv ve = restPosition[e2]-restPosition[e1];

					if(vp.norm2()>epsilonSq && ve.norm2()>epsilonSq)
                    {
                        ei.setEdgeSpring(restPosition.ref(), v1, v2, e1, e2, (Real)ff->d_bendingStiffness.getValue());
                    }
					else
						ei.is_activated = ei.is_initialized = false;
                }
                else
                    ei.is_activated = ei.is_initialized = false;

            }

        }

        ff->d_edgeSprings.endEdit();
    }

}

template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    using namespace sofa::component::topology;
    const sofa::helper::vector<unsigned int> &triangleAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTriangleCreation(triangleAdded, elems, ancestors, coefs);
}

template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> &triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}

template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyPointDestruction(const sofa::helper::vector<unsigned int> &tab)
{
    if(ff)
    {
        bool debug_mode = false;

        unsigned int last = ff->_topology->getNbPoints() -1;
        unsigned int i,j;

        helper::vector<EdgeSpring>& edgeInf = *(ff->d_edgeSprings.beginEdit());

        //make a reverse copy of tab
        sofa::helper::vector<unsigned int> lastIndexVec;
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

            const sofa::helper::vector<unsigned int> &shell= ff->_topology->getTrianglesAroundVertex(lastIndexVec[i]);
            for (j=0; j<shell.size(); ++j)
            {
                core::topology::BaseMeshTopology::EdgesInTriangle tej = ff->_topology->getEdgesInTriangle(shell[j]);
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

        ff->d_edgeSprings.endEdit();
    }
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyPointRenumbering(const sofa::helper::vector<unsigned int> &newIndices)
{
    if(ff)
    {
        helper::vector<EdgeSpring>& edgeInf = *(ff->d_edgeSprings.beginEdit());
        for (unsigned int i = 0; i < ff->_topology->getNbEdges(); ++i)
        {
            if(edgeInf[i].is_activated)
            {
                edgeInf[i].replaceIndices(newIndices);
            }
        }
        ff->d_edgeSprings.endEdit();
    }
}

template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::PointsRemoved* e)
{
    const sofa::helper::vector<unsigned int> & tab = e->getArray();
    applyPointDestruction(tab);
}

template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::PointsRenumbering* e)
{
    const sofa::helper::vector<unsigned int> &newIndices = e->getIndexArray();
    applyPointRenumbering(newIndices);
}

template<class DataTypes>
FastTriangularBendingSprings<DataTypes>::FastTriangularBendingSprings(/*double _ks, double _kd*/)
    : d_bendingStiffness(initData(&d_bendingStiffness,(SReal) 1.0,"bendingStiffness","bending stiffness of the material"))
    , d_minDistValidity(initData(&d_minDistValidity,(SReal) 0.000001,"minDistValidity","Distance under which a spring is not valid"))
    , d_edgeSprings(initData(&d_edgeSprings, "edgeInfo", "Internal edge data"))
    , d_edgeHandler(NULL)
{
    // Create specific handler for EdgeData
    d_edgeHandler = new TriangularBSEdgeHandler(this, &d_edgeSprings);
}

template<class DataTypes>
FastTriangularBendingSprings<DataTypes>::~FastTriangularBendingSprings()
{
    if(d_edgeHandler) delete d_edgeHandler;
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::init()
{
    this->Inherited::init();

    _topology = this->getContext()->getMeshTopology();

    if (_topology->getNbTriangles()==0)
    {
        msg_error() << "ERROR(FastTriangularBendingSprings): object must have a Triangular Set Topology."<<msgendl;
        return;
    }
    d_edgeSprings.createTopologicalEngine(_topology,d_edgeHandler);
    d_edgeSprings.linkToPointDataArray();
    d_edgeSprings.linkToTriangleDataArray();
    d_edgeSprings.registerTopologicalData();

    this->reinit();
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::reinit()
{
    using namespace sofa::component::topology;
    /// prepare to store info in the edge array
    helper::vector<EdgeSpring>& edgeInf = *(d_edgeSprings.beginEdit());
    edgeInf.resize(_topology->getNbEdges());

    // set edge tensor to 0
    for (unsigned int i=0; i<_topology->getNbEdges(); ++i)
    {

        d_edgeHandler->applyCreateFunction(i, edgeInf[i],
                _topology->getEdge(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }

    // create edge tensor by calling the triangle creation function
    sofa::helper::vector<unsigned int> triangleAdded;
    for (unsigned int i=0; i<_topology->getNbTriangles(); ++i)
        triangleAdded.push_back(i);

    d_edgeHandler->applyTriangleCreation(triangleAdded,
            (const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>)0,
            (const sofa::helper::vector<sofa::helper::vector<unsigned int> >)0,
            (const sofa::helper::vector<sofa::helper::vector<double> >)0);

    d_edgeSprings.endEdit();
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
    const helper::vector<EdgeSpring>& edgeInf = d_edgeSprings.getValue();
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
    const helper::vector<EdgeSpring>& edgeInf = d_edgeSprings.getValue();
    const Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    df.resize(dx.size());
    for(unsigned i=0; i<edgeInf.size(); i++ )
    {
        edgeInf[i].addDForce(df.wref(),dx,kFactor);
    }
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *mat, SReal scale, unsigned int &offset)
{
    const helper::vector<EdgeSpring>& springs = d_edgeSprings.getValue();
    for(unsigned i=0; i< springs.size() ; i++)
    {
        springs[i].addStiffness( mat, offset, scale, this);
    }
}




template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    unsigned int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    vparams->drawTool()->saveLastState();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();

    const helper::vector<EdgeSpring>& edgeInf = d_edgeSprings.getValue();
    sofa::defaulttype::RGBAColor color = sofa::defaulttype::RGBAColor::green();
    std::vector<sofa::defaulttype::Vector3> vertices;

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
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::addStiffness( sofa::defaulttype::BaseMatrix *bm, unsigned int offset, SReal scale, core::behavior::ForceField< _DataTypes>* ff ) const
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
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::replaceIndex( unsigned oldIndex, unsigned newIndex )
{
    for(unsigned i=0; i<4; i++)
        if( vid[i] == oldIndex )
            vid[i] = newIndex;
}

template<class _DataTypes>
void FastTriangularBendingSprings<_DataTypes>::EdgeSpring::replaceIndices( const helper::vector<unsigned> &newIndices )
{
    for(unsigned i=0; i<4; i++)
        vid[i] = newIndices[vid[i]];
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif //#ifndef SOFA_COMPONENT_FORCEFIELD_FastTriangularBendingSprings_INL

