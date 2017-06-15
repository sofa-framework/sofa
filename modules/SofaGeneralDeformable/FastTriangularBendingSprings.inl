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

#include <sofa/helper/gl/template.h>
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

        helper::WriteAccessor<Data<helper::vector<EdgeSpring> > > edgeData(ff->edgeSprings);
        
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
                    else   // shell[1] == triangleAdded[i]
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
						ei.setEdgeSpring( restPosition.ref(), v1, v2, e1, e2, (Real)ff->f_bendingStiffness.getValue() );
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
        helper::vector<EdgeSpring>& edgeData = *(ff->edgeSprings.beginEdit());
        for (unsigned int i=0; i<triangleRemoved.size(); ++i)
        {
            /// describe the jth edge index of triangle no i
            EdgesInTriangle te = ff->_topology->getEdgesInTriangle(triangleRemoved[i]);
            /// describe the jth vertex index of triangle no i
            //Triangle t = ff->_topology->getTriangle(triangleRemoved[i]);

			double epsilonSq = ff->d_minDistValidity.getValue();
			epsilonSq *= epsilonSq;

            for(unsigned int j=0; j<3; ++j)
            {
                EdgeSpring &ei = edgeData[te[j]]; // ff->edgeInfo
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
						ei.setEdgeSpring(restPosition.ref(), v1, v2, e1, e2, (Real)ff->f_bendingStiffness.getValue());
                    }
					else
						ei.is_activated = ei.is_initialized = false;
                }
                else
                    ei.is_activated = ei.is_initialized = false;

            }

        }

        ff->edgeSprings.endEdit();
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

        helper::vector<EdgeSpring>& edgeInf = *(ff->edgeSprings.beginEdit());

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

        ff->edgeSprings.endEdit();
    }
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyPointRenumbering(const sofa::helper::vector<unsigned int> &newIndices)
{
    if(ff)
    {
        helper::vector<EdgeSpring>& edgeInf = *(ff->edgeSprings.beginEdit());
        for (int i = 0; i < ff->_topology->getNbEdges(); ++i)
        {
            if(edgeInf[i].is_activated)
            {
                edgeInf[i].replaceIndices(newIndices);
            }
        }
        ff->edgeSprings.endEdit();
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
    : f_bendingStiffness(initData(&f_bendingStiffness,(SReal) 1.0,"bendingStiffness","bending stiffness of the material"))
    , d_minDistValidity(initData(&d_minDistValidity,(SReal) 0.000001,"minDistValidity","Distance under which a spring is not valid"))
    , edgeSprings(initData(&edgeSprings, "edgeInfo", "Internal edge data"))
    , edgeHandler(NULL)
{
    // Create specific handler for EdgeData
    edgeHandler = new TriangularBSEdgeHandler(this, &edgeSprings);
    //serr<<"FastTriangularBendingSprings<DataTypes>::FastTriangularBendingSprings"<<sendl;
}

template<class DataTypes>
FastTriangularBendingSprings<DataTypes>::~FastTriangularBendingSprings()
{
    if(edgeHandler) delete edgeHandler;
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::init()
{
    //serr << "initializing FastTriangularBendingSprings" << sendl;
    this->Inherited::init();

    _topology = this->getContext()->getMeshTopology();

    if (_topology->getNbTriangles()==0)
    {
        serr << "ERROR(FastTriangularBendingSprings): object must have a Triangular Set Topology."<<sendl;
        return;
    }
    edgeSprings.createTopologicalEngine(_topology,edgeHandler);
    edgeSprings.linkToPointDataArray();
    edgeSprings.linkToTriangleDataArray();
    edgeSprings.registerTopologicalData();

    this->reinit();
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::reinit()
{
    using namespace sofa::component::topology;
    /// prepare to store info in the edge array
    helper::vector<EdgeSpring>& edgeInf = *(edgeSprings.beginEdit());
    edgeInf.resize(_topology->getNbEdges());
    int i;
    // set edge tensor to 0
    for (i=0; i<_topology->getNbEdges(); ++i)
    {

        edgeHandler->applyCreateFunction(i, edgeInf[i],
                _topology->getEdge(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }

    // create edge tensor by calling the triangle creation function
    sofa::helper::vector<unsigned int> triangleAdded;
    for (i=0; i<_topology->getNbTriangles(); ++i)
        triangleAdded.push_back(i);

    edgeHandler->applyTriangleCreation(triangleAdded,
            (const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>)0,
            (const sofa::helper::vector<sofa::helper::vector<unsigned int> >)0,
            (const sofa::helper::vector<sofa::helper::vector<double> >)0);

    edgeSprings.endEdit();
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
    const helper::vector<EdgeSpring>& edgeInf = edgeSprings.getValue();
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
    const helper::vector<EdgeSpring>& edgeInf = edgeSprings.getValue();
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
    const helper::vector<EdgeSpring>& springs = edgeSprings.getValue();
    for(unsigned i=0; i< springs.size() ; i++)
    {
        springs[i].addStiffness( mat, offset, scale, this);
    }
}




template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    unsigned int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    //VecCoord& x_rest = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    //int nbTriangles=_topology->getNbTriangles();

    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    //unsigned int nb_to_draw = 0;

    const helper::vector<EdgeSpring>& edgeInf = edgeSprings.getValue();

    glColor4f(0,1,0,1);
    glBegin(GL_LINES);
    for(i=0; i<edgeInf.size(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            //nb_to_draw+=1;
            helper::gl::glVertexT(x[edgeInf[i].vid[EdgeSpring::A]]);
            helper::gl::glVertexT(x[edgeInf[i].vid[EdgeSpring::B]]);

        }
    }
    glEnd();
    
    glPopAttrib();
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif //#ifndef SOFA_COMPONENT_FORCEFIELD_FastTriangularBendingSprings_INL

