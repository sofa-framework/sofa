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

#include <sofa/component/forcefield/FastTriangularBendingSprings.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging

#include <sofa/helper/gl/template.h>
#include <sofa/component/topology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace	sofa::component::topology;
using namespace core::topology;





using namespace core::behavior;
using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyCreateFunction(unsigned int , EdgeSpring &ei, const Edge &, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        ei.is_activated=false;
        ei.is_initialized=false;

    }
}



template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyTriangleCreation(const sofa::helper::vector<unsigned int> &triangleAdded, const sofa::helper::vector<Triangle> &, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &, const sofa::helper::vector<sofa::helper::vector<double> > &)
{
    if (ff)
    {


        unsigned int nb_activated = 0;

        typename MechanicalState::ReadVecCoord restPosition = ff->mstate->readRestPositions();

        helper::WriteAccessor<Data<vector<EdgeSpring> > > edgeData(ff->edgeSprings);

        for (unsigned int i=0; i<triangleAdded.size(); ++i)
        {

            /// edges of the new triangle
            EdgesInTriangle te2 = ff->_topology->getEdgesInTriangle(triangleAdded[i]);
            /// vertices of the new triangle
            Triangle t2 = ff->_topology->getTriangle(triangleAdded[i]);

            // for each edge in the new triangle
            for(unsigned int j=0; j<3; ++j)
            {

                EdgeSpring &ei = edgeData[te2[j]]; // edge spring
                if(!(ei.is_initialized))
                {
                    unsigned int edgeIndex = te2[j];

                    const sofa::helper::vector< unsigned int > shell = ff->_topology->getTrianglesAroundEdge(edgeIndex);

                    if (shell.size()==2)   // there is another triangle attached to this edge, so a spring is needed
                    {

                        nb_activated+=1;

                        // the other triangle and its edges
                        EdgesInTriangle te1;
                        Triangle t1;
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
                        Edge edge = ff->_topology->getEdge(edgeIndex);                  // indices of the vertices of the current edge

                        ei.setEdgeSpring( restPosition.ref(), t1[i1], t2[i2], edge[0], edge[1], (Real)ff->f_bendingStiffness.getValue() );

//                        ei.m1 = t1[i1];
//                        ei.m2 = t2[i2];

//                        //FastTriangularBendingSprings<DataTypes> *fftest= (FastTriangularBendingSprings<DataTypes> *)param;
//                        ei.ks=m_ks; //(fftest->ks).getValue();
//                        ei.kd=m_kd; //(fftest->kd).getValue();

//                        Coord u = (*restPosition)[ei.m1] - (*restPosition)[ei.m2];

//                        Real d = u.norm();

//                        ei.restlength=(double) d;

//                        ei.is_activated=true;

                    }
                    else
                    {

                        ei.is_activated=false;

                    }

                }
            }

        }

        ff->edgeSprings.endEdit();
    }

}




template< class DataTypes>
void FastTriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyTriangleDestruction(const sofa::helper::vector<unsigned int> &triangleRemoved)
{
    if (ff)
    {

//        double m_ks=ff->getKs(); // typename DataTypes::
//        double m_kd=ff->getKd(); // typename DataTypes::

        //unsigned int u,v;

        typename MechanicalState::ReadVecCoord restPositions = ff->mstate->readRestPositions();
        helper::vector<EdgeSpring>& edgeData = *(ff->edgeSprings.beginEdit());

        for (unsigned int i=0; i<triangleRemoved.size(); ++i)
        {
            /// describe the jth edge index of triangle no i
            EdgesInTriangle te = ff->_topology->getEdgesInTriangle(triangleRemoved[i]);
            /// describe the jth vertex index of triangle no i
            Triangle t = ff->_topology->getTriangle(triangleRemoved[i]);


            for(unsigned int j=0; j<3; ++j)
            {

                EdgeSpring &ei = edgeData[te[j]]; // ff->edgeInfo
                if(ei.is_initialized)
                {

                    unsigned int edgeIndex = te[j];

                    const sofa::helper::vector< unsigned int > shell = ff->_topology->getTrianglesAroundEdge(edgeIndex);
                    if (shell.size()==3)
                    {

                        EdgesInTriangle te1;
                        Triangle t1;
                        EdgesInTriangle te2;
                        Triangle t2;

                        if(shell[0] == triangleRemoved[i])
                        {
                            te1 = ff->_topology->getEdgesInTriangle(shell[1]);
                            t1 = ff->_topology->getTriangle(shell[1]);
                            te2 = ff->_topology->getEdgesInTriangle(shell[2]);
                            t2 = ff->_topology->getTriangle(shell[2]);

                        }
                        else
                        {

                            if(shell[1] == triangleRemoved[i])
                            {

                                te1 = ff->_topology->getEdgesInTriangle(shell[2]);
                                t1 = ff->_topology->getTriangle(shell[2]);
                                te2 = ff->_topology->getEdgesInTriangle(shell[0]);
                                t2 = ff->_topology->getTriangle(shell[0]);

                            }
                            else   // shell[2] == triangleRemoved[i]
                            {

                                te1 = ff->_topology->getEdgesInTriangle(shell[0]);
                                t1 = ff->_topology->getTriangle(shell[0]);
                                te2 = ff->_topology->getEdgesInTriangle(shell[1]);
                                t2 = ff->_topology->getTriangle(shell[1]);

                            }
                        }

                        int i1 = ff->_topology->getEdgeIndexInTriangle(te1, edgeIndex);
                        int i2 = ff->_topology->getEdgeIndexInTriangle(te2, edgeIndex);

                        Edge edge = ff->_topology->getEdge(edgeIndex);
                        ei.setEdgeSpring(restPositions.ref(), t1[i1], t2[i2], edge[0], edge[1], (Real)ff->f_bendingStiffness.getValue());

                        //                        ei.m1 = t1[i1];
                        //                        ei.m2 = t2[i2];

                        //                        //FastTriangularBendingSprings<DataTypes> *fftest= (FastTriangularBendingSprings<DataTypes> *)param;
                        //                        ei.ks=m_ks; //(fftest->ks).getValue();
                        //                        ei.kd=m_kd; //(fftest->kd).getValue();

                        //                        Coord u = (*restPosition)[ei.m1] - (*restPosition)[ei.m2];
                        //                        Real d = u.norm();

                        //                        ei.restlength=(double) d;

                        //                        ei.is_activated=true;

                    }
                    else
                    {

                        ei.is_activated=false;
                        ei.is_initialized = false;

                    }

                }
                else
                {

                    ei.is_activated=false;
                    ei.is_initialized = false;

                }
            }

        }

        ff->edgeSprings.endEdit();
    }

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

        sofa::helper::vector<unsigned int> lastIndexVec;
        for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
        {

            lastIndexVec.push_back(last - i_init);
        }

        for ( i = 0; i < tab.size(); ++i)
        {

            unsigned int i_next = i;
            bool is_reached = false;
            while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
            {

                i_next += 1 ;
                is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
            }

            if(is_reached)
            {

                lastIndexVec[i_next] = lastIndexVec[i];

            }

            const sofa::helper::vector<unsigned int> &shell= ff->_topology->getTrianglesAroundVertex(lastIndexVec[i]);
            for (j=0; j<shell.size(); ++j)
            {

                Triangle tj = ff->_topology->getTriangle(shell[j]);

                int vertexIndex = ff->_topology->getVertexIndexInTriangle(tj, lastIndexVec[i]);

                EdgesInTriangle tej = ff->_topology->getEdgesInTriangle(shell[j]);

                unsigned int ind_j = tej[vertexIndex];

                edgeInf[ind_j].replaceIndex( last, tab[i]);
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
FastTriangularBendingSprings<DataTypes>::FastTriangularBendingSprings(/*double _ks, double _kd*/)
    : f_bendingStiffness(initData(&f_bendingStiffness,(double) 1.0,"bendingStiffness","bending stiffness of the material"))
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
            (const sofa::helper::vector<Triangle>)0,
            (const sofa::helper::vector<sofa::helper::vector<unsigned int> >)0,
            (const sofa::helper::vector<sofa::helper::vector<double> >)0);

    edgeSprings.endEdit();
}

template <class DataTypes>
double FastTriangularBendingSprings<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    return m_potentialEnergy;
}


template<class DataTypes>
void FastTriangularBendingSprings<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
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

    df.resize(dx.size());
    for(unsigned i=0; i<edgeInf.size(); i++ )
    {
        edgeInf[i].addDForce(df.wref(),dx,(Real)mparams->kFactor());
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
    unsigned int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = *this->mstate->getX();
    //VecCoord& x_rest = *this->mstate->getX0();
    //int nbTriangles=_topology->getNbTriangles();

    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    unsigned int nb_to_draw = 0;

    const helper::vector<EdgeSpring>& edgeInf = edgeSprings.getValue();

    glColor4f(0,1,0,1);
    glBegin(GL_LINES);
    for(i=0; i<edgeInf.size(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            nb_to_draw+=1;
            helper::gl::glVertexT(x[edgeInf[i].vid[EdgeSpring::A]]);
            helper::gl::glVertexT(x[edgeInf[i].vid[EdgeSpring::B]]);

        }
    }
    glEnd();

    glPopAttrib();
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif //#ifndef SOFA_COMPONENT_FORCEFIELD_FastTriangularBendingSprings_INL

