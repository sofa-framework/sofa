/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef OGLTETRAHEDRALMODEL_INL_
#define OGLTETRAHEDRALMODEL_INL_

#include "OglTetrahedralModel.h"

#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <limits>

namespace sofa
{
namespace component
{
namespace visualmodel
{

template<class DataTypes>
OglTetrahedralModel<DataTypes>::OglTetrahedralModel()
    : m_topology(NULL)
    , m_positions(initData(&m_positions, "position", "Vertices coordinates"))
    , modified(false)
    , lastMeshRev(-1)
    , useTopology(false)
    , depthTest(initData(&depthTest, (bool) true, "depthTest", "Set Depth Test"))
    , blending(initData(&blending, (bool) true, "blending", "Set Blending"))
{
}

template<class DataTypes>
OglTetrahedralModel<DataTypes>::~OglTetrahedralModel()
{
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    m_topology = context->getMeshTopology();
    

    if (!m_topology)
    {// currently OglTetrahedralMedal has to use topology to initialize data
        serr << "OglTetrahedralModel : Error : no BaseMeshTopology found." << sendl;
        return;
    }
    // for now, no mesh file will be loaded directly from OglTetrahedralModel component
    // so force useTopology and modified to be true to enable the first time data loading from topology
    useTopology = true;
    modified = true;
    VisualModel::init();
    updateVisual();
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::updateVisual()
{
    
    if((modified && !m_positions.getValue().empty())
       || useTopology)
    {
        // update mesh either when data comes from useTopology initially or vertices
        // get modified
        if( useTopology ) 
        {
            sofa::core::topology::TopologyModifier* topoMod;
            getContext()->get(topoMod);
        
            if( topoMod ) 
            {// topology will be handled by handleTopologyChange() with topologyModifier
                useTopology = false;
                computeMesh();
            }
            else if( topoMod==NULL&& m_topology->getRevision()!=lastMeshRev ) 
            {
                computeMesh();
            }
        }
        modified = false;
    }
    m_tetrahedrons.updateIfDirty();
    m_positions.updateIfDirty();
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::computeMesh()
{
    using sofa::core::behavior::BaseMechanicalState;
    // update m_positions
    if( m_topology->hasPos() ) 
    {
        if( this->f_printLog.getValue() ) 
            sout<< "OglTetrahedralModel: copying "<< m_topology->getNbPoints() << "points from topology." <<sendl;
        helper::WriteAccessor<  Data<sofa::defaulttype::ResizableExtVector<Coord> > > position = m_positions;
        position.resize(m_topology->getNbPoints());
        for( unsigned int i = 0; i<position.size(); i++ ) {
            position[i][0] = (Real)m_topology->getPX(i);
            position[i][1] = (Real)m_topology->getPY(i);
            position[i][2] = (Real)m_topology->getPZ(i);
        }
    }
    else if( BaseMechanicalState* mstate = m_topology->getContext()->getMechanicalState() )
    {
        if( this->f_printLog.getValue() ) 
            sout<<"OglTetrahedralModel: copying "<< mstate->getSize()<< " points from mechanical state." <<sendl;
        helper::WriteAccessor< Data<sofa::defaulttype::ResizableExtVector<Coord> > > position = m_positions;
        position.resize(mstate->getSize());
        for( unsigned int i = 0; i<position.size(); i++ ) 
        {
            position[i][0] = (Real)mstate->getPX(i);
            position[i][1] = (Real)mstate->getPY(i);
            position[i][2] = (Real)mstate->getPZ(i);
        }
    }
    else
    {
        serr<<"OglTetrahedralModel: can not update vertices!"<< sendl;
    }
    lastMeshRev = m_topology->getRevision();
    // update m_tetrahedrons
    const core::topology::BaseMeshTopology::SeqTetrahedra& inputTetrahedrons = m_topology->getTetrahedra();
    if( this->f_printLog.getValue() ) 
        sout<<"OglTetrahedralModel: copying "<< inputTetrahedrons.size() << " tetrahedrons from topology." <<sendl;
    helper::WriteAccessor< Data< sofa::defaulttype::ResizableExtVector<Tetrahedron> > > tetrahedrons = m_tetrahedrons;
    tetrahedrons.resize(inputTetrahedrons.size());
    for( unsigned int i = 0; i<inputTetrahedrons.size(); i++ ) {
        tetrahedrons[i] = inputTetrahedrons[i];
    }
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::drawTransparent(const core::visual::VisualParams* vparams)
{
  using sofa::component::topology::TetrahedronSetTopologyContainer;
    if (!vparams->displayFlags().getShowVisualModels()) return;

    if(blending.getValue())
        glEnable(GL_BLEND);
    if(depthTest.getValue())
        glDepthMask(GL_FALSE);

    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    //core::topology::BaseMeshTopology::SeqHexahedra::const_iterator it;
    core::topology::BaseMeshTopology::SeqTetrahedra::const_iterator it;

#ifdef GL_LINES_ADJACENCY_EXT

    const core::topology::BaseMeshTopology::SeqTetrahedra& vec = m_topology->getTetrahedra();


    //const VecCoord& x =nodes->read(core::ConstVecCoordId::position())->getValue();
    Coord v;

    glBegin(GL_LINES_ADJACENCY_EXT);

    const sofa::defaulttype::ResizableExtVector<Coord> position = m_positions.getValue();
    for( it = vec.begin(); it!=vec.end(); it++ ) {
        // for every tetrahedron, get its four vertices
        for( unsigned int i = 0; i<4; i++ ) {
          /*topo->getPointsOnBorder()*/
          //int vertex_index = t->getVertexIndexInTetrahedron((*it), i);
          // double x1 =  m_topology->getPX((*it)[i]);
          // double y1 = m_topology->getPY((*it)[i]);
          // double z1 = m_topology->getPZ((*it)[i]);
          v = position[(*it)[i]];
          glVertex3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]);
        }
    }

    ////sout<<"the number of tetras is:"<<nr_of_tetras<<sendl;
    //for(it = vec.begin() ; it != vec.end() ; it++)
    ////for( int i = 0; i<nr_of_tetras; i++  )
    //{
    //    
    //    for (unsigned int i=0 ; i< 4 ; i++)
    //    {
    //        v = x[(*it)[i]];
    //        glVertex3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]);
    //    }
    //}
    glEnd();
    /*
    	const core::topology::BaseMeshTopology::SeqHexahedra& vec = topo->getHexahedra();

    	VecCoord& x =nodes->read(core::ConstVecCoordId::position())->getValue();
    	Coord v;


    	const unsigned int hexa2tetrahedra[24] = { 0, 5, 1, 6,
    										   0, 1, 3, 6,
    										   1, 3, 6, 2,
    										   6, 3, 0, 7,
    										   6, 7, 0, 5,
    										   7, 5, 4, 0 };



    	glBegin(GL_LINES_ADJACENCY_EXT);
    	for(it = vec.begin() ; it != vec.end() ; it++)
    	{

    		for (unsigned int i=0 ; i<6 ; i++)
    		{
    			for (unsigned int j=0 ; j<4 ; j++)
    			{
    				//glVertex3f((GLfloat)x[(*it)[hexa2tetrahedra[i][j]]][0], (GLfloat)x[(*it)[hexa2tetrahedra[i][j]]][1], (GLfloat)x[(*it)[hexa2tetrahedra[i][j]]][2]);
    				glVertex3f((GLfloat)x[(*it)[hexa2tetrahedra[i*4 + j]]][0], (GLfloat)x[(*it)[hexa2tetrahedra[i*4 + j]]][1], (GLfloat)x[(*it)[hexa2tetrahedra[i*4 + j]]][2]);
    			}
    		}
    	}
    	glEnd();
    	*/
#else

#endif
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::computeBBox(const core::ExecParams * params, bool /*onlyVisible*/)
{
    if ( m_topology)
    {
        const core::topology::BaseMeshTopology::SeqTetrahedra& vec = m_topology->getTetrahedra();
        core::topology::BaseMeshTopology::SeqTetrahedra::const_iterator it;
        Coord v;
        const sofa::defaulttype::ResizableExtVector<Coord>& position = m_positions.getValue();
        const SReal max_real = std::numeric_limits<SReal>::max();
        const SReal min_real = std::numeric_limits<SReal>::min();

        SReal maxBBox[3] = {min_real,min_real,min_real};
        SReal minBBox[3] = {max_real,max_real,max_real};

        for(it = vec.begin() ; it != vec.end() ; ++it)
        {
            for (unsigned int i=0 ; i< 4 ; i++)
            {
                v = position[(*it)[i]];
                //v = x[(*it)[i]];

                if (minBBox[0] > v[0]) minBBox[0] = v[0];
                if (minBBox[1] > v[1]) minBBox[1] = v[1];
                if (minBBox[2] > v[2]) minBBox[2] = v[2];
                if (maxBBox[0] < v[0]) maxBBox[0] = v[0];
                if (maxBBox[1] < v[1]) maxBBox[1] = v[1];
                if (maxBBox[2] < v[2]) maxBBox[2] = v[2];
            }
        }

        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
    }
}

}
}
}

#endif //OGLTETRAHEDRALMODEL_H_
