/*
 * OglShaderVisualModel.cpp
 *
 *  Created on: 9 févr. 2009
 *      Author: froy
 */

#include <sofa/component/visualmodel/OglShaderVisualModel.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/QuadSetTopologyChange.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::core::componentmodel::topology;
using namespace sofa::core::componentmodel::behavior;

SOFA_DECL_CLASS(OglShaderVisualModel)

int OglShaderVisualModelClass = core::RegisterObject("Visual model for OpenGL display using Glew extensions")
        .add< OglShaderVisualModel >()
        ;

OglShaderVisualModel::OglShaderVisualModel()
{
    // TODO Auto-generated constructor stub

}

OglShaderVisualModel::~OglShaderVisualModel()
{
    // TODO Auto-generated destructor stub
}

void OglShaderVisualModel::init()
{
    OglModel::init();
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    shader = context->core::objectmodel::BaseContext::get<OglShader>();

    vrestpositions.setContext( this->getContext());
    vrestpositions.setID( std::string("restPosition"));
    vrestpositions.setIndexShader( 0);
    vrestpositions.init();
    ResizableExtVector<Coord>& vrestpos = * ( vrestpositions.beginEdit() );
    vrestpos.resize ( vertices.size() );
    for ( unsigned int i = 0; i < vertices.size(); i++ )
    {
        vrestpos[i] = vertices[i];
    }
    vrestpositions.endEdit();


    vrestnormals.setContext( this->getContext());
    vrestnormals.setID( "restNormal");
    vrestnormals.setIndexShader( 0);
    vrestnormals.init();
    ResizableExtVector<Coord>& vrestnorm = * ( vrestnormals.beginEdit() );
    vrestnorm.resize ( vnormals.size() );
    for ( unsigned int i = 0; i < vnormals.size(); i++ )
    {
        vrestnorm[i] = vnormals[i];
    }
    vrestnormals.endEdit();
}


void OglShaderVisualModel::initVisual()
{
    OglModel::initVisual();

    //Store other attributes
    if(shader)
    {
        vrestpositions.initVisual();
        vrestnormals.initVisual();

    }
}

} //namespace visualmodel

} //namespace component

} //namespace sofa
