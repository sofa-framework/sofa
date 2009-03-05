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
#include <sofa/component/topology/PointSetGeometryAlgorithms.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/defaulttype/VecTypes.h>

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

void OglShaderVisualModel::handleTopologyChange()
{
    VisualModelImpl::handleTopologyChange();

    //TODO// update the rest position when inserting a point. Then, call computeNormals() to update the attributes.
    // Not done here because we don't have the rest position of the model.
    // For the moment, the only class using dynamic topology is HexaToTriangleTopologicalMapping which update itself the attributes... TODO !
}



void OglShaderVisualModel::computeRestNormals()
{
    if (vertNormIdx.empty())
    {
        ResizableExtVector<Coord>& vrestpos = * ( vrestpositions.beginEdit() );
        int nbn = vrestpos.size();
//    serr << "nb of visual vertices"<<nbn<<sendl;
//    serr << "nb of visual triangles"<<triangles.size()<<sendl;

        ResizableExtVector<Coord>& restNormals = * ( vrestnormals.beginEdit() );

        restNormals.resize(nbn);
        for (int i = 0; i < nbn; i++)
            restNormals[i].clear();

        for (unsigned int i = 0; i < triangles.size() ; i++)
        {

            const Coord  v1 = vrestpos[triangles[i][0]];
            const Coord  v2 = vrestpos[triangles[i][1]];
            const Coord  v3 = vrestpos[triangles[i][2]];
            Coord n = cross(v2-v1, v3-v1);

            n.normalize();
            restNormals[triangles[i][0]] += n;
            restNormals[triangles[i][1]] += n;
            restNormals[triangles[i][2]] += n;
        }
        for (unsigned int i = 0; i < quads.size() ; i++)
        {
            const Coord & v1 = vrestpos[quads[i][0]];
            const Coord & v2 = vrestpos[quads[i][1]];
            const Coord & v3 = vrestpos[quads[i][2]];
            const Coord & v4 = vrestpos[quads[i][3]];
            Coord n1 = cross(v2-v1, v4-v1);
            Coord n2 = cross(v3-v2, v1-v2);
            Coord n3 = cross(v4-v3, v2-v3);
            Coord n4 = cross(v1-v4, v3-v4);
            n1.normalize(); n2.normalize(); n3.normalize(); n4.normalize();
            restNormals[quads[i][0]] += n1;
            restNormals[quads[i][1]] += n2;
            restNormals[quads[i][2]] += n3;
            restNormals[quads[i][3]] += n4;
        }
        for (unsigned int i = 0; i < restNormals.size(); i++)
        {
            restNormals[i].normalize();
        }
    }
}


} //namespace visualmodel

} //namespace component

} //namespace sofa
