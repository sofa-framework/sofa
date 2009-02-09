/*
 * OglShaderVisualModel.cpp
 *
 *  Created on: 9 févr. 2009
 *      Author: froy
 */

#include <sofa/component/visualmodel/OglShaderVisualModel.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

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

    vrestpositions.resize(vertices.size());
    vrestnormals.resize(vnormals.size());

    for (unsigned int i = 0; i < vertices.size(); i++)
    {
        vrestpositions[i] = vertices[i];
    }

    for (unsigned int i = 0; i < vnormals.size(); i++)
    {
        vrestnormals[i] = vnormals[i];
    }

}


void OglShaderVisualModel::initVisual()
{
    OglModel::initVisual();

    //Store other attributes
    if(shader)
    {
        unsigned int restPositionsSize = vrestpositions.size() * sizeof(vrestpositions[0]);
        unsigned int restNormalsSize = vrestnormals.size() * sizeof(vrestnormals[0]);
        unsigned int totalSize = restPositionsSize + restNormalsSize;

        glGenBuffers(1, &abo);

        glBindBuffer(GL_ARRAY_BUFFER, abo);

        glBufferData(GL_ARRAY_BUFFER,
                totalSize,
                NULL,
                GL_DYNAMIC_DRAW);

        //Rest Positions
        glBufferSubData(GL_ARRAY_BUFFER,
                0,
                restPositionsSize,
                vrestpositions.getData());

        glEnableVertexAttribArray(shader->getAttribute(0, "restPosition"));
        glVertexAttribPointer(shader->getAttribute(0, "restPosition"), 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + 0);

        //Rest Normals
        glBufferSubData(GL_ARRAY_BUFFER,
                restPositionsSize,
                restNormalsSize,
                vrestnormals.getData());

        glEnableVertexAttribArray(shader->getAttribute(0, "restNormal"));
        glVertexAttribPointer(shader->getAttribute(0, "restNormal"), 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + 0 + restPositionsSize);

        glBindBuffer(GL_ARRAY_BUFFER, 0);


    }
}


} //namespace visualmodel

} //namespace component

} //namespace sofa
