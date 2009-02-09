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

    for (unsigned int i = 0; i < vertices.size(); i++)
    {
        vrestpositions[i] = vertices[i];
    }
}


void OglShaderVisualModel::initVisual()
{
    OglModel::initVisual();

    //Store other attributes
    //Rest Positions
    if(shader)
    {
        GLuint abotemp;
        glGenBuffers(1, &abotemp);

        glBindBuffer(GL_ARRAY_BUFFER, abotemp);

        glBufferData(GL_ARRAY_BUFFER,
                vrestpositions.size() * sizeof(vrestpositions[0]),
                NULL,
                GL_DYNAMIC_DRAW);

        glBufferSubData(GL_ARRAY_BUFFER,
                0,
                vrestpositions.size()*sizeof(vrestpositions[0]),
                vrestpositions.getData());

        //for (unsigned int i=0 ; i<vrestpositions.size() ; i++)
        //	std::cout << vrestpositions[i] << std::endl;

        glEnableVertexAttribArray(shader->getAttribute(0, "restPosition"));
        glVertexAttribPointer(shader->getAttribute(0, "restPosition"), 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}


} //namespace visualmodel

} //namespace component

} //namespace sofa
