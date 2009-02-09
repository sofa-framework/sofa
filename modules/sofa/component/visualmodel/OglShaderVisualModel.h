/*
 * GlslModel.h
 *
 *  Created on: 9 févr. 2009
 *      Author: froy
 */

#ifndef OGLSHADERVISUALMODEL_H_
#define OGLSHADERVISUALMODEL_H_

#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/component/visualmodel/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_COMPONENT_VISUALMODEL_API OglShaderVisualModel : public OglModel
{
protected:

    typedef ExtVec3fTypes::Coord Coord;

    GLuint abo;
    OglShader* shader;

    ResizableExtVector<Coord> vrestpositions;
    ResizableExtVector<Coord> vrestnormals;

public:
    OglShaderVisualModel();
    virtual ~OglShaderVisualModel();

    void init();
    void initVisual();
};

} //namespace visualmodel

} //namespace component

} //namespace sofa

#endif /* OGLSHADERVISUALMODEL_H_ */
