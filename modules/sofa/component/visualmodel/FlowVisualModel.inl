/*
 * FlowVisualModel.h
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#ifndef FLOWVISUALMODEL_INL_
#define FLOWVISUALMODEL_INL_

#include "FlowVisualModel.h"
#include <sofa/helper/system/glut.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

template <class DataTypes>
FlowVisualModel<DataTypes>::FlowVisualModel()
    :viewVelocityFactor(initData(&viewVelocityFactor, double(0.001), "viewVelocityFactor", "Set factor for velocity arrows"))
    ,velocityMin(initData(&velocityMin, double(-1.0), "velocityMin", "Set the minimum value of velocity for drawing,"))
    ,velocityMax(initData(&velocityMax, double(1.0), "velocityMax", "Set the maximum value of velocity for drawing,"))
{

}

template <class DataTypes>
FlowVisualModel<DataTypes>::~FlowVisualModel()
{

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    fstate = context->core::objectmodel::BaseContext::get<FluidState>();
    if (!fstate)
        std::cerr << "WARNING: FlowVisualModel has no binding FluidState" <<endl;

    this->getContext()->get(m_triTopo);

    if (m_triTopo == NULL)
        std::cerr << "WARNING: FlowVisualModel has no binding TriangleSetTopology" <<endl;

    this->getContext()->get(m_triGeo);

    if (m_triGeo == NULL)
        std::cerr << "WARNING: FlowVisualModel has no binding TriangleSetGeometry" <<endl;



}

template <class DataTypes>
void FlowVisualModel<DataTypes>::initVisual()
{

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::draw()
{
    if (!getContext()->getShowVisualModels()) return;

    glDisable(GL_LIGHTING);
    const VecDeriv& v = *this->fstate->getV();
    //VecCoord& y = *this->fstate->getX();

    VecDeriv vnorm;
    double vmax2 = v[0].norm2();
    double vmin = v[0].norm();

    //search vmax
    for (unsigned int i=1 ; i<v.size() ; i++)
    {
        if (v[i].norm2() > vmax2)
            vmax2=v[i].norm2();
    }

    double vmax = (velocityMax.getValue() < sqrt(vmax2)) ? sqrt(vmax2) : velocityMax.getValue();

    if (velocityMin.getValue() < 0.0)
    {
        //search vmin
        for (unsigned int i=1 ; i<v.size() ; i++)
        {
            if (v[i].norm() < vmin)
                vmin=v[i].norm();
        }
    }
    else vmin = velocityMin.getValue();

    //normalize v
    for (unsigned int i=0 ; i<v.size() ; i++)
    {
        if (vmax > 0.0)
            vnorm.push_back(v[i]/vmax);
        else vnorm.push_back(v[i]);
    }

    core::componentmodel::topology::BaseMeshTopology::SeqTriangles triangles =  m_triTopo->getTriangles();
    unsigned int nbPoints =  m_triTopo->getNbPoints();
    VecCoord colors;
    helper::vector<unsigned int> weight;
    colors.resize(nbPoints);
    weight.resize(nbPoints);
    std::fill( weight.begin(), weight.end(), 0 );
    //accumulate v/colors
    for(unsigned int i=0 ; i<triangles.size() ; i++)
    {
        core::componentmodel::topology::BaseMeshTopology::Triangle t = (triangles[i]);
        for(unsigned int j=0 ; j<3 ; j++)
        {
            int indColor;
            if(vmax > 0.0)
                indColor = (int)(64*((v[i].norm()-vmin)/(vmax-vmin)));
            else indColor = 0;

            colors[t[j]]+= ColorMap[indColor];
            weight[t[j]]++;
        }
    }
    //VecCoord& x = *this->fstate->getX();
    VecCoord x;

    //Show un truc
    glBegin(GL_TRIANGLES);
    for(unsigned int i=0 ; i<triangles.size() ; i++)
    {
        core::componentmodel::topology::BaseMeshTopology::Triangle t = (triangles[i]);

        Coord p0 = m_triGeo->getPointPosition(t[0]);
        Coord p1 = m_triGeo->getPointPosition(t[1]);
        Coord p2 = m_triGeo->getPointPosition(t[2]);

        //compute barycenter of each triangle
        Coord pb;
        pb[0] = (p0[0] + p1[0] + p2[0])/3;
        pb[1] = (p0[1] + p1[1] + p2[1])/3;
        pb[2] = (p0[2] + p1[2] + p2[2])/3;
        x.push_back(pb);

        glColor3dv((colors[t[0]]/weight[t[0]]).ptr() );
        glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
        glColor3dv((colors[t[1]]/weight[t[1]]).ptr() );
        glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
        glColor3dv((colors[t[2]]/weight[t[2]]).ptr() );
        glVertex3f((GLfloat)p2[0], (GLfloat)p2[1], (GLfloat)p2[2]);
    }
    glEnd();

    //Show Velocity
    glBegin(GL_LINES);
    for(unsigned int i=0 ; i<x.size() ; i++)
    {
        if (v[i].norm() > 0.0)
        {
            Coord p0 = x[i];
            Coord p1 = x[i] + vnorm[i]*viewVelocityFactor.getValue();

            glColor3f(1.0,1.0,1.0);
            glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
            glColor3f(1.0,1.0,1.0);
            glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
        }
    }
    glEnd();

    //Draw color scale

    double xMargin = 50.0;
    double yMargin = 20.0;
    GLint currentViewport[4];
    glGetIntegerv(GL_VIEWPORT, currentViewport);
    Vec2d scaleSize = Vec2d(20, 200);
    Vec2d scalePosition = Vec2d(currentViewport[2] - xMargin - scaleSize[0], currentViewport[3] - yMargin - scaleSize[1]);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,currentViewport[2],0,currentViewport[3]);
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushMatrix();
    double step = scaleSize[1]/64;
    glBegin(GL_QUADS);
    for (unsigned int i=0 ; i<63 ; i++)
    {
        glColor3dv(ColorMap[i].ptr());
        glVertex2d(scalePosition[0], scalePosition[1] + i*step);
        glVertex2d(scalePosition[0] + scaleSize[0], scalePosition[1] + i*step);
        glColor3dv(ColorMap[i+1].ptr());
        glVertex2d(scalePosition[0] + scaleSize[0], scalePosition[1] + (i+1)*step);
        glVertex2d(scalePosition[0], scalePosition[1] + (i+1)*step);
    }
    glEnd();
    glPopMatrix();

    //Draw color scale values
    unsigned int NUMBER_OF_VALUES = 10;
    glColor3f(1.0,1.0,1.0);
    step = scaleSize[1]/(NUMBER_OF_VALUES - 1);
    double stepValue = (vmax-vmin)/(NUMBER_OF_VALUES - 1);
    double textScaleSize = 0.06;
    double xMarginWithScale = 5;

    for (unsigned int i=0 ; i<NUMBER_OF_VALUES  ; i++)
    {
        glPushMatrix();
        glTranslatef(scalePosition[0] + scaleSize[0] + xMarginWithScale, scalePosition[1] + i*step, 0);
        glScalef(textScaleSize, textScaleSize, textScaleSize);
        double intpart;
        double decpart = modf((vmin + i*stepValue), &intpart);
        std::ostringstream oss;
        oss << intpart << "." << (ceil(decpart*100));
        std::string tmp = oss.str();


        const char* s = tmp.c_str();
        while(*s)
        {
            glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
            s++;
        }
        glPopMatrix();
    }
    glPopMatrix();
}



}

}

}

#endif /* FLOWVISUALMODEL_INL_ */
