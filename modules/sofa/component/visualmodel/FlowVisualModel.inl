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
const double FlowVisualModel<DataTypes>::STREAMLINE_NUMBER_OF_POINTS_BY_TRIANGLE = 5.0;

template <class DataTypes>
FlowVisualModel<DataTypes>::FlowVisualModel()
    :meanEdgeLength(0.0)
    ,showVelocityLines(initData(&showVelocityLines, bool(true), "showVelocityLines", "Show velocities lines"))
    ,viewVelocityFactor(initData(&viewVelocityFactor, double(0.001), "viewVelocityFactor", "Set factor for velocity arrows"))
    ,velocityMin(initData(&velocityMin, double(-1.0), "velocityMin", "Set the minimum value of velocity for drawing,"))
    ,velocityMax(initData(&velocityMax, double(1.0), "velocityMax", "Set the maximum value of velocity for drawing"))
    ,showStreamLines(initData(&showStreamLines, bool(true), "showStreamLines", "Set stream lines"))
    ,streamlineSeeds(initData(&streamlineSeeds, helper::vector<Coord>(), "streamlineSeeds", "Set streamlineSeeds for Stream Lines"))
    ,streamlineMaxNumberOfPoints(initData(&streamlineMaxNumberOfPoints, (unsigned int) 50 , "streamlineMaxNumberOfPoints", "Set the maximum number of points for each stream line"))
    ,streamlineDtNumberOfPointsPerTriangle(initData(&streamlineDtNumberOfPointsPerTriangle, (double) 5.0 , "streamlineDtNumberOfPointsPerTriangle", "Set the number of points for each step (equals ~ a triangle)"))

    ,showColorScale(initData(&showColorScale, bool(true), "showColorScale", "Set color scale"))
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

    reinit();

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::initVisual()
{

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::reinit()
{
//	unsigned int seedsSize = streamlineSeeds.getValue().size();
//	streamLines.resize(seedSize);
    //fill flags to false
    meanEdgeLength = 0.0;
    int size = m_triTopo->getNbEdges();
    for (int i=0 ; i<size ; i++)
        meanEdgeLength += m_triGeo->computeEdgeLength(i);
    meanEdgeLength /= size;
}

template <class DataTypes>
bool FlowVisualModel<DataTypes>::isInDomain(unsigned int index, typename DataTypes::Coord p)
{
    const VecCoord& x = *this->fstate->getX();

    //test if p is in the mesh
    //1-find closest point from seed to mesh
    bool found = false;

    helper::vector<unsigned int> triangles;
    unsigned int indTest;

    if (x.size() > 0)
    {
        //if a triangle was not already found, test all the points
        if (streamLines[index].currentTriangleID == sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID)
        {
            unsigned int indexClosestPoint = 0;
            double dist = (p - x[0]).norm();
            for (unsigned int i=1 ; i<x.size() ; i++)
            {
                double d = (p - x[i]).norm();
                if (d < dist)
                {
                    indexClosestPoint = i;
                    dist = d;
                }
            }

            //2-get its TriangleShell
            triangles = m_triTopo->getTriangleVertexShell(indexClosestPoint);
            //3-check if the seed is in one of these triangles

            for (unsigned int i=0 ; i<triangles.size() && !found ; i++)
            {
                found = m_triGeo->isPointInsideTriangle(triangles[i], false, p, indTest);
                streamLines[index].currentTriangleID = triangles[i];
            }
        }
        else
        {
            const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& currentTriangle = m_triTopo->getTriangle(streamLines[index].currentTriangleID);
            for (unsigned int i=0 ; i<3 && !found; i++)
            {
                triangles = m_triTopo->getTriangleVertexShell(currentTriangle[i]);
                for (unsigned int i=0 ; i<triangles.size() && !found ; i++)
                {
                    found = m_triGeo->isPointInsideTriangle(triangles[i], false, p, indTest);
                    streamLines[index].currentTriangleID = triangles[i];
                }
            }

        }
    }

    return found;
}

template <class DataTypes>
typename DataTypes::Coord FlowVisualModel<DataTypes>::interpolateVelocity(unsigned int index, Coord p, bool &atEnd)
{
    const VecCoord& v = *this->fstate->getV();

    if (!isInDomain(index,p))
    {
        atEnd = true;
        return Coord();
    }


    //compute the velocity at "currentPos" position
    /*Coord p0 = m_triGeo->getPointPosition(t[0]);
    Coord p1 = m_triGeo->getPointPosition(t[1]);
    Coord p2 = m_triGeo->getPointPosition(t[2]);
    double coeff0 = (p0-currentPos).norm();
    double coeff1 = (p1-currentPos).norm();
    double coeff2 = (p2-currentPos).norm();*/

    //Coord velocitySeed = (v[t[0]]*coeff0 + v[t[1]]*coeff1 + v[t[2]]*coeff2)/(coeff0 + coeff1+ coeff2) ;
    Coord velocitySeed = v[streamLines[index].currentTriangleID];

    return velocitySeed;
}

template <class DataTypes>
void FlowVisualModel<DataTypes>::computeStreamLine(unsigned int index, unsigned int maxNbPoints, double dt)
{
    StreamLine &streamLine = streamLines[index];
    streamLines[index].positions.clear();

    core::componentmodel::topology::BaseMeshTopology::Triangle t;
    Coord currentPos = streamlineSeeds.getValue()[index];
    bool finished = false;

    while(streamLine.positions.size() < maxNbPoints && !finished)
    {
        streamLine.positions.push_back(currentPos);
        //p'k  	=  	pk  	+  	(1)/(2)hv(pk)
        //pk+1 	= 	pk 	+ 	hv(p'k)

        Coord v1 =	interpolateVelocity(index, currentPos, finished);
        Coord nextPositionPrime = currentPos + v1*dt*0.5;
        Coord v2 =	interpolateVelocity(index, nextPositionPrime, finished);
        Coord nextPosition = currentPos + v2*dt;
        currentPos = nextPosition;
    }
}

template <class DataTypes>
void FlowVisualModel<DataTypes>::draw()
{
    if (!getContext()->getShowVisualModels()) return;

    glDisable(GL_LIGHTING);
    VecDeriv& v = *this->fstate->getV();
    //VecCoord& y = *this->fstate->getX();

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


    core::componentmodel::topology::BaseMeshTopology::SeqTriangles triangles =  m_triTopo->getTriangles();

    unsigned int nbPoints =  m_triTopo->getNbPoints();

    helper::vector<Vec3f> colors;
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
            unsigned int indColor = 0;
            if((vmax-vmin) > 0.0 && i < v.size())
                indColor = (unsigned int)(64* ((v[i].norm()-vmin)/(vmax-vmin)));
            else indColor = 0;

            colors[t[j]]+= ColorMap[indColor];
            weight[t[j]]++;
        }
    }
    //VecCoord& x = *this->fstate->getX();
    x.clear();
    x.resize(0);

    //Show colored triangles
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

        glColor3fv((colors[t[0]]/weight[t[0]]).ptr() );
        glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
        glColor3fv((colors[t[1]]/weight[t[1]]).ptr() );
        glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
        glColor3fv((colors[t[2]]/weight[t[2]]).ptr() );
        glVertex3f((GLfloat)p2[0], (GLfloat)p2[1], (GLfloat)p2[2]);
    }
    glEnd();

    //Show Velocity
    if (showVelocityLines.getValue())
    {
        glBegin(GL_LINES);
        for(unsigned int i=0 ; i<v.size() ; i++)
        {
            if (v[i].norm() > 0.0 && vmax > 0.0)
            {

                Coord p0 = x[i];
                Coord p1 = x[i] + v[i]/vmax*viewVelocityFactor.getValue();

                glColor3f(1.0,1.0,1.0);
                glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
                glColor3f(1.0,1.0,1.0);
                glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
            }
        }
        glEnd();
    }

    //Draw StreamLines
    if (showStreamLines.getValue())
    {
        helper::vector<Coord> seeds = streamlineSeeds.getValue();
        unsigned int seedsSize = streamlineSeeds.getValue().size();
        streamLines.clear();
        streamLines.resize(seedsSize);

        for (unsigned int i=0 ; i<seedsSize ; i++)
        {
            double dtStreamLine = (meanEdgeLength / vmax) * (1.0/streamlineDtNumberOfPointsPerTriangle.getValue());
            streamLines[i].currentTriangleID = sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID;
            computeStreamLine(i,streamlineMaxNumberOfPoints.getValue(), dtStreamLine) ;
            glLineWidth(2);
            glBegin(GL_LINE_STRIP);
            for(unsigned int j=0 ; j<streamLines[i].positions.size() ; j++)
            {
                glColor3f(1.0,1.0,1.0);
                glVertex3f((GLfloat) streamLines[i].positions[j][0], (GLfloat) streamLines[i].positions[j][1], (GLfloat) streamLines[i].positions[j][2]);
            }
            glEnd();
        }
    }

    //Draw color scale
    if (showColorScale.getValue())
    {
        double xMargin = 50.0;
        double yMargin = 20.0;
        GLint currentViewport[4];
        glGetIntegerv(GL_VIEWPORT, currentViewport);
        Vec2d scaleSize = Vec2d(20, 200);
        Vec2d scalePosition = Vec2d(currentViewport[2] - xMargin - scaleSize[0], currentViewport[3] - yMargin - scaleSize[1]);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0,currentViewport[2],0,currentViewport[3]);
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

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
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }


}



}

}

}

#endif /* FLOWVISUALMODEL_INL_ */
