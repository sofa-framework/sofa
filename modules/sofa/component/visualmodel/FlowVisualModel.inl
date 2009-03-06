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
using namespace sofa::core::componentmodel::topology;

template <class DataTypes>
const double FlowVisualModel<DataTypes>::STREAMLINE_NUMBER_OF_POINTS_BY_TRIANGLE = 5.0;

template <class DataTypes>
FlowVisualModel<DataTypes>::FlowVisualModel()
    :m_tetraTopo(NULL), m_tetraGeo(NULL), meanEdgeLength(0.0)
    ,m_tag2D(initData(&m_tag2D, (std::string) "", "tag2D", "Set tag which defines 2D mesh"))
    ,m_tag3D(initData(&m_tag3D, (std::string) "", "tag3D", "Set tag which defines 3D mesh"))
    ,showVelocityLines(initData(&showVelocityLines, bool(true), "showVelocityLines", "Show velocities lines"))
    ,viewVelocityFactor(initData(&viewVelocityFactor, double(0.001), "viewVelocityFactor", "Set factor for velocity arrows"))
    ,velocityMin(initData(&velocityMin, double(-1.0), "velocityMin", "Set the minimum value of velocity for drawing"))
    ,velocityMax(initData(&velocityMax, double(1.0), "velocityMax", "Set the maximum value of velocity for drawing"))
    ,showStreamLines(initData(&showStreamLines, bool(true), "showStreamLines", "Set stream lines"))
    ,streamlineSeeds(initData(&streamlineSeeds, helper::vector<Coord>(), "streamlineSeeds", "Set streamlineSeeds for Stream Lines"))
    ,streamlineMaxNumberOfPoints(initData(&streamlineMaxNumberOfPoints, (unsigned int) 50 , "streamlineMaxNumberOfPoints", "Set the maximum number of points for each stream line"))
    ,streamlineDtNumberOfPointsPerTriangle(initData(&streamlineDtNumberOfPointsPerTriangle, (double) 5.0 , "streamlineDtNumberOfPointsPerTriangle", "Set the number of points for each step (equals ~ a triangle)"))
    ,showColorScale(initData(&showColorScale, bool(true), "showColorScale", "Set color scale"))
    ,showTetras(initData(&showTetras, bool(false), "showTetras", "Show Tetras"))
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

    //locate necessary objects with tags
    core::objectmodel::TagSet::const_iterator tagIt = this->getTags().begin();

    core::objectmodel::Tag tag2D(m_tag2D.getValue());
    core::objectmodel::Tag tag3D(m_tag3D.getValue());

    context->core::objectmodel::BaseContext::get(fstate3D, tag3D, core::objectmodel::BaseContext::SearchRoot);
    //3D
    if (!fstate3D)
    {
        std::cerr << "WARNING: FlowVisualModel has no binding FluidState3D, will considerer 2D model" <<endl;
    }
    else
    {
        this->getContext()->get (m_tetraTopo, tag3D,core::objectmodel::BaseContext::SearchRoot);

        if (m_tetraTopo == NULL)
        {
            std::cerr << "WARNING: FlowVisualModel has no binding TetrahedraSetTopology" <<endl;
            return;
        }
        else
        {
            this->getContext()->get (m_tetraGeo, tag3D, core::objectmodel::BaseContext::SearchRoot);
            if (m_tetraGeo == NULL)
            {
                std::cerr << "WARNING: FlowVisualModel has no binding TetrahedraSetGeometry" <<endl;
                return;
            }
        }
        this->getContext()->get(shader, tag3D,core::objectmodel::BaseContext::SearchRoot);
        if(!shader)
        {
            std::cerr << "WARNING: FlowVisualModel has no binding Shader ; no volumic rendering" <<endl;
        }
    }

    context->core::objectmodel::BaseContext::get(fstate2D, tag2D, core::objectmodel::BaseContext::SearchRoot);

    //FluidState2D
    if (!fstate2D)
    {
        std::cerr << "WARNING: FlowVisualModel has no binding FluidState2D" <<endl;
        return;
    }
    else
    {

        this->getContext()->get (m_triTopo, tag2D,core::objectmodel::BaseContext::SearchRoot);
        if (m_triTopo == NULL)
        {
            std::cerr << "WARNING: FlowVisualModel has no binding TriangleSetTopology" <<endl;
            return;
        }

        this->getContext()->get (m_triGeo, tag2D,core::objectmodel::BaseContext::SearchRoot);
        if (m_triGeo == NULL)
        {
            std::cerr << "WARNING: FlowVisualModel has no binding TriangleSetGeometry" <<endl;
            return;
        }
    }

    reinit();

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::initVisual()
{
    const BaseMeshTopology::SeqEdges& edges = m_triTopo->getEdges();

    for (unsigned int i=0 ; i<edges.size() ; i++)
        meanEdgeLength += m_triGeo->computeEdgeLength(i);
    meanEdgeLength /= edges.size();

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::reinit()
{

}

template <class DataTypes>
unsigned int FlowVisualModel<DataTypes>::getIndexClosestPoint(const VecCoord &x, typename DataTypes::Coord p)
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
    return indexClosestPoint;
}

template <class DataTypes>
bool FlowVisualModel<DataTypes>::isInDomain(unsigned int index, typename DataTypes::Coord p)
{
    //test if p is in the mesh
    //1-find closest point from seed to mesh
    bool found = false;
    BaseMeshTopology::TriangleID triangleID = BaseMeshTopology::InvalidID;
    helper::vector<BaseMeshTopology::TriangleID> triangles;
    //const VecCoord& x = *this->fstate2D->getV();

    unsigned int indTest;

    if (x.size() > 0)
    {
        //if a triangle was not already found, test all the points
        if (streamLines[index].currentTriangleID == BaseMeshTopology::InvalidID)
        {
            unsigned int indexClosestPoint = getIndexClosestPoint(*this->fstate2D->getX(), p);

            //2-get its TriangleShell
            triangles = m_triTopo->getTriangleVertexShell(indexClosestPoint);
            //3-check if the seed is in one of these triangles
            streamLines[index].trianglesAroundLastPoint.clear();
            for (unsigned int i=0 ; i<triangles.size() ; i++)
            {
                if (!found)
                {
                    if ( (found = m_triGeo->isPointInsideTriangle(triangles[i], false, p, indTest)) )
                    {
                        triangleID = triangles[i];
                    }
                }

                //fill the set of triangles
                streamLines[index].trianglesAroundLastPoint.insert(triangles[i]);
            }
        }
        else
        {
            //test if the new point is still in the current triangle
            if((found = m_triGeo->isPointInTriangle(streamLines[index].currentTriangleID, false, p, indTest)) )
            {
                triangleID = streamLines[index].currentTriangleID;
            }
            //find the new triangle (if any) and compute the new triangles set
            else
            {
                streamLines[index].trianglesAroundLastPoint.clear();
                const BaseMeshTopology::Triangle& currentTriangle = m_triTopo->getTriangle(streamLines[index].currentTriangleID);

                for (unsigned int i=0 ; i<3; i++)
                {
                    triangles = m_triTopo->getTriangleVertexShell(currentTriangle[i]);
                    for (unsigned int i=0 ; i<triangles.size() ; i++)
                    {
                        if (!found)
                        {
                            if ( (found = m_triGeo->isPointInsideTriangle(triangles[i], false, p, indTest)) )
                            {
                                triangleID = triangles[i];
                            }
                        }
                        streamLines[index].trianglesAroundLastPoint.insert(triangles[i]);
                    }
                }
            }
        }
    }

    if (found)
        streamLines[index].currentTriangleID = triangleID;

    return found;
}

template <class DataTypes>
typename DataTypes::Coord FlowVisualModel<DataTypes>::interpolateVelocity(unsigned int index, Coord p, bool &atEnd)
{
    if (!isInDomain(index,p))
    {
        atEnd = true;
        return Coord();
    }

    helper::vector<BaseMeshTopology::TriangleID> triangles;
    double distCoeff=0.0;
    double sumDistCoeff=0.0;
    Coord velocitySeed;

    for(helper::set<BaseMeshTopology::TriangleID>::iterator it = streamLines[index].trianglesAroundLastPoint.begin() ; it != streamLines[index].trianglesAroundLastPoint.end() ; it++)
    {
        unsigned int ind = (*it);
        distCoeff = 1/(p-x[ind]).norm2();
        velocitySeed += velocityAtVertex[ind]*distCoeff;
        sumDistCoeff += distCoeff;
    }
    velocitySeed /= sumDistCoeff;

    //velocitySeed = Coord(0.0,0.0,0.0);
    //Coord velocitySeed = (v[t[0]]*coeff0 + v[t[1]]*coeff1 + v[t[2]]*coeff2)/(coeff0 + coeff1+ coeff2) ;
    Coord velocitySeed2 = velocityAtVertex[streamLines[index].currentTriangleID];

    //std::cout << velocitySeed << " " << velocitySeed2 << std::endl;

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
void FlowVisualModel<DataTypes>::interpolateVelocityAtVertices()
{
    unsigned int indexClosestPoint;
    unsigned int nbPoints =  m_triTopo->getNbPoints();
    helper::vector<unsigned int> weight;

    velocityAtVertex.resize(nbPoints);
    weight.resize(nbPoints);
    std::fill( weight.begin(), weight.end(), 0 );
    std::fill( velocityAtVertex.begin(), velocityAtVertex.end(), Coord() );

    core::componentmodel::topology::BaseMeshTopology::SeqTriangles triangles =  m_triTopo->getTriangles();

    if (!m_tetraTopo)
    {
        VecDeriv& v2d = *this->fstate2D->getV();

        for(unsigned int i=0 ; i<triangles.size() ; i++)
        {
            core::componentmodel::topology::BaseMeshTopology::Triangle t = (triangles[i]);
            for(unsigned int j=0 ; j<3 ; j++)
            {
                velocityAtVertex[t[j]] += v2d[i];
                weight[t[j]]++;
            }
        }
        for(unsigned int i=0 ; i<nbPoints ; i++)
        {
            velocityAtVertex[i] /= weight[i];
        }
    }
    else
    {
        VecDeriv& v3d = *this->fstate3D->getV();

        for (unsigned int i=0 ; i<nbPoints ; i++)
        {
            Coord p = m_triGeo->getPointPosition(i);
            indexClosestPoint = getIndexClosestPoint(*this->fstate3D->getX(), p);

            helper::vector<BaseMeshTopology::TetraID> tetras = m_tetraTopo->getTetraVertexShell(indexClosestPoint);
            //VecCoord ptetras;
            for (unsigned int j=0 ; j<tetras.size() ; j++)
            {
                velocityAtVertex[i] += v3d[tetras[j]];
                weight[i]++;
            }
            velocityAtVertex[i] /= weight[i];
        }

    }


}

template <class DataTypes>
void FlowVisualModel<DataTypes>::drawTetra()
{
    VecCoord& tetrasX = *this->fstate3D->getX();
    VecDeriv& v3d = *this->fstate3D->getV();
    //glEnable (GL_BLEND);

    //glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);


    if(shader)
        shader->start();
    //Tetra
    if(m_tetraTopo)
    {
        //draw tetrahedra
        core::componentmodel::topology::BaseMeshTopology::SeqTetras tetrahedra=  m_tetraTopo->getTetras();
        glPointSize(10.0);
        for (unsigned int i=0 ; i<tetrahedra.size() ; i++)
        {
            /*
            #ifdef GL_LINES_ADJACENCY_EXT
            glBegin(GL_LINES_ADJACENCY_EXT);
            #else
            glBegin(GL_POINTS);
            #endif

            for (unsigned int j=0 ; j< 4 ; j++)
            {
            	Coord v = tetrasX[(*it)[j]];
            	glColor4f(1.0,1.0,1.0,1.0);
            	//glColor4f(j%3,(j+1)%3,(j+2)%3,1.0);
            	glVertex3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]);

            }

            glEnd();
            */
            const BaseMeshTopology::Tetra tetra = m_tetraTopo->getTetra(i);
            Coord center(0.0,0.0,0.0);

            for (unsigned int j=0 ; j< 4 ; j++)
            {
                Coord v = tetrasX[tetra[j]];
                center += v*0.25;
            }
            glBegin(GL_LINES);
            glColor3f(0.0,1.0,0.0);
            glVertex3f(center[0], center[1],center[2]);
            glVertex3f(center[0] + v3d[i][0]/vmax* viewVelocityFactor.getValue(), center[1] + v3d[i][1]/vmax* viewVelocityFactor.getValue(),center[2]+ v3d[i][2]/vmax* viewVelocityFactor.getValue());
            glEnd();

        }
    }
    if(shader)
        shader->stop();

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::draw()
{
    if (!getContext()->getShowVisualModels()) return;
    glDepthMask(GL_TRUE);
    glDisable(GL_LIGHTING);
    interpolateVelocityAtVertices();
    //VecCoord& x = *this->fstate2D->getX();

    double vmax2 = velocityAtVertex[0].norm2();
    vmin = velocityAtVertex[0].norm();

    //search vmax
    for (unsigned int i=1 ; i<velocityAtVertex.size() ; i++)
    {
        if (velocityAtVertex[i].norm2() > vmax2)
            vmax2=velocityAtVertex[i].norm2();
    }

    vmax = (velocityMax.getValue() < sqrt(vmax2)) ? sqrt(vmax2) : velocityMax.getValue();

    if (velocityMin.getValue() < 0.0)
    {
        //search vmin
        for (unsigned int i=1 ; i<velocityAtVertex.size() ; i++)
        {
            if (velocityAtVertex[i].norm() < vmin)
                vmin=velocityAtVertex[i].norm();
        }
        if (vmin < 0.0) vmin = 0.0;
    }
    else vmin = velocityMin.getValue();

    {
        core::componentmodel::topology::BaseMeshTopology::SeqTriangles triangles =  m_triTopo->getTriangles();

        int indColor0, indColor1, indColor2;
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

            if((vmax-vmin) > 0.0)
            {
                indColor0 = (int)(64* ((velocityAtVertex[t[0]].norm()-vmin)/(vmax-vmin)));
                indColor1 = (int)(64* ((velocityAtVertex[t[1]].norm()-vmin)/(vmax-vmin)));
                indColor2 = (int)(64* ((velocityAtVertex[t[2]].norm()-vmin)/(vmax-vmin)));
                if (indColor0 < 0) indColor0 = 0;
                if (indColor1 < 0) indColor1 = 0;
                if (indColor2 < 0) indColor2 = 0;
                if (indColor0 >= 64) indColor0 = 63;
                if (indColor1 >= 64) indColor1 = 63;
                if (indColor2 >= 64) indColor2 = 63;

            }
            else indColor0 = indColor1 = indColor2 = 0;
            glColor3f(ColorMap[indColor0][0], ColorMap[indColor0][1], ColorMap[indColor0][2]);
            glVertex3f((GLfloat)p0[0], (GLfloat)p0[1], (GLfloat)p0[2]);
            glColor3f(ColorMap[indColor1][0], ColorMap[indColor1][1], ColorMap[indColor1][2]);
            glVertex3f((GLfloat)p1[0], (GLfloat)p1[1], (GLfloat)p1[2]);
            glColor3f(ColorMap[indColor2][0], ColorMap[indColor2][1], ColorMap[indColor2][2]);
            glVertex3f((GLfloat)p2[0], (GLfloat)p2[1], (GLfloat)p2[2]);
        }
        glEnd();

        VecCoord& x2D = *this->fstate2D->getX();

        //Show Velocity
        if (showVelocityLines.getValue())
        {
            glBegin(GL_LINES);
            for(unsigned int i=0 ; i<velocityAtVertex.size() ; i++)
            {
                if (velocityAtVertex[i].norm() > 0.0 && vmax > 0.0)
                {
                    Coord p0 = x2D[i];
                    Coord p1 = x2D[i] + velocityAtVertex[i]/vmax*viewVelocityFactor.getValue();

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
                streamLines[i].currentTriangleID = BaseMeshTopology::InvalidID;
                computeStreamLine(i,streamlineMaxNumberOfPoints.getValue(), dtStreamLine) ;
                glPointSize(10.0);
                glBegin(GL_POINTS);
                glVertex3f(seeds[i][0], seeds[i][1], seeds[i][2]);
                glEnd();
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

        if (showTetras.getValue())
            drawTetra();

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
            glEnable(GL_DEPTH_TEST);
            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
        }
    }

}



}

}

}

#endif /* FLOWVISUALMODEL_INL_ */
