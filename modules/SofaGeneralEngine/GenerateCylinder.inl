/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_GENERATECYLINDER_INL
#define SOFA_COMPONENT_ENGINE_GENERATECYLINDER_INL

#include "GenerateCylinder.h"
#include <sofa/helper/rmath.h> //M_PI

namespace sofa
{

namespace component
{

namespace engine
{


    const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};

template <class DataTypes>
GenerateCylinder<DataTypes>::GenerateCylinder()
    : f_outputTetrahedraPositions ( initData (&f_outputTetrahedraPositions, "output_TetrahedraPosition", "output array of 3d points of tetrahedra mesh") )
    , f_outputTrianglesPositions ( initData (&f_outputTrianglesPositions, "output_TrianglesPosition", "output array of 3d points of triangle mesh") )
    , f_tetrahedra( initData (&f_tetrahedra, "tetrahedra", "output mesh tetrahedra") )
    , f_triangles( initData (&f_triangles, "triangles", "output triangular mesh") )
    , f_bezierTriangleWeight( initData (&f_bezierTriangleWeight, "BezierTriangleWeights", "weights of rational Bezier triangles") )
    , f_isBezierTriangleRational( initData (&f_isBezierTriangleRational, "isBezierTriangleRational", "booleans indicating if each Bezier triangle is rational or integral") )
    , f_bezierTriangleDegree( initData (&f_bezierTriangleDegree, "BezierTriangleDegree", "order of Bezier triangles") )
    , f_bezierTetrahedronWeight( initData (&f_bezierTetrahedronWeight, "BezierTetrahedronWeights", "weights of rational Bezier tetrahedra") )
    , f_isBezierTetrahedronRational( initData (&f_isBezierTetrahedronRational, "isBezierTetrahedronRational", "booleans indicating if each Bezier tetrahedron is rational or integral") )
    , f_bezierTetrahedronDegree( initData (&f_bezierTetrahedronDegree, "BezierTetrahedronDegree", "order of Bezier tetrahedra") )
    , f_radius( initData (&f_radius,(Real)0.2, "radius", "input cylinder radius") )
    , f_height( initData (&f_height,(Real)1.0, "height", "input cylinder height") )
    , f_origin( initData (&f_origin,Coord(), "origin", "cylinder origin point") )
    , f_openSurface( initData (&f_openSurface,true, "openSurface", "if the cylinder is open at its 2 ends") )
    , f_resolutionCircumferential( initData (&f_resolutionCircumferential,(size_t)6, "resCircumferential", "Resolution in the circumferential direction") )
   , f_resolutionRadial( initData (&f_resolutionRadial,(size_t)3, "resRadial", "Resolution in the radial direction") )
  , f_resolutionHeight( initData (&f_resolutionHeight,(size_t)5, "resHeight", "Resolution in the height direction") )
{
    addAlias(&f_outputTetrahedraPositions,"position");
    addAlias(&f_outputTetrahedraPositions,"output_position");
}


template <class DataTypes>
void GenerateCylinder<DataTypes>::init()
{
    addInput(&f_radius);
    addInput(&f_height);
    addInput(&f_origin);

    addInput(&f_resolutionCircumferential);
    addInput(&f_resolutionRadial);
    addInput(&f_resolutionHeight);

    addOutput(&f_triangles);
    addOutput(&f_outputTrianglesPositions);
    addOutput(&f_bezierTriangleWeight);
    addOutput(&f_isBezierTriangleRational);


    addOutput(&f_tetrahedra);
    addOutput(&f_outputTetrahedraPositions);
    addOutput(&f_bezierTetrahedronWeight);
    addOutput(&f_isBezierTetrahedronRational);
    setDirtyValue();
}

template <class DataTypes>
void GenerateCylinder<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void GenerateCylinder<DataTypes>::update()
{
    const Real radius = f_radius.getValue();
    const Real height = f_height.getValue();
    const Coord origin = f_origin.getValue();

    const size_t freqTheta=f_resolutionCircumferential.getValue();
    const size_t freqR=f_resolutionRadial.getValue();
    const size_t freqZ=f_resolutionHeight.getValue();

    cleanDirty();

    helper::WriteOnlyAccessor<Data<VecCoord> > out = f_outputTetrahedraPositions;
    helper::WriteOnlyAccessor<Data<SeqTetrahedra> > tetras = f_tetrahedra;

    size_t  nbVertices= (freqTheta*freqR+1)*freqZ;
    out.resize(nbVertices);

    std::vector<bool> onSurface;
    onSurface.resize(nbVertices);
    std::fill(onSurface.begin(),onSurface.end(),false);

    Real zValue,r,theta,xValue,yValue;
    size_t i,j,k,index;
    Coord pos;

    for(index=0,i=0;i<freqZ;i++) {
        // vertex index = i*(freQTheta*freqR+1)
        zValue=i*height/(freqZ-1);
        pos=Coord(0,0,zValue);
        pos+=origin;
        out[index++]=pos;
        for(j=1;j<=freqR;++j) {
            r=j*radius/(freqR);
            for(k=0;k<freqTheta;++k) {
                theta=(Real)(k*2*M_PI/freqTheta);
                xValue= r*cos(theta);
                yValue= r*sin(theta);
                pos=Coord(xValue,yValue,zValue);
                pos+=origin;
                if (j==freqR)
                    onSurface[index]=true;
                out[index++]=pos;

            }
        }
    }


    size_t nbTetrahedra=3*freqTheta*(freqZ-1) + 6*(freqR-1)*freqTheta*(freqZ-1);
    tetras.resize(nbTetrahedra);

    size_t  offsetZ=(freqTheta*freqR+1);
    size_t prevk;
    index=0;
    size_t prism[6];
    size_t hexahedron[8];

    for(i=1;i<freqZ;i++) {
        size_t  centerIndex0=i*offsetZ;
        prevk=freqTheta;

        for(k=1;k<=freqTheta;++k) {
            /// create triangular prism
            prism[0]=centerIndex0;
            prism[1]=centerIndex0+prevk;
            prism[2]=centerIndex0+k;
            prism[3]=prism[0]-offsetZ;
            prism[4]=prism[1]-offsetZ;
            prism[5]=prism[2]-offsetZ;
            /// decompose triangular prism into 3 tetrahedra
            tetras[index++]=Tetrahedron(prism[1],prism[0],prism[2],prism[3]);
            tetras[index++]=Tetrahedron(prism[1],prism[2],prism[4],prism[3]);
            tetras[index++]=Tetrahedron(prism[3],prism[4],prism[5],prism[2]);

            prevk=k;
        }

        for(j=1;j<freqR;++j) {
            prevk=freqTheta;
            for(k=1;k<=freqTheta;++k) {
                /// create hexahedron
                hexahedron[0]=centerIndex0+k;
                hexahedron[1]=centerIndex0+prevk;
                hexahedron[2]=centerIndex0+k+freqTheta;
                hexahedron[3]=centerIndex0+prevk+freqTheta;
                hexahedron[4]=hexahedron[0]-offsetZ;
                hexahedron[5]=hexahedron[1]-offsetZ;
                hexahedron[6]=hexahedron[2]-offsetZ;
                hexahedron[7]=hexahedron[3]-offsetZ;
                /// decompose hexahedron into 6 tetra
                tetras[index++]=Tetrahedron(hexahedron[0],hexahedron[5],hexahedron[4],hexahedron[7]);
                tetras[index++]=Tetrahedron(hexahedron[0],hexahedron[1],hexahedron[5],hexahedron[3]);
                tetras[index++]=Tetrahedron(hexahedron[5],hexahedron[0],hexahedron[3],hexahedron[7]);
                tetras[index++]=Tetrahedron(hexahedron[0],hexahedron[3],hexahedron[7],hexahedron[2]);
                tetras[index++]=Tetrahedron(hexahedron[7],hexahedron[0],hexahedron[2],hexahedron[4]);
                tetras[index++]=Tetrahedron(hexahedron[6],hexahedron[7],hexahedron[2],hexahedron[4]);
                prevk=k;
            }
            centerIndex0+=freqTheta;
        }

    }
    if (f_bezierTetrahedronDegree.getValue()>1) {

        size_t degreeTetrahedron=f_bezierTetrahedronDegree.getValue();
        // fill the bezier tetrahedra weight to 1 for integral  tetrahedron vertices
        sofa::helper::vector<Real> & bezierTetrahedronWeight=*(f_bezierTetrahedronWeight.beginEdit());
        // initialize the weight to 1
        bezierTetrahedronWeight.resize(out.size());
        std::fill(bezierTetrahedronWeight.begin(),bezierTetrahedronWeight.end(),(Real)1.0);
        // initialize the rational flag for each tetrahedron to false
        helper::WriteOnlyAccessor<Data <sofa::helper::vector<bool> > >  isRationalSpline=f_isBezierTetrahedronRational;
        isRationalSpline.resize(nbTetrahedra);
        std::fill(isRationalSpline.begin(),isRationalSpline.end(),false);
        // set the tetrahedra next to the circular surface to true
        size_t nbSurfaceNodes,tetraRank;

        // parse edges
        std::map<Edge,size_t> edgeMap;
        std::map<Edge,size_t>::iterator item;
        SeqTetrahedra::iterator itt;
        //size_t pointOffset=freqTheta*freqZ;
        Real ctheta=(Real)cos(M_PI/freqTheta);
        Coord posTmp,posTmp2;
        std::vector<Edge> edgeArray;
        for (tetraRank=0,itt=tetras.begin();itt!=tetras.end();++itt,++tetraRank) {
            nbSurfaceNodes=0;
            for (i=0;i<4;++i){
                if (onSurface[(*itt)[i]])
                    ++nbSurfaceNodes;
            }
            // if the tetrahedron has at least 2 vertices on the circular surface then the bezier tetrahedron is rational
            if (nbSurfaceNodes>=2)
                isRationalSpline[tetraRank]=true;
            // parse the tetrahedron edge
            for (i=0;i<6;++i){

                Edge e,se;
                e[0]=(*itt)[edgesInTetrahedronArray[i][0]];
                e[1]=(*itt)[edgesInTetrahedronArray[i][1]];
                if (e[0]>e[1]){
                    se[0]=e[1];se[1]=e[0];
                } else {
                    se=e;
                }
                if ((item=edgeMap.find(se))==edgeMap.end()){
                    edgeMap.insert(std::pair<Edge,size_t>(se,edgeArray.size()));
                    edgeArray.push_back(se);
                    e=se;

                    // add Bezier points along the edge

                    if ( (out[e[0]][2]==out[e[1]][2]) &&(onSurface[e[0]])&&(onSurface[e[1]])) {
                        // the edge is along a circle
                        if (degreeTetrahedron==2) {
                            pos=(out[e[0]]+out[e[1]])/2.0;
                            pos[2]=0;
                            pos*=radius/(pos.norm()*ctheta);
                            pos[2]=out[e[1]][2];
                            out.push_back(pos);
                            bezierTetrahedronWeight.push_back((Real)ctheta);
                        } else if (degreeTetrahedron==3) {
                            posTmp=(out[e[0]]+out[e[1]])/2.0;
                            posTmp[2]=0;
                            posTmp*=radius/(posTmp.norm()*ctheta);
                            posTmp[2]=out[e[1]][2];
                            pos=(2*ctheta*posTmp+out[e[0]])/(1+2*ctheta);
                            out.push_back(pos);
                            pos=(2*ctheta*posTmp+out[e[1]])/(1+2*ctheta);
                            out.push_back(pos);
                            bezierTetrahedronWeight.push_back((Real)(1+2*ctheta)/3.0f);
                            bezierTetrahedronWeight.push_back((Real)(1+2*ctheta)/3.0f);
                        } else {
                            for (j=1;j<degreeTetrahedron;++j) {
                                // interpolated position
                                pos= ((Real) j*out[e[1]]+(Real)(degreeTetrahedron-j)*out[e[0]])/degreeTetrahedron;
                                out.push_back(pos);
                                // weight is 1
                                bezierTetrahedronWeight.push_back((Real)1.0f);
                            }
                        }


                    } else if ( (out[e[0]][2]!=out[e[1]][2]) &&  (out[e[0]][1]!=out[e[1]][1]) && (onSurface[e[0]])&&(onSurface[e[1]]) ) {
                        // the edge is along a diagonal and therefore is curved
                        // the edge is along a circle
                        if (degreeTetrahedron==2) {
                            pos=(out[e[0]]+out[e[1]])/2.0;
                            pos[2]=0;
                            pos*=radius/(pos.norm()*ctheta);
                            pos[2]=out[e[1]][2];
                            out.push_back(pos);
                            bezierTetrahedronWeight.push_back((Real)ctheta);
                        } else if (degreeTetrahedron==3) {
                            posTmp=(out[e[0]]+out[e[1]])/2.0;
                            posTmp[2]=0;
                            posTmp*=radius/(posTmp.norm()*ctheta);
                            posTmp[2]=out[e[0]][2];
                            posTmp2=out[e[0]];
                            posTmp2[2]=out[e[1]][2];
                            pos=(2*ctheta*posTmp+posTmp2)/(1+2*ctheta);
                            out.push_back(pos);
                            posTmp[2]=out[e[1]][2];
                            posTmp2=out[e[1]];
                            posTmp2[2]=out[e[0]][2];
                            pos=(2*ctheta*posTmp+posTmp2)/(1+2*ctheta);

                            out.push_back(pos);
                            bezierTetrahedronWeight.push_back((Real)(1+2*ctheta)/3.0f);
                            bezierTetrahedronWeight.push_back((Real)(1+2*ctheta)/3.0f);
                        } else {
                            for (j=1;j<degreeTetrahedron;++j) {
                                // interpolated position
                                pos= ((Real) j*out[e[1]]+(Real)(degreeTetrahedron-j)*out[e[0]])/degreeTetrahedron;
                                out.push_back(pos);
                                // weight is 1
                                bezierTetrahedronWeight.push_back((Real)1.0f);
                            }
                        }

                    } else {
                        // as default the edge is straight : compute control point by degree elevation
                        for (j=1;j<degreeTetrahedron;++j) {
                            // interpolated position
                            pos= ((Real) j*out[e[1]]+(Real)(degreeTetrahedron-j)*out[e[0]])/degreeTetrahedron;
                            out.push_back(pos);
                            // weight is 1
                            bezierTetrahedronWeight.push_back((Real)1.0f);
                        }
                    }
                }

            }
        }
        // add points inside triangles
        if (f_bezierTetrahedronDegree.getValue()>2) {
            std::map<Triangle,size_t> triangleMap;
            std::map<Triangle,size_t>::iterator ittm;
            size_t  v[3],triangleRank;

            for (triangleRank=0,itt=tetras.begin();itt!=tetras.end();++itt) {

                // parse the tetrahedron edge
                for (i=0;i<4;++i){


                    if (i%2)
                    {
                        v[0]=(*itt)[(i+1)%4];
                        v[1]=(*itt)[(i+2)%4];
                        v[2]=(*itt)[(i+3)%4];
                    }
                    else
                    {
                        v[0]=(*itt)[(i+1)%4];
                        v[2]=(*itt)[(i+2)%4];
                        v[1]=(*itt)[(i+3)%4];
                    }

                    // permute v such that v[0] has the smallest index
                    while ((v[0]>v[1]) || (v[0]>v[2]))
                    {
                        size_t val=v[0];
                        v[0]=v[1];
                        v[1]=v[2];
                        v[2]=val;
                    }

                    // check if a triangle with an opposite orientation already exists
                    Triangle tr = Triangle(v[0], v[2], v[1]);

                    if (triangleMap.find(tr) == triangleMap.end())
                    {
                        // triangle not in triangleMap so create a new one
                        tr = Triangle(v[0], v[1], v[2]);
                        if (triangleMap.find(tr) == triangleMap.end())
                        {
                            triangleMap[tr] = triangleRank++;
                        }
                        else
                        {
                            serr << "ERROR: duplicate triangle " << tr << " in tetra " << i <<" : " << (*itt) << sendl;
                        }
                        // tests if all vertices are on the circular surface
                        nbSurfaceNodes=0;

                        for (j=1;j<4;++j){
                            if (onSurface[(*itt)[(i+j)%4]])
                                ++nbSurfaceNodes;
                        }
                        if (nbSurfaceNodes==3) {
                            // this is the more complex case as the bezier triangle is rational and non affine
                            // ONLY the case for degree=3 is considered here.
                            for(j=0;(out[tr[(j+1)%3]][2]==out[tr[(j+2)%3]][2])||(out[tr[(j+1)%3]][1]==out[tr[(j+2)%3]][1]);++j);
                            for(k=1;out[tr[(j+k)%3]][2]!=out[tr[j]][2];++k);
                            posTmp=(out[tr[j]]+out[tr[(j+k)%3]])/2.0;
                            posTmp[2]=0;
                            posTmp*=radius/(posTmp.norm()*ctheta);
                            posTmp[2]=out[tr[(j+1)%3]][2];
                            posTmp2=posTmp;
                            posTmp2[2]=out[tr[(j+2)%3]][2];
                            pos=(out[tr[j]]+ctheta*posTmp+ctheta*posTmp2)/(1+2*ctheta);
                            out.push_back(pos);
                            bezierTetrahedronWeight.push_back((Real)(1+2*ctheta)/3.0f);
                        } else {
                            // this is the affine case. The triangular control points are simply built from a tesselation of the triangles
                            for (j=1;j<(degreeTetrahedron-1);++j) {
                                for (k=1;k<(degreeTetrahedron-j);++k) {
                                    pos= out[tr[0]]*k/degreeTetrahedron+(Real)(degreeTetrahedron-j-k)*out[tr[1]]/degreeTetrahedron+
                                        (Real)(j)*out[tr[2]]/degreeTetrahedron;
                                    out.push_back(pos);
                                    // weight is 1
                                    bezierTetrahedronWeight.push_back((Real)1.0f);
                                }

                            }
                        }

                    }
                }
            }
        }
        }


    bool open=f_openSurface.getValue();
    helper::WriteOnlyAccessor<Data<VecCoord> > outTrian = f_outputTrianglesPositions;
    SeqTriangles  &triangles = *(f_triangles.beginEdit());

    if (open) {
        nbVertices= (freqTheta)*(freqZ);
    } else
        nbVertices= (freqTheta)*(freqZ+2*(freqR-1))+2;
    outTrian.resize(nbVertices);

    index=0;



    for(i=0;i<freqZ;i++) {
        // vertex index = i*(freQTheta*freqR+1)
        zValue=i*height/(freqZ-1);
        for(k=0;k<freqTheta;++k) {
            theta=(Real)(k*2*M_PI/freqTheta);
            xValue= radius*cos(theta);
            yValue= radius*sin(theta);
            pos=Coord(xValue,yValue,zValue);
            pos+=origin;
            outTrian[index++]=pos;
        }
    }
    if (!open) {
        for (i=0;i<2;++i) {
            zValue=i*height;
            pos=Coord(0,0,zValue)+origin;
            outTrian[index++]=pos;
            for(j=1;j<freqR;++j) {
                r=j*radius/(freqR);

                for(k=0;k<freqTheta;++k) {
                    theta=(Real)(k*2*M_PI/freqTheta);
                    xValue= r*cos(theta);
                    yValue= r*sin(theta);
                    pos=Coord(xValue,yValue,zValue);
                    pos+=origin;
                    outTrian[index++]=pos;
                }
            }
        }
    }


    size_t nbTriangles;
    if (open) {
        nbTriangles=2*freqTheta*(freqZ-1);
    } else {
        nbTriangles=2*freqTheta*(freqZ-1)+2*freqTheta+4*(freqR-1)*freqTheta;
    }

    triangles.resize(nbTriangles);


    index=0;


    for(i=1;i<freqZ;i++) {

        for(k=0;k<freqTheta;++k) {
            /// create triangles
            triangles[index++]=Triangle(i*freqTheta+k,(i-1)*freqTheta+k,(i-1)*freqTheta+(k+1)%freqTheta);
            triangles[index++]=Triangle(i*freqTheta+k,(i-1)*freqTheta+(k+1)%freqTheta,i*freqTheta+(k+1)%freqTheta);
        }

    }
    if (!open) {
        for (i=0;i<2;++i) {
            size_t centerIndex0=freqTheta*freqZ+i*(freqTheta*(freqR-1)+1);
            // inner ring with triangles linking the center vertex
            for(k=0;k<freqTheta;++k) {
                /// create triangular prism
                if (i==1)
                    triangles[index++]=Triangle( centerIndex0,centerIndex0+1+k, centerIndex0+1+(k+1)%freqTheta);
                else
                    triangles[index++]=Triangle( centerIndex0, centerIndex0+1+(k+1)%freqTheta,centerIndex0+1+k);
            }
            for(j=0;j<(freqR-2);++j) {

                for(k=0;k<freqTheta;++k) {
                    if (i==1) {
                        triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,centerIndex0+1+(j+1)*freqTheta+k,
                            centerIndex0+1+(j+1)*freqTheta+(k+1)%freqTheta);
                        triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,centerIndex0+1+(j+1)*freqTheta+(k+1)%freqTheta,
                            centerIndex0+1+(j)*freqTheta+(k+1)%freqTheta);
                    }
                    else {
                        triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,centerIndex0+1+(j+1)*freqTheta+(k+1)%freqTheta,
                            centerIndex0+1+(j+1)*freqTheta+k);
                        triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,centerIndex0+1+(j)*freqTheta+(k+1)%freqTheta,
                            centerIndex0+1+(j+1)*freqTheta+(k+1)%freqTheta);
                    }
                }
            }
            j=freqR-2;
            // last ring
            for(k=0;k<freqTheta;++k) {
                if (i==1) {
                    triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,i*(freqZ-1)*freqTheta+k,
                        i*(freqZ-1)*freqTheta+(k+1)%freqTheta);
                    triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,i*(freqZ-1)*freqTheta+(k+1)%freqTheta,
                        centerIndex0+1+(j)*freqTheta+(k+1)%freqTheta);
                }
                else {
                    triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,i*(freqZ-1)*freqTheta+(k+1)%freqTheta,
                        i*(freqZ-1)*freqTheta+k);
                    triangles[index++]=Triangle(centerIndex0+1+j*freqTheta+k,centerIndex0+1+(j)*freqTheta+(k+1)%freqTheta,
                        i*(freqZ-1)*freqTheta+(k+1)%freqTheta);
                }
            }
        }
    }

    if (f_bezierTriangleDegree.getValue()>1) {

        size_t degreeTriangle=f_bezierTriangleDegree.getValue();
        // fill the bezier triangle weight to 1 for regular triangle vertices
        sofa::helper::vector<Real> & bezierTriangleWeight=*(f_bezierTriangleWeight.beginEdit());
        // initialize the weight to 1
        bezierTriangleWeight.resize(outTrian.size());
        std::fill(bezierTriangleWeight.begin(),bezierTriangleWeight.end(),(Real)1.0);
        // initialize the rational flag for each triangle to false
        helper::WriteOnlyAccessor<Data <sofa::helper::vector<bool> > >  isRationalSpline=f_isBezierTriangleRational;
        isRationalSpline.resize(nbTriangles);
        // sets the first set of triangles to be rational
        std::fill(isRationalSpline.begin(),isRationalSpline.end(),false);


        // parse edges
        std::map<Edge,size_t> edgeMap;
        std::map<Edge,size_t>::iterator item;
        SeqTriangles::iterator itt;
        size_t pointOffset=freqTheta*freqZ;
        Real ctheta=(Real)cos(M_PI/freqTheta);
        Coord posTmp,posTmp2;
        size_t nbCircularSurfaceNodes;
        std::vector<Edge> edgeArray;
        for (itt=triangles.begin();itt!=triangles.end();++itt) {
            nbCircularSurfaceNodes=0;
            for (i=0;i<3;++i){
                if ((*itt)[i]<(freqTheta*freqZ))
                    ++nbCircularSurfaceNodes;
            }
            // if the triangle has at least 2 vertices on the circular surface then the bezier triangle is rational
            if (nbCircularSurfaceNodes>=2)
                isRationalSpline[itt-triangles.begin()]=true;
            for (i=0;i<3;++i){
                Edge e,se;
                e[0]=(*itt)[(i+1)%3];
                e[1]=(*itt)[(i+2)%3];
                if (e[0]>e[1]){
                    se[0]=e[1];se[1]=e[0];
                } else {
                    se=e;
                }
                if ((item=edgeMap.find(se))==edgeMap.end()){
                    edgeMap.insert(std::pair<Edge,size_t>(se,edgeArray.size()));
                    edgeArray.push_back(e);
                    // add Bezier points along the edge
                    if ( (outTrian[e[0]][2]==outTrian[e[1]][2]) &&(e[0]<pointOffset)&&(e[1]<pointOffset)) {
                        // the edge is along a circle
                        if (degreeTriangle==2) {
                            pos=(outTrian[e[0]]+outTrian[e[1]])/2.0;
                            pos[2]=0;
                            pos*=radius/(pos.norm()*ctheta);
                            pos[2]=outTrian[e[1]][2];
                            outTrian.push_back(pos);
                            bezierTriangleWeight.push_back((Real)ctheta);
                        } else if (degreeTriangle==3) {
                            posTmp=(outTrian[e[0]]+outTrian[e[1]])/2.0;
                            posTmp[2]=0;
                            posTmp*=radius/(posTmp.norm()*ctheta);
                            posTmp[2]=outTrian[e[1]][2];
                            pos=(2*ctheta*posTmp+outTrian[e[0]])/(1+2*ctheta);
                            outTrian.push_back(pos);
                            pos=(2*ctheta*posTmp+outTrian[e[1]])/(1+2*ctheta);
                            outTrian.push_back(pos);
                            bezierTriangleWeight.push_back((Real)(1+2*ctheta)/3.0f);
                            bezierTriangleWeight.push_back((Real)(1+2*ctheta)/3.0f);
                        } else {
                            for (j=1;j<degreeTriangle;++j) {
                                // interpolated position
                                pos= ((Real) j*outTrian[e[1]]+(Real)(degreeTriangle-j)*outTrian[e[0]])/degreeTriangle;
                                outTrian.push_back(pos);
                                // weight is 1
                                bezierTriangleWeight.push_back((Real)1.0f);
                            }
                        }


                    } else if ( (outTrian[e[0]][2]!=outTrian[e[1]][2]) &&  (outTrian[e[0]][1]!=outTrian[e[1]][1]) &&(e[0]<pointOffset)&&(e[1]<pointOffset)) {
                        // the edge is along a diagonal and therefore is curved
                        // the edge is along a circle
                        if (degreeTriangle==2) {
                            pos=(outTrian[e[0]]+outTrian[e[1]])/2.0;
                            pos[2]=0;
                            pos*=radius/(pos.norm()*ctheta);
                            pos[2]=outTrian[e[1]][2];
                            outTrian.push_back(pos);
                            bezierTriangleWeight.push_back((Real)ctheta);
                        } else if (degreeTriangle==3) {
                            posTmp=(outTrian[e[0]]+outTrian[e[1]])/2.0;
                            posTmp[2]=0;
                            posTmp*=radius/(posTmp.norm()*ctheta);
                            posTmp[2]=outTrian[e[0]][2];
                            posTmp2=outTrian[e[0]];
                            posTmp2[2]=outTrian[e[1]][2];
                            pos=(2*ctheta*posTmp+posTmp2)/(1+2*ctheta);
                            outTrian.push_back(pos);
                            posTmp[2]=outTrian[e[1]][2];
                            posTmp2=outTrian[e[1]];
                            posTmp2[2]=outTrian[e[0]][2];
                            pos=(2*ctheta*posTmp+posTmp2)/(1+2*ctheta);

                            outTrian.push_back(pos);
                            bezierTriangleWeight.push_back((Real)(1+2*ctheta)/3.0f);
                            bezierTriangleWeight.push_back((Real)(1+2*ctheta)/3.0f);
                        } else {
                            for (j=1;j<degreeTriangle;++j) {
                                // interpolated position
                                pos= ((Real) j*outTrian[e[1]]+(Real)(degreeTriangle-j)*outTrian[e[0]])/degreeTriangle;
                                outTrian.push_back(pos);
                                // weight is 1
                                bezierTriangleWeight.push_back((Real)1.0f);
                            }
                        }

                    } else {
                        // as default the edge is straight : compute control point by degree elevation
                        for (j=1;j<degreeTriangle;++j) {
                            // interpolated position
                            pos= ((Real) j*outTrian[e[1]]+(Real)(degreeTriangle-j)*outTrian[e[0]])/degreeTriangle;
                            outTrian.push_back(pos);
                            // weight is 1
                            bezierTriangleWeight.push_back((Real)1.0f);
                        }
                    }
                }

            }
        }
        if (degreeTriangle>2) {
            for (j=0,itt=triangles.begin();itt!=triangles.end();++itt,++j) {
                if (j<(2*freqTheta*(freqZ-1))){

                    // add points inside triangles
                    // find the edge along the axis z of the cylinder
                    for(i=0;(outTrian[(*itt)[(i+1)%3]][2]==outTrian[(*itt)[(i+2)%3]][2])||(outTrian[(*itt)[(i+1)%3]][1]==outTrian[(*itt)[(i+2)%3]][1]);++i);
            //		posTmp=outTrian[(*itt)[(i+1)%3]];
            //		posTmp2=outTrian[(*itt)[(i+2)%3]];
                    for(k=1;outTrian[(*itt)[(i+k)%3]][2]!=outTrian[(*itt)[i]][2];++k);
                    posTmp=(outTrian[(*itt)[i]]+outTrian[(*itt)[(i+k)%3]])/2.0;
                    posTmp[2]=0;
                    posTmp*=radius/(posTmp.norm()*ctheta);
                    posTmp[2]=outTrian[(*itt)[(i+1)%3]][2];
                    posTmp2=posTmp;
                    posTmp2[2]=outTrian[(*itt)[(i+2)%3]][2];
                    pos=(outTrian[(*itt)[i]]+ctheta*posTmp+ctheta*posTmp2)/(1+2*ctheta);
                    outTrian.push_back(pos);
            //		msg_info()<<" central point norm ="<<sqrt(pos[0]*pos[0]+pos[1]*pos[1])<<std::endl;
                    bezierTriangleWeight.push_back((Real)(1+2*ctheta)/3.0f);
                } else{
                    outTrian.push_back((outTrian[(*itt)[0]]+outTrian[(*itt)[1]]+outTrian[(*itt)[2]])/3.0f);
                    bezierTriangleWeight.push_back(1.0f);
                }
            }
        }

        f_bezierTriangleWeight.endEdit();
    }
    f_triangles.endEdit();
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
