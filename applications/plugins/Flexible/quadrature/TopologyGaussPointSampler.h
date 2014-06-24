/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_TopologyGaussPointSAMPLER_H
#define SOFA_TopologyGaussPointSAMPLER_H

#include "../initFlexible.h"
#include "../quadrature/BaseGaussPointSampler.h"
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/NumericalIntegrationDescriptor.h>

namespace sofa
{
namespace component
{
namespace engine
{

using helper::vector;

/**
 * This class samples an object represented by a mesh
 */


class SOFA_Flexible_API TopologyGaussPointSampler : public BaseGaussPointSampler
{
public:
    typedef BaseGaussPointSampler Inherited;
    SOFA_CLASS(TopologyGaussPointSampler,Inherited);

    /** @name  GaussPointSampler types */
    //@{
    typedef Inherited::Real Real;
    typedef Inherited::Coord Coord;
    typedef Inherited::SeqPositions SeqPositions;
    typedef Inherited::raPositions raPositions;
    typedef Inherited::waPositions waPositions;
    //@}

    /** @name  topology */
    //@{
    Data< SeqPositions > f_inPosition;
    typedef sofa::core::topology::BaseMeshTopology Topo;
    Topo* parentTopology;
    Data< vector<unsigned int> > f_cell;

    typedef topology::TetrahedronSetGeometryAlgorithms<defaulttype::StdVectorTypes<Coord,Coord,Real> > TetraGeoAlg;
    TetraGeoAlg* tetraGeoAlgo;
    //@}

    Data< vector<Real> > f_fineVolumes;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const TopologyGaussPointSampler* = NULL) { return std::string();    }

    virtual void init()
    {
        Inherited::init();

        if( !parentTopology )
        {
            this->getContext()->get(parentTopology,core::objectmodel::BaseContext::SearchUp);
            if(!this->parentTopology) serr<<"MeshTopology not found"<<sendl;
        }

        if(!this->tetraGeoAlgo)
        {
            this->getContext()->get(tetraGeoAlgo,core::objectmodel::BaseContext::SearchUp);
        }

        addInput(&f_inPosition);
        addInput(&f_fineVolumes);
        addOutput(&f_cell);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:
    TopologyGaussPointSampler()    :   Inherited()
      , f_inPosition(initData(&f_inPosition,SeqPositions(),"inPosition","input node positions"))
      , parentTopology( 0 )
      , f_cell(initData(&f_cell,"cell","cell index associated with each sample"))
      , tetraGeoAlgo( 0 )
      , f_fineVolumes(initData(&f_fineVolumes,"fineVolumes","input cell volumes (typically computed from a fine model)"))
    {
    }

    virtual ~TopologyGaussPointSampler()
    {

    }

    virtual void update()
    {
        cleanDirty();

        if(!this->parentTopology) return;

        raPositions parent(this->f_inPosition);
        if(!parent.size()) return;

        const Topo::SeqTetrahedra& tetrahedra = this->parentTopology->getTetrahedra();
        const Topo::SeqHexahedra& cubes = this->parentTopology->getHexahedra();
        const Topo::SeqTriangles& triangles = this->parentTopology->getTriangles();
        const Topo::SeqQuads& quads = this->parentTopology->getQuads();
        const Topo::SeqEdges& edges = this->parentTopology->getEdges();

        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);

        helper::WriteAccessor<Data< vector<unsigned int> > > cel(this->f_cell);

        if ( tetrahedra.empty() && cubes.empty() )
        {
            if ( triangles.empty() && quads.empty() )
            {
                if ( edges.empty() ) return;
                //no 3D elements, nor 2D elements -> map on 1D elements

                if(this->f_method.getValue().getSelectedId() == ELASTON || this->f_order.getValue()==1) // one point at center
                {
                    pos.resize ( edges.size() );
                    vol.resize ( pos.size() );
                    cel.resize ( pos.size() );
                    for (unsigned int i=0; i<edges.size(); i++ )
                    {
                        const Coord& p1=parent[edges[i][0]],p2=parent[edges[i][1]];
                        pos[i] = (p1+p2)*0.5;
                        vol[i].resize(1);  if(f_fineVolumes.getValue().size()>i) vol[i][0]=f_fineVolumes.getValue()[i]; else  vol[i][0]=(p1-p2).norm();
                        cel[i] = i;
                        // to do : volume integrals for elastons
                    }
                }
                else serr<<"Requested quadrature method not yet implemented"<<sendl;
            }
            else
            {
                // no 3D elements -> map on 2D elements
                if(this->f_method.getValue().getSelectedId() == ELASTON || this->f_order.getValue()==1) // one point at center
                {
                    pos.resize ( triangles.size() +quads.size() );
                    vol.resize ( pos.size() );
                    cel.resize ( pos.size() );
                    for ( unsigned int i = 0; i < triangles.size(); i++ )
                    {
                        const Coord& p1=parent[triangles[i][0]],p2=parent[triangles[i][1]],p3=parent[triangles[i][2]];
                        pos[i] = (p1+p2+p3)/(Real)3.;
                        vol[i].resize(1);  if(f_fineVolumes.getValue().size()>i) vol[i][0]=f_fineVolumes.getValue()[i]; else  vol[i][0] = cross(p2-p1,p3-p1).norm()*0.5;
                        cel[i] = i;
                        // to do : volume integrals for elastons
                    }
                    int c0 = triangles.size();
                    for ( unsigned int i = 0; i < quads.size(); i++ )
                    {
                        const Coord& p1=parent[quads[i][0]],p2=parent[quads[i][1]],p3=parent[quads[i][2]],p4=parent[quads[i][3]];
                        pos[i+c0] = (p1+p2+p3+p4)*0.25;
                        vol[i+c0].resize(1);  if(f_fineVolumes.getValue().size()>i+c0) vol[i+c0][0]=f_fineVolumes.getValue()[i+c0]; else  vol[i+c0][0] = cross(p2-p1,p3-p1).norm();
                        cel[i+c0] = i+c0;
                        // to do : volume integrals for elastons
                    }
                }
                else if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE || this->f_order.getValue()==2)
                {
                    pos.resize ( 4*quads.size() );
                    vol.resize ( pos.size() );
                    cel.resize ( pos.size() );

                    // 4 points per quad at [ +-1/sqrt(3), +-1/sqrt(3)], weight = volume/4
                    const Real offset = 0.5/sqrt(3.0);
                    unsigned int count=0;
                    for ( unsigned int i = 0; i < quads.size(); i++ )
                    {
                        const Coord& p1=parent[quads[i][0]],p2=parent[quads[i][1]],p3=parent[quads[i][2]],p4=parent[quads[i][3]];
                        Coord u=(p2-p1),v=(p4-p1);
                        Real V;
                        if(f_fineVolumes.getValue().size()>i) V=f_fineVolumes.getValue()[i]*0.25; else  V=u.norm()*v.norm()*0.25;
                        u*=offset; v*=offset;
                        const Coord c = (p1+p2+p3+p4)*0.25;
                        for (int gx2=-1; gx2<=1; gx2+=2)
                            for (int gx3=-1; gx3<=1; gx3+=2)
                            {
                                pos[count] = c + u*gx3 + v*gx2;
                                vol[count].resize(1); vol[count][0] = V;
                                cel[count] = i;
                                //getCubeVolumes(vol[i+c0],p1,p2,p3,p4,this->f_order.getValue());
                                //
                                count++;
                            }
                    }
                }
                else serr<<"Requested quadrature method not yet implemented"<<sendl;
            }
        }
        else
        {
            // map on 3D elements
            if(this->f_method.getValue().getSelectedId() == ELASTON || this->f_order.getValue()==1) // one point at center
            {
                pos.resize ( tetrahedra.size() +cubes.size() );
                vol.resize ( pos.size() );
                cel.resize ( pos.size() );
                for ( unsigned int i = 0; i < tetrahedra.size(); i++ )
                {
                    const Coord& p1=parent[tetrahedra[i][0]],p2=parent[tetrahedra[i][1]],p3=parent[tetrahedra[i][2]],p4=parent[tetrahedra[i][3]];
                    pos[i] = (p1+p2+p3+p4)*0.25;
                    vol[i].resize(1); if(f_fineVolumes.getValue().size()>i) vol[i][0]=f_fineVolumes.getValue()[i]; else  vol[i][0] = fabs(dot(cross(p4-p1,p3-p1),p2-p1))/(Real)6.;
                    cel[i] = i;
                    // to do : volume integrals for elastons
                }
                int c0 = tetrahedra.size();
                for ( unsigned int i = 0; i < cubes.size(); i++ )
                {
                    const Coord& p1=parent[cubes[i][0]],p2=parent[cubes[i][1]],p3=parent[cubes[i][2]],p4=parent[cubes[i][3]],p5=parent[cubes[i][4]],p6=parent[cubes[i][5]],p7=parent[cubes[i][6]],p8=parent[cubes[i][7]];
                    pos[i+c0] = (p1+p2+p3+p4+p5+p6+p7+p8)*0.125;
                    cel[i+c0] = i+c0;
                    getCubeVolumes(vol[i+c0],p1,p2,p3,p4,this->f_order.getValue());
                    if(f_fineVolumes.getValue().size()>i+c0) { Real fact=f_fineVolumes.getValue()[i+c0]/vol[i+c0][0];   for (unsigned int j=0; j<vol[i+c0].size(); j++) vol[i+c0][j]*=fact; }
                    //vol[i+c0].resize(1); if(f_fineVolumes.getValue().size()>i+c0) vol[i+c0][0]=f_fineVolumes.getValue()[i+c0]; else  vol[i+c0][0] = fabs(dot(cross(p4-p1,p3-p1),p2-p1));
                }
            }
            else if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE)
            {
                if(tetraGeoAlgo)  // retrieve gauss points from geometry algorithm
                {
                    typedef topology::NumericalIntegrationDescriptor<Real,4> IntegrationDescriptor;
                    IntegrationDescriptor::QuadraturePointArray qpa=tetraGeoAlgo->getTetrahedronNumericalIntegrationDescriptor().getQuadratureMethod( (IntegrationDescriptor::QuadratureMethod)0,this->f_order.getValue());
                    pos.resize ( qpa.size()*tetrahedra.size());
                    vol.resize ( pos.size() );
                    cel.resize ( pos.size() );
                    unsigned int count=0;
                    for ( unsigned int i = 0; i < tetrahedra.size(); i++ )
                    {
                        const Coord p[4]={parent[tetrahedra[i][0]],parent[tetrahedra[i][1]],parent[tetrahedra[i][2]],parent[tetrahedra[i][3]]};
                        Real V; if(f_fineVolumes.getValue().size()>i) V=f_fineVolumes.getValue()[i]; else  V = fabs(dot(cross(p[3]-p[0],p[2]-p[0]),p[1]-p[0]))/(Real)6.;
                        for ( unsigned int j = 0; j < qpa.size(); j++ )
                        {
                            pos[count]=Coord(); for ( unsigned int k = 0; k < qpa[j].first.size(); k++ ) pos[count]+=p[k]*qpa[j].first[k];
                            vol[count].resize(1); vol[count][0] = V*qpa[j].second*(Real)6.0;        // todo: change weight in TetrahedronGeoAlgo to correspond to tet volume weighting (not 6xtet volume weighting)
                            cel[count] = i;
                            count++;
                        }
                    }
                }
                else   if(this->f_order.getValue()==2)  // else use built in gauss points sampling
                {
                    pos.resize ( 4*tetrahedra.size() + 8*cubes.size() );
                    vol.resize ( pos.size() );
                    cel.resize ( pos.size() );
                    unsigned int count=0;

                    // 4 points per tet at [ (5+3sqrt(5))/20, (5-sqrt(5))/20, (5-sqrt(5))/20, (5-sqrt(5))/20 ], weight = volume/4
                    if(tetrahedra.size())
                    {
                        const Real offsetB = (5.0-sqrt(5.0))/20.0;
                        const Real offsetA = 1.0-4.0*offsetB;
                        for ( unsigned int i = 0; i < tetrahedra.size(); i++ )
                        {
                            const Coord p[4]={parent[tetrahedra[i][0]],parent[tetrahedra[i][1]],parent[tetrahedra[i][2]],parent[tetrahedra[i][3]]};
                            Real V; if(f_fineVolumes.getValue().size()>i) V=f_fineVolumes.getValue()[i]/4.0; else  V = fabs(dot(cross(p[3]-p[0],p[2]-p[0]),p[1]-p[0]))/(Real)24.;
                            for ( unsigned int j = 0; j < 4; j++ )
                            {
                                pos[count] = (p[0]+p[1]+p[2]+p[3])*offsetB + p[j]*offsetA;
                                vol[count].resize(1); vol[count][0] = V;
                                cel[count] = i;
                                count++;
                                // to do : volume integrals for elastons
                            }
                        }
                    }

                    // 8 points per cube at [ +-1/sqrt(3), +-1/sqrt(3), +-1/sqrt(3) ], weight = volume/8
                    if(cubes.size())
                    {
                        const Real offset = 0.5/sqrt(3.0);
                        for ( unsigned int i = 0; i < cubes.size(); i++ )
                        {
                            const Coord& p1=parent[cubes[i][0]],p2=parent[cubes[i][1]],p3=parent[cubes[i][2]],p4=parent[cubes[i][3]],p5=parent[cubes[i][4]],p6=parent[cubes[i][5]],p7=parent[cubes[i][6]],p8=parent[cubes[i][7]];
                            Coord u=(p2-p1),v=(p5-p1),w=(p4-p1);
                            Real V;
                            if(f_fineVolumes.getValue().size()>i) V=f_fineVolumes.getValue()[i]/(Real)8; else  V=u.norm()*v.norm()*w.norm()/(Real)8.;

                            u*=offset; v*=offset; w*=offset;
                            const Coord c = (p1+p2+p3+p4+p5+p6+p7+p8)*0.125;
                            for (int gx1=-1; gx1<=1; gx1+=2)
                                for (int gx2=-1; gx2<=1; gx2+=2)
                                    for (int gx3=-1; gx3<=1; gx3+=2)
                                    {
                                        pos[count] = c + u*gx3 + v*gx2 + w*gx1;
                                        vol[count].resize(1); vol[count][0] = V;
                                        cel[count] = i;
                                        //getCubeVolumes(vol[i+c0],p1,p2,p3,p4,this->f_order.getValue());
                                        //
                                        count++;
                                    }
                        }
                    }
                }
                else serr<<"Requested quadrature order not implemented here: use GeometryAlgorithms"<<sendl;
            }
            else serr<<"Requested quadrature method not yet implemented"<<sendl;

        }

        if(this->f_printLog.getValue()) if(pos.size())    std::cout<<"TopologyGaussPointSampler: "<< pos.size() <<" generated samples"<<std::endl;
    }


    // returns integrated volumes and moments across an element
    inline void getCubeVolumes(vector<Real> &V, const Coord& p1,const Coord& p2,const Coord& p3,const Coord& p4, const unsigned int order)
    {
        Coord u=p2-p1,v=p3-p1,w=p4-p1;
        Vec<3,Real> l;  for(unsigned int i=0; i<3; i++) l[i]=helper::rmax(helper::rmax(helper::rabs(u[i]),helper::rabs(v[i])),helper::rabs(w[i]));
        Vec<3,Real> l2;  for(unsigned int i=0; i<3; i++) l2[i]=l[i]*l[i];
        Vec<3,Real> l3;  for(unsigned int i=0; i<3; i++) l3[i]=l2[i]*l[i];
        Vec<3,Real> l5;  for(unsigned int i=0; i<3; i++) l5[i]=l3[i]*l2[i];


        unsigned int dim=(order+1)*(order+2)*(order+3)/6;          V.resize(dim);
        if(order>4) {serr<<"integration order higher than 4 not supported"<<sendl;}
        unsigned int count=0;

        // order 0
        V[count]=l[0]*l[1]*l[2]; count++;
        if(order==0) return;
        // order 1
        V[count]=0; count++;
        V[count]=0; count++;
        V[count]=0; count++;
        if(order==1) return;
        // order 2
        V[count]=l3[0]*l[1]*l[2]/(Real)12.; count++; //x^2
        V[count]=0; count++;
        V[count]=0; count++;
        V[count]=l[0]*l3[1]*l[2]/(Real)12.; count++; //y^2
        V[count]=0; count++;
        V[count]=l[0]*l[1]*l3[2]/(Real)12.; count++; //z^2
        if(order==2) return;
        // order 3
        for(unsigned int i=0; i<10; i++) {V[count]=0; count++;}
        if(order==3) return;
        // order 4
        V[count]=l5[0]*l[1]*l[2]/(Real)80.; count++;       // x^4
        V[count]=l3[0]*l3[1]*l[2]/(Real)144.; count++;     // x^2*y^2
        V[count]=l3[0]*l[1]*l3[2]/(Real)144.; count++;     // x^2*z^2
        V[count]=l[0]*l5[1]*l[2]/(Real)80.; count++;       // y^4
        V[count]=l[0]*l3[1]*l3[2]/(Real)144.; count++;     // y^2*z^2
        V[count]=l[0]*l[1]*l5[2]/(Real)80.; count++;       // z^4
        for(unsigned int i=0; i<9; i++) {V[count]=0; count++;}
    }




};

}
}
}

#endif
