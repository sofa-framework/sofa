/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_TopologyGaussPointSAMPLER_H
#define SOFA_TopologyGaussPointSAMPLER_H

#include <Flexible/config.h>
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
    typedef Inherited::VTransform VTransform;
    typedef Inherited::Transform Transform;
    typedef Inherited::volumeIntegralType volumeIntegralType;
    //@}

    /** @name  topology */
    //@{
    Data< SeqPositions > f_inPosition;
    typedef sofa::core::topology::BaseMeshTopology Topo;
    Topo* parentTopology;
    Data< helper::vector<unsigned int> > f_cell;
    Data< helper::vector<unsigned int> > f_indices;

    typedef topology::TetrahedronSetGeometryAlgorithms<defaulttype::StdVectorTypes<Coord,Coord,Real> > TetraGeoAlg;
    TetraGeoAlg* tetraGeoAlgo;
    //@}

    /** @name orientation data */
    //@{
    Data< SeqPositions > f_orientation; // = rest deformation gradient orientation in each cell (Euler angles)
    Data< bool > f_useLocalOrientation;
    //@}

    Data< helper::vector<Real> > f_fineVolumes;

    virtual void init()
    {
        Inherited::init();

        if( !parentTopology )
        {
            this->getContext()->get(parentTopology,core::objectmodel::BaseContext::SearchUp);
            if(!this->parentTopology) serr<<"MeshTopology not found"<<sendl;
        }

        addInput(&f_inPosition);
        addInput(&f_fineVolumes);
        addInput(&f_indices);
        addOutput(&f_position);
        addOutput(&f_cell);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:
    TopologyGaussPointSampler()    :   Inherited()
      , f_inPosition(initData(&f_inPosition,SeqPositions(),"inPosition","input node positions"))
      , parentTopology( 0 )
      , f_cell(initData(&f_cell,"cell","cell index associated with each sample"))
      , f_indices(initData(&f_indices,"indices","list of cells where sampling is performed (all by default)"))
      , tetraGeoAlgo( 0 )
      , f_orientation(initData(&f_orientation,"orientation","input orientation (Euler angles) inside each cell"))
      , f_useLocalOrientation(initData(&f_useLocalOrientation,false,"useLocalOrientation","tells if orientations are defined in the local basis on each cell"))
      , f_fineVolumes(initData(&f_fineVolumes,"fineVolumes","input cell volumes (typically computed from a fine model)"))
    {
    }

    virtual ~TopologyGaussPointSampler()
    {

    }

    inline bool isInIndices(const std::set<unsigned int>& indices,const unsigned int& index)
    {
        if(!indices.size()) return true;
        if(indices.find(index)!=indices.end()) return true; else return false;
    }

    virtual void update()
    {
        this->updateAllInputsIfDirty();
        cleanDirty();

        if( !parentTopology ) return;

        raPositions parent(f_inPosition);

        if(!parent.size()) return;

        // convert to set to speed up search
        std::set<unsigned int> indices; for(size_t i=0;i<f_indices.getValue().size();i++) indices.insert(f_indices.getValue()[i]);

        const Topo::SeqTetrahedra& tetrahedra = parentTopology->getTetrahedra();
        const Topo::SeqHexahedra& cubes = parentTopology->getHexahedra();
        const Topo::SeqTriangles& triangles = parentTopology->getTriangles();
        const Topo::SeqQuads& quads = parentTopology->getQuads();
        const Topo::SeqEdges& edges = parentTopology->getEdges();

        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);
        helper::WriteOnlyAccessor<Data< VTransform > > transforms(this->f_transforms);
        helper::WriteOnlyAccessor<Data< helper::vector<unsigned int> > > cel(f_cell);
        pos.clear();
        vol.clear();
        transforms.clear();
        cel.clear();

        if ( tetrahedra.empty() && cubes.empty() )
        {
            if ( triangles.empty() && quads.empty() )
            {
                if ( edges.empty() ) return;
                //no 3D elements, nor 2D elements -> map on 1D elements

                if(this->f_method.getValue().getSelectedId() == ELASTON || this->f_order.getValue()==1) // one point at center
                {
                    for (unsigned int i=0; i<edges.size(); i++ ) if(isInIndices(indices,i))
                    {
                        const Coord& p1=parent[edges[i][0]],p2=parent[edges[i][1]];
                        pos.push_back( (p1+p2)*0.5 );
                        vol.push_back(volumeIntegralType(1,(f_fineVolumes.getValue().size()>i)?f_fineVolumes.getValue()[i]:(p1-p2).norm()));
                        cel.push_back(i);
                        transforms.push_back(getLocalFrame1D(p1,p2));
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
                    for ( unsigned int i = 0; i < triangles.size(); i++ ) if(isInIndices(indices,i))
                    {
                        const Coord& p1=parent[triangles[i][0]],p2=parent[triangles[i][1]],p3=parent[triangles[i][2]];
                        pos.push_back( (p1+p2+p3)/(Real)3. );
                        vol.push_back(volumeIntegralType(1,(f_fineVolumes.getValue().size()>i)?f_fineVolumes.getValue()[i]:cross(p2-p1,p3-p1).norm()*0.5));
                        cel.push_back( i );
                        // to do : volume integrals for elastons
                        // compute local orientation given user input and local element frame
                        Transform U=this->getUserOrientation(i),R= getLocalFrame2D(p1,p2,p3);
                        transforms.push_back( this->f_useLocalOrientation.getValue()? reorientGlobalOrientation(R*U,R) : reorientGlobalOrientation(U,R) );
                    }
                    int c0 = triangles.size();
                    for ( unsigned int i = 0; i < quads.size(); i++ ) if(isInIndices(indices,i))
                    {
                        const Coord& p1=parent[quads[i][0]],p2=parent[quads[i][1]],p3=parent[quads[i][2]],p4=parent[quads[i][3]];
                        pos.push_back( (p1+p2+p3+p4)*0.25 );
                        vol.push_back(volumeIntegralType(1,(f_fineVolumes.getValue().size()>i+c0)?f_fineVolumes.getValue()[i+c0]:cross(p2-p1,p3-p1).norm()));
                        cel.push_back( i+c0 );
                        // to do : volume integrals for elastons
                        // compute local orientation given user input and local element frame
                        Transform U=this->getUserOrientation(i+c0),R= getLocalFrame2D(p1,p2,p4);
                        transforms.push_back( this->f_useLocalOrientation.getValue() ? reorientGlobalOrientation(R*U,R) : reorientGlobalOrientation(U,R) );
                    }
                }
                else if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE || this->f_order.getValue()==2)
                {
                    // 4 points per quad at [ +-1/sqrt(3), +-1/sqrt(3)], weight = volume/4
                    const Real offset = 0.5/sqrt(3.0);
                    for ( unsigned int i = 0; i < quads.size(); i++ ) if(isInIndices(indices,i))
                    {
                        const Coord& p1=parent[quads[i][0]],p2=parent[quads[i][1]],p3=parent[quads[i][2]],p4=parent[quads[i][3]];
                        Coord u=(p2-p1),v=(p4-p1);
                        Real V;
                        if(f_fineVolumes.getValue().size()>i) V=f_fineVolumes.getValue()[i]*0.25; else  V=u.norm()*v.norm()*0.25;
                        u*=offset; v*=offset;
                        const Coord c = (p1+p2+p3+p4)*0.25;
                        // compute local orientation given user input and local element frame
                        Transform U=this->getUserOrientation(i),R= getLocalFrame2D(p1,p2,p4);
                        Transform M = this->f_useLocalOrientation.getValue() ? reorientGlobalOrientation(R*U,R) : reorientGlobalOrientation(U,R);

                        for (int gx2=-1; gx2<=1; gx2+=2)
                            for (int gx3=-1; gx3<=1; gx3+=2)
                            {
                                pos.push_back( c + u*gx3 + v*gx2 );
                                vol.push_back(volumeIntegralType(1,V));
                                cel.push_back( i );
                                transforms.push_back( M );
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
                for ( unsigned int i = 0; i < tetrahedra.size(); i++ ) if(isInIndices(indices,i))
                {
                    const Coord& p1=parent[tetrahedra[i][0]],p2=parent[tetrahedra[i][1]],p3=parent[tetrahedra[i][2]],p4=parent[tetrahedra[i][3]];
                    pos.push_back( (p1+p2+p3+p4)*0.25 );
                    vol.push_back(volumeIntegralType(1,(f_fineVolumes.getValue().size()>i)?f_fineVolumes.getValue()[i]:fabs(dot(cross(p4-p1,p3-p1),p2-p1))/(Real)6.));
                    cel.push_back( i );
                    // to do : volume integrals for elastons
                    // compute local orientation given user input and local element frame
                    Transform U=this->getUserOrientation(i);
                    transforms.push_back(this->f_useLocalOrientation.getValue() ? getLocalFrame2D(p1,p2,p3) * U : U);
                }
                int c0 = tetrahedra.size();
                for ( unsigned int i = 0; i < cubes.size(); i++ ) if(isInIndices(indices,i))
                {
                    const Coord& p1=parent[cubes[i][0]],p2=parent[cubes[i][1]],p3=parent[cubes[i][2]],p4=parent[cubes[i][3]],p5=parent[cubes[i][4]],p6=parent[cubes[i][5]],p7=parent[cubes[i][6]],p8=parent[cubes[i][7]];
                    pos.push_back( (p1+p2+p3+p4+p5+p6+p7+p8)*0.125 );
                    cel.push_back( i+c0 );
                    Coord u=(p2-p1),v=(p5-p1),w=(p4-p1);
                    Real V;
                    if(f_fineVolumes.getValue().size()>i+c0) V=f_fineVolumes.getValue()[i+c0]; else  V=u.norm()*v.norm()*w.norm();
                    vol.push_back(volumeIntegralType(1,V));
                    // volumeIntegralType v; getCubeVolumes(v,p1,p2,p4,p5,0);
                    // if(f_fineVolumes.getValue().size()>i+c0) { Real fact=f_fineVolumes.getValue()[i+c0]/v[0];   for (unsigned int j=0; j<v.size(); j++) v[j]*=fact; }
                    // vol.push_back(v);
                    // compute local orientation given user input and local element frame
                    Transform U=this->getUserOrientation(i+c0);
                    transforms.push_back(this->f_useLocalOrientation.getValue() ? getLocalFrame2D(p1,p2,p5) * U : U);
                }
            }
            else if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE)
            {
                if(!this->tetraGeoAlgo) this->getContext()->get(tetraGeoAlgo,core::objectmodel::BaseContext::SearchUp);
                if(tetraGeoAlgo)  // retrieve gauss points from geometry algorithm
                {
                    typedef topology::NumericalIntegrationDescriptor<Real,4> IntegrationDescriptor;
                    IntegrationDescriptor::QuadraturePointArray qpa=tetraGeoAlgo->getTetrahedronNumericalIntegrationDescriptor().getQuadratureMethod( (IntegrationDescriptor::QuadratureMethod)3,this->f_order.getValue());
                    for ( unsigned int i = 0; i < tetrahedra.size(); i++ ) if(isInIndices(indices,i))
                    {
                        const Coord p[4]={parent[tetrahedra[i][0]],parent[tetrahedra[i][1]],parent[tetrahedra[i][2]],parent[tetrahedra[i][3]]};
                        Real V; if(f_fineVolumes.getValue().size()>i) V=f_fineVolumes.getValue()[i]; else  V = fabs(dot(cross(p[3]-p[0],p[2]-p[0]),p[1]-p[0]))/(Real)6.;
                        // compute local orientation given user input and local element frame
                        Transform U=this->getUserOrientation(i);
                        Transform M=this->f_useLocalOrientation.getValue() ? getLocalFrame2D(p[0],p[1],p[2]) * U : U;
                        for ( unsigned int j = 0; j < qpa.size(); j++ )
                        {
                            Coord P; for ( unsigned int k = 0; k < qpa[j].first.size(); k++ ) P+=p[k]*qpa[j].first[k];
                            pos.push_back(P);
                            vol.push_back(volumeIntegralType(1,V*qpa[j].second*(Real)6.0)); // todo: change weight in TetrahedronGeoAlgo to correspond to tet volume weighting (not 6xtet volume weighting)
                            cel.push_back( i );
                            transforms.push_back( M );
                        }
                    }
                }
                else   if(this->f_order.getValue()==2)  // else use built in gauss points sampling
                {
                    // 4 points per tet at [ (5+3sqrt(5))/20, (5-sqrt(5))/20, (5-sqrt(5))/20, (5-sqrt(5))/20 ], weight = volume/4
                    if(tetrahedra.size())
                    {
                        const Real offsetB = (5.0-sqrt(5.0))/20.0;
                        const Real offsetA = 1.0-4.0*offsetB;
                        for ( unsigned int i = 0; i < tetrahedra.size(); i++ ) if(isInIndices(indices,i))
                        {
                            const Coord p[4]={parent[tetrahedra[i][0]],parent[tetrahedra[i][1]],parent[tetrahedra[i][2]],parent[tetrahedra[i][3]]};
                            Real V; if(f_fineVolumes.getValue().size()>i) V=f_fineVolumes.getValue()[i]/4.0; else  V = fabs(dot(cross(p[3]-p[0],p[2]-p[0]),p[1]-p[0]))/(Real)24.;
                            // compute local orientation given user input and local element frame
                            Transform U=this->getUserOrientation(i);
                            Transform M = this->f_useLocalOrientation.getValue()? getLocalFrame2D(p[0],p[1],p[2]) * U : U;
                            for ( unsigned int j = 0; j < 4; j++ )
                            {
                                pos.push_back( (p[0]+p[1]+p[2]+p[3])*offsetB + p[j]*offsetA );
                                vol.push_back(volumeIntegralType(1,V));
                                cel.push_back( i );
                                // to do : volume integrals for elastons
                                transforms.push_back( M );
                            }
                        }
                    }
                }
                else serr<<"Requested quadrature order not implemented here: use GeometryAlgorithms"<<sendl;

                int c0 = tetrahedra.size();
                if(this->f_order.getValue()==2)  // else use built in gauss points sampling
                {

                    // 8 points per cube at [ +-1/sqrt(3), +-1/sqrt(3), +-1/sqrt(3) ], weight = volume/8
                    if(cubes.size())
                    {
                        const Real offset = 0.5/sqrt(3.0);
                        for ( unsigned int i = 0; i < cubes.size(); i++ ) if(isInIndices(indices,i))
                        {
                            const Coord& p1=parent[cubes[i][0]],p2=parent[cubes[i][1]],p3=parent[cubes[i][2]],p4=parent[cubes[i][3]],p5=parent[cubes[i][4]],p6=parent[cubes[i][5]],p7=parent[cubes[i][6]],p8=parent[cubes[i][7]];
                            Coord u=(p2-p1),v=(p5-p1),w=(p4-p1);
                            Real V;
                            if(f_fineVolumes.getValue().size()>i+c0) V=f_fineVolumes.getValue()[i+c0]/(Real)8; else  V=u.norm()*v.norm()*w.norm()/(Real)8.;
                            //getCubeVolumes(vol[i+c0],p1,p2,p4,p5,this->f_order.getValue());
                            // compute local orientation given user input and local element frame
                            Transform U=this->getUserOrientation(i+c0);
                            Transform M = this->f_useLocalOrientation.getValue() ? getLocalFrame2D(p1,p2,p5) * U : U;
                            u*=offset; v*=offset; w*=offset;
                            const Coord c = (p1+p2+p3+p4+p5+p6+p7+p8)*0.125;
                            for (int gx1=-1; gx1<=1; gx1+=2)
                                for (int gx2=-1; gx2<=1; gx2+=2)
                                    for (int gx3=-1; gx3<=1; gx3+=2)
                                    {
                                        pos.push_back( c + u*gx3 + v*gx2 + w*gx1 );
                                        vol.push_back(volumeIntegralType(1,V));
                                        cel.push_back( i+c0 );
                                        transforms.push_back( M) ;
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
    inline void getCubeVolumes(volumeIntegralType &V, const Coord& p1,const Coord& p2,const Coord& p3,const Coord& p4, const unsigned int order)
    {
        Coord u=p2-p1,v=p3-p1,w=p4-p1;
        defaulttype::Vec<3,Real> l;  for(unsigned int i=0; i<3; i++) l[i]=helper::rmax(helper::rmax(helper::rabs(u[i]),helper::rabs(v[i])),helper::rabs(w[i]));
        defaulttype::Vec<3,Real> l2;  for(unsigned int i=0; i<3; i++) l2[i]=l[i]*l[i];
        defaulttype::Vec<3,Real> l3;  for(unsigned int i=0; i<3; i++) l3[i]=l2[i]*l[i];
        defaulttype::Vec<3,Real> l5;  for(unsigned int i=0; i<3; i++) l5[i]=l3[i]*l2[i];


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


    // local coordinate systems
    static Transform getLocalFrame1D( const Coord &p1,const Coord &p2)
    {
        Transform R;
        R(0) = p2-p1; R(0).normalize();
        R(1) = Coord(1,0,0); if(dot(R(0),R(1))==1.) R(1) = Coord(0,1,0); if(dot(R(0),R(1))==1.) R(1) = Coord(0,0,1);
        R(2) = cross(R(0),R(1)); R(2).normalize();
        R(1) = cross(R(2),R(0));
        R.transpose();
        return R;
    }
    static Transform getLocalFrame2D( const Coord &p1, const Coord &p2,const Coord &p3)
    {
        Transform R;
        R(0) = p2-p1; R(0).normalize();
        R(1) = p3-p1;
        R(2) = cross(R(0),R(1)); R(2).normalize();
        R(1)=cross(R(2),R(0));
        R.transpose();
        return R;
    }

    // align user provided global orientation to 2D manifold
    static Transform reorientGlobalOrientation( const Transform& Global, const Transform& Local )
    {
        Transform R;
        R(2)=Local.transposed()(2); // third axis = face normal

        Transform GT = Global.transposed();

        R(0)=GT(0) - R(2)*dot(R(2),GT(0));

        Real norm = R(0).norm();

        if(norm<std::numeric_limits<Real>::epsilon()) { R(0)=GT(1) - R(2)*dot(R(2),GT(1)); R(0).normalize(); }
        else R(0).normalizeWithNorm( norm );

        R(1) = cross(R(2),R(0));
        R.transpose();
        return R;
    }


    // user provided orientation
    Transform getUserOrientation(const unsigned int index) const
    {
        Coord orient; if(this->f_orientation.getValue().size()) orient=(this->f_orientation.getValue().size()>index)?this->f_orientation.getValue()[index]:this->f_orientation.getValue()[0];
        helper::Quater<Real> q = helper::Quater< Real >::createQuaterFromEuler(orient * (Real)M_PI / (Real)180.0);
        Transform R; q.toMatrix(R);
        return R;
    }


};

}
}
}

#endif
