/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef CLOSESTPOINTREGISTRATIONFORCEFIELD_INL
#define CLOSESTPOINTREGISTRATIONFORCEFIELD_INL

#include "ClosestPointRegistrationForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/Mapping.inl>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <iostream>
#include <map>

#ifdef USING_OMP_PRAGMAS
    #include <omp.h>
#endif

#include <sofa/component/loader/MeshObjLoader.h>
#include <sofa/component/engine/NormalsFromPoints.h>
#include <limits>
#include <set>
#include <iterator>
#include <sofa/helper/gl/Color.h>

using std::cerr;
using std::endl;



namespace sofa
{

namespace component
{

namespace forcefield
{


using namespace helper;

template <class DataTypes>
ClosestPointRegistrationForceField<DataTypes>::ClosestPointRegistrationForceField(core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , ks(initData(&ks,(Real)100.0,"stiffness","uniform stiffness for the all springs."))
    , kd(initData(&kd,(Real)5.0,"damping","uniform damping for the all springs."))
    , cacheSize(initData(&cacheSize,(unsigned int)5,"cacheSize","number of closest points used in the cache to speed up closest point computation."))
    , blendingFactor(initData(&blendingFactor,(Real)0,"blendingFactor","blending between projection (=0) and attraction (=1) forces."))
    , outlierThreshold(initData(&outlierThreshold,(Real)2.5,"outlierThreshold","suppress outliers when distance > (meandistance + threshold*stddev)."))
    , normalThreshold(initData(&normalThreshold,(Real)0.5,"normalThreshold","suppress outliers when normal.closestPointNormal < threshold."))
    , projectToPlane(initData(&projectToPlane,true,"projectToPlane","project closest points in the plane defined by the normal."))
    , rejectBorders(initData(&rejectBorders,true,"rejectBorders","ignore border vertices."))
    , springs(initData(&springs,"spring","index, stiffness, damping"))
    , sourceTriangles(initData(&sourceTriangles,"sourceTriangles","Triangles of the source mesh."))
    , sourceNormals(initData(&sourceNormals,"sourceNormals","Normals of the source mesh."))
    , targetPositions(initData(&targetPositions,"position","Vertices of the target mesh."))
    , targetNormals(initData(&targetNormals,"normals","Normals of the target mesh."))
    , targetTriangles(initData(&targetTriangles,"triangles","Triangles of the target mesh."))
    , showArrowSize(initData(&showArrowSize,0.01f,"showArrowSize","size of the axis."))
    , drawMode(initData(&drawMode,0,"drawMode","The way springs will be drawn:\n- 0: Line\n- 1:Cylinder\n- 2: Arrow."))
    , drawColorMap(initData(&drawColorMap,false,"drawColorMap","Hue mapping of distances to closest point"))
{
}

template <class DataTypes>
ClosestPointRegistrationForceField<DataTypes>::~ClosestPointRegistrationForceField()
{
}


template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::reinit()
{
    for (unsigned int i=0;i<springs.getValue().size();++i)
    {
        (*springs.beginEdit())[i].ks = (Real) ks.getValue();
        (*springs.beginEdit())[i].kd = (Real) kd.getValue();
    }
}

template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::init()
{
    this->Inherit::init();
    core::objectmodel::BaseContext* context = this->getContext();

    if(!(this->mstate)) this->mstate = dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes> *>(context->getMechanicalState());

    // Get source triangles
    if(!sourceTriangles.getValue().size()) {
        sofa::component::loader::MeshObjLoader *meshobjLoader;
        this->getContext()->get( meshobjLoader, core::objectmodel::BaseContext::Local);
        if (meshobjLoader) {sourceTriangles.virtualSetLink(meshobjLoader->triangles); sout<<"imported triangles from "<<meshobjLoader->getName()<<sendl;}
    }
    // Get source normals
    if(!sourceNormals.getValue().size()) serr<<"normals of the source model not found"<<sendl;

    // add a spring for every input point
    const VecCoord& x = *this->mstate->getX(); 			//RDataRefVecCoord x(*this->getMState()->read(core::ConstVecCoordId::position()));
    this->clearSprings(x.size());
    for(unsigned int i=0;i<x.size();i++) this->addSpring(i, (Real) ks.getValue(),(Real) kd.getValue());
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::detectBorder(vector<bool> &border,const helper::vector< tri > &triangles)
{
    unsigned int nbp=border.size();
    unsigned int nbt=triangles.size();
    for(unsigned int i=0;i<nbp;i++) border[i]=false;

    if(!nbt) return;
    vector<vector< unsigned int> > ngbTriangles((int)nbp);
    for(unsigned int i=0;i<nbt;i++) for(unsigned int j=0;j<3;j++)	ngbTriangles[triangles[i][j]].push_back(i);
    for(unsigned int i=0;i<nbp;i++) if(ngbTriangles[i].size()==0) border[i]=true;
    for(unsigned int i=0;i<nbt;i++)
        for(unsigned int j=0;j<3;j++)
        {
            unsigned int id1=triangles[i][j],id2=triangles[i][(j==2)?0:j+1];
            if(!border[id1] || !border[id2]) {
                bool bd=true;
                for(unsigned int i1=0;i1<ngbTriangles[id1].size() && bd;i1++)
                    for(unsigned int i2=0;i2<ngbTriangles[id2].size() && bd;i2++)
                        if(ngbTriangles[id1][i1]!=i)
                            if(ngbTriangles[id1][i1]==ngbTriangles[id2][i2])
                                bd=false;
                if(bd) border[id1]=border[id2]=true;
            }
        }
}


template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::initSource()
{
    // build k-d tree
    const VecCoord&  p = *this->mstate->getX();
    sourceKdTree.build(p);

    // detect border
    if(sourceBorder.size()!=p.size()) { sourceBorder.resize(p.size()); detectBorder(sourceBorder,sourceTriangles.getValue()); }
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::initTarget()
{
    // build k-d tree
    const VecCoord&  p = targetPositions.getValue();
    targetKdTree.build(p);

    // detect border
    if(targetBorder.size()!=p.size()) { targetBorder.resize(p.size()); detectBorder(targetBorder,targetTriangles.getValue()); }
}



template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::updateClosestPoints()
{
    const VecCoord& x = *this->mstate->getX();
    const VecCoord&  tp = targetPositions.getValue();

    unsigned int nbs=x.size(),nbt=tp.size();

    distanceSet emptyset;
    if(nbs!=closestSource.size()) {initSource();  closestSource.resize(nbs);	closestSource.fill(emptyset); cacheDist.resize(nbs); cacheDist.fill((Real)0.); cacheDist2.resize(nbs); cacheDist2.fill((Real)0.); previousX.assign(x.begin(),x.end());}
    if(nbt!=closestTarget.size()) {initTarget();  closestTarget.resize(nbt);	closestTarget.fill(emptyset);}

    if(nbs==0 || nbt==0) return;

    // closest target points from source points
    if(blendingFactor.getValue()<1) {

    //unsigned int count=0;
#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(int i=0;i<(int)nbs;i++)
        {
            Real dx=(previousX[i]-x[i]).norm();
            //  closest point caching [cf. Simon96 thesis]
            if(dx>=cacheDist[i] || closestSource[i].size()==0)
            {
                targetKdTree.getNClosest(closestSource[i],x[i],this->cacheSize.getValue() );
                typename distanceSet::iterator it0=closestSource[i].begin(), it1=it0; it1++;
                typename distanceSet::reverse_iterator itn=closestSource[i].rbegin();
                cacheDist[i] =((itn->first)-(it0->first))*(Real)0.5;
                cacheDist2[i]=((it1->first)-(it0->first))*(Real)0.5;
                previousX[i]=x[i];
            }
            else if(dx>=cacheDist2[i]) // in the cache -> update N-1 distances
            {
                targetKdTree.updateCachedDistances(closestSource[i],x[i]);
                //count++;
            }
        }
    //std::cout<<(Real)count*(Real)100./(Real)nbs<<" % cached"<<std::endl;
    }
    // closest source points from target points
    if(blendingFactor.getValue()>0)
    {
        initSource();
#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(int i=0;i<(int)nbt;i++)
            sourceKdTree.getNClosest(closestTarget[i],tp[i],1);
    }


    this->sourceIgnored.resize(nbs); sourceIgnored.fill(false);
    this->targetIgnored.resize(nbt); targetIgnored.fill(false);

    // prune outliers
    if(outlierThreshold.getValue()!=0) {
        Real mean=0,stdev=0,count=0;
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) {count++; stdev+=closestSource[i].begin()->first; mean+=(Real)(closestSource[i].begin()->first); }
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) {count++; stdev+=closestTarget[i].begin()->first; mean+=(Real)(closestTarget[i].begin()->first); }
        mean=mean/count; stdev=(Real)sqrt(stdev/count-mean*mean);
        mean+=stdev*outlierThreshold.getValue();
        mean*=mean;
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) if(closestSource[i].begin()->first>mean) sourceIgnored[i]=true;
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) if(closestTarget[i].begin()->first>mean) targetIgnored[i]=true;
    }
    if(rejectBorders.getValue()) {
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) if(targetBorder[closestSource[i].begin()->second]) sourceIgnored[i]=true;
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) if(sourceBorder[closestTarget[i].begin()->second]) targetIgnored[i]=true;
    }
    if(normalThreshold.getValue()>(Real)-1. && sourceNormals.getValue().size()!=0 && targetNormals.getValue().size()!=0) {
        ReadAccessor< Data< VecCoord > > sn(sourceNormals);
        ReadAccessor< Data< VecCoord > > tn(targetNormals);
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) if(dot(sn[i],tn[closestSource[i].begin()->second])<normalThreshold.getValue()) sourceIgnored[i]=true;
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) if(dot(tn[i],sn[closestTarget[i].begin()->second])<normalThreshold.getValue()) targetIgnored[i]=true;
    }
}




template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/,DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v )
{
    if(ks.getValue()==0) return;

    VecDeriv&        f = *_f.beginEdit();           //WDataRefVecDeriv f(_f);
    const VecCoord&  x = _x.getValue();			//RDataRefVecCoord x(_x);
    const VecDeriv&  v = _v.getValue();			//RDataRefVecDeriv v(_v);
    ReadAccessor< Data< VecCoord > > tn(targetNormals);
    ReadAccessor< Data< VecCoord > > tp(targetPositions);

    const vector<Spring>& s= this->springs.getValue();
    this->dfdx.resize(s.size());
    this->closestPos.resize(s.size());

    updateClosestPoints();

    m_potentialEnergy = 0;

    // get attraction/ projection factors
    Real attrF=(Real) blendingFactor.getValue();
    if(attrF<(Real)0.) attrF=(Real)0.;
    if(attrF>(Real)1.) attrF=(Real)1.;
    Real projF=((Real)1.-attrF);

    if(tp.size()==0)
        for (unsigned int i=0; i<s.size(); i++)
            closestPos[i]=x[i];
    else {

        // count number of attractors
        cnt.resize(s.size()); cnt.fill(0);  if(attrF>0) for (unsigned int i=0; i<tp.size(); i++) if(!targetIgnored[i]) cnt[closestTarget[i].begin()->second]++;

        // compute targetpos = projF*closestto + attrF* sum closestfrom / count

        // projection to point or plane
        if(projF>0) {
            for (unsigned int i=0; i<s.size(); i++)
                if(!sourceIgnored[i]) {
                    unsigned int id=closestSource[i].begin()->second;
                    if(projectToPlane.getValue() && tn.size()!=0)	closestPos[i]=(x[i]+tn[id]*dot(tp[id]-x[i],tn[id]))*projF;
                    else closestPos[i]=tp[id]*projF;
                    if(!cnt[i]) closestPos[i]+=x[i]*attrF;
                }
                else {
                    closestPos[i]=x[i]*projF;
                    if(!cnt[i]) closestPos[i]+=x[i]*attrF;
                }
        }
        else for (unsigned int i=0; i<s.size(); i++) { if(!cnt[i]) closestPos[i]=x[i]; else closestPos[i].fill(0); }

        // attraction
        if(attrF>0)
            for (unsigned int i=0; i<tp.size(); i++)
                if(!targetIgnored[i])	{
                    unsigned int id=closestTarget[i].begin()->second;
                    closestPos[id]+=tp[i]*attrF/(Real)cnt[id];
                    sourceIgnored[id]=false;
                }
    }

    for (unsigned int i=0; i<s.size(); i++)
    {
        //serr<<"addForce() between "<<springs[i].m1<<" and "<<closestPos[springs[i].m1]<<sendl;
        this->addSpringForce(m_potentialEnergy,f,x,v, i, s[i]);
    }
    _f.endEdit();

}



template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addSpringForce(double& potentialEnergy, VecDeriv& f,const  VecCoord& p,const VecDeriv& v,int i, const Spring& spring)
{
    int a = spring.m1;
    Coord u = this->closestPos[i]-p[a];
    Real d = u.norm();
    if( d>1.0e-4 )
    {
        Real inverseLength = 1.0f/d;
        u *= inverseLength;
        Real elongation = (Real)d;
        potentialEnergy += elongation * elongation * spring.ks / 2;
        /*          serr<<"addSpringForce, p = "<<p<<sendl;
        serr<<"addSpringForce, new potential energy = "<<potentialEnergy<<sendl;*/
        Deriv relativeVelocity = -v[a];
        Real elongationVelocity = dot(u,relativeVelocity);
        Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
        Deriv force = u*forceIntensity;
        f[a]+=force;
        Mat& m = this->dfdx[i];
        Real tgt = forceIntensity * inverseLength;
        for( int j=0; j<N; ++j )
        {
            // anisotropic
            //for( int k=0; k<N; ++k ) m[j][k] = tgt * u[j] * u[k];

            // isotropic
            for( int k=0; k<N; ++k ) m[j][k] = ((Real)spring.ks-tgt) * u[j] * u[k];
            m[j][j] += tgt;
        }
    }
    else // null length, no force and no stiffness
    {
        Mat& m = this->dfdx[i];
        for( int j=0; j<N; ++j )
        {
            for( int k=0; k<N; ++k )
            {
                m[j][k] = 0;
            }
        }
    }
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addSpringDForce(VecDeriv& df,const  VecDeriv& dx, int i, const Spring& spring, double kFactor, double /*bFactor*/)
{
    const int a = spring.m1;
    const Coord d = -dx[a];
    Deriv dforce = this->dfdx[i]*d;
    dforce *= kFactor;
    df[a]+=dforce;
    //serr<<"addSpringDForce, a="<<a<<", b="<<b<<", dforce ="<<dforce<<sendl;
}



template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams,DataVecDeriv& _df , const DataVecDeriv&  _dx )
{

    VecDeriv& df = *_df.beginEdit();		//WDataRefVecDeriv df(_df);
    const VecDeriv&  dx = _dx.getValue();	// RDataRefVecDeriv dx(_dx);

    double kFactor       =  mparams->kFactor();
    double bFactor       =  mparams->bFactor();

    if(ks.getValue()==0) return;

    const vector<Spring>& s = this->springs.getValue();

    //serr<<"addDForce, dx = "<<dx<<sendl;
    //serr<<"addDForce, df before = "<<f<<sendl;
    for (unsigned int i=0; i<s.size(); i++)
    {
        this->addSpringDForce(df,dx, i, s[i], kFactor, bFactor);
    }
    //serr<<"addDForce, df = "<<f<<sendl;
    _df.endEdit();
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams,const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if(ks.getValue()==0) return;

    double kFact = mparams->kFactor();

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat = matrix->getMatrix(this->mstate);
    if (!mat) return;
    const vector<Spring >& ss = this->springs.getValue();
    const unsigned int n = ss.size() < this->dfdx.size() ? ss.size() : this->dfdx.size();
    for (unsigned int e=0; e<n; e++)
    {
        const Spring& s = ss[e];
        unsigned p1 = mat.offset+Deriv::total_size*s.m1;
        const Mat& m = this->dfdx[e];
        for(int i=0; i<N; i++)
            for (int j=0; j<N; j++)
            {
                Real k = (Real)(m[i][j]*kFact);
                mat.matrix->add(p1+i,p1+j, -k);
            }
    }
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(ks.getValue()==0) return;

    if (this->closestPos.size()!=springs.getValue().size()) return;
    if (!vparams->displayFlags().getShowForceFields() && !drawColorMap.getValue()) return;

    ReadAccessor< Data< VecCoord > > x(*this->getMState()->read(core::ConstVecCoordId::position()));
    //const VecCoord& x = *this->mstate->getX();
    const vector<Spring>& springs = this->springs.getValue();

    if (vparams->displayFlags().getShowForceFields())
    {
        std::vector< Vector3 > points;
        for (unsigned int i=0; i<springs.size(); i++)
            if(!sourceIgnored[i])
            {
                Vector3 point1 = DataTypes::getCPos(x[springs[i].m1]);
                Vector3 point2 = DataTypes::getCPos(this->closestPos[i]);
                points.push_back(point1);
                points.push_back(point2);
            }

        const Vec<4,float> c(0,1,0.5,1);
        if (showArrowSize.getValue()==0 || drawMode.getValue() == 0)	vparams->drawTool()->drawLines(points, 1, c);
        else if (drawMode.getValue() == 1)	for (unsigned int i=0;i<points.size()/2;++i) vparams->drawTool()->drawCylinder(points[2*i+1], points[2*i], showArrowSize.getValue(), c);
        else if (drawMode.getValue() == 2)	for (unsigned int i=0;i<points.size()/2;++i) vparams->drawTool()->drawArrow(points[2*i+1], points[2*i], showArrowSize.getValue(), c);
        else serr << "No proper drawing mode found!" << sendl;
    }

    if(drawColorMap.getValue())
    {
        std::vector< Real > dists(x.size());  for (unsigned int i=0; i<dists.size(); i++) dists[i]=0.;
        for (unsigned int i=0; i<springs.size(); i++)
            if(!sourceIgnored[i])
            {
                Vector3 point1 = DataTypes::getCPos(x[springs[i].m1]);
                Vector3 point2 = DataTypes::getCPos(this->closestPos[i]);
                dists[springs[i].m1]=(point2-point1).norm();
            }
        Real max=0; for (unsigned int i=0; i<dists.size(); i++) if(max<dists[i]) max=dists[i];

        glPushAttrib( GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT);
        glDisable( GL_LIGHTING);

        ReadAccessor< Data< helper::vector< tri > > > t(sourceTriangles);
        if(t.size()) // mesh visu
        {
            glBegin( GL_TRIANGLES);
            for ( unsigned int i = 0; i < t.size(); i++)
            {
                for ( unsigned int j = 0; j < 3; j++)
                {
                    const unsigned int& indexP = t[i][j];
                    sofa::helper::gl::Color::setHSVA(dists[indexP]*240./max,1.,.8,1.);
                    glVertex3d(x[indexP][0],x[indexP][1],x[indexP][2]);
                }
            }
            glEnd();
        }

        else // point visu
        {
            glPointSize( 10);
            glBegin( GL_POINTS);
            for (unsigned int i=0; i<dists.size(); i++)
            {
                sofa::helper::gl::Color::setHSVA(dists[i]*240./max,1.,.8,1.);
                glVertex3d(x[i][0],x[i][1],x[i][2]);
            }
            glEnd();
            glPointSize( 1);
        }

        glPopAttrib();
    }

}


}
}
} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_ClosestPointRegistrationForceField_INL */
