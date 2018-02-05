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
#ifndef SOFA_COMPONENT_FORCEFIELD_INTERACTION_ELLIPSOIDFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_INTERACTION_ELLIPSOIDFORCEFIELD_INL

#include <SofaGeneralObjectInteraction/InteractionEllipsoidForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

// v = sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1
// dv/dxj = xj/rj� * 1/sqrt(x0�/r0�+x1�/r1�+x2�/r2�)

// f  = -stiffness * v * (dv/dp) / norm(dv/dp)

// fi = -stiffness * (sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1) * (xi/ri�) / sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4)

// dfi/dxj = -stiffness * [ d(sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1)/dxj *   (xi/ri�) / sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4)
//                          +  (sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1)     * d(xi/ri�)/dxj / sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4)
//                          +  (sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1)     *  (xi/ri�) * d(1/sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4))/dxj ]
// dfi/dxj = -stiffness * [ xj/rj� * 1/sqrt(x0�/r0�+x1�/r1�+x2�/r2�) * (xi/ri�) / sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4)
//                          +  (sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1)       * (i==j)/ri� / sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4)
//                          +  (sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1)       * (xi/ri�) * (-1/2*2xj/rj^4*1/(x0�/r0^4+x1�/r1^4+x2�/r2^4) ]
// dfi/dxj = -stiffness * [ xj/rj� * 1/sqrt(x0�/r0�+x1�/r1�+x2�/r2�) * (xi/ri�) / sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4)
//                          +  (sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1)       * (i==j)/ri� / sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4)
//                          +  (sqrt(x0�/r0�+x1�/r1�+x2�/r2�)-1)       * (xi/ri�) * (-xj/rj^4*1/(x0�/r0^4+x1�/r1^4+x2�/r2^4) ]

// dfi/dxj = -stiffness * [ (xj/rj�) * (xi/ri�) * 1/(sqrt(x0�/r0�+x1�/r1�+x2�/r2�) * sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4))
//                          +  v       * (i==j) / (ri�*sqrt(x0�/r0^4+x1�/r1^4+x2�/r2^4))
//                          +  v       * (xi/ri�) * (xj/rj�) * 1/(rj�*(x0�/r0^4+x1�/r1^4+x2�/r2^4) ]


template<class DataTypes1, class DataTypes2>
void InteractionEllipsoidForceField<DataTypes1, DataTypes2>::init()
{
    Inherit1::init();
    vars.pos6D = this->mstate2->read(core::VecCoordId::position())->getValue()[object2_dof_index.getValue()];
    if(object2_invert.getValue())
        vars.pos6D = DataTypes2::inverse(vars.pos6D);
    initCalcF();
}

template<class DataTypes1, class DataTypes2>
void InteractionEllipsoidForceField<DataTypes1, DataTypes2>::reinit()
{
    Inherit1::reinit();
    vars.pos6D = this->mstate2->read(core::VecCoordId::position())->getValue()[object2_dof_index.getValue()];
    if(object2_invert.getValue())
        vars.pos6D = DataTypes2::inverse(vars.pos6D);
    initCalcF();
}

template<class DataTypes1, class DataTypes2>
void InteractionEllipsoidForceField<DataTypes1, DataTypes2>::initCalcF()
{
    vars.stiffness = this->stiffness.getValue();
    vars.stiffabs =  helper::rabs(vars.stiffness);
    vars.damping = this->damping.getValue();
    helper::ReadAccessor< DataVecCoord1 > vr = this->vradius;
    helper::ReadAccessor< DataVecCoord1 > vcenter = this->center;
    vars.nelems = (vr.size() > vcenter.size()) ? vr.size() : vcenter.size();
    vars.vr.resize(vars.nelems);
    vars.vcenter.resize(vars.nelems);
    vars.vinv_r2.resize(vars.nelems);
    for (unsigned int e=0; e<vars.nelems; ++e)
    {
        Coord1 r;
        if (e < vr.size()) r = vr[e];
        else if (!vr.empty()) r = vr[vr.size()-1];
        else
        {
            for (int j=0; j<N; j++) r[j] = 1.0f;
        }
        Coord1 center;
        if (e < vcenter.size()) center = vcenter[e];
        else if (!vcenter.empty()) center = vcenter[vcenter.size()-1];
        vars.vr[e] = r;
        vars.vcenter[e] = center;
        for (int j=0; j<N; j++) vars.vinv_r2[e][j] = 1/(vars.vr[e][j]*vars.vr[e][j]);
    }
    //printf("\n **********************");
    //printf("\n vars.inv_r2 = %f %f %f", vars.inv_r2[0], vars.inv_r2[1], vars.inv_r2[2]);
}

template<class DataTypes1, class DataTypes2>
bool InteractionEllipsoidForceField<DataTypes1, DataTypes2>::calcF(const Coord1& p1, const Deriv1& v1, Deriv1 &f1, Mat& dfdx)
{
    Coord1 bdp;
    Real1 bnorm2 = -1;
    int be = -1;
    for (unsigned int e=0; e<vars.nelems; ++e)
    {
        Coord1 dp = p1 - vars.vcenter[e];
        Real1 norm2 = 0;
        for (int j=0; j<N; j++) norm2 += (dp[j]*dp[j])*vars.vinv_r2[e][j];
        if (be == -1 || norm2 < bnorm2)
        {
            bnorm2 = norm2;
            be = e;
            bdp = dp;
        }
    }
    //Real1 d = (norm2-1)*s2;
    if ((bnorm2-1)*vars.stiffness<0)
    {
        int e = be;
        Coord1 dp = bdp;
        Real1 norm2 = bnorm2;

        //printf("\n norm2 = %f", norm2);
        //printf("\n dp = %f %f %f   p1 = %f %f %f   vars.center = %f %f %f", dp[0], dp[1], dp[2],
        //	p1.x(),p1.y(), p1.z(), vars.center.x(), vars.center.y(), vars.center.z());

        Real1 norm = helper::rsqrt(norm2);
        Real1 v = norm-1;
        Deriv1 grad;
        for (int j=0; j<N; j++) grad[j] = dp[j]*vars.vinv_r2[e][j];
        Real1 gnorm2 = grad.norm2();
        Real1 gnorm = helper::rsqrt(gnorm2);
        //grad /= gnorm; //.normalize();
        Real1 forceIntensity = -vars.stiffabs*v/gnorm;
        Real1 dampingIntensity = vars.damping*helper::rabs(v);
        Deriv1 force = grad*forceIntensity - v1*dampingIntensity;
        f1=force;
        Real1 fact1 = -vars.stiffabs / (norm * gnorm);
        Real1 fact2 = -vars.stiffabs*v / gnorm;
        Real1 fact3 = -vars.stiffabs*v / gnorm2;
        for (int ci = 0; ci < N; ++ci)
        {
            for (int cj = 0; cj < N; ++cj)
                dfdx[ci][cj] = grad[ci]*grad[cj] * (fact1 + fact3*vars.vinv_r2[e][cj]);
            dfdx[ci][ci] += fact2*vars.vinv_r2[e][ci];
        }
        return true;
    }
    else
        return false;
}

template<class DataTypes1, class DataTypes2>
void InteractionEllipsoidForceField<DataTypes1, DataTypes2>::addForce(
    const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv1& dataf1, DataVecDeriv2& dataf2,
    const DataVecCoord1& datax1, const DataVecCoord2& datax2,
    const DataVecDeriv1& datav1, const DataVecDeriv2& datav2)
{
    helper::WriteAccessor< DataVecDeriv1 > f1 = dataf1;
    Deriv2 f2;

    helper::ReadAccessor< DataVecCoord1 >  p1 = datax1;
    helper::ReadAccessor< DataVecCoord2 >  p2 = datax2;
    helper::ReadAccessor< DataVecDeriv1 >  v1 = datav1;
    helper::ReadAccessor< DataVecDeriv2 >  v2 = datav2;

    vars.pos6D = p2[object2_dof_index.getValue()];
    if(object2_invert.getValue())
        vars.pos6D = DataTypes2::inverse(vars.pos6D);

    sofa::defaulttype::Quat Cq = vars.pos6D.getOrientation();
    sofa::defaulttype::Vec3d Cx = (Coord1) vars.pos6D.getCenter();
    Deriv2 V6D = v2[object2_dof_index.getValue()];
    sofa::defaulttype::Vec3d Cv = (sofa::defaulttype::Vec3d) getVCenter(V6D);
    Cv.clear();

    initCalcF();

    sofa::helper::vector<Contact>* contacts = this->contacts.beginEdit();
    contacts->clear();
    f1.resize(p1.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Coord1 p1Xform = Cq.inverseRotate(p1[i] - Cx);

        Deriv1 v1Xform = Cq.inverseRotate(v1[i]);
        Deriv1 f1Xform;
        f1Xform.clear();
        Mat dfdx;
        if (calcF(p1Xform, v1Xform, f1Xform, dfdx))
        {
            //printf("\n p1[%d] : %f %f %f p1Xform[%d] : %f %f %f",i, p1[i].x(), p1[i].y(), p1[i].z(),i, p1Xform.x(),p1Xform.y(),p1Xform.z());

            Contact c;
            c.pos = p1[i];
            c.index = i;
            c.m = dfdx;

            sofa::defaulttype::Vec3d contactForce =  Cq.rotate(f1Xform);
            c.force = contactForce;
            f1[i]+=contactForce;
            f2.getVCenter() -= contactForce;
            c.bras_levier = p1[i] - Cx;
            //f2.getVOrientation() -= c.bras_levier.cross(contactForce);
            contacts->push_back(c);
            /*
            printf("\n contactForce : %f %f %f, bras_Levier : %f %f %f, f2 = %f %f %f - %f %f %f",
                contactForce.x(), contactForce.y(), contactForce.z(),
                c.bras_levier.x(), c.bras_levier.y(), c.bras_levier.z(),
                f2.getVCenter().x(), f2.getVCenter().y(), f2.getVCenter().z(),
                f2.getVOrientation().x(), f2.getVOrientation().y(), f2.getVOrientation().z());
             */

        }
    }
    /*
    printf("\n f2 = %f %f %f - %f %f %f",
        f2.getVCenter().x(), f2.getVCenter().y(), f2.getVCenter().z(),
        f2.getVOrientation().x(), f2.getVOrientation().y(), f2.getVOrientation().z());
     */
    /*
    printf("\n verify addForce2 : ");
    addForce2(f1, f2, p1, p2, v1, v2);
    printf("\n f2 = %f %f %f - %f %f %f",
        f2.getVCenter().x(), f2.getVCenter().y(), f2.getVCenter().z(),
        f2.getVOrientation().x(), f2.getVOrientation().y(), f2.getVOrientation().z());
     */

    if (object2_forces.getValue())
    {
        helper::WriteAccessor< DataVecDeriv2 > wf2 = dataf2;
        wf2.resize(p2.size());
        wf2[object2_dof_index.getValue()] += f2;
    }
    this->contacts.endEdit();
}

template<class DataTypes1, class DataTypes2>
void InteractionEllipsoidForceField<DataTypes1, DataTypes2>::addForce2(DataVecDeriv1& dataf1, DataVecDeriv2& dataf2, const DataVecCoord1& datap1, const DataVecCoord2& datap2, const DataVecDeriv1& datav1, const DataVecDeriv2& /*datav2*/)
{
    helper::WriteAccessor< DataVecDeriv1 > f1 = dataf1;
    helper::WriteAccessor< DataVecDeriv2 > f2 = dataf2;

    helper::ReadAccessor< DataVecCoord1 >  p1 = datap1;
    helper::ReadAccessor< DataVecCoord2 >  p2 = datap2;
    helper::ReadAccessor< DataVecDeriv1 >  v1 = datav1;

    Coord2 C = p2[object2_dof_index.getValue()];

    if(object2_invert.getValue())
    {
        C = DataTypes2::inverse(C);
    }

    sofa::defaulttype::Quat Cq = C.getOrientation();
    sofa::defaulttype::Vec3d Cx = (sofa::defaulttype::Vec3d) C.getCenter();

    f1.clear();
    f2.clear();
    f1.resize(p1.size());
    f2.resize(p2.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Coord1 p1Xform = Cq.inverseRotate(p1[i] - Cx);

        Deriv1 v1Xform = Cq.inverseRotate(v1[i]);
        Deriv1 f1Xform;
        f1Xform.clear();
        Mat dfdx;
        if (calcF(p1Xform, v1Xform, f1Xform, dfdx))
        {


            sofa::defaulttype::Vec3d contactForce =  Cq.rotate(f1Xform);
            sofa::defaulttype::Vec3d bras_levier;
            bras_levier = p1[i] - Cx;
            f1[i]+=contactForce;
            getVCenter(f2[object2_dof_index.getValue()]) -= contactForce;
            getVOrientation(f2[object2_dof_index.getValue()]) -= bras_levier.cross(contactForce);


        }
    }



}


template<class DataTypes1, class DataTypes2>
void InteractionEllipsoidForceField<DataTypes1, DataTypes2>::addDForce(
    const sofa::core::MechanicalParams* mparams, DataVecDeriv1& datadf1, DataVecDeriv2& datadf2,
    const DataVecDeriv1& datadx1, const DataVecDeriv2& datadx2)

{
    const SReal kFactor = mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    helper::WriteAccessor< DataVecDeriv1 > df1 = datadf1;
    Deriv2 df2;
    helper::ReadAccessor< DataVecDeriv1 >  dx1 = datadx1;
    helper::ReadAccessor< DataVecDeriv2 >  dx2 = datadx2;
    Deriv2 dx2i;
    if (object2_forces.getValue())
    {
        dx2i = dx2[object2_dof_index.getValue()];
    }

    const sofa::defaulttype::Quat Cq = vars.pos6D.getOrientation();

    df1.resize(dx1.size());
    const sofa::helper::vector<Contact>& contacts = this->contacts.getValue();
    //printf("\n");
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        assert((unsigned)c.index<dx1.size());
        sofa::defaulttype::Vec3d du;
        du = (sofa::defaulttype::Vec3d) dx1[c.index] - (sofa::defaulttype::Vec3d) getVCenter(dx2i); //- c.bras_levier.cross(dx2i.getVOrientation());
        Deriv1 dforce = c.m * Cq.inverseRotate(du);
        dforce *= kFactor;
        Deriv1 DF = Cq.rotate(dforce);
        df1[c.index] += DF;
        df2.getVCenter()  -= DF;
        //df2.getVOrientation()  -= c.bras_levier.cross(DF);
        //printf(" bras_levier[%d] = %f %f %f  - ", i, c.bras_levier.x(), c.bras_levier.y(), c.bras_levier.z());
    }

    if (object2_forces.getValue())
    {
        helper::WriteAccessor< DataVecDeriv2 > wdf2 = datadf2;
        wdf2.resize(dx2.size());
        wdf2[object2_dof_index.getValue()] += df2;
    }

}

template <class DataTypes1, class DataTypes2>
SReal InteractionEllipsoidForceField<DataTypes1, DataTypes2>::getPotentialEnergy(const sofa::core::MechanicalParams* /*mparams*/, const DataVecCoord1& /*x1*/, const DataVecCoord2& /*x2*/) const
{
    serr<<"InteractionEllipsoidForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}

template<class DataTypes1, class DataTypes2>
void InteractionEllipsoidForceField<DataTypes1, DataTypes2>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;
    if (!bDraw.getValue()) return;
    Real1 cx2=0, cy2=0, cz2=0;

    cx2=(Real1)vars.pos6D.getCenter()[0];
    cy2=(Real1)vars.pos6D.getCenter()[1];
    cz2=(Real1)vars.pos6D.getCenter()[2];

    for (unsigned int e=0; e<vars.nelems; ++e)
    {

        Real1 cx1=0, cy1=0, cz1=0;
        DataTypes1::get(cx1, cy1, cz1, vars.vcenter[e] );

        Real1 rx=1, ry=1, rz=1;
        DataTypes1::get(rx, ry, rz, vars.vr[e]);
        glEnable(GL_CULL_FACE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor3f(color.getValue()[0],color.getValue()[1],color.getValue()[2]);

        sofa::defaulttype::Quat q=vars.pos6D.getOrientation();
#ifdef SOFA_FLOAT
        GLfloat R[4][4];
#elif PS3
        double R[4][4];
#else
        GLdouble R[4][4];
#endif

        glPushMatrix();
        //if(!object2_invert.getValue())
        {
            sofa::defaulttype::Quat q1=q.inverse();
            q1.buildRotationMatrix(R);
            helper::gl::glTranslate(cx2, cy2, cz2);
            helper::gl::glMultMatrix( &(R[0][0]) );
        }
        sofa::defaulttype::Vector3 center(cx1, cy1, cz1);
        sofa::defaulttype::Vector3 radii(rx, ry, (stiffness.getValue()>0 ? rz : -rz));

        vparams->drawTool()->drawEllipsoid(center, radii);

        glTranslated(-cx2, -cy2, -cz2);

        glPopMatrix();
        glDisable(GL_CULL_FACE);
        glDisable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL);

        /*if(object2_invert.getValue())
        {
            glPushMatrix();
            q.buildRotationMatrix(R);
            helper::gl::glMultMatrix( &(R[0][0]) );
            helper::gl::glTranslate(-cx2, -cy2, -cz2);
        }*/

        const sofa::helper::vector<Contact>& contacts = this->contacts.getValue();
        const double fscale = 1000.0f/this->stiffness.getValue();
        glColor4f (1,0.5f,0.5f,1);
        glBegin (GL_LINES);
        for (unsigned int i=0; i<contacts.size(); i++)
        {
            glVertex3d(contacts[i].pos[0],contacts[i].pos[1],contacts[i].pos[2] );
            glVertex3d(contacts[i].pos[0]+contacts[i].force[0]*fscale,
                    contacts[i].pos[1]+contacts[i].force[1]*fscale,
                    contacts[i].pos[2]+contacts[i].force[2]*fscale );
        }
        glEnd();
        /*if(object2_invert.getValue())
        {
            glPopMatrix();
        }*/
    }
#endif /* SOFA_NO_OPENGL */
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
