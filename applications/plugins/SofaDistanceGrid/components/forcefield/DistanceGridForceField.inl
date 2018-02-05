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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_INL

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include "DistanceGridForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>



namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
void DistanceGridForceField<DataTypes>::init()
{
    Inherit::init();


    if (fileDistanceGrid.getValue().empty())
    {
        if (grid==NULL)
            msg_error() << "DistanceGridForceField requires an input filename." ;
        /// the grid has already been set
        return;
    }
    msg_info() << " creating "<<nx.getValue()<<"x"<<ny.getValue()<<"x"<<nz.getValue()<< msgendl
               << " DistanceGrid from file '"<< fileDistanceGrid.getValue() <<"'.";

    msg_info_when(scale.getValue()!=1.0) << " scale="<<scale.getValue();

    msg_info_when(box.getValue()[0][0]<box.getValue()[1][0])
            <<" bbox=<"<<box.getValue()[0]<<">-<"<<box.getValue()[0]<<">";

    grid = DistanceGrid::loadShared(fileDistanceGrid.getFullPath(), scale.getValue(), 0.0,
                                    nx.getValue(),ny.getValue(),nz.getValue(),
                                    box.getValue()[0],box.getValue()[1]);

    if (this->stiffnessArea.getValue() != 0 && this->mstate)
    {
        core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
        if (topology && topology->getNbTriangles() > 0)
        {
            const core::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();
            Real sumArea = 0;
            Real sumSArea = 0;
            const VecCoord& p1 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
            pOnBorder.resize(p1.size(), false);
            for (unsigned int ti = 0; ti < triangles.size(); ++ti)
            {
                helper::fixed_array<unsigned int,3> t = triangles[ti];
                Coord B = p1[t[1]]-p1[t[0]];
                Coord C = p1[t[2]]-p1[t[0]];
                Coord tN = cross(B, C);
                Real area = tN.norm()/2;
                Coord sN = grid->grad((p1[t[0]]+p1[t[1]]+p1[t[2]])*(1.0/3.0));
                sumArea += area;
                sumSArea = (sN*tN)*0.5f;
                pOnBorder[t[0]] = true;
                pOnBorder[t[1]] = true;
                pOnBorder[t[2]] = true;
            }
            msg_info() << "Surface area : " << sumArea << " ( mean " << sumArea / triangles.size() << " per triangle )" ;
            flipNormals = (sumSArea < 0);
        }
        else
        {
            serr << "No triangles found in topology" << sendl;
        }
    }

    if (this->stiffnessVolume.getValue() != 0 && this->mstate)
    {
        core::topology::BaseMeshTopology* topology = this->mstate->getContext()->getMeshTopology();
        if (topology && topology->getNbTetrahedra() > 0)
        {
            const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = topology->getTetrahedra();
            Real sumVolume = 0;
            const VecCoord& p1 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
            for (unsigned int ti = 0; ti < tetrahedra.size(); ++ti)
            {
                helper::fixed_array<unsigned int,4> t = tetrahedra[ti];
                Coord A = p1[t[1]]-p1[t[0]];
                Coord B = p1[t[2]]-p1[t[0]];
                Coord C = p1[t[3]]-p1[t[0]];
                Real volume = (A*cross(B, C))/6.0f;
                sumVolume += volume;
            }
            msg_info() << "Volume : " << sumVolume << " ( mean " << sumVolume / tetrahedra.size() << " per tetra )" ;
        }
        else
        {
            serr << "No tetrahedra found in topology" << sendl;
        }
    }

}

template<class DataTypes>
void DistanceGridForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV )
{
    VecDeriv& f1 = *(dataF.beginEdit());
    const VecCoord& p1=dataX.getValue();
    const VecDeriv& v1=dataV.getValue();


    if (!grid) return;
    //this->dfdd.resize(p1.size());
    f1.resize(p1.size());

    sofa::helper::vector<Contact>& contacts = *this->contacts.beginEdit();
    contacts.clear();

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

    const Real stiffIn = stiffnessIn.getValue();
    const Real stiffOut = stiffnessOut.getValue();
    const Real damp = damping.getValue();
    const Real maxdist = maxDist.getValue();
    unsigned int nbIn = 0;
    for (unsigned int i=ibegin; i<iend; i++)
    {
        if (i < pOnBorder.size() && !pOnBorder[i]) continue;
        Real d = grid->teval(p1[i]);
        if(d > 0)
        {
            if (d >= maxdist || stiffOut == 0) continue;
            Deriv grad = grid->tgrad(p1[i]);
            Real forceIntensity = -stiffOut * d;
            Real dampingIntensity = forceIntensity * damp;
            Deriv force = grad * forceIntensity - v1[i]*dampingIntensity;
            f1[i]+=force;
            Contact c;
            c.index = i;
            c.normal = grad;
            c.fact = forceIntensity;
            contacts.push_back(c);
        }
        else if (d < 0)
        {
            if (-d >= maxdist || stiffIn == 0) continue;
            Deriv grad = grid->tgrad(p1[i]);
            Real forceIntensity = -stiffIn * d;
            Real dampingIntensity = forceIntensity * damp;
            Deriv force = grad * forceIntensity - v1[i]*dampingIntensity;
            f1[i]+=force;
            Contact c;
            c.index = i;
            c.normal = grad;
            c.fact = forceIntensity;
            contacts.push_back(c);
            nbIn++;
        }
    }

    dmsg_info() << " number of points " << nbIn ;

    this->contacts.endEdit();

    sofa::helper::vector<TContact>& tcontacts = *this->tcontacts.beginEdit();
    tcontacts.clear();

    const Real stiffA = stiffnessArea.getValue();
    const Real minA = minArea.getValue();
    if (stiffA != 0)
    {
        core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
        if (topology && topology->getNbTriangles() > 0)
        {
            const core::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();
            for (unsigned int ti = 0; ti < triangles.size(); ++ti)
            {
                helper::fixed_array<unsigned int,3> t = triangles[ti];
                Coord B = p1[t[1]]-p1[t[0]];
                Coord C = p1[t[2]]-p1[t[0]];
                Coord tN = cross(B, C);
                Coord tcenter = (p1[t[0]]+p1[t[1]]+p1[t[2]])*(1.0/3.0);
                Coord sN = grid->tgrad(tcenter);
                if (flipNormals) sN = -sN;
                sN.normalize();
                Real area = (tN * sN) * 0.5f;

                if (area < minA)
                {

                    // lets consider A = (0,0,0)
                    // area = 0.5*(sNx*tNx + sNy * tNy + sNz * tNz)
                    //      = 0.5*(sNx*(By*Cz-Bz*Cy) + sNy * (Bz*Cx-Bx*Cz) + sNz * (Bx*Cy-By*Cx))

                    // d(area) / dBx = 0.5*(sNz*Cy-sNy*Cz)
                    // d(area) / dBy = 0.5*(sNx*Cz-sNz*Cx)
                    // d(area) / dBz = 0.5*(sNy*Cx-sNx*Cy)
                    // d(area) / dB = 0.5*cross(C,sN)
                    // d(area) / dC = 0.5*cross(sN,B)

                    Real forceIntensity = (stiffA * (minA-area));
                    Coord fB = cross(C,sN)*forceIntensity; f1[t[1]] += fB;
                    Coord fC = cross(sN,B)*forceIntensity; f1[t[2]] += fC;
                    Coord fA = -(fB+fC);                   f1[t[0]] += fA;

                    TContact c;
                    c.index = t;
                    c.fact = minA-area;
                    c.normal = sN;
                    c.B = B;
                    c.C = C;
                    tcontacts.push_back(c);
                }
            }
        }
    }
    this->tcontacts.endEdit();

    sofa::helper::vector<VContact>& vcontacts = *this->vcontacts.beginEdit();
    vcontacts.clear();

    const Real stiffV = stiffnessVolume.getValue();
    const Real minV = minVolume.getValue();
    if (stiffV != 0)
    {
        core::topology::BaseMeshTopology* topology = this->mstate->getContext()->getMeshTopology();
        if (topology && topology->getNbTetrahedra() > 0)
        {
            const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = topology->getTetrahedra();
            const Real v1_6 = (Real)(1.0/6.0);
            for (unsigned int ti = 0; ti < tetrahedra.size(); ++ti)
            {
                helper::fixed_array<unsigned int,4> t = tetrahedra[ti];
                Coord A = p1[t[1]]-p1[t[0]];
                Coord B = p1[t[2]]-p1[t[0]];
                Coord C = p1[t[3]]-p1[t[0]];
                Real volume = (A*cross(B, C))*v1_6;

                if (volume < minV)
                {
                    // vol = 1/6*(A(BxC))
                    //      = 1/6*(Ax*(By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))

                    // d(vol) / dBx = 1/6*(Az*Cy-Ay*Cz)
                    // d(vol) / dBy = 1/6*(Ax*Cz-Az*Cx)
                    // d(vol) / dBz = 1/6*(Ay*Cx-Ax*Cy)
                    // d(vol) / dA = 1/6*cross(B,C)
                    // d(vol) / dB = 1/6*cross(C,A)
                    // d(vol) / dC = 1/6*cross(A,B)

                    Real forceIntensity = v1_6*(stiffV * (minV-volume));
                    Coord fA = cross(B,C)*forceIntensity; f1[t[1]] += fA;
                    Coord fB = cross(C,A)*forceIntensity; f1[t[2]] += fB;
                    Coord fC = cross(A,B)*forceIntensity; f1[t[3]] += fC;
                    Coord f0 = -(fA+fB+fC);               f1[t[0]] += f0;

                    VContact c;
                    c.index = t;
                    c.fact = minV-volume;
                    c.A = A;
                    c.B = B;
                    c.C = C;
                    vcontacts.push_back(c);
                }
            }
        }
    }
    this->vcontacts.endEdit();

    dataF.endEdit();

}

template<class DataTypes>
void DistanceGridForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv&   datadF , const DataVecDeriv&   datadX )
{
    VecDeriv& df1      = *(datadF.beginEdit());
    const VecCoord& dx1=   datadX.getValue()  ;
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    if (!grid)
        return;

    const sofa::helper::vector<Contact>& contacts = this->contacts.getValue();
    const sofa::helper::vector<TContact>& tcontacts = this->tcontacts.getValue();
    const sofa::helper::vector<VContact>& vcontacts = this->vcontacts.getValue();

    if (contacts.empty() && tcontacts.empty() && vcontacts.empty())
        return;

    df1.resize(dx1.size());
    const Real fact = (Real)(kFactor);

    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = (this->contacts.getValue())[i];
        Coord du = dx1[c.index];

        Real dd = du * c.normal;
        Deriv dforce = c.normal * (dd * c.fact * fact);
        df1[c.index] += dforce;
    }

    const Real factA = (Real)( -this->stiffnessArea.getValue()* kFactor );
    for (unsigned int i=0; i<tcontacts.size(); i++)
    {
        const TContact& c = (this->tcontacts.getValue())[i];
        const helper::fixed_array<unsigned int,3>& t = c.index;
        Coord dB = dx1[t[1]]-dx1[t[0]];
        Coord dC = dx1[t[2]]-dx1[t[0]];

        // d(area) = dot(0.5*cross(C,sN), dB) + dot(0.5*cross(sN,B), dC)
        Real darea = 0.5f*(cross(c.C,c.normal)*dB + cross(c.normal,c.B)*dC);

        // fB = (C x sN) * (stiffA * (minA-area));
        // dfB = (C x sN) * (-stiffA * d(area)) + (dC x sN) * (stiffA * (minA-area));
        Coord dfB = cross(c.C, c.normal) * (factA * darea) - cross (dC, c.normal) * (factA*c.fact);
        Coord dfC = cross(c.normal, c.B) * (factA * darea) - cross (c.normal, dB) * (factA*c.fact);
        Coord dfA = -(dfB+dfC);
        df1[t[0]] += dfA;
        df1[t[1]] += dfB;
        df1[t[2]] += dfC;
    }

    const Real factV = (Real)( - this->stiffnessVolume.getValue()*(1.0/6.0)* kFactor );
    for (unsigned int i=0; i<vcontacts.size(); i++)
    {
        const Real v1_6 = (Real)(1.0/6.0);
        const VContact& c = (this->vcontacts.getValue())[i];
        const helper::fixed_array<unsigned int,4>& t = c.index;
        Coord dA = dx1[t[1]]-dx1[t[0]];
        Coord dB = dx1[t[2]]-dx1[t[0]];
        Coord dC = dx1[t[3]]-dx1[t[0]];
        // d(vol) = 1/6*(dot(cross(B,C), dA) + dot(cross(C,A), dB) + dot(cross(A,B), dC))
        Real dvolume = v1_6*(dA*cross(c.B,c.C) + dB*cross(c.C,c.A) + dC*cross(c.A,c.B));

        // fA = (1/6)*(B x C)*(stiffV * (minV-volume))
        // dfA = (stiffV*1/6) * ((dB x C + B x dC) * (minV-volume) - (B x C) * dvol)
        // dfB = (stiffV*1/6) * ((dC x A + C x dA) * (minV-volume) - (C x A) * dvol)
        // dfC = (stiffV*1/6) * ((dA x B + A x dB) * (minV-volume) - (A x B) * dvol)
        Coord dfA = cross(c.B, c.C) * (factV * dvolume) - (cross(dB, c.C) + cross(c.B, dC)) * (factV*c.fact);
        Coord dfB = cross(c.C, c.A) * (factV * dvolume) - (cross(dC, c.A) + cross(c.C, dA)) * (factV*c.fact);
        Coord dfC = cross(c.A, c.B) * (factV * dvolume) - (cross(dA, c.B) + cross(c.A, dB)) * (factV*c.fact);
        Coord df0 = -(dfA+dfB+dfC);
        df1[t[1]] += dfA;
        df1[t[2]] += dfB;
        df1[t[3]] += dfC;
        df1[t[0]] += df0;
    }

    datadF.endEdit();
}

template<class DataTypes>
void DistanceGridForceField<DataTypes>::addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    unsigned int &offset = r.offset;
    sofa::defaulttype::BaseMatrix* mat = r.matrix;

    if (r)
    {
        if (!grid) return;
        const sofa::helper::vector<Contact>& contacts = this->contacts.getValue();
        if (contacts.empty()) return;
        for (unsigned int i=0; i<contacts.size(); i++)
        {
            const Contact& c = contacts[i];
            const int p = c.index;
            const Real fact = (Real)(c.fact * -kFactor);
            const Deriv& normal = c.normal;
            for (int l=0; l<Deriv::total_size; ++l)
                for (int c=0; c<Deriv::total_size; ++c)
                {
                    SReal coef = normal[l] * fact * normal[c];
                    mat->add(offset + p*Deriv::total_size + l, offset + p*Deriv::total_size + c, coef);
                }
        }
    }
}

template<class DataTypes>
void DistanceGridForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    drawDistanceGrid(vparams);
}

template<class DataTypes>
void DistanceGridForceField<DataTypes>::drawDistanceGrid(const core::visual::VisualParams* vparams,float size)
{
    if (!grid) return;
    if (size == 0.0f) size = (float)drawSize.getValue();

    const VecCoord& p1 = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    std::vector< defaulttype::Vector3 > pointsLineIn;
    std::vector< defaulttype::Vector3 > pointsLineOut;
    // lines for points penetrating the distancegrid

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

    const Real stiffIn = stiffnessIn.getValue();
    const Real stiffOut = stiffnessOut.getValue();
    const Real maxdist = maxDist.getValue();

    defaulttype::Vector3 point1,point2;
    for (unsigned int i=ibegin; i<iend; i++)
    {
        if (i < pOnBorder.size() && !pOnBorder[i]) continue;
        Real d = grid->teval(p1[i]);
        if (d > 0)
        {
            if (d >= maxdist || stiffOut == 0) continue;
        }
        else if (d < 0)
        {
            if (-d >= maxdist || stiffIn == 0) continue;
        }
        else continue;
        Coord p2 = p1[i];
        Deriv normal = grid->tgrad(p1[i]); //normal.normalize();
        p2 += normal*(-d);
        point1 = DataTypes::getCPos(p1[i]);
        point2 = DataTypes::getCPos(p2);
        if (d > 0)
        {
            pointsLineOut.push_back(point1);
            pointsLineOut.push_back(point2);
        }
        else //if (d < 0)
        {
            pointsLineIn.push_back(point1);
            pointsLineIn.push_back(point2);
        }
    }
    if (!pointsLineIn.empty())
        vparams->drawTool()->drawLines(pointsLineIn, 1, defaulttype::Vec<4,float>(1,0,0,1));
    if (!pointsLineOut.empty())
        vparams->drawTool()->drawLines(pointsLineOut, 1, defaulttype::Vec<4,float>(1,0,1,1));

    const sofa::helper::vector<TContact>& tcontacts = this->tcontacts.getValue();
    if (!tcontacts.empty())
    {
        std::vector< defaulttype::Vector3 > pointsTri;
        for (unsigned int i=0; i<tcontacts.size(); i++)
        {
            const TContact& c = (this->tcontacts.getValue())[i];
            defaulttype::Vector3 p;
            for (int j=0; j<3; ++j)
            {
                p = DataTypes::getCPos(p1[c.index[j]]);
                pointsTri.push_back(p);
            }
        }
        vparams->drawTool()->drawTriangles(pointsTri, defaulttype::Vec<4,double>(1.0,0.2,0.2,0.5));
    }
    const sofa::helper::vector<VContact>& vcontacts = this->vcontacts.getValue();
    if (!vcontacts.empty())
    {
        std::vector< defaulttype::Vector3 > pointsTet;
        for (unsigned int i=0; i<vcontacts.size(); i++)
        {
            const VContact& c = (this->vcontacts.getValue())[i];
            const helper::fixed_array<unsigned int,4>& t = c.index;
            defaulttype::Vector3 p[4];
            Coord pc = (p1[t[0]]+p1[t[1]]+p1[t[2]]+p1[t[3]])*0.25f;
            for (int j=0; j<4; ++j)
            {
                Coord pj = p1[c.index[j]];
                pj += (pc-pj)*0.2f;
                p[j] = DataTypes::getCPos(pj);
            }
            pointsTet.push_back(p[0]);
            pointsTet.push_back(p[1]);
            pointsTet.push_back(p[2]);
            pointsTet.push_back(p[0]);
            pointsTet.push_back(p[2]);
            pointsTet.push_back(p[3]);
            pointsTet.push_back(p[0]);
            pointsTet.push_back(p[3]);
            pointsTet.push_back(p[1]);
            pointsTet.push_back(p[1]);
            pointsTet.push_back(p[3]);
            pointsTet.push_back(p[2]);
        }
        vparams->drawTool()->drawTriangles(pointsTet, defaulttype::Vec<4,double>(0.8,0.8,0,0.25));
    }

    if (drawPoints.getValue())
    {
        std::vector< defaulttype::Vector3 > distancePointsIn;
        std::vector< defaulttype::Vector3 > distancePointsOut;

        for (int i=0; i < grid->getNx(); i++)
            for (int j=0; j < grid->getNy(); j++)
                for (int k=0; k < grid->getNz(); k++)
                {
                    Coord cellCoord = grid->coord(i,j,k);
                    if (grid->teval(cellCoord) < 0.0)
                        distancePointsIn.push_back(cellCoord);
                    else
                        distancePointsOut.push_back(cellCoord);
                }

        if (distancePointsIn.size())
            vparams->drawTool()->drawPoints(distancePointsIn, (float)drawSize.getValue(), defaulttype::Vec<4,double>(0.8,0.2,0.2,1.0));
        if (distancePointsOut.size())
            vparams->drawTool()->drawPoints(distancePointsOut, (float)drawSize.getValue()*1.2f, defaulttype::Vec<4,double>(0.2,0.8,0.2,1.0));
    }

}
} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_INL
