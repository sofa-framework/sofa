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
#define SOFA_COMPONENT_ENGINE_INERTIAALIGN_CPP

#include "InertiaAlign.h"
#include <sofa/core/ObjectFactory.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
namespace sofa
{

namespace component
{


using namespace sofa::defaulttype;

int InertiaAlignClass = core::RegisterObject("An engine computing inertia matrix and the principal direction of a mesh.")
        .add< InertiaAlign >()
        ;


SOFA_DECL_CLASS(InertiaAlign)

InertiaAlign::InertiaAlign()
    : targetC( initData(&targetC,"targetCenter","input: the gravity center of the target mesh") )
    , sourceC( initData(&sourceC,"sourceCenter","input: the gravity center of the source mesh") )
    , targetInertiaMatrix( initData(&targetInertiaMatrix,"targetInertiaMatrix","input: the inertia matrix of the target mesh") )
    , sourceInertiaMatrix( initData(&sourceInertiaMatrix,"sourceInertiaMatrix","input: the inertia matrix of the source mesh") )
    , m_positiont( initData(&m_positiont,"targetPosition","input: positions of the target vertices") )
    , m_positions( initData(&m_positions,"sourcePosition","input: positions of the source vertices") )
{

}


InertiaAlign::~InertiaAlign()
{
}

void InertiaAlign::init()
{
    //Activate an output data
    m_positions.setPersistent(false);
    m_positiont.setPersistent(false);

    sourceInertiaMatrix.setPersistent(false);
    // Allow to edit an output data

    helper::WriteAccessor<Data<helper::vector<sofa::defaulttype::Vec<3,SReal> > > > waPositions = m_positions;


    Eigen::MatrixXd eigenSourceInertiaMatrix(3,3);
    Eigen::MatrixXd eigenTargetInertiaMatrix(3,3);

    SReal Sxx,Syy,Szz,Txx,Tyy,Tzz;
    for(unsigned int i=0; i<3; i++ )
    {
        for( unsigned int j=0; j<3; j++ )
        {
            eigenSourceInertiaMatrix(i,j) = sourceInertiaMatrix.getValue().col(j)[i];
            eigenTargetInertiaMatrix(i,j) = targetInertiaMatrix.getValue().col(j)[i];

        }
    }
    sout << "Source intertia matrix"<<sendl<<eigenSourceInertiaMatrix << sendl;
    sout << "Target intertia matrix"<<sendl<<eigenTargetInertiaMatrix << sendl;

    Eigen::EigenSolver<Eigen::Matrix3d> solverTarget(eigenTargetInertiaMatrix);
    Eigen::EigenSolver<Eigen::Matrix3d> solverSource(eigenSourceInertiaMatrix);

    /* Creation of the following transformation matrix:
     *
     *U.x V.x W.x tx
     *U.y V.y W.Y ty
     *U.Z V.Z W.z tz
     * 0   0   0   1
     *
     *With U,V,W The proper vectors of the Source inertia matrix (represent the principal axis of inertia)
     *And tx,ty,tz the translation beetween the source object and the world
     */

    Eigen::Matrix3d eigenvectorsTarget,tmp ;
    Eigen::Vector3d eigenvaluesTarget;
    Eigen::Matrix3cd complexEigenvectorsTarget = solverTarget.eigenvectors();
    Eigen::Vector3cd complexEigenvaluesTarget = solverTarget.eigenvalues();

    Eigen::Matrix3d eigenvectorsSource ;
    Eigen::Vector3d eigenvaluesSource ;
    Eigen::Matrix3cd complexEigenvectorsSource = solverSource.eigenvectors();
    Eigen::Vector3cd complexEigenvaluesSource = solverSource.eigenvalues();

    for(unsigned int i = 0; i<3; i++)
    {
        for(unsigned int j=0;j<3;j++)
        {
            eigenvectorsSource(i,j)= real(complexEigenvectorsSource(i,j));
            eigenvectorsTarget(i,j)= real(complexEigenvectorsTarget(i,j));
        }
        eigenvaluesTarget(i) = real(complexEigenvaluesTarget(i));
        eigenvaluesSource(i) = real(complexEigenvaluesSource(i));
    }

    SReal minSource = eigenvaluesSource(0);

    //Compute the length of the principal axes
    Sxx = eigenvaluesSource[0];
    Syy = eigenvaluesSource[1];
    Szz = eigenvaluesSource[2];
    Txx = eigenvaluesTarget[0];
    Tyy = eigenvaluesTarget[1];
    Tzz = eigenvaluesTarget[2];

    SReal Lxs = sqrt(abs(-Sxx+Syy+Szz)/2);
    SReal Lys = sqrt(abs( Sxx-Syy+Szz)/2);
    SReal Lzs = sqrt(abs( Sxx+Syy-Szz)/2);

    SReal Lxt = sqrt(abs(-Txx+Tyy+Tzz)/2);
    SReal Lyt = sqrt(abs( Txx-Tyy+Tzz)/2);
    SReal Lzt = sqrt(abs( Txx+Tyy-Tzz)/2);

    //Source inversion of axes (to be sure that u is on u', v on v' and w on w')
    //Check thats the proper values are in the same order for the two objects
    //If not switch the proper values.
    //Inversion of u and v
    if (eigenvaluesSource[0] < eigenvaluesSource[1])
    {
        eigenvaluesSource(0)= eigenvaluesSource(1);
        eigenvaluesSource(1)=minSource;
        minSource = eigenvaluesSource(0);
        for(unsigned int i = 0; i<3; i++){
            tmp(i) = eigenvectorsSource(i,1);
            eigenvectorsSource(i,1) =   eigenvectorsSource(i,0);
            eigenvectorsSource(i,0) = tmp(i);
        }

    }
    //Inversion of u and w
    if (eigenvaluesSource[0]<eigenvaluesSource[2])
    {
        eigenvaluesSource(0)= eigenvaluesSource(2);
        eigenvaluesSource(2)=minSource;
        minSource = eigenvaluesSource(0);
        for(unsigned int i = 0; i<3; i++){
            tmp(i) = eigenvectorsSource(i,2);
            eigenvectorsSource(i,2) =   eigenvectorsSource(i,0);
            eigenvectorsSource(i,0) = tmp(i);
        }
    }
    //Inversion of v and w
    minSource = eigenvaluesSource(1);

    if (eigenvaluesSource[1]<eigenvaluesSource[2])
    {
        eigenvaluesSource(1)= eigenvaluesSource(2);
        eigenvaluesSource(2)=minSource;
        minSource = eigenvaluesSource(1);
        for(unsigned int i = 0; i<3; i++){
            tmp(i) = eigenvectorsSource(i,2);
            eigenvectorsSource(i,2) =   eigenvectorsSource(i,1);
            eigenvectorsSource(i,1) = tmp(i);
        }
    }


    SReal minTarget = eigenvaluesTarget(0);

    //Target Inversion of axes
    //Inversion of u' and v'
    if (eigenvaluesTarget[0]<eigenvaluesTarget[1])
    {
        eigenvaluesTarget(0)= eigenvaluesTarget(1);
        eigenvaluesTarget(1)=minTarget;
        minTarget = eigenvaluesTarget(0);
        for(unsigned int i = 0; i<3; i++){
            tmp(i) = eigenvectorsTarget(i,1);
            eigenvectorsTarget(i,1) =   eigenvectorsTarget(i,0);
            eigenvectorsTarget(i,0) = tmp(i);
        }
    }
    //Inversion of u' and w'
    if (eigenvaluesTarget[0]<eigenvaluesTarget[2])
    {
        eigenvaluesTarget(0)= eigenvaluesTarget(2);
        eigenvaluesTarget(2)=minTarget;
        minTarget = eigenvaluesTarget(0);
        for(unsigned int i = 0; i<3; i++){
            tmp(i) = eigenvectorsTarget(i,2);
            eigenvectorsTarget(i,2) =   eigenvectorsTarget(i,0);
            eigenvectorsTarget(i,0) = tmp(i);
        }
    }
    //Inversion of v' and w'
    minTarget = eigenvaluesTarget(1);

    if (eigenvaluesTarget[1]<eigenvaluesTarget[2])
    {
        eigenvaluesTarget(1)= eigenvaluesTarget(2);
        eigenvaluesTarget(2)=minTarget;
        minTarget = eigenvaluesTarget(1);
        for(unsigned int i = 0; i<3; i++){
            tmp(i) = eigenvectorsTarget(i,2);
            eigenvectorsTarget(i,2) =   eigenvectorsTarget(i,1);
            eigenvectorsTarget(i,1) = tmp(i);
        }
    }
    //Compute the length of the axes.
    //The scale along x is Lxs/Lxt
    //The ratio 1/5m have been supressed due to the ratio beetween the source/target

    Sxx = eigenvaluesSource[0];
    Syy = eigenvaluesSource[1];
    Szz = eigenvaluesSource[2];
    Txx = eigenvaluesTarget[0];
    Tyy = eigenvaluesTarget[1];
    Tzz = eigenvaluesTarget[2];

    Lxs = sqrt(abs(-Sxx+Syy+Szz)/2);
    Lys = sqrt(abs( Sxx-Syy+Szz)/2);
    Lzs = sqrt(abs( Sxx+Syy-Szz)/2);

    Lxt = sqrt(abs(-Txx+Tyy+Tzz)/2);
    Lyt = sqrt(abs( Txx-Tyy+Tzz)/2);
    Lzt = sqrt(abs( Txx+Tyy-Tzz)/2);


    sout << "EigenVectorsSource =" << sendl;
    for(unsigned int i=0;i<3;i++)
    {
        for(unsigned int j=0;j<3;j++)
            sout << eigenvectorsSource(i,j) << " ";
        sout << sendl;

    }

    SReal scale_u = Lxs / Lxt;
    SReal scale_v = Lys / Lyt;
    SReal scale_w = Lzs / Lzt;

    sout << "Scale X = " << scale_u << "Scale Y = "<< scale_v << "Scale Z = " << scale_w << sendl;


    sout << "Lxs = " <<Lxs<< "Lys = " <<Lys<< "Lzs = " <<Lzs << sendl;
    sout << "Lxt = " <<Lxt<< "Lyt = " <<Lyt<< "Lzt = " <<Lzt << sendl;
    /*Creation of the two 4x4 transformation matrix:
     *
     *MTransformSource :
     *
     *U.x V.x W.x tx
     *U.y V.y W.Y ty
     *U.z V.z W.z tz
     * 0   0   0   1
     *
     *
     *Then inverse R = MtransformSource [0..2][0..2]
     *t = -R*t
     *
     * with U,V and W the eigenvectors of the source inertia matrix in the world coordinates.
     *And tx,ty,tz represents the translation beetween the center of the world and the inertia center of the object S
     *
     *
     *MTransformTarget :
     *
     *U'.x V'.x W'.x tx
     *U'.y V'.y W'.Y ty
     *U'.z V'.z W'.z tz
     *  0    0    0   1
     *

     *
     * with U',V' and W' the eigenvectors of the target inertia matrix in the world coordinates.
     *And tx,ty,tz represents the translation beetween the center of the world and the inertia center of the object T
     *
     *
     */



    Eigen::Matrix3d MRotationSource, MRotationTarget;
    defaulttype::Matrix4 MTransformSource,MTransformTarget,MTransformTargetTest,MTransform,Mscale;
    MRotationSource =  eigenvectorsSource;
    MRotationTarget =  eigenvectorsTarget;
    Eigen::Vector3d u,v,w;
    SReal sdirect;
    Mscale(0,0)=scale_u;
    Mscale(1,1)=scale_v;
    Mscale(2,2)=scale_w;
    Mscale(3,3)=1;
    for(unsigned int i=0;i<3;i++)
    {
        u(i) = MRotationSource(i,0);
        v(i) = MRotationSource(i,1);
        w(i) = MRotationSource(i,2);

    }
    sdirect = u.cross(v).dot(w);


    for(unsigned int i=0;i<3;i++)
    {
        if(sdirect < 0)
        {
            MRotationSource(i,2) = -MRotationSource(i,2);
        }
    }


    sout <<"Source Directe? "<< sdirect << sendl;

    for(unsigned int i=0;i<3;i++)
    {
        u(i) = MRotationTarget(i,0);
        v(i) = MRotationTarget(i,1);
        w(i) = MRotationTarget(i,2);

    }

    //Checks if the axes are direct or not
    sdirect = u.cross(v).dot(w);



    for(unsigned int i=0;i<3;i++)
    {
        if(sdirect == -1)
        {
            MRotationTarget(i,2) = -MRotationTarget(i,2);
        }
    }

    sout <<"Target Directe? "<< sdirect << sendl;

    //Normalised last line
    MTransformTarget(3,3) = 1;

    for(unsigned int i=0;i<3;i++)
    {
        //TODO : vérifier ces histoires de Lxs/Lxt

        MTransformTarget(i,0) = MRotationTarget(i,0)*Lxt;
        MTransformTarget(i,1) = MRotationTarget(i,1)*Lyt;
        MTransformTarget(i,2) = MRotationTarget(i,2)*Lzt;
        MTransformTarget(3,i) = 0;

    }
    Eigen::Vector3d TargetTransform;
    for(unsigned int i=0;i<3;i++)
    {
        TargetTransform(i)= -(*targetC.beginEdit())[0];
    }
    MTransformTarget(0,3) = -(*targetC.beginEdit())[0];
    MTransformTarget(1,3) = -(*targetC.beginEdit())[1];
    MTransformTarget(2,3) = -(*targetC.beginEdit())[2];


    //Construction of S
    MTransformSource(3,3) = 1;
    Vector4 TranslationSource;

    MTransformSource(0,3) = -(*sourceC.beginEdit())[0];
    MTransformSource(1,3) = -(*sourceC.beginEdit())[1];
    MTransformSource(2,3) = -(*sourceC.beginEdit())[2];

    for(unsigned int i=0;i<3;i++)
    {
        //TODO : vérifier ces histoires de Lxs/Lxt

        MTransformSource(i,0) = MRotationSource(i,0) *Lxs;
        MTransformSource(i,1) = MRotationSource(i,1) *Lys;
        MTransformSource(i,2) = MRotationSource(i,2) *Lzs;
        MTransformSource(3,i) = 0;
        TranslationSource(i)  = MTransformSource(i,3);

    }

    sout << "MTransformSource before inversion ="<< sendl;
    for(unsigned int i=0;i<4;i++)
    {
        for(unsigned int j=0;j<4;j++)
        {
            sout <<MTransformSource(i,j)<< " ";
        }
        sout << sendl;
    }

    MTransformSource = inverseTransform(MTransformSource);
    sout << "MTransformSource after inversion ="<< sendl;
    for(unsigned int i=0;i<4;i++)
    {
        for(unsigned int j=0;j<4;j++)
        {
            sout <<MTransformSource(i,j)<< " ";
        }
        sout << sendl;
    }
    defaulttype::Matrix4 MTranslation;
    for(unsigned int i=0;i<4;i++)
    {
        for(unsigned int j=0;j<4;j++)
        {
            MTranslation(i,j)=0;
            if(i==j)
                MTranslation(i,i) = 1;
        }
    }




    int indice_min =0;
    SReal distance, distance_min;

    distance = std::numeric_limits<SReal>::max();
    distance_min = std::numeric_limits<SReal>::max();

    for (unsigned int i=0;i<4;i++)
        for (unsigned int j=0;j<4;j++)
            MTransformTargetTest(i,j) = MTransformTarget(i,j);

    for(unsigned int i=0;i<4;i++)//Test of the 4 possible permutations
    {
        switch (i)
        {
        case 0 :
            for(unsigned int j=0;j<3;j++)
            {
                MTransformTargetTest(j,0) = MTransformTarget(j,0) ;
                MTransformTargetTest(j,1) = MTransformTarget(j,1) ;
                MTransformTargetTest(j,2) = MTransformTarget(j,2) ;
            }
            break;
        case 1 :

            for(unsigned int j=0;j<3;j++)
            {
                MTransformTargetTest(j,0) = MTransformTarget(j,0) ;
                MTransformTargetTest(j,1) =-MTransformTarget(j,1) ;
                MTransformTargetTest(j,2) =-MTransformTarget(j,2) ;
            }
            break;
        case 2 :

            for(unsigned int j=0;j<3;j++)
            {
                MTransformTargetTest(j,0) =-MTransformTarget(j,0) ;
                MTransformTargetTest(j,1) = MTransformTarget(j,1) ;
                MTransformTargetTest(j,2) =-MTransformTarget(j,2) ;
            }
            break;
        case 3:

            for(unsigned int j=0;j<3;j++)
            {
                MTransformTargetTest(j,0) =-MTransformTarget(j,0) ;
                MTransformTargetTest(j,1) =-MTransformTarget(j,1) ;
                MTransformTargetTest(j,2) = MTransformTarget(j,2) ;

            }
            break;

        }
        MTransform = MTransformTargetTest * MTransformSource ;

        positionDistSource = (*m_positions.beginEdit());
        for (size_t j = 0; j < waPositions.size(); j++)
        {
            defaulttype::Vector4 pointS,pointT;
            pointS(0) = (*m_positions.beginEdit())[j][0];
            pointS(1) = (*m_positions.beginEdit())[j][1];
            pointS(2) = (*m_positions.beginEdit())[j][2];
            pointS(3) = 1;
            pointT = MTransform * pointS  ;
            (*m_positions.beginEdit())[j][0] =pointT(0);
            (*m_positions.beginEdit())[j][1] =pointT(1);
            (*m_positions.beginEdit())[j][2] =pointT(2);
        }

        distance =computeDistances(*m_positions.beginEdit(),*m_positiont.beginEdit());
        if (distance < distance_min)
        {
            indice_min = i;
            distance_min = distance;
        }
        (*m_positions.beginEdit()) = positionDistSource;
    }
    sout << "Indice choisi : " << indice_min << " avec une distance de " << distance_min << sendl;
    //Compute the best transformation
    switch(indice_min)
    {
    case 0 ://Nothing is inverted
        for(unsigned int j=0;j<3;j++)
        {
            MTransformTarget(j,0) = MTransformTarget(j,0) ;
            MTransformTarget(j,1) = MTransformTarget(j,1) ;
            MTransformTarget(j,2) = MTransformTarget(j,2) ;
        }
        break;
    case 1 ://v and w are inverted

        for(unsigned int j=0;j<3;j++)
        {
            MTransformTarget(j,0) = MTransformTarget(j,0) ;
            MTransformTarget(j,1) =-MTransformTarget(j,1) ;
            MTransformTarget(j,2) =-MTransformTarget(j,2) ;
        }
        break;
    case 2 ://u and w are inverted

        for(unsigned int j=0;j<3;j++)
        {
            MTransformTarget(j,0) =-MTransformTarget(j,0) ;
            MTransformTarget(j,1) = MTransformTarget(j,1) ;
            MTransformTarget(j,2) =-MTransformTarget(j,2) ;
        }
        break;
    case 3:// u and v are inverted

        for(unsigned int j=0;j<3;j++)
        {
            MTransformTarget(j,0) =-MTransformTarget(j,0) ;
            MTransformTarget(j,1) =-MTransformTarget(j,1) ;
            MTransformTarget(j,2) = MTransformTarget(j,2) ;
        }
        break;
    }

    for(unsigned int k=0;k<4;k++)
    {
       for(unsigned int j=0;j<4;j++)
       {
         sout << MTransformTarget(k,j) << " ";
       }
       sout << sendl;
    }
    MTransformSource = MTransformTarget * MTransformSource;


    for(unsigned int k=0;k<4;k++)
    {
       for(unsigned int j=0;j<4;j++)
       {
         sout << MTransformSource(k,j) << " ";
       }
       sout << sendl;
    }

    Eigen::Matrix4d MDeterminantTest;
    for(unsigned int k=0;k<3;k++)
    {
       for(unsigned int j=0;j<3;j++)
       {
           MDeterminantTest(k,j)=MTransformSource(k,j);
       }
    }
    if(MDeterminantTest.determinant()<0)
        sout << "The MTransformSourceMatrix is not a Transformation matrix" << sendl;
    for (size_t i = 0; i < waPositions.size(); i++)
    {
        defaulttype::Vector4 pointS,pointT;
        pointS(0) = (*m_positions.beginEdit())[i][0];
        pointS(1) = (*m_positions.beginEdit())[i][1];
        pointS(2) = (*m_positions.beginEdit())[i][2];
        pointS(3) = 1;
        pointT = MTransformSource * pointS  ;
        (*m_positions.beginEdit())[i][0] =pointT(0);
        (*m_positions.beginEdit())[i][1] =pointT(1);
        (*m_positions.beginEdit())[i][2] =pointT(2);
    }
    //After this step, this data cannot be modified
    m_positions.endEdit();

    m_positiont.endEdit();

}



//Hausdorff distance
//good approx but slow

/**
 * Compute the distance from a point to a point cloud
 */

SReal InertiaAlign::distance(sofa::defaulttype::Vec<3,SReal> p, helper::vector<sofa::defaulttype::Vec<3,SReal> > S)
{
    SReal min = std::numeric_limits<SReal>::max();

    for (unsigned int i = 0 ; i < S.size(); i++)
    {
        SReal d = (p-S[i]).norm();
        if (d<min) min = d;
    }

    return min;
}

/**
 * Compute distances between both point clouds (symmetrical and non-symmetrical distances)
 */
SReal InertiaAlign::computeDistances( helper::vector<sofa::defaulttype::Vec<3,SReal> > S, helper::vector<sofa::defaulttype::Vec<3,SReal> > T)
{
    SReal maxST = 0.0;
    for (unsigned int i = 0 ; i < S.size(); i++)
    {
        SReal d = InertiaAlign::distance(S[i], T);
        if (d>maxST) maxST = d;
    }

    SReal maxTS = 0.0;
    for (unsigned int i = 0 ; i < T.size(); i++)
    {
        SReal d = InertiaAlign::distance(T[i], S);
        if (d>maxTS) maxTS = d;
    }

    if (maxTS > maxST)
        return maxST/S.size();
    else
        return maxTS/S.size();

}

Matrix4 InertiaAlign::inverseTransform(Matrix4 transformToInvert)
{
    Matrix3 rotationToInvert;
    Matrix3 rotationInverted;
    Vector3 Translation;
    Matrix4 transformInverted;

    sout << "Before inversion"<< sendl;
    for(unsigned int i=0;i<4;i++)
    {
        for(unsigned int j=0;j<4;j++)
        {
            sout << transformToInvert(i,j)<< " ";
        }
        sout << sendl;
    }
    for(unsigned int i=0;i<3;i++)
    {
        for(unsigned int j=0;j<3;j++)
        {
            rotationToInvert(i,j) = transformToInvert(i,j);
        }
        Translation(i)=transformToInvert(i,3);
    }

    bool bS = invertMatrix(rotationInverted,rotationToInvert);

    sout << "Translation = " << Translation;
    if(!bS)
    {
        sout <<"Error : Source transformation matrix is not invertible"<<sendl;
    }
    else //Compute R-1 * t
    {
        Translation = (-1*rotationInverted * Translation);
    }
    sout << "Translation = " << Translation;
    for(unsigned int i=0;i<3;i++)
    {
        for(unsigned int j=0;j<3;j++)
        {
            transformInverted(i,j) = rotationInverted(i,j);
        }
        transformInverted(i,3)= Translation(i);
    }
    transformInverted(3,3)=1;

    sout << "After inversion"<< sendl;
    for(unsigned int i=0;i<4;i++)
    {
        for(unsigned int j=0;j<4;j++)
        {
            sout << transformInverted(i,j)<< " ";
        }
        sout << sendl;
    }
    return transformInverted;

}

SReal InertiaAlign::abs(SReal a)
{
    return sqrt(a*a);
}



} // namespace component

} // namespace sofa
