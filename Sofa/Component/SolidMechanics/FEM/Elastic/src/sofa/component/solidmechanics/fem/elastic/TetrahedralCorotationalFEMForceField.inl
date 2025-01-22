/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/solidmechanics/fem/elastic/TetrahedralCorotationalFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.inl>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/decompose.h>
#include <sofa/core/topology/TopologyData.inl>


namespace sofa::component::solidmechanics::fem::elastic
{

template< class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::createTetrahedronInformation(Index tetrahedronIndex, TetrahedronInformation& tInfo,
        const core::topology::BaseMeshTopology::Tetrahedron& tetra,
        const sofa::type::vector<Index>& ancestors,
        const sofa::type::vector<SReal>& coefs)
{
    SOFA_UNUSED(tInfo);
    SOFA_UNUSED(tetra);
    SOFA_UNUSED(ancestors);
    SOFA_UNUSED(coefs);

    const core::topology::BaseMeshTopology::Tetrahedron t=this->l_topology->getTetrahedron(tetrahedronIndex);
    Index a = t[0];
    Index b = t[1];
    Index c = t[2];
    Index d = t[3];

    switch(method)
    {
    case SMALL :
        computeMaterialStiffness(tetrahedronIndex,a,b,c,d);
        initSmall(tetrahedronIndex,a,b,c,d);
        break;
    case LARGE :
        computeMaterialStiffness(tetrahedronIndex,a,b,c,d);
        initLarge(tetrahedronIndex,a,b,c,d);

        break;
    case POLAR :
        computeMaterialStiffness(tetrahedronIndex,a,b,c,d);
        initPolar(tetrahedronIndex,a,b,c,d);
        break;
    }
}

template< class DataTypes>
TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedralCorotationalFEMForceField()
    : d_tetrahedronInfo(initData(&d_tetrahedronInfo, "tetrahedronInfo", "Internal tetrahedron data"))
    , d_method(initData(&d_method, std::string("large"), "method", "\"small\", \"large\" (by QR) or \"polar\" displacements"))
    , d_localStiffnessFactor(core::objectmodel::BaseObject::initData(&d_localStiffnessFactor, "localStiffnessFactor", "Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]"))
    , d_updateStiffnessMatrix(core::objectmodel::BaseObject::initData(&d_updateStiffnessMatrix, false, "updateStiffnessMatrix", ""))
    , d_assembling(core::objectmodel::BaseObject::initData(&d_assembling, false, "computeGlobalMatrix", ""))
    , d_drawing(initData(&d_drawing, true, "drawing", " draw the forcefield if true"))
    , d_drawColor1(initData(&d_drawColor1, sofa::type::RGBAColor(0.0f, 0.0f, 1.0f, 1.0f), "drawColor1", " draw color for faces 1"))
    , d_drawColor2(initData(&d_drawColor2, sofa::type::RGBAColor(0.0f, 0.5f, 1.0f, 1.0f), "drawColor2", " draw color for faces 2"))
    , d_drawColor3(initData(&d_drawColor3, sofa::type::RGBAColor(0.0f, 1.0f, 1.0f, 1.0f), "drawColor3", " draw color for faces 3"))
    , d_drawColor4(initData(&d_drawColor4, sofa::type::RGBAColor(0.5f, 1.0f, 1.0f, 1.0f), "drawColor4", " draw color for faces 4"))
    , d_computeVonMisesStress(initData(&d_computeVonMisesStress, false, "computeVonMisesStress", "compute and display von Mises stress: 0: no computations, 1: using corotational strain, 2: using full Green strain. Set listening=1"))
    , d_vonMisesPerElement(initData(&d_vonMisesPerElement, "vonMisesPerElement", "von Mises Stress per element"))
    , d_vonMisesPerNode(initData(&d_vonMisesPerNode, "vonMisesPerNode", "von Mises Stress per node"))
    , m_vonMisesColorMap(nullptr)
{
    this->addAlias(&d_assembling, "assembling");

    tetrahedronInfo.setOriginalData(&d_tetrahedronInfo);
    f_method.setOriginalData(&d_method);
    _localStiffnessFactor.setOriginalData(&d_localStiffnessFactor);
    _updateStiffnessMatrix.setOriginalData(&d_updateStiffnessMatrix);
    _assembling.setOriginalData(&d_assembling);
    f_drawing.setOriginalData(&d_drawing);
    drawColor1.setOriginalData(&d_drawColor1);
    drawColor2.setOriginalData(&d_drawColor2);
    drawColor3.setOriginalData(&d_drawColor3);
    drawColor4.setOriginalData(&d_drawColor4);


}


template <class DataTypes>
TetrahedralCorotationalFEMForceField<DataTypes>::~TetrahedralCorotationalFEMForceField()
{
    if (m_vonMisesColorMap != nullptr)
        delete m_vonMisesColorMap;
}

template <class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::init()
{
    BaseLinearElasticityFEMForceField<DataTypes>::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    if (this->l_topology->getNbTetrahedra() == 0)
    {
        msg_warning() << "No tetrahedra found in linked Topology.";
    }

    reinit(); // compute per-element stiffness matrices and other precomputed values

    if (m_vonMisesColorMap == nullptr)
    {
        m_vonMisesColorMap = new helper::ColorMap(256, std::string("Blue to Red"));
    }
}


template <class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::reinit()
{

    if (d_method.getValue() == "small")
        this->setMethod(SMALL);
    else if (d_method.getValue() == "polar")
        this->setMethod(POLAR);
    else this->setMethod(LARGE);

    // Need to initialize the _stiffnesses vector before using it
    const std::size_t sizeMO=this->mstate->getSize();
    if(d_assembling.getValue())
    {
        _stiffnesses.resize( sizeMO * 3 );
    }

    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    tetrahedronInf.resize(this->l_topology->getNbTetrahedra());

    for (std::size_t i=0; i<this->l_topology->getNbTetrahedra(); ++i)
    {
        createTetrahedronInformation(i, tetrahedronInf[i],
                this->l_topology->getTetrahedron(i),  (const std::vector< Index > )0,
                (const std::vector< SReal >)0);
    }

    d_tetrahedronInfo.createTopologyHandler(this->l_topology);
    d_tetrahedronInfo.setCreationCallback([this](Index tetrahedronIndex, TetrahedronInformation& tInfo,
                                                 const core::topology::BaseMeshTopology::Tetrahedron& tetra,
                                                 const sofa::type::vector<Index>& ancestors,
                                                 const sofa::type::vector<SReal>& coefs)
    {
        createTetrahedronInformation(tetrahedronIndex, tInfo, tetra, ancestors, coefs);
    });

    d_tetrahedronInfo.endEdit();
}


template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& p = d_x.getValue();

    switch(method)
    {
    case SMALL :
    {
        for(Size i = 0 ; i<this->l_topology->getNbTetrahedra(); ++i)
        {
            accumulateForceSmall( f, p, i );
        }
        break;
    }
    case LARGE :
    {
        for(Size i = 0 ; i<this->l_topology->getNbTetrahedra(); ++i)
        {
            accumulateForceLarge( f, p, i );
        }
        break;
    }
    case POLAR :
    {
        for(Size i = 0 ; i<this->l_topology->getNbTetrahedra(); ++i)
        {
            accumulateForcePolar( f, p, i );
        }
        break;
    }
    }
    d_f.endEdit();

    if (d_computeVonMisesStress.getValue())
        computeVonMisesStress();
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    const auto& tetraArray = this->l_topology->getTetrahedra();

    switch(method)
    {
    case SMALL :
    {
        for(Size tetraId = 0; tetraId<tetraArray.size(); ++tetraId)
        {
            const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetraArray[tetraId];
            applyStiffnessSmall( df, dx, tetraId, tetra[0], tetra[1], tetra[2], tetra[3], kFactor );
        }
        break;
    }
    case LARGE :
    {
        for(Size tetraId = 0; tetraId<tetraArray.size(); ++tetraId)
        {
            const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetraArray[tetraId];
            applyStiffnessLarge( df, dx, tetraId, tetra[0], tetra[1], tetra[2], tetra[3], kFactor );
        }
        break;
    }
    case POLAR :
    {
        for(Size tetraId = 0; tetraId<tetraArray.size(); ++tetraId)
        {
            const core::topology::BaseMeshTopology::Tetrahedron& tetra = tetraArray[tetraId];
            applyStiffnessPolar( df, dx, tetraId, tetra[0], tetra[1], tetra[2], tetra[3], kFactor );
        }
        break;
    }
    }

    d_df.endEdit();
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeStrainDisplacement( StrainDisplacementTransposed &J, Coord a, Coord b, Coord c, Coord d )
{
    // shape functions matrix
    type::Mat<2, 3, Real> M;

    M[0][0] = b[1];
    M[0][1] = c[1];
    M[0][2] = d[1];
    M[1][0] = b[2];
    M[1][1] = c[2];
    M[1][2] = d[2];
    J[0][0] = J[1][3] = J[2][5]   = - peudo_determinant_for_coef( M );
    M[0][0] = b[0];
    M[0][1] = c[0];
    M[0][2] = d[0];
    J[0][3] = J[1][1] = J[2][4]   = peudo_determinant_for_coef( M );
    M[1][0] = b[1];
    M[1][1] = c[1];
    M[1][2] = d[1];
    J[0][5] = J[1][4] = J[2][2]   = - peudo_determinant_for_coef( M );

    M[0][0] = c[1];
    M[0][1] = d[1];
    M[0][2] = a[1];
    M[1][0] = c[2];
    M[1][1] = d[2];
    M[1][2] = a[2];
    J[3][0] = J[4][3] = J[5][5]   = peudo_determinant_for_coef( M );
    M[0][0] = c[0];
    M[0][1] = d[0];
    M[0][2] = a[0];
    J[3][3] = J[4][1] = J[5][4]   = - peudo_determinant_for_coef( M );
    M[1][0] = c[1];
    M[1][1] = d[1];
    M[1][2] = a[1];
    J[3][5] = J[4][4] = J[5][2]   = peudo_determinant_for_coef( M );

    M[0][0] = d[1];
    M[0][1] = a[1];
    M[0][2] = b[1];
    M[1][0] = d[2];
    M[1][1] = a[2];
    M[1][2] = b[2];
    J[6][0] = J[7][3] = J[8][5]   = - peudo_determinant_for_coef( M );
    M[0][0] = d[0];
    M[0][1] = a[0];
    M[0][2] = b[0];
    J[6][3] = J[7][1] = J[8][4]   = peudo_determinant_for_coef( M );
    M[1][0] = d[1];
    M[1][1] = a[1];
    M[1][2] = b[1];
    J[6][5] = J[7][4] = J[8][2]   = - peudo_determinant_for_coef( M );

    M[0][0] = a[1];
    M[0][1] = b[1];
    M[0][2] = c[1];
    M[1][0] = a[2];
    M[1][1] = b[2];
    M[1][2] = c[2];
    J[9][0] = J[10][3] = J[11][5]   = peudo_determinant_for_coef( M );
    M[0][0] = a[0];
    M[0][1] = b[0];
    M[0][2] = c[0];
    J[9][3] = J[10][1] = J[11][4]   = - peudo_determinant_for_coef( M );
    M[1][0] = a[1];
    M[1][1] = b[1];
    M[1][2] = c[1];
    J[9][5] = J[10][4] = J[11][2]   = peudo_determinant_for_coef( M );


    // 0
    J[0][1] = J[0][2] = J[0][4] = J[1][0] =  J[1][2] =  J[1][5] =  J[2][0] =  J[2][1] =  J[2][3]  = 0;
    J[3][1] = J[3][2] = J[3][4] = J[4][0] =  J[4][2] =  J[4][5] =  J[5][0] =  J[5][1] =  J[5][3]  = 0;
    J[6][1] = J[6][2] = J[6][4] = J[7][0] =  J[7][2] =  J[7][5] =  J[8][0] =  J[8][1] =  J[8][3]  = 0;
    J[9][1] = J[9][2] = J[9][4] = J[10][0] = J[10][2] = J[10][5] = J[11][0] = J[11][1] = J[11][3] = 0;

    //m_deq( J, 1.2 ); //hack for stability ??
}

template<class DataTypes>
typename TetrahedralCorotationalFEMForceField<DataTypes>::Real TetrahedralCorotationalFEMForceField<DataTypes>::peudo_determinant_for_coef ( const type::Mat<2, 3, Real>&  M )
{
    return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacementTransposed &J, const Transformation& Rot )
{
    type::MatNoInit<6, 12, Real> Jt;
    Jt.transpose(J);

    type::MatNoInit<12, 12, Real> JKJt;
    JKJt = J * K * Jt;

    type::MatNoInit<12, 12, Real> RR, RRt;
    RR.clear();
    RRt.clear();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            RR[i][j] = RR[i + 3][j + 3] = RR[i + 6][j + 6] = RR[i + 9][j + 9] = Rot[i][j];
            RRt[i][j] = RRt[i + 3][j + 3] = RRt[i + 6][j + 6] = RRt[i + 9][j + 9] = Rot[j][i];
        }
    }

    S = RR * JKJt;
    SR = S * RRt;
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d)
{

    const VecReal& localStiffnessFactor = d_localStiffnessFactor.getValue();

    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    computeMaterialStiffness(i, tetrahedronInf[i].materialMatrix, a, b, c, d, (localStiffnessFactor.empty() ? 1.0f : localStiffnessFactor[i*localStiffnessFactor.size()/this->l_topology->getNbTetrahedra()]));

    d_tetrahedronInfo.endEdit();
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeMaterialStiffness(int tetrahedronId, MaterialStiffness& materialMatrix, Index&a, Index&b, Index&c, Index&d, SReal localStiffnessFactor)
{
    const Real youngModulusElement = this->getYoungModulusInElement(tetrahedronId);
    const Real youngModulus = youngModulusElement*(Real)localStiffnessFactor;
    const Real poissonRatio = this->getPoissonRatioInElement(tetrahedronId);

    materialMatrix[0][0] = materialMatrix[1][1] = materialMatrix[2][2] = 1;
    materialMatrix[0][1] = materialMatrix[0][2] = materialMatrix[1][0] =
            materialMatrix[1][2] = materialMatrix[2][0] = materialMatrix[2][1] = poissonRatio/(1-poissonRatio);
    materialMatrix[0][3] = materialMatrix[0][4] = materialMatrix[0][5] = 0;
    materialMatrix[1][3] = materialMatrix[1][4] = materialMatrix[1][5] = 0;
    materialMatrix[2][3] = materialMatrix[2][4] = materialMatrix[2][5] = 0;
    materialMatrix[3][0] = materialMatrix[3][1] = materialMatrix[3][2] =
            materialMatrix[3][4] = materialMatrix[3][5] = 0;
    materialMatrix[4][0] = materialMatrix[4][1] = materialMatrix[4][2] =
            materialMatrix[4][3] = materialMatrix[4][5] = 0;
    materialMatrix[5][0] = materialMatrix[5][1] = materialMatrix[5][2] =
            materialMatrix[5][3] = materialMatrix[5][4] = 0;
    materialMatrix[3][3] = materialMatrix[4][4] = materialMatrix[5][5] =
            (1-2*poissonRatio)/(2*(1-poissonRatio));
    materialMatrix *= (youngModulus*(1-poissonRatio))/((1+poissonRatio)*(1-2*poissonRatio));

    // divide by 36 times volumes of the element
    const VecCoord X0=this->mstate->read(core::vec_id::read_access::restPosition)->getValue();

    Coord A = (X0)[b] - (X0)[a];
    Coord B = (X0)[c] - (X0)[a];
    Coord C = (X0)[d] - (X0)[a];
    Coord AB = cross(A, B);
    Real volumes6 = fabs( dot( AB, C ) );
    if (volumes6<0)
    {
        msg_error() << "Negative volume for tetra " << a << ',' << b << ',' << c << ',' << d << "> = " << volumes6 / 6;
    }
    //	materialMatrix  /= (volumes6);//*6 christian
    // @TODO: in TetrahedronFEMForceField, the stiffness matrix is divided by 6 compared to the code in TetrahedralCorotationalFEMForceField. Check which is the correct one...
    // FF:  there is normally  a factor 1/6v in the strain-displacement matrix. Times transpose makes 1/36v². Integrating across the volume multiplies by v, so the factor is 1/36v
    materialMatrix  /= (volumes6*6);

}

template<class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacementTransposed &J )
{
    // Unit of K = unit of youngModulus / unit of volume = Pa / m^3 = kg m^-4 s^-2
    // Unit of J = m^2
    // Unit of JKJt =  kg s^-2
    // Unit of displacement = m
    // Unit of force = kg m s^-2

    /* We have these zeros
                                  K[0][3]   K[0][4]   K[0][5]
                                  K[1][3]   K[1][4]   K[1][5]
                                  K[2][3]   K[2][4]   K[2][5]
    K[3][0]   K[3][1]   K[3][2]             K[3][4]   K[3][5]
    K[4][0]   K[4][1]   K[4][2]   K[4][3]             K[4][5]
    K[5][0]   K[5][1]   K[5][2]   K[5][3]   K[5][4]


              J[0][1]   J[0][2]             J[0][4]
    J[1][0]             J[1][2]                       J[1][5]
    J[2][0]   J[2][1]             J[2][3]
              J[3][1]   J[3][2]             J[3][4]
    J[4][0]             J[4][2]                       J[4][5]
    J[5][0]   J[5][1]             J[5][3]
              J[6][1]   J[6][2]             J[6][4]
    J[7][0]             J[7][2]                       J[7][5]
    J[8][0]   J[8][1]             J[8][3]
              J[9][1]   J[9][2]             J[9][4]
    J[10][0]            J[10][2]                      J[10][5]
    J[11][0]  J[11][1]            J[11][3]
    */

    type::VecNoInit<6,Real> JtD;
    JtD[0] =   J[ 0][0]*Depl[ 0]+/*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
            J[ 3][0]*Depl[ 3]+/*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
            J[ 6][0]*Depl[ 6]+/*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
            J[ 9][0]*Depl[ 9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
    JtD[1] = /*J[ 0][1]*Depl[ 0]+*/J[ 1][1]*Depl[ 1]+/*J[ 2][1]*Depl[ 2]+*/
            /*J[ 3][1]*Depl[ 3]+*/J[ 4][1]*Depl[ 4]+/*J[ 5][1]*Depl[ 5]+*/
            /*J[ 6][1]*Depl[ 6]+*/J[ 7][1]*Depl[ 7]+/*J[ 8][1]*Depl[ 8]+*/
            /*J[ 9][1]*Depl[ 9]+*/J[10][1]*Depl[10] /*J[11][1]*Depl[11]*/;
    JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/J[ 2][2]*Depl[ 2]+
            /*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/J[ 5][2]*Depl[ 5]+
            /*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/J[ 8][2]*Depl[ 8]+
            /*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/J[11][2]*Depl[11]  ;
    JtD[3] =   J[ 0][3]*Depl[ 0]+  J[ 1][3]*Depl[ 1]+/*J[ 2][3]*Depl[ 2]+*/
            J[ 3][3]*Depl[ 3]+  J[ 4][3]*Depl[ 4]+/*J[ 5][3]*Depl[ 5]+*/
            J[ 6][3]*Depl[ 6]+  J[ 7][3]*Depl[ 7]+/*J[ 8][3]*Depl[ 8]+*/
            J[ 9][3]*Depl[ 9]+  J[10][3]*Depl[10] /*J[11][3]*Depl[11]*/;
    JtD[4] = /*J[ 0][4]*Depl[ 0]+*/J[ 1][4]*Depl[ 1]+  J[ 2][4]*Depl[ 2]+
            /*J[ 3][4]*Depl[ 3]+*/J[ 4][4]*Depl[ 4]+  J[ 5][4]*Depl[ 5]+
            /*J[ 6][4]*Depl[ 6]+*/J[ 7][4]*Depl[ 7]+  J[ 8][4]*Depl[ 8]+
            /*J[ 9][4]*Depl[ 9]+*/J[10][4]*Depl[10]+  J[11][4]*Depl[11]  ;
    JtD[5] =   J[ 0][5]*Depl[ 0]+/*J[ 1][5]*Depl[ 1]*/ J[ 2][5]*Depl[ 2]+
            J[ 3][5]*Depl[ 3]+/*J[ 4][5]*Depl[ 4]*/ J[ 5][5]*Depl[ 5]+
            J[ 6][5]*Depl[ 6]+/*J[ 7][5]*Depl[ 7]*/ J[ 8][5]*Depl[ 8]+
            J[ 9][5]*Depl[ 9]+/*J[10][5]*Depl[10]*/ J[11][5]*Depl[11];

    type::VecNoInit<6,Real> KJtD;
    KJtD[0] =   K[0][0]*JtD[0]+  K[0][1]*JtD[1]+  K[0][2]*JtD[2]
            /*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
    KJtD[1] =   K[1][0]*JtD[0]+  K[1][1]*JtD[1]+  K[1][2]*JtD[2]
            /*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
    KJtD[2] =   K[2][0]*JtD[0]+  K[2][1]*JtD[1]+  K[2][2]*JtD[2]
            /*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
    KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
        K[3][3]*JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
    KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
        /*K[4][3]*JtD[3]+*/K[4][4]*JtD[4] /*K[4][5]*JtD[5]*/;
    KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
        /*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/K[5][5]*JtD[5]  ;

    F[ 0] =   J[ 0][0]*KJtD[0]+/*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
            J[ 0][3]*KJtD[3]+/*J[ 0][4]*KJtD[4]+*/J[ 0][5]*KJtD[5]  ;
    F[ 1] = /*J[ 1][0]*KJtD[0]+*/J[ 1][1]*KJtD[1]+/*J[ 1][2]*KJtD[2]+*/
            J[ 1][3]*KJtD[3]+  J[ 1][4]*KJtD[4] /*J[ 1][5]*KJtD[5]*/;
    F[ 2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/J[ 2][2]*KJtD[2]+
            /*J[ 2][3]*KJtD[3]+*/J[ 2][4]*KJtD[4]+  J[ 2][5]*KJtD[5]  ;
    F[ 3] =   J[ 3][0]*KJtD[0]+/*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
            J[ 3][3]*KJtD[3]+/*J[ 3][4]*KJtD[4]+*/J[ 3][5]*KJtD[5]  ;
    F[ 4] = /*J[ 4][0]*KJtD[0]+*/J[ 4][1]*KJtD[1]+/*J[ 4][2]*KJtD[2]+*/
            J[ 4][3]*KJtD[3]+  J[ 4][4]*KJtD[4] /*J[ 4][5]*KJtD[5]*/;
    F[ 5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/J[ 5][2]*KJtD[2]+
            /*J[ 5][3]*KJtD[3]+*/J[ 5][4]*KJtD[4]+  J[ 5][5]*KJtD[5]  ;
    F[ 6] =   J[ 6][0]*KJtD[0]+/*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
            J[ 6][3]*KJtD[3]+/*J[ 6][4]*KJtD[4]+*/J[ 6][5]*KJtD[5]  ;
    F[ 7] = /*J[ 7][0]*KJtD[0]+*/J[ 7][1]*KJtD[1]+/*J[ 7][2]*KJtD[2]+*/
            J[ 7][3]*KJtD[3]+  J[ 7][4]*KJtD[4] /*J[ 7][5]*KJtD[5]*/;
    F[ 8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/J[ 8][2]*KJtD[2]+
            /*J[ 8][3]*KJtD[3]+*/J[ 8][4]*KJtD[4]+  J[ 8][5]*KJtD[5]  ;
    F[ 9] =   J[ 9][0]*KJtD[0]+/*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
            J[ 9][3]*KJtD[3]+/*J[ 9][4]*KJtD[4]+*/J[ 9][5]*KJtD[5]  ;
    F[10] = /*J[10][0]*KJtD[0]+*/J[10][1]*KJtD[1]+/*J[10][2]*KJtD[2]+*/
            J[10][3]*KJtD[3]+  J[10][4]*KJtD[4] /*J[10][5]*KJtD[5]*/;
    F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/J[11][2]*KJtD[2]+
            /*J[11][3]*KJtD[3]+*/J[11][4]*KJtD[4]+  J[11][5]*KJtD[5]  ;
}

template<class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacementTransposed &J, SReal fact )
{

    // Unit of K = unit of youngModulus / unit of volume = Pa / m^3 = kg m^-4 s^-2
    // Unit of J = m^2
    // Unit of JKJt =  kg s^-2
    // Unit of displacement = m
    // Unit of force = kg m s^-2

    /* We have these zeros
                                  K[0][3]   K[0][4]   K[0][5]
                                  K[1][3]   K[1][4]   K[1][5]
                                  K[2][3]   K[2][4]   K[2][5]
    K[3][0]   K[3][1]   K[3][2]             K[3][4]   K[3][5]
    K[4][0]   K[4][1]   K[4][2]   K[4][3]             K[4][5]
    K[5][0]   K[5][1]   K[5][2]   K[5][3]   K[5][4]


              J[0][1]   J[0][2]             J[0][4]
    J[1][0]             J[1][2]                       J[1][5]
    J[2][0]   J[2][1]             J[2][3]
              J[3][1]   J[3][2]             J[3][4]
    J[4][0]             J[4][2]                       J[4][5]
    J[5][0]   J[5][1]             J[5][3]
              J[6][1]   J[6][2]             J[6][4]
    J[7][0]             J[7][2]                       J[7][5]
    J[8][0]   J[8][1]             J[8][3]
              J[9][1]   J[9][2]             J[9][4]
    J[10][0]            J[10][2]                      J[10][5]
    J[11][0]  J[11][1]            J[11][3]
    */

    type::VecNoInit<6,Real> JtD;
    JtD[0] =   J[ 0][0]*Depl[ 0]+/*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
            J[ 3][0]*Depl[ 3]+/*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
            J[ 6][0]*Depl[ 6]+/*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
            J[ 9][0]*Depl[ 9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
    JtD[1] = /*J[ 0][1]*Depl[ 0]+*/J[ 1][1]*Depl[ 1]+/*J[ 2][1]*Depl[ 2]+*/
            /*J[ 3][1]*Depl[ 3]+*/J[ 4][1]*Depl[ 4]+/*J[ 5][1]*Depl[ 5]+*/
            /*J[ 6][1]*Depl[ 6]+*/J[ 7][1]*Depl[ 7]+/*J[ 8][1]*Depl[ 8]+*/
            /*J[ 9][1]*Depl[ 9]+*/J[10][1]*Depl[10] /*J[11][1]*Depl[11]*/;
    JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/J[ 2][2]*Depl[ 2]+
            /*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/J[ 5][2]*Depl[ 5]+
            /*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/J[ 8][2]*Depl[ 8]+
            /*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/J[11][2]*Depl[11]  ;
    JtD[3] =   J[ 0][3]*Depl[ 0]+  J[ 1][3]*Depl[ 1]+/*J[ 2][3]*Depl[ 2]+*/
            J[ 3][3]*Depl[ 3]+  J[ 4][3]*Depl[ 4]+/*J[ 5][3]*Depl[ 5]+*/
            J[ 6][3]*Depl[ 6]+  J[ 7][3]*Depl[ 7]+/*J[ 8][3]*Depl[ 8]+*/
            J[ 9][3]*Depl[ 9]+  J[10][3]*Depl[10] /*J[11][3]*Depl[11]*/;
    JtD[4] = /*J[ 0][4]*Depl[ 0]+*/J[ 1][4]*Depl[ 1]+  J[ 2][4]*Depl[ 2]+
            /*J[ 3][4]*Depl[ 3]+*/J[ 4][4]*Depl[ 4]+  J[ 5][4]*Depl[ 5]+
            /*J[ 6][4]*Depl[ 6]+*/J[ 7][4]*Depl[ 7]+  J[ 8][4]*Depl[ 8]+
            /*J[ 9][4]*Depl[ 9]+*/J[10][4]*Depl[10]+  J[11][4]*Depl[11]  ;
    JtD[5] =   J[ 0][5]*Depl[ 0]+/*J[ 1][5]*Depl[ 1]*/ J[ 2][5]*Depl[ 2]+
            J[ 3][5]*Depl[ 3]+/*J[ 4][5]*Depl[ 4]*/ J[ 5][5]*Depl[ 5]+
            J[ 6][5]*Depl[ 6]+/*J[ 7][5]*Depl[ 7]*/ J[ 8][5]*Depl[ 8]+
            J[ 9][5]*Depl[ 9]+/*J[10][5]*Depl[10]*/ J[11][5]*Depl[11];


    type::VecNoInit<6,Real> KJtD;
    KJtD[0] =   K[0][0]*JtD[0]+  K[0][1]*JtD[1]+  K[0][2]*JtD[2]
            /*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
    KJtD[1] =   K[1][0]*JtD[0]+  K[1][1]*JtD[1]+  K[1][2]*JtD[2]
            /*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
    KJtD[2] =   K[2][0]*JtD[0]+  K[2][1]*JtD[1]+  K[2][2]*JtD[2]
            /*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
    KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
        K[3][3]*JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
    KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
        /*K[4][3]*JtD[3]+*/K[4][4]*JtD[4] /*K[4][5]*JtD[5]*/;
    KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
        /*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/K[5][5]*JtD[5]  ;

    KJtD *= fact;

    F[ 0] =   J[ 0][0]*KJtD[0]+/*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
            J[ 0][3]*KJtD[3]+/*J[ 0][4]*KJtD[4]+*/J[ 0][5]*KJtD[5]  ;
    F[ 1] = /*J[ 1][0]*KJtD[0]+*/J[ 1][1]*KJtD[1]+/*J[ 1][2]*KJtD[2]+*/
            J[ 1][3]*KJtD[3]+  J[ 1][4]*KJtD[4] /*J[ 1][5]*KJtD[5]*/;
    F[ 2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/J[ 2][2]*KJtD[2]+
            /*J[ 2][3]*KJtD[3]+*/J[ 2][4]*KJtD[4]+  J[ 2][5]*KJtD[5]  ;
    F[ 3] =   J[ 3][0]*KJtD[0]+/*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
            J[ 3][3]*KJtD[3]+/*J[ 3][4]*KJtD[4]+*/J[ 3][5]*KJtD[5]  ;
    F[ 4] = /*J[ 4][0]*KJtD[0]+*/J[ 4][1]*KJtD[1]+/*J[ 4][2]*KJtD[2]+*/
            J[ 4][3]*KJtD[3]+  J[ 4][4]*KJtD[4] /*J[ 4][5]*KJtD[5]*/;
    F[ 5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/J[ 5][2]*KJtD[2]+
            /*J[ 5][3]*KJtD[3]+*/J[ 5][4]*KJtD[4]+  J[ 5][5]*KJtD[5]  ;
    F[ 6] =   J[ 6][0]*KJtD[0]+/*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
            J[ 6][3]*KJtD[3]+/*J[ 6][4]*KJtD[4]+*/J[ 6][5]*KJtD[5]  ;
    F[ 7] = /*J[ 7][0]*KJtD[0]+*/J[ 7][1]*KJtD[1]+/*J[ 7][2]*KJtD[2]+*/
            J[ 7][3]*KJtD[3]+  J[ 7][4]*KJtD[4] /*J[ 7][5]*KJtD[5]*/;
    F[ 8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/J[ 8][2]*KJtD[2]+
            /*J[ 8][3]*KJtD[3]+*/J[ 8][4]*KJtD[4]+  J[ 8][5]*KJtD[5]  ;
    F[ 9] =   J[ 9][0]*KJtD[0]+/*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
            J[ 9][3]*KJtD[3]+/*J[ 9][4]*KJtD[4]+*/J[ 9][5]*KJtD[5]  ;
    F[10] = /*J[10][0]*KJtD[0]+*/J[10][1]*KJtD[1]+/*J[10][2]*KJtD[2]+*/
            J[10][3]*KJtD[3]+  J[10][4]*KJtD[4] /*J[10][5]*KJtD[5]*/;
    F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/J[11][2]*KJtD[2]+
            /*J[11][3]*KJtD[3]+*/J[11][4]*KJtD[4]+  J[11][5]*KJtD[5]  ;
}

//////////////////////////////////////////////////////////////////////
////////////////////  small displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::initSmall(int i, Index&a, Index&b, Index&c, Index&d)
{
    const VecCoord X0=this->mstate->read(core::vec_id::read_access::restPosition)->getValue();

    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    computeStrainDisplacement(tetrahedronInf[i].strainDisplacementTransposedMatrix, (X0)[a], (X0)[b], (X0)[c], (X0)[d] );

    d_tetrahedronInfo.endEdit();

    this->printStiffnessMatrix(i);////////////////////////////////////////////////////////////////
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall( Vector& f, const Vector & p,Index elementIndex )
{

    const core::topology::BaseMeshTopology::Tetrahedron t=this->l_topology->getTetrahedron(elementIndex);
    const VecCoord& X0=this->mstate->read(core::vec_id::read_access::restPosition)->getValue();

    const auto& [a, b, c, d] = t.array();

    // displacements
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] =  (X0)[b][0] - (X0)[a][0] - p[b][0]+p[a][0];
    D[4] =  (X0)[b][1] - (X0)[a][1] - p[b][1]+p[a][1];
    D[5] =  (X0)[b][2] - (X0)[a][2] - p[b][2]+p[a][2];
    D[6] =  (X0)[c][0] - (X0)[a][0] - p[c][0]+p[a][0];
    D[7] =  (X0)[c][1] - (X0)[a][1] - p[c][1]+p[a][1];
    D[8] =  (X0)[c][2] - (X0)[a][2] - p[c][2]+p[a][2];
    D[9] =  (X0)[d][0] - (X0)[a][0] - p[d][0]+p[a][0];
    D[10] = (X0)[d][1] - (X0)[a][1] - p[d][1]+p[a][1];
    D[11] = (X0)[d][2] - (X0)[a][2] - p[d][2]+p[a][2];

    // compute force on element
    Displacement F;

    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = d_tetrahedronInfo.getValue();

    if(!d_assembling.getValue())
    {
        computeForce( F, D,tetrahedronInf[elementIndex].materialMatrix,tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix );
    }
    else
    {
        Transformation Rot;
        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;


        StiffnessMatrix JKJt,tmp;
        computeStiffnessMatrix(JKJt,tmp,tetrahedronInf[elementIndex].materialMatrix,tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix,Rot);


        //erase the stiffness matrix at each time step
        if(elementIndex==0)
        {
            for(Size i=0; i<_stiffnesses.size(); ++i)
            {
                _stiffnesses[i].resize(0);
            }
        }

        for(int i=0; i<12; ++i)
        {
            int row = t[i/3]*3+i%3;

            for(int j=0; j<12; ++j)
            {
                if(JKJt[i][j]!=0)
                {

                    int col = t[j/3]*3+j%3;
                    // search if the vertex is already take into account by another element
                    typename CompressedValue::iterator result = _stiffnesses[row].end();
                    for(typename CompressedValue::iterator it=_stiffnesses[row].begin(); it!=_stiffnesses[row].end()&&result==_stiffnesses[row].end(); ++it)
                    {
                        if( (*it).first == col )
                            result = it;
                    }

                    if( result==_stiffnesses[row].end() )
                        _stiffnesses[row].push_back( Col_Value(col,JKJt[i][j] )  );
                    else
                        (*result).second += JKJt[i][j];
                }
            }
        }

        F = JKJt * D;
    }

    f[a] += Deriv( F[0], F[1], F[2] );
    f[b] += Deriv( F[3], F[4], F[5] );
    f[c] += Deriv( F[6], F[7], F[8] );
    f[d] += Deriv( F[9], F[10], F[11] );
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessSmall( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact )
{
    Displacement X;

    X[0] = x[a][0];
    X[1] = x[a][1];
    X[2] = x[a][2];

    X[3] = x[b][0];
    X[4] = x[b][1];
    X[5] = x[b][2];

    X[6] = x[c][0];
    X[7] = x[c][1];
    X[8] = x[c][2];

    X[9] = x[d][0];
    X[10] = x[d][1];
    X[11] = x[d][2];

    Displacement F;

    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = d_tetrahedronInfo.getValue();

    computeForce( F, X,tetrahedronInf[i].materialMatrix,tetrahedronInf[i].strainDisplacementTransposedMatrix, fact);

    f[a] += Deriv( -F[0], -F[1],  -F[2] );
    f[b] += Deriv( -F[3], -F[4],  -F[5] );
    f[c] += Deriv( -F[6], -F[7],  -F[8] );
    f[d] += Deriv( -F[9], -F[10], -F[11] );
}

//////////////////////////////////////////////////////////////////////
////////////////////  large displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    const Coord edgex = (p[b]-p[a]).normalized();
          Coord edgey = p[c]-p[a];
    const Coord edgez = cross( edgex, edgey ).normalized();
                edgey = cross( edgez, edgex ); //edgey is unit vector because edgez and edgex are orthogonal unit vectors

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];
}

template <class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::getElementRotation(Transformation& R, Index elementIdx)
{
    type::vector<TetrahedronInformation>& tetraInf = *(d_tetrahedronInfo.beginEdit());
    TetrahedronInformation *tinfo = &tetraInf[elementIdx];
    Transformation r01,r21;
    r01=tinfo->initialTransformation;
    r21=tinfo->rotation*r01;
    R=r21;
}

template <class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::getRotation(Transformation& R, Index nodeIdx)
{
    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetraInf = d_tetrahedronInfo.getValue();
    const int numNeiTetra=this->l_topology->getTetrahedraAroundVertex(nodeIdx).size();
    Transformation r;
    r.clear();

    for(int i=0; i<numNeiTetra; i++)
    {
        int tetraIdx=this->l_topology->getTetrahedraAroundVertex(nodeIdx)[i];
        const TetrahedronInformation *tinfo = &tetraInf[tetraIdx];
        Transformation r01,r21;
        r01=tinfo->initialTransformation;
        r21=tinfo->rotation*r01;
        r+=r21;
    }
    R=r*(1.0f/numNeiTetra);

    //orthogonalization
    Coord ex,ey,ez;
    for(int i=0; i<3; i++)
    {
        ex[i]=R[0][i];
        ey[i]=R[1][i];
    }
    ex.normalize();
    ey.normalize();

    ez=cross(ex,ey);
    ez.normalize();

    ey=cross(ez,ex);
    ey.normalize();

    for(int i=0; i<3; i++)
    {
        R[0][i]=ex[i];
        R[1][i]=ey[i];
        R[2][i]=ez[i];
    }
}

template <class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::getElementStiffnessMatrix(Real* stiffness, Index elementIndex)
{
    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetraInf = d_tetrahedronInfo.getValue();
    Transformation Rot;
    StiffnessMatrix JKJt,tmp;
    Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
    Rot[0][1]=Rot[0][2]=0;
    Rot[1][0]=Rot[1][2]=0;
    Rot[2][0]=Rot[2][1]=0;
    computeStiffnessMatrix(JKJt,tmp,tetraInf[elementIndex].materialMatrix, tetraInf[elementIndex].strainDisplacementTransposedMatrix,Rot);
    for(int i=0; i<12; i++)
    {
        for(int j=0; j<12; j++)
            stiffness[i*12+j]=JKJt(i,j);
    }
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::initLarge(int i, Index&a, Index&b, Index&c, Index&d)
{
    // Rotation matrix (initial Tetrahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    const VecCoord& X0=this->mstate->read(core::vec_id::read_access::restPosition)->getValue();

    Transformation R_0_1;
    computeRotationLarge( R_0_1, (X0), a, b, c);

    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    tetrahedronInf[i].rotatedInitialElements[0] = R_0_1*(X0)[a];
    tetrahedronInf[i].rotatedInitialElements[1] = R_0_1*(X0)[b];
    tetrahedronInf[i].rotatedInitialElements[2] = R_0_1*(X0)[c];
    tetrahedronInf[i].rotatedInitialElements[3] = R_0_1*(X0)[d];

    tetrahedronInf[i].initialTransformation = R_0_1;

    tetrahedronInf[i].rotatedInitialElements[1] -= tetrahedronInf[i].rotatedInitialElements[0];
    tetrahedronInf[i].rotatedInitialElements[2] -= tetrahedronInf[i].rotatedInitialElements[0];
    tetrahedronInf[i].rotatedInitialElements[3] -= tetrahedronInf[i].rotatedInitialElements[0];
    tetrahedronInf[i].rotatedInitialElements[0] = Coord(0,0,0);

    computeStrainDisplacement( tetrahedronInf[i].strainDisplacementTransposedMatrix,tetrahedronInf[i].rotatedInitialElements[0], tetrahedronInf[i].rotatedInitialElements[1],tetrahedronInf[i].rotatedInitialElements[2],tetrahedronInf[i].rotatedInitialElements[3] );


    type::Mat<4, 4, Real> matVert;
    const core::topology::BaseMeshTopology::Tetrahedron tetra = { a, b, c, d };
    for (Index k = 0; k < 4; k++) {
        Index ix = tetra[k];
        matVert[k][0] = 1.0;
        for (Index l = 1; l < 4; l++)
            matVert[k][l] = X0[ix][l - 1];
    }

    [[maybe_unused]] const bool canInvert = type::invertMatrix(tetrahedronInf[i].elemShapeFun, matVert);

    tetrahedronInfo.endEdit();
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceLarge( Vector& f, const Vector & p, Index elementIndex )
{
    const core::topology::BaseMeshTopology::Tetrahedron t=this->l_topology->getTetrahedron(elementIndex);

    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    // Rotation matrix (deformed and displaced Tetrahedron/world)
    Transformation R_0_2;
    computeRotationLarge( R_0_2, p, t[0],t[1],t[2]);
    tetrahedronInf[elementIndex].rotation.transpose(R_0_2);

    // positions of the deformed and displaced Tetrahedron in its frame
    type::fixed_array<Coord,4> deforme;
    for(int i=0; i<4; ++i)
        deforme[i] = R_0_2*p[t[i]];

    deforme[1][0] -= deforme[0][0];
    deforme[2][0] -= deforme[0][0];
    deforme[2][1] -= deforme[0][1];
    deforme[3] -= deforme[0];

    // displacement
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] = tetrahedronInf[elementIndex].rotatedInitialElements[1][0] - deforme[1][0];
    D[4] = 0;
    D[5] = 0;
    D[6] = tetrahedronInf[elementIndex].rotatedInitialElements[2][0] - deforme[2][0];
    D[7] = tetrahedronInf[elementIndex].rotatedInitialElements[2][1] - deforme[2][1];
    D[8] = 0;
    D[9] = tetrahedronInf[elementIndex].rotatedInitialElements[3][0] - deforme[3][0];
    D[10] = tetrahedronInf[elementIndex].rotatedInitialElements[3][1] - deforme[3][1];
    D[11] =tetrahedronInf[elementIndex].rotatedInitialElements[3][2] - deforme[3][2];

    Displacement F;
    if(d_updateStiffnessMatrix.getValue())
    {
        StrainDisplacementTransposed& J = tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix;
        J[0][0] = J[1][3] = J[2][5]   = ( - deforme[2][1]*deforme[3][2] );
        J[1][1] = J[0][3] = J[2][4]   = ( deforme[2][0]*deforme[3][2] - deforme[1][0]*deforme[3][2] );
        J[2][2] = J[0][5] = J[1][4]   = ( deforme[2][1]*deforme[3][0] - deforme[2][0]*deforme[3][1] + deforme[1][0]*deforme[3][1] - deforme[1][0]*deforme[2][1] );

        J[3][0] = J[4][3] = J[5][5]   = ( deforme[2][1]*deforme[3][2] );
        J[4][1] = J[3][3] = J[5][4]  = ( - deforme[2][0]*deforme[3][2] );
        J[5][2] = J[3][5] = J[4][4]   = ( - deforme[2][1]*deforme[3][0] + deforme[2][0]*deforme[3][1] );

        J[7][1] = J[6][3] = J[8][4]  = ( deforme[1][0]*deforme[3][2] );
        J[8][2] = J[6][5] = J[7][4]   = ( - deforme[1][0]*deforme[3][1] );

        J[11][2] = J[9][5] = J[10][4] = ( deforme[1][0]*deforme[2][1] );
    }

    if(!d_assembling.getValue())
    {
        // compute force on element
        computeForce( F, D, tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix);
        for(int i=0; i<12; i+=3)
            f[t[i/3]] += tetrahedronInf[elementIndex].rotation * Deriv( F[i], F[i+1],  F[i+2] );
    }
    else
    {
        tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix[6][0] = 0;
        tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix[9][0] = 0;
        tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix[10][1] = 0;

        StiffnessMatrix RJKJt, RJKJtRt;
        computeStiffnessMatrix(RJKJt,RJKJtRt,tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix,tetrahedronInf[elementIndex].rotation);

        //erase the stiffness matrix at each time step
        if(elementIndex==0)
        {
            for(Size i=0; i<_stiffnesses.size(); ++i)
            {
                _stiffnesses[i].resize(0);
            }
        }

        for(int i=0; i<12; ++i)
        {
            int row = t[i/3]*3+i%3;

            for(int j=0; j<12; ++j)
            {
                int col = t[j/3]*3+j%3;

                // search if the vertex is already take into account by another element
                typename CompressedValue::iterator result = _stiffnesses[row].end();
                for(typename CompressedValue::iterator it=_stiffnesses[row].begin(); it!=_stiffnesses[row].end()&&result==_stiffnesses[row].end(); ++it)
                {
                    if( (*it).first == col )
                    {
                        result = it;
                    }
                }

                if( result==_stiffnesses[row].end() )
                {
                    _stiffnesses[row].push_back( Col_Value(col,RJKJtRt[i][j] )  );
                }
                else
                {
                    (*result).second += RJKJtRt[i][j];
                }
            }
        }

        F = RJKJt*D;

        for(int i=0; i<12; i+=3)
            f[t[i/3]] += Deriv( F[i], F[i+1],  F[i+2] );
    }

    d_tetrahedronInfo.endEdit();
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessLarge( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact)
{
    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = d_tetrahedronInfo.getValue();

    Transformation R_0_2;
    R_0_2.transpose(tetrahedronInf[i].rotation);

    Displacement X;
    Coord x_2;

    x_2 = R_0_2*x[a];
    X[0] = x_2[0];
    X[1] = x_2[1];
    X[2] = x_2[2];

    x_2 = R_0_2*x[b];
    X[3] = x_2[0];
    X[4] = x_2[1];
    X[5] = x_2[2];

    x_2 = R_0_2*x[c];
    X[6] = x_2[0];
    X[7] = x_2[1];
    X[8] = x_2[2];

    x_2 = R_0_2*x[d];
    X[9] = x_2[0];
    X[10] = x_2[1];
    X[11] = x_2[2];

    Displacement F;

    computeForce( F, X,tetrahedronInf[i].materialMatrix, tetrahedronInf[i].strainDisplacementTransposedMatrix, fact);

    f[a] += tetrahedronInf[i].rotation * Deriv( -F[0], -F[1],  -F[2] );
    f[b] += tetrahedronInf[i].rotation * Deriv( -F[3], -F[4],  -F[5] );
    f[c] += tetrahedronInf[i].rotation * Deriv( -F[6], -F[7],  -F[8] );
    f[d] += tetrahedronInf[i].rotation * Deriv( -F[9], -F[10], -F[11] );
}

//////////////////////////////////////////////////////////////////////
////////////////////  polar decomposition method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::initPolar(int i, Index& a, Index&b, Index&c, Index&d)
{
    const VecCoord X0=this->mstate->read(core::vec_id::read_access::restPosition)->getValue();

    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    Transformation A;
    A[0] = (X0)[b]-(X0)[a];
    A[1] = (X0)[c]-(X0)[a];
    A[2] = (X0)[d]-(X0)[a];
    tetrahedronInf[i].initialTransformation = A;

    Transformation R_0_1;
    helper::Decompose<Real>::polarDecomposition(A, R_0_1);

    tetrahedronInf[i].rotatedInitialElements[0] = R_0_1*(X0)[a];
    tetrahedronInf[i].rotatedInitialElements[1] = R_0_1*(X0)[b];
    tetrahedronInf[i].rotatedInitialElements[2] = R_0_1*(X0)[c];
    tetrahedronInf[i].rotatedInitialElements[3] = R_0_1*(X0)[d];

    computeStrainDisplacement( tetrahedronInf[i].strainDisplacementTransposedMatrix,tetrahedronInf[i].rotatedInitialElements[0], tetrahedronInf[i].rotatedInitialElements[1],tetrahedronInf[i].rotatedInitialElements[2],tetrahedronInf[i].rotatedInitialElements[3] );

    d_tetrahedronInfo.endEdit();
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForcePolar( Vector& f, const Vector & p, Index elementIndex )
{
    const core::topology::BaseMeshTopology::Tetrahedron t=this->l_topology->getTetrahedron(elementIndex);

    Transformation A;
    A[0] = p[t[1]]-p[t[0]];
    A[1] = p[t[2]]-p[t[0]];
    A[2] = p[t[3]]-p[t[0]];

    Transformation R_0_2;
    helper::Decompose<Real>::polarDecomposition(A, R_0_2);

    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    tetrahedronInf[elementIndex].rotation.transpose( R_0_2 );

    // positions of the deformed and displaced Tetrahedre in its frame
    type::fixed_array<Coord, 4>  deforme;
    for(int i=0; i<4; ++i)
        deforme[i] = R_0_2 * p[t[i]];

    // displacement
    Displacement D;
    D[0] = tetrahedronInf[elementIndex].rotatedInitialElements[0][0] - deforme[0][0];
    D[1] = tetrahedronInf[elementIndex].rotatedInitialElements[0][1] - deforme[0][1];
    D[2] = tetrahedronInf[elementIndex].rotatedInitialElements[0][2] - deforme[0][2];
    D[3] = tetrahedronInf[elementIndex].rotatedInitialElements[1][0] - deforme[1][0];
    D[4] = tetrahedronInf[elementIndex].rotatedInitialElements[1][1] - deforme[1][1];
    D[5] = tetrahedronInf[elementIndex].rotatedInitialElements[1][2] - deforme[1][2];
    D[6] = tetrahedronInf[elementIndex].rotatedInitialElements[2][0] - deforme[2][0];
    D[7] = tetrahedronInf[elementIndex].rotatedInitialElements[2][1] - deforme[2][1];
    D[8] = tetrahedronInf[elementIndex].rotatedInitialElements[2][2] - deforme[2][2];
    D[9] = tetrahedronInf[elementIndex].rotatedInitialElements[3][0] - deforme[3][0];
    D[10] = tetrahedronInf[elementIndex].rotatedInitialElements[3][1] - deforme[3][1];
    D[11] = tetrahedronInf[elementIndex].rotatedInitialElements[3][2] - deforme[3][2];

    Displacement F;
    if(d_updateStiffnessMatrix.getValue())
    {
        // shape functions matrix
        computeStrainDisplacement( tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix, deforme[0],deforme[1],deforme[2],deforme[3]  );
    }

    if(!d_assembling.getValue())
    {
        computeForce( F, D, tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix );
        for(int i=0; i<12; i+=3)
            f[t[i/3]] += tetrahedronInf[elementIndex].rotation * Deriv( F[i], F[i+1],  F[i+2] );
    }
    else
    {
        msg_error() << "TODO(TetrahedralCorotationalFEMForceField): support for assembling system matrix when using polar method.";
    }

    d_tetrahedronInfo.endEdit();
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessPolar( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact )
{
    type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(d_tetrahedronInfo.beginEdit());

    Transformation R_0_2;
    R_0_2.transpose( tetrahedronInf[i].rotation );

    Displacement X;
    Coord x_2;

    x_2 = R_0_2*x[a];
    X[0] = x_2[0];
    X[1] = x_2[1];
    X[2] = x_2[2];

    x_2 = R_0_2*x[b];
    X[3] = x_2[0];
    X[4] = x_2[1];
    X[5] = x_2[2];

    x_2 = R_0_2*x[c];
    X[6] = x_2[0];
    X[7] = x_2[1];
    X[8] = x_2[2];

    x_2 = R_0_2*x[d];
    X[9] = x_2[0];
    X[10] = x_2[1];
    X[11] = x_2[2];

    Displacement F;

    computeForce( F, X, tetrahedronInf[i].materialMatrix, tetrahedronInf[i].strainDisplacementTransposedMatrix, fact);

    f[a] -= tetrahedronInf[i].rotation * Deriv( F[0], F[1],  F[2] );
    f[b] -= tetrahedronInf[i].rotation * Deriv( F[3], F[4],  F[5] );
    f[c] -= tetrahedronInf[i].rotation * Deriv( F[6], F[7],  F[8] );
    f[d] -= tetrahedronInf[i].rotation * Deriv( F[9], F[10], F[11] );

    d_tetrahedronInfo.endEdit();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

	if( !onlyVisible ) return;

	helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::vec_id::write_access::position);

	static const Real max_real = std::numeric_limits<Real>::max();
	static const Real min_real = std::numeric_limits<Real>::lowest();
	Real maxBBox[3] = {min_real,min_real,min_real};
	Real minBBox[3] = {max_real,max_real,max_real};
    for (std::size_t i=0; i<x.size(); i++)
	{
		for (int c=0; c<3; c++)
		{
			if (x[i][c] > maxBBox[c]) maxBBox[c] = (Real)x[i][c];
			else if (x[i][c] < minBBox[c]) minBBox[c] = (Real)x[i][c];
		}
	}

	this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox,maxBBox));
}


template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeVonMisesStress()
{
    typename core::behavior::MechanicalState<DataTypes>* mechanicalObject;
    this->getContext()->get(mechanicalObject);
    const VecCoord& X = mechanicalObject->read(core::vec_id::read_access::position)->getValue();

    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetras = this->l_topology->getTetrahedra();
    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = tetrahedronInfo.getValue();

    helper::WriteAccessor<Data<type::vector<Real> > > vME = d_vonMisesPerElement;
    vME.resize(tetras.size());

    for (Size i = 0; i < this->l_topology->getNbTetrahedra(); ++i)
    {
        type::Vec<6, Real> vStrain;
        type::Mat<3, 3, Real> gradU;
        Transformation R_0_2;
        Displacement D;

        const auto& tetra = tetras[i];
        const TetrahedronInformation& tetraInfo = tetrahedronInf[i];

        computeRotationLarge(R_0_2, X, tetra[0], tetra[1], tetra[2]);

        // positions of the deformed and displaced Tetrahedron in its frame
        type::fixed_array<Coord, 4> deforme;
        for (int j = 0; j < 4; ++j)
            deforme[j] = R_0_2 * X[tetra[j]];

        deforme[1][0] -= deforme[0][0];
        deforme[2][0] -= deforme[0][0];
        deforme[2][1] -= deforme[0][1];
        deforme[3] -= deforme[0];
        
        // displacement
        D[0] = 0;
        D[1] = 0;
        D[2] = 0;
        D[3] = tetraInfo.rotatedInitialElements[1][0] - deforme[1][0];
        D[4] = 0;
        D[5] = 0;
        D[6] = tetraInfo.rotatedInitialElements[2][0] - deforme[2][0];
        D[7] = tetraInfo.rotatedInitialElements[2][1] - deforme[2][1];
        D[8] = 0;
        D[9] = tetraInfo.rotatedInitialElements[3][0] - deforme[3][0];
        D[10] = tetraInfo.rotatedInitialElements[3][1] - deforme[3][1];
        D[11] = tetraInfo.rotatedInitialElements[3][2] - deforme[3][2];

        const type::Mat<4, 4, Real>& shf = tetraInfo.elemShapeFun;

        /// compute gradU
        for (Index k = 0; k < 3; k++) {
            for (Index l = 0; l < 3; l++) {
                gradU[k][l] = 0.0;
                for (Index m = 0; m < 4; m++)
                    gradU[k][l] += shf[l + 1][m] * D[3 * m + k];
            }
        }

        type::Mat<3, 3, Real> strain = Real(0.5) * (gradU + gradU.transposed());

        for (Index j = 0; j < 3; j++)
            vStrain[j] = strain[j][j];
        vStrain[3] = strain[1][2];
        vStrain[4] = strain[0][2];
        vStrain[5] = strain[0][1];

        Real lambda = tetraInfo.materialMatrix[0][1];
        Real mu = tetraInfo.materialMatrix[3][3];

        /// stress
        type::VecNoInit<6, Real> s; // VoigtTensor

        Real traceStrain = 0.0;
        for (Index k = 0; k < 3; k++) {
            traceStrain += vStrain[k];
            s[k] = vStrain[k] * 2 * mu;
        }

        for (Index k = 3; k < 6; k++)
            s[k] = vStrain[k] * 2 * mu;

        for (Index k = 0; k < 3; k++)
            s[k] += lambda * traceStrain;

        vME[i] = helper::rsqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] - s[0] * s[1] - s[1] * s[2] - s[2] * s[0] + 3 * s[3] * s[3] + 3 * s[4] * s[4] + 3 * s[5] * s[5]);
        if (vME[i] < 1e-10)
            vME[i] = 0.0;
    }

    helper::WriteAccessor<Data<type::vector<Real> > > vMN = d_vonMisesPerNode;
    vMN.resize(X.size());

    /// compute the values of vonMises stress in nodes
    for (Index dof = 0; dof < X.size(); dof++) {
        core::topology::BaseMeshTopology::TetrahedraAroundVertex tetrasAroundDOF = this->l_topology->getTetrahedraAroundVertex(dof);

        vMN[dof] = 0.0;
        for (size_t at = 0; at < tetrasAroundDOF.size(); at++)
            vMN[dof] += vME[tetrasAroundDOF[at]];
        if (!tetrasAroundDOF.empty())
            vMN[dof] /= Real(tetrasAroundDOF.size());
    }
}


template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,true);

    if (d_computeVonMisesStress.getValue())
    {
        Real minVM = (Real)1e20, maxVM = (Real)-1e20;
        Real minVMN = (Real)1e20, maxVMN = (Real)-1e20;
        helper::ReadAccessor<Data<type::vector<Real> > > vM = d_vonMisesPerElement;
        helper::ReadAccessor<Data<type::vector<Real> > > vMN = d_vonMisesPerNode;
        
        if (m_vonMisesColorMap == nullptr)
            return;
        
        for (size_t i = 0; i < vM.size(); i++)
        {
            minVM = (vM[i] < minVM) ? vM[i] : minVM;
            maxVM = (vM[i] > maxVM) ? vM[i] : maxVM;
        }
        if (maxVM < prevMaxStress)
        {
            maxVM = prevMaxStress;
        }
        for (size_t i = 0; i < vMN.size(); i++)
        {
            minVMN = (vMN[i] < minVMN) ? vMN[i] : minVMN;
            maxVMN = (vMN[i] > maxVMN) ? vMN[i] : maxVMN;
        }
        
        // Draw nodes (if node option enabled)
        if (vMN.size() == x.size())
        {
            std::vector<sofa::type::RGBAColor> nodeColors(x.size());
            std::vector<type::Vec3> pts(x.size());
            helper::ColorMap::evaluator<Real> evalColor = m_vonMisesColorMap->getEvaluator(minVMN, maxVMN);
            for (size_t nd = 0; nd < x.size(); nd++) {
                pts[nd] = x[nd];
                nodeColors[nd] = evalColor(vMN[nd]);
            }
            vparams->drawTool()->drawPoints(pts, 10, nodeColors);
        }
    }
    
    if (d_drawing.getValue())
    {
        const sofa::Size nbrTetra = this->l_topology->getNbTetrahedra();
        std::vector< type::Vec3 > points[4];
        points[0].reserve(nbrTetra * 3);
        points[1].reserve(nbrTetra * 3);
        points[2].reserve(nbrTetra * 3);
        points[3].reserve(nbrTetra * 3);

        for(Size i = 0 ; i< nbrTetra; ++i)
        {
            const core::topology::BaseMeshTopology::Tetrahedron t=this->l_topology->getTetrahedron(i);

            const auto& [a, b, c, d] = t.array();
            Coord center = (x[a] + x[b] + x[c] + x[d]) * 0.125;
            Coord pa = (x[a] + center) * (Real)0.666667;
            Coord pb = (x[b] + center) * (Real)0.666667;
            Coord pc = (x[c] + center) * (Real)0.666667;
            Coord pd = (x[d] + center) * (Real)0.666667;

            points[0].push_back(pa);
            points[0].push_back(pb);
            points[0].push_back(pc);

            points[1].push_back(pb);
            points[1].push_back(pc);
            points[1].push_back(pd);

            points[2].push_back(pc);
            points[2].push_back(pd);
            points[2].push_back(pa);

            points[3].push_back(pd);
            points[3].push_back(pa);
            points[3].push_back(pb);
        }

        vparams->drawTool()->drawTriangles(points[0], d_drawColor1.getValue());
        vparams->drawTool()->drawTriangles(points[1], d_drawColor2.getValue());
        vparams->drawTool()->drawTriangles(points[2], d_drawColor3.getValue());
        vparams->drawTool()->drawTriangles(points[3], d_drawColor4.getValue());
    }
    
    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,false);
}


template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal k, unsigned int &offset)
{
    // Build Matrix Block for this ForceField
    Transformation Rot;
    StiffnessMatrix JKJt,tmp;

    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = d_tetrahedronInfo.getValue();

    Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
    Rot[0][1]=Rot[0][2]=0;
    Rot[1][0]=Rot[1][2]=0;
    Rot[2][0]=Rot[2][1]=0;
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetras = this->l_topology->getTetrahedra();
    for(int IT=0 ; IT != (int)tetras.size() ; ++IT)
    {
        if (method == SMALL)
            computeStiffnessMatrix(JKJt,tmp,tetrahedronInf[IT].materialMatrix,tetrahedronInf[IT].strainDisplacementTransposedMatrix,Rot);
        else
            computeStiffnessMatrix(JKJt,tmp,tetrahedronInf[IT].materialMatrix,tetrahedronInf[IT].strainDisplacementTransposedMatrix,tetrahedronInf[IT].rotation);

        const core::topology::BaseMeshTopology::Tetrahedron t=tetras[IT];
        Inherit1::addToMatrix(mat, offset, t, tmp, -k);
    }
}

template <class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    StiffnessMatrix JKJt, RJKJtRt;
    sofa::type::Mat<3, 3, Real> localMatrix(type::NOINIT);

    static constexpr Transformation identity = []
    {
        Transformation i;
        i.identity();
        return i;
    }();

    const type::vector<TetrahedronInformation>& tetrahedronInf = d_tetrahedronInfo.getValue();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->l_topology->getTetrahedra();

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (std::size_t tetraId = 0; tetraId != tetrahedra.size(); ++tetraId)
    {
        const auto& rotation = method == SMALL ? identity : tetrahedronInf[tetraId].rotation;
        computeStiffnessMatrix(JKJt, RJKJtRt, tetrahedronInf[tetraId].materialMatrix,
                               tetrahedronInf[tetraId].strainDisplacementTransposedMatrix, rotation);

        const core::topology::BaseMeshTopology::Tetrahedron tetra = tetrahedra[tetraId];

        static constexpr auto S = DataTypes::deriv_total_size; // size of node blocks
        for (sofa::Size n1 = 0; n1 < core::topology::BaseMeshTopology::Tetrahedron::size(); ++n1)
        {
            for (sofa::Size n2 = 0; n2 < core::topology::BaseMeshTopology::Tetrahedron::size(); ++n2)
            {
                RJKJtRt.getsub(S * n1, S * n2, localMatrix); //extract the submatrix corresponding to the coupling of nodes n1 and n2
                dfdx(S * tetra[n1], S * tetra[n2]) += -localMatrix;
            }
        }
    }
}
template <class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::printStiffnessMatrix(int idTetra)
{

    const type::vector<typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = d_tetrahedronInfo.getValue();

    Transformation Rot;
    StiffnessMatrix JKJt,tmp;

    Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
    Rot[0][1]=Rot[0][2]=0;
    Rot[1][0]=Rot[1][2]=0;
    Rot[2][0]=Rot[2][1]=0;

    computeStiffnessMatrix(JKJt,tmp,tetrahedronInf[idTetra].materialMatrix,tetrahedronInf[idTetra].strainDisplacementTransposedMatrix,Rot);
}

} // namespace sofa::component::solidmechanics::fem::elastic
