/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_INL

#include "TetrahedronDiffusionFEMForceField.h"
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/core/visual/VisualParams.h>

#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/TopologySubsetData.inl>

#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/behavior/MultiMatrixAccessor.h>


#include <iostream>
#include <fstream>
#include <string.h>

#include <SofaSimulationTree/GNode.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

   using namespace sofa::defaulttype;
   using namespace sofa::component::topology;
   using namespace core::topology;
   using namespace core::objectmodel;
   using core::topology::BaseMeshTopology;
   typedef BaseMeshTopology::EdgesInTetrahedron		EdgesInTetrahedron;


template< class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::computeEdgeDiffusionCoefficient()
{
    edgeDiffusionCoefficient.clear();
    edgeDiffusionCoefficient.resize(nbEdges);

    unsigned int i,j,k,l, nbTetra;
    typename DataTypes::Real val1,volume;
    typename DataTypes::Real diff;
    Vec3 point[4],shapeVector[4];

    for (i=0; i<nbEdges; ++i)
        edgeDiffusionCoefficient[i] = 0;

    nbTetra = topology->getNbTetrahedra();
    const typename TetrahedronDiffusionFEMForceField<DataTypes>::MechanicalTypes::VecCoord position = this->mechanicalObject->read(core::ConstVecCoordId::position())->getValue();

    typename DataTypes::Real anisotropyRatio = this->d_transverseAnisotropyRatio.getValue();
    bool isotropicDiffusion = (anisotropyRatio == 1.0);

    for (i=0; i<nbTetra; ++i)
    {
        // get a reference on the edge set of the ith added tetrahedron
        const EdgesInTetrahedron &te= topology->getEdgesInTetrahedron(i);
        //get a reference on the vertex set of the ith added tetrahedron
        const Tetrahedron &t= topology->getTetrahedron(i);

        // store points
        for(j=0; j<4; ++j)
            point[j]= position[t[j]];

        // compute 6 times the rest volume
        volume = dot(cross(point[1]-point[0], point[2]-point[0]), point[0]-point[3]);

        // store shape vectors
        for(j=0;j<4;++j)
        {
            if ((j%2)==0)
                shapeVector[j] = -cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
            else
                shapeVector[j] = cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
        }

        diff=(d_tetraDiffusionCoefficient.getValue())[i]*fabs(volume)/6;

        // isotropic case
        if (isotropicDiffusion)
        {
            for(j=0;j<6;++j)
            {
                /// local indices of the edge
                k = topology->getLocalEdgesInTetrahedron(j)[0];
                l = topology->getLocalEdgesInTetrahedron(j)[1];

                val1 = dot(shapeVector[k],shapeVector[l])*diff;
                edgeDiffusionCoefficient[te[j]] += val1;
            }
        }
        // anisotropic case
        else
        {
            Vec3 direction = d_transverseAnisotropyDirectionArray.getValue()[i];
            direction.norm();

            for(j=0;j<6;++j)
            {
                /// local indices of the edge
                k = topology->getLocalEdgesInTetrahedron(j)[0];
                l = topology->getLocalEdgesInTetrahedron(j)[1];

                val1= dot(shapeVector[k],shapeVector[l]+direction * ((anisotropyRatio-1)*dot(direction,shapeVector[l])))*diff;
                edgeDiffusionCoefficient[te[j]] += val1;
            }
        }
    }
}


// --------------------------------------------------------------------------------------
// --- constructor
// --------------------------------------------------------------------------------------
template <class DataTypes>
TetrahedronDiffusionFEMForceField<DataTypes>::TetrahedronDiffusionFEMForceField()
    : d_constantDiffusionCoefficient(initData(&d_constantDiffusionCoefficient, (Real)1.0, "constantDiffusionCoefficient","Constant diffusion coefficient")),
      d_tetraDiffusionCoefficient( initData(&d_tetraDiffusionCoefficient, "tetraDiffusionCoefficient","Diffusion coefficient for each tetrahedron, by default equal to constantDiffusionCoefficient.")),
      d_1DDiffusion(initData(&d_1DDiffusion, false, "scalarDiffusion","if true, diffuse only on the first dimension.")),
      d_transverseAnisotropyRatio(initData(&d_transverseAnisotropyRatio, (Real)1.0, "anisotropyRatio","Anisotropy ratio (r²>1).\n Default is 1.0 = isotropy.")),
      d_transverseAnisotropyDirectionArray(initData(&d_transverseAnisotropyDirectionArray, "transverseAnisotropyArray","Data to handle topology on tetrahedra")),
      d_tagMeshMechanics(initData(&d_tagMeshMechanics, std::string("meca"),"tagMechanics","Tag of the Mechanical Object.")),
      d_drawConduc( initData(&d_drawConduc, (bool)false, "drawConduc","To display conductivity map."))
{
    this->f_listening.setValue(true);
}


template <class DataTypes>
TetrahedronDiffusionFEMForceField<DataTypes>::~TetrahedronDiffusionFEMForceField()
{
}


// --------------------------------------------------------------------------------------
// --- Initialization stage
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::init()
{
    this->Inherited::init();
    // Test if topology is present and is a appropriate
    topology = this->getContext()->getMeshTopology();

    // Save the number of edges in the mesh
    nbEdges = topology->getNbEdges();

    if (!topology)
    {
        serr  << "Error: TetrahedralMitchellShaefferForceField is not able to acces topology!" << sendl;
        return;
    }
    if (topology->getNbTetrahedra()==0)
    {
        serr << "ERROR(TetrahedronDiffusionFEMForceField): object must have a Tetrahedral Set Topology."<<sendl;
        return;
    }

    /// Initialize all the diffusion coefficients (for tetras) to the value given by the user.
    sofa::helper::vector <Real>& tetraDiff = *(d_tetraDiffusionCoefficient.beginEdit());
    loadedDiffusivity = false;

    /// case no potential vector input
    if(tetraDiff.size()==0)
    {
        tetraDiff.resize(topology->getNbTetrahedra());
        for(int i=0; i<topology->getNbTetrahedra(); i++)
            tetraDiff[i]=this->d_constantDiffusionCoefficient.getValue();
    }
    /// possible input tetrahedral diffusion coefficient
    else
    {
        loadedDiffusivity = true;
        if((int)(tetraDiff.size())!=topology->getNbTetrahedra())
        {
            serr<<"ERROR(TetrahedronDiffusionFEMForceField): error wrong size of potential input vector."<<sendl;
            return;
        }

        msg_info() << "diffusion coefficient is loaded per tetra";
    }
    d_tetraDiffusionCoefficient.endEdit();

    /// Get the mechanical object containing the mesh position in 3D
    Tag mechanicalTag(d_tagMeshMechanics.getValue());
    this->getContext()->get(mechanicalObject, mechanicalTag,sofa::core::objectmodel::BaseContext::SearchUp);
    if (mechanicalObject==NULL)
    {
        serr<<"ERROR(TetrahedronDiffusionFEMForceField): cannot find the mechanical object."<<sendl;
        msg_info() <<"mechanicalObj = "<<mechanicalObject ;

        return;
    }

    if(d_transverseAnisotropyRatio.getValue()!=1.0)
    {
        msg_info() << "(TetrahedronDiffusionFEMForceField) anisotropic (r="<<sqrt(d_transverseAnisotropyRatio.getValue())<<") diffusion.";
    }
    else
    {
        msg_info() << "(TetrahedronDiffusionFEMForceField) isotropic diffusion.";
    }

    // prepare to store info in the edge array
    this->computeEdgeDiffusionCoefficient();
}


template <class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::reinit()
{
}


template <class DataTypes>
SReal TetrahedronDiffusionFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    serr<<"TetrahedronDiffusionFEMForceField::getPotentialEnergy not implemented yet"<<sendl;
    return 0;
}


template <class DataTypes>
typename TetrahedronDiffusionFEMForceField<DataTypes>::VectorReal TetrahedronDiffusionFEMForceField<DataTypes>::getDiffusionCoefficient()
{
    // Constant diffusion in input: return this single value
    if(!loadedDiffusivity)
    {
        sofa::helper::vector<Real> output;
        output.resize(1);
        output[0] = d_constantDiffusionCoefficient.getValue();
        return output;
    }
    // Tetrahedral diffusion coefficient in input: return the entire vector
    else
        return d_tetraDiffusionCoefficient.getValue();
}


template <class DataTypes>
typename TetrahedronDiffusionFEMForceField<DataTypes>::Real TetrahedronDiffusionFEMForceField<DataTypes>::getTetraDiffusionCoefficient(unsigned int i)
{
    sofa::helper::vector<Real> tetraDiff = this->d_tetraDiffusionCoefficient.getValue();
    if((int) i <= topology->getNbTetrahedra())
    {
        return tetraDiff[i];
    }
    else
    {
        serr<<  "Tetra i is larger than topology->getNbTetrahedra() " << sendl;
        return -1;
    }
}


template <class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::setDiffusionCoefficient(const Real val)
{
    // Save the new diffusion coefficient: d_constantDiffusionCoefficient
    Real& _constantDiffusion = *(d_constantDiffusionCoefficient.beginEdit());
    _constantDiffusion = val;
    d_constantDiffusionCoefficient.endEdit();

    // Save the new diffusion coefficient for each tetrahedron
    sofa::helper::vector <Real>& tetraDiff = *(d_tetraDiffusionCoefficient.beginEdit());
    tetraDiff.clear();
    tetraDiff.resize(topology->getNbTetrahedra());
    for(int i=0; i<topology->getNbTetrahedra(); i++)
        tetraDiff[i] = d_constantDiffusionCoefficient.getValue();
    d_tetraDiffusionCoefficient.endEdit();

    // Integrate it on each edge of the mesh
    this->computeEdgeDiffusionCoefficient();
}


template <class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::setDiffusionCoefficient(const sofa::helper::vector<Real> val)
{
    // Check that the flag of input tetra diffusion coefficient is true
    if(!loadedDiffusivity)
        loadedDiffusivity = true;

    // Save the new diffusion coefficient for each tetrahedron
    sofa::helper::vector <Real>& tetraDiff = *(d_tetraDiffusionCoefficient.beginEdit());
    for(int i=0; i<topology->getNbTetrahedra(); i++)
        tetraDiff[i] = val[i];
    d_tetraDiffusionCoefficient.endEdit();

    // Integrate it on each edge of the mesh
    this->computeEdgeDiffusionCoefficient();
}


template <class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::addForce (const core::MechanicalParams* /*mparams*/, DataVecDeriv& dataf, const DataVecCoord& datax, const DataVecDeriv& /*v*/)
{
    helper::AdvancedTimer::stepBegin("addForceDiffusion");

    VecDeriv& f = *dataf.beginEdit();
    const VecCoord& x = datax.getValue();

    unsigned int v0,v1;

    Coord dp;

    for(unsigned int i=0; i<nbEdges; i++ )
    {
        v0=topology->getEdge(i)[0];
        v1=topology->getEdge(i)[1];

        //Case 1D Diffusion
        if (d_1DDiffusion.getValue())
        {
            dp[0] = (x[v1][0]-x[v0][0]) * edgeDiffusionCoefficient[i];

            f[v1][0]+=dp[0];
            f[v0][0]-=dp[0];
        }
        //Case 3D Diffusion
        else
        {
            dp = (x[v1]-x[v0]) * edgeDiffusionCoefficient[i];

            f[v1]+=dp;
            f[v0]-=dp;
        }
    }

    dataf.endEdit() ;
    sofa::helper::AdvancedTimer::stepEnd("addForceDiffusion");
}


template <class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv&   datadF , const DataVecDeriv&   datadX)
{
    sofa::helper::AdvancedTimer::stepBegin("addDForceDiffusion");
    VecDeriv& df = *datadF.beginEdit();
    const VecDeriv& dx=datadX.getValue();
    Real kFactor = mparams->kFactor();

    unsigned int v0,v1;

    Coord dp;

    for(int i=0; i<(int)nbEdges; i++ )
    {
        v0=topology->getEdge(i)[0];
        v1=topology->getEdge(i)[1];

        if (d_1DDiffusion.getValue())
        {
            dp[0] = (dx[v1][0]-dx[v0][0]) * edgeDiffusionCoefficient[i] * kFactor;

            df[v1][0]+=dp[0];
            df[v0][0]-=dp[0];
        }
        else
        {
            dp = (dx[v1]-dx[v0]) * edgeDiffusionCoefficient[i] * kFactor;

            df[v1]+=dp;
            df[v0]-=dp;
        }
    }
    datadF.endEdit() ;
    sofa::helper::AdvancedTimer::stepEnd("addDForceDiffusion");
}


template <class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::helper::AdvancedTimer::stepBegin("addKToMatrix");
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = r.matrix;

    if((int)(mat->colSize()) != (topology->getNbPoints()*N) || (int)(mat->rowSize()) != (topology->getNbPoints()*N))
    {
        serr<<"Wrong size of the input Matrix: need resize in addKToMatrix function."<<sendl;
        mat->resize(topology->getNbPoints()*N,topology->getNbPoints()*N);
    }

    Real kFactor = mparams->kFactor();
    unsigned int &offset = r.offset;

    unsigned int v0,v1;


    for(int i=0; i<(int)nbEdges; i++ )
    {
        v0=topology->getEdge(i)[0];
        v1=topology->getEdge(i)[1];

        mat->add(offset+N*v1, offset+N*v0, -kFactor * edgeDiffusionCoefficient[i]);
        mat->add(offset+N*v0, offset+N*v1, -kFactor * edgeDiffusionCoefficient[i]);
        mat->add(offset+N*v0, offset+N*v0, kFactor * edgeDiffusionCoefficient[i]);
        mat->add(offset+N*v1, offset+N*v1, kFactor * edgeDiffusionCoefficient[i]);
    }
    sofa::helper::AdvancedTimer::stepEnd("addKToMatrix");
}


template<class DataTypes>
void TetrahedronDiffusionFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields())
        return;
    if (!this->mstate)
        return;

    vparams->drawTool()->saveLastState();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    //draw the conductivity
    if (d_drawConduc.getValue())
    {
        const typename TetrahedronDiffusionFEMForceField<DataTypes>::MechanicalTypes::VecCoord restPosition =
        this->mechanicalObject->read(core::ConstVecCoordId::restPosition())->getValue();
        vparams->drawTool()->setLightingEnabled(false);

        unsigned int nbr = topology->getNbTriangles();
        sofa::helper::vector<unsigned int> surfaceTri;

        for (unsigned int i=0; i<nbr; ++i)
        {
            if((topology->getTetrahedraAroundTriangle(i)).size() == 1)
                surfaceTri.push_back(i);
        }

        sofa::defaulttype::Vec4f colorLine(1.0, 0.0, 0.0, 1.0);
        helper::vector<sofa::defaulttype::Vector3> vertices;

        for (unsigned int i=0; i<surfaceTri.size(); ++i)
        {
            sofa::defaulttype::Vector3 point[3];
            Triangle tri = topology->getTriangle(surfaceTri[i]);
            for (unsigned int j=0; j<3; ++j)
                point[j] = restPosition[tri[j]];

            for (unsigned int j = 0; j<3; j++)
            {
                vertices.push_back(point[j]);
                vertices.push_back(point[(j+1)%3]);
            }
        }

        vparams->drawTool()->drawLines(vertices, 1, colorLine);

        unsigned int nbrTetra = topology->getNbTetrahedra();
        Real maxDiffusion = 0.0;
        for (unsigned int i = 0; i<nbrTetra; ++i)
        {
            if (d_tetraDiffusionCoefficient.getValue()[i] > maxDiffusion)
                maxDiffusion = d_tetraDiffusionCoefficient.getValue()[i];
        }

        colorLine = sofa::defaulttype::Vec4f(0.2, 0.2, 0.2, 1.0);
        vertices.clear();
        for (unsigned int i = 0; i<nbrTetra; ++i)
        {
            Real Ratio = d_tetraDiffusionCoefficient.getValue()[i] / maxDiffusion;
            sofa::defaulttype::Vec4f tetraColor(0.0, Ratio, 0.5-Ratio, 1.0);

            Tetrahedron tetra = topology->getTetrahedron(i);
            sofa::defaulttype::Vec<3,SReal> point[4];

            for (unsigned int j = 0; j<4; j++)
                point[j] = restPosition[tetra[j]];

            vparams->drawTool()->drawTetrahedron(point[0],point[1],point[2],point[3],tetraColor);

            for (unsigned int j = 0; j<4; j++)
            {
                vertices.push_back(point[j]);
                vertices.push_back(point[(j+1)%4]);

                vertices.push_back(point[(j+1)%4]);
                vertices.push_back(point[(j+2)%4]);

                vertices.push_back(point[(j+2)%4]);
                vertices.push_back(point[j]);
            }
        }
        vparams->drawTool()->drawLines(vertices, 1, colorLine);

    }

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);


    vparams->drawTool()->restoreLastState();
}


} // namespace forcefield

} // namespace Components

} // namespace Sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_INL
