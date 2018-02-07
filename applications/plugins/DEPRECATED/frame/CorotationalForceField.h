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
#ifndef FRAME_COROTATIONALFORCEFIELD_H
#define FRAME_COROTATIONALFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>
#include "initFrame.h"
#include "GridMaterial.h"
#include "Blending.h"

namespace sofa
{

namespace component
{

namespace forcefield
{

using helper::vector;

using namespace sofa::defaulttype;
/** Compute corotational strain and apply material law
*/
template <class DataTypes>
class CorotationalForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CorotationalForceField,DataTypes),SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::Real Real;
    static const int material_dimensions = DataTypes::material_dimensions;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef typename DataTypes::MaterialFrame Frame;
    typedef vector<Frame> VecFrame;

    typedef defaulttype::CStrain<DataTypes,true> StrainType;
    typedef typename StrainType::StrainVec StrainVec;
    typedef typename StrainType::StrStr StrStrMat;
    typedef typename StrainType::Strain Strain;
    typedef typename StrainType::Stress Stress;

    static const unsigned strainenergy_size = StrainType::strainenergy_size;
    typedef typename StrainType::StrainEnergyVec StrainEnergyVec;

    //                typedef material::GridMaterial<material::MaterialTypes<material_dimensions,Real> > Material;
    typedef material::Material<material::MaterialTypes<material_dimensions,Real> > Material;
    typedef typename Material::StrStr StressStrainMatrix;
    typedef typename Material::VecMaterialCoord VecMaterialCoord;


public:
    CorotationalForceField(core::behavior::MechanicalState<DataTypes> *mm = NULL);
    virtual ~CorotationalForceField();


    // -- ForceField interface
    void init();
    void addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& f , const DataVecCoord& x , const DataVecDeriv& v);
    void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&   df , const DataVecDeriv&   dx );


    //        virtual void draw();


protected :

    VecFrame rotation; ///< rotation matrices
    vector<Strain> strain, strainRate, stress, strainChange, stressChange;
    Material* material;
    typedef defaulttype::SampleData<DataTypes, true> SampleData;
    SampleData* sampleData;

    typedef typename StrainType::StrainEnergyVec  VecInteg; // vec or order VecIntegOrder
    vector< VecInteg > integFactors;
    vector< StressStrainMatrix > stressStrainMatrices;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FRAME_COROTATIONALFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_FRAME_API CorotationalForceField<DeformationGradient331dTypes>;
extern template class SOFA_FRAME_API CorotationalForceField<DeformationGradient332dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_FRAME_API CorotationalForceField<DeformationGradient331fTypes>;
extern template class SOFA_FRAME_API CorotationalForceField<DeformationGradient332fTypes>;
#endif
#endif

} //

} //

} // namespace sofa

#endif
