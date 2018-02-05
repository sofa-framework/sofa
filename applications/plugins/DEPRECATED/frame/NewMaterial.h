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
#ifndef SOFA_COMPONENT_MATERIAL_MATERIAL_H
#define SOFA_COMPONENT_MATERIAL_MATERIAL_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include "initFrame.h"
#include "DeformationGradientTypes.h"
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{
namespace component
{
namespace material
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

/** \brief Material as a stress-strain relationship.
 */
template<class TMaterialTypes>
class SOFA_FRAME_API Material : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(Material, TMaterialTypes), Inherited);

    typedef TMaterialTypes MaterialTypes;
    typedef typename MaterialTypes::Real Real;          ///< Real
    typedef typename MaterialTypes::Coord MaterialCoord;
    typedef vector<MaterialCoord> VecMaterialCoord;
    typedef Vec<6,Real> Str;            ///< Strain or stress tensor
    typedef vector<Str> VecStr;      ///< Vector of strain or stress tensors
    //typedef Mat<6,10,Real> El2Str;            ///< second-order Elaston strain or stress
    //typedef vector<El2Str> VecEl2Str;      ///< Vector of elaston strain or stress
    typedef Mat<6,6,Real> StrStr;      ///< Stress-strain matrix
    typedef vector<StrStr> VecStrStr;      ///< Vector of Stress-strain matrices

    virtual ~Material() {}

    typedef DeformationGradientTypes<3,3,1,Real> D331;
    typedef typename CStrain<D331,true>::Strain Strain1;
    typedef vector<Strain1> VecStrain1;
    typedef DeformationGradientTypes<3,3,2,Real> D332;
    typedef typename CStrain<D332,true>::Strain Strain4;
    typedef vector<Strain4> VecStrain4;
    typedef typename CStrain<D332,false>::Strain Strain10;
    typedef vector<Strain10> VecStrain10;

    /** @name Stress
     *   Compute stress at each point based on local strain and strain rate.
     */
    //@{
    /** Compute stress at each point based on local strain and strain rate. */
    virtual Real getBulkModulus(const unsigned int sampleindex) const = 0;
    virtual bool computeVolumeIntegrationFactors(const unsigned int sampleindex,const MaterialCoord& point,const unsigned int order,vector<Real>& moments)=0;
    virtual void computeStress  ( VecStrain1& stress, VecStrStr* stressStrainMatrices, const VecStrain1& strain, const VecStrain1& strainRate, const VecMaterialCoord& point )=0;
    virtual void computeStressChange  ( VecStrain1& stressChange, const VecStrain1& strainChange, const VecMaterialCoord& point )=0;
    virtual void computeStress  ( VecStrain4& stress, VecStrStr* stressStrainMatrices, const VecStrain4& strain, const VecStrain4& strainRate, const VecMaterialCoord& point )=0;
    virtual void computeStressChange  ( VecStrain4& stressChange, const VecStrain4& strainChange, const VecMaterialCoord& point )=0;
    virtual void computeStress  ( VecStrain10& stress, VecStrStr* stressStrainMatrices, const VecStrain10& strain, const VecStrain10& strainRate, const VecMaterialCoord& point )=0;
    virtual void computeStressChange  ( VecStrain10& stressChange, const VecStrain10& strainChange, const VecMaterialCoord& point )=0;
    //@}

    /// get the StressStrain matrices at the given points, assuming null strain or linear material
    virtual void getStressStrainMatrix( StrStr& matrix, const MaterialCoord& point ) const =0;

    inline Str hookeStress  ( const Str& strain, Real stressDiagonal, Real stressOffDiagonal, Real shear  ) const
    {
        return Str(
                stressDiagonal * strain[0] + stressOffDiagonal * strain[1] + stressOffDiagonal * strain[2],
                stressOffDiagonal * strain[0] + stressDiagonal * strain[1] + stressOffDiagonal * strain[2],
                stressOffDiagonal * strain[0] + stressOffDiagonal * strain[1] + stressDiagonal * strain[2],
                shear * strain[3],
                shear * strain[4],
                shear * strain[5]
                );
    }

    inline void fillHookeMatrix  ( StrStr& m, Real stressDiagonal, Real stressOffDiagonal, Real shear  ) const
    {
        m.fill(0);
        m[0][0] = m[1][1] = m[2][2] = stressDiagonal;
        m[0][1] = m[0][2] = m[1][0] = m[1][2] = m[2][0] = m[2][1] = stressOffDiagonal;
        m[3][3] = m[4][4] = m[5][5] = shear;
    }


    //    virtual void computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& strainRate );


    //    /** \brief Compute stress based on local strain and strain rate at each point.
    //      The stress-strain relation may depend on strain rate (time derivative of strain).
    //      The stress-strain matrices are written if the pointer is not null.
    //    */
    //    virtual void computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& strainRate ) = 0;
    //
    //    /** \brief Compute elaston stress based on local strain and strain rate at each point.
    //      The stress-strain relation may depend on strain rate (time derivative of strain).
    //      The stress-strain matrices are written if the pointer is not null.
    //    */
    //    virtual void computeStress  ( VecEl2Str& stress, VecStrStr* stressStrainMatrices, const VecEl2Str& strain, const VecEl2Str& strainRate ) = 0;

    //typedef defaulttype::DeformationGradient<3,3,1,Real> DeformationGradient331;
    //typedef typename DeformationGradient331::SampleIntegVector SampleIntegVector331;
    //typedef vector<SampleIntegVector331>  VecSampleIntegVector331;
    //typedef typename DeformationGradient331::Strain            Strain331;
    //typedef vector<Strain331>  VecStrain331;

    ///** \brief Compute stress based on local strain and strain rate at each point.
    //*/
    //virtual void computeStress  ( VecStrain331& stress, const VecStrain331& strain, const VecStrain331& strainRate, const VecSampleIntegVector331& integ ) = 0;

    ///** \brief Compute stress change based on strain change
    // */
    //virtual void computeStressChange  ( VecStrain331& stressChange, const VecStrain331& strainChange, const VecSampleIntegVector331& integ ) = 0;


    //typedef defaulttype::DeformationGradient<3,3,2,Real> DeformationGradient332;
    //typedef typename DeformationGradient332::SampleIntegVector SampleIntegVector332;
    //typedef vector<SampleIntegVector332>  VecSampleIntegVector332;
    //typedef typename DeformationGradient332::Strain            Strain332;
    //typedef vector<Strain332>  VecStrain332;

    ///** \brief Compute stress based on local strain and strain rate at each point.
    //*/
    //virtual void computeStress  ( VecStrain332& stress, const VecStrain332& strain, const VecStrain332& strainRate, const VecSampleIntegVector332& integ ) = 0;

    ///** \brief Compute stress change based on strain change
    // */
    //virtual void computeStressChange  ( VecStrain332& stressChange, const VecStrain332& strainChange, const VecSampleIntegVector332& integ ) = 0;






};



template<int N_, class R>
struct MaterialTypes
{
    typedef R Real;
    static const int N=N_ ;  ///< Number of parameters of the material coordinates
    static const int StrDim = N*(N+1)/2;             ///< Number of independent entries in the symmetric DxD strain tensor
    typedef Vec<N,Real> Coord;

    //    typedef defaulttype::Vec<StrDim,R> Str;       ///< Strain or stress tensor in Voigt (i.e. vector) notation
    //    typedef helper::vector<Str> VecStr;
    //
    //    /** Strain or stress tensor in Voigt (i.e. vector) notation for an elaston.
    //    The first column is the strain (or stress), the other columns are its derivatives in the space directions (TODO: check this)
    //    */
    //    typedef defaulttype::Mat<StrDim,D*D+1,R> ElStr;
    //    typedef helper::vector<ElStr> VecElStr;
    //
    //    typedef defaulttype::Mat<StrDim,StrDim,R> StrStr;  ///< Stress-strain matrix
    //    typedef helper::vector<StrStr> VecStrStr;

    static const char* Name();
};

typedef MaterialTypes<3,float> Material3f;
typedef MaterialTypes<3,double> Material3d;


//typedef Rigid3fTypes Material3f;
//typedef Rigid3dTypes Material3d;

#ifdef SOFA_FLOAT
template<> inline const char* Material3d::Name() { return "Material3d"; }
template<> inline const char* Material3f::Name() { return "Material3"; }
#else
template<> inline const char* Material3d::Name() { return "Material3"; }
template<> inline const char* Material3f::Name() { return "Material3f"; }
#endif



} //

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FEM_BASEMATERIAL_H
