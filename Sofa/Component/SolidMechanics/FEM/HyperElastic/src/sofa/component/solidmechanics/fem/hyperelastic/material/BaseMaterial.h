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

#include <sofa/component/solidmechanics/fem/hyperelastic/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Vec.h>

namespace sofa::component::solidmechanics::fem::hyperelastic::material
{

/**
 * Generic material class
 */
class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API BaseMaterial : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseMaterial,core::objectmodel::BaseObject);

    BaseMaterial() {}
    ~BaseMaterial() override {}

    void init() override
    {
        this->core::objectmodel::BaseObject::init();
    }

    //virtual VecN computeStress (VecN & strain,int idElement,int id_QP){return stress in the i-th quadrature point}
    //So here needed the shapefunctionvalue *  ,  quadratureformular*  (verifie if shapfunctionvalue compute with the local method)
    // The same principe for computing the strain given the displacement

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doComputeStress" internally,
     * which is the method to override from now on.
     * 
     **/  
    virtual void computeStress (type::Vec3 & stress, type::Vec3 & strain, unsigned int & elementIndex) final {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doComputeStress(stress, strain, elementIndex);
    }

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doComputeDStress" internally,
     * which is the method to override from now on.
     * 
     **/  
    virtual void computeDStress (type::Vec3 & dstress, type::Vec3 & dstrain) final {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doComputeDStress(dstress, dstrain);
    }

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doComputeStress" internally,
     * which is the method to override from now on.
     * 
     **/  
    virtual void computeStress (unsigned int elementIndex) final {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doComputeStress(elementIndex);
    };

protected:
    virtual void doComputeStress(type::Vec3 & stress, type::Vec3 & strain, unsigned int & idElement) = 0;
    virtual void doComputeDStress(type::Vec3 & dstress, type::Vec3 & dstrain) = 0;
    virtual void doComputeStress(unsigned int elementIndex) = 0;

private:
	BaseMaterial(const BaseMaterial& n) ;
	BaseMaterial& operator=(const BaseMaterial& n) ;
	
};

} // namespace sofa::component::solidmechanics::fem::hyperelastic::material
