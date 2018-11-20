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
#ifndef SOFA_COMPONENT_ENGINE_MEAN_COMPUTATION_H
#define SOFA_COMPONENT_ENGINE_MEAN_COMPUTATION_H

#include <MultiThreading/config.h>


#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
//#include <sofa/defaulttype/Vec.h>
//#include <sofa/core/topology/BaseMeshTopology.h>
//#include <sofa/defaulttype/VecTypes.h>
//#include <sofa/defaulttype/RigidTypes.h>

#include <SofaBaseMechanics/MechanicalObject.h>

namespace sofa
{

    namespace component
    {

        namespace engine
        {

            /**
            * This class merge 2 coordinate vectors.
            */
            template <class DataTypes>
            class MeanComputation : public virtual core::objectmodel::BaseObject
            {
                typedef typename DataTypes::Coord         Coord;
                typedef typename DataTypes::VecCoord      VecCoord;
                typedef typename DataTypes::Real          Real;
                //typedef sofa::defaulttype::Vec<1, Real>                       Coord1D;
                //typedef sofa::defaulttype::Vec<2, Real>                       Coord2D;
                //typedef sofa::defaulttype::Vec<3, Real>                       Coord3D;
                typedef sofa::defaulttype::ResizableExtVector <Coord>       ResizableExtVectorCoord;
                typedef sofa::defaulttype::ResizableExtVector <VecCoord>    ResizableExtVectorVecCoord;
                //typedef sofa::helper::vector <Coord3D>    VecCoord3D;

            public:
                SOFA_CLASS(SOFA_TEMPLATE(MeanComputation, DataTypes), core::objectmodel::BaseObject);
                //typedef typename DataTypes::VecCoord VecCoord;

            protected:

                MeanComputation();

                ~MeanComputation() {}

                void compute();

            public:
                void init() override;

                void reinit() override;

                virtual void handleEvent(core::objectmodel::Event* event) override;

                virtual std::string getTemplateName() const override
                {
                    return templateName(this);
                }

                static std::string templateName(const MeanComputation<DataTypes>* = NULL)
                {
                    return DataTypes::Name();
                }

            private:

                Data<VecCoord> d_result;

                //std::vector<component::container::MechanicalObject<DataTypes>*> _inputMechObjs;
                
                sofa::helper::vector< Data< VecCoord >* > _inputs;
                
                size_t _resultSize;
          
            };


        } // namespace engine

    } // namespace component

} // namespace sofa


#endif  /* SOFA_COMPONENT_ENGINE_MEAN_COMPUTATION_H */
