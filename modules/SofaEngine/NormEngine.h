 
/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_NORMENGINE_H
#define SOFA_COMPONENT_ENGINE_NORMENGINE_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace engine
{

/// convert a vector of Vecs in a vector of their l-norms
template <class TDataType>
class NormEngine : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NormEngine,TDataType),core::DataEngine);

    typedef TDataType DataType;
    typedef typename DataType::value_type Real;
    typedef helper::vector<DataType> VecData;
    typedef helper::vector<Real> VecReal;

    NormEngine();

    virtual ~NormEngine() {}

    void init();

    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const NormEngine<TDataType>* = NULL)
    {
        return defaulttype::DataTypeInfo<TDataType>::name();
    }


protected:

    Data<VecData> d_input;
    Data<VecReal> d_output;
    Data<int> d_normType;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_NORMENGINE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API NormEngine<defaulttype::Vec3d>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API NormEngine<defaulttype::Vec3f>;
#endif
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
