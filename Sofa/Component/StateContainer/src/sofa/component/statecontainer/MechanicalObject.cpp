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
#define SOFA_COMPONENT_CONTAINER_MECHANICALOBJECT_CPP

#include <sofa/component/statecontainer/MechanicalObject.inl>
#include <sofa/type/Quat.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::component::statecontainer
{

using namespace core::behavior;
using namespace defaulttype;

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
        .add< MechanicalObject<Vec3Types> >(true) // default template
        .add< MechanicalObject<Vec2Types> >()
        .add< MechanicalObject<Vec1Types> >()
        .add< MechanicalObject<Vec6Types> >()
        .add< MechanicalObject<Rigid3Types> >()
        .add< MechanicalObject<Rigid2Types> >()

        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
template class SOFA_COMPONENT_STATECONTAINER_API MechanicalObject<Vec3Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MechanicalObject<Vec2Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MechanicalObject<Vec1Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MechanicalObject<Vec6Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MechanicalObject<Rigid3Types>;
template class SOFA_COMPONENT_STATECONTAINER_API MechanicalObject<Rigid2Types>;






template<>
void MechanicalObject<defaulttype::Rigid3Types>::applyRotation (const type::Quat<SReal> q)
{
    helper::WriteAccessor< Data<VecCoord> > x = *this->write(core::VecCoordId::position());

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i].getCenter() = q.rotate(x[i].getCenter());
        x[i].getOrientation() = q * x[i].getOrientation();
    }
}

template<>
void MechanicalObject<defaulttype::Rigid3Types>::addFromBaseVectorDifferentSize(core::VecId dest, const linearalgebra::BaseVector* src, unsigned int &offset )
{

    if (dest.type == sofa::core::V_COORD)
    {

        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(core::VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = src->size()/coordDim;

        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<3; ++j)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i+offset],j, tmp + src->element(i*coordDim+j));
            }

            type::Quat<double> q_src;
            type::Quat<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue(vDest[i+offset], j+3, tmp);
            }
        }
        offset += nbEntries;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(core::VecDerivId(dest));

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = src->size()/derivDim;
        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<derivDim; ++j)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i+offset],j, tmp + src->element(i*derivDim+j));
            }
        }
        offset += nbEntries;
    }


}

template<>
void MechanicalObject<defaulttype::Rigid3Types>::addFromBaseVectorSameSize(core::VecId dest, const linearalgebra::BaseVector* src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(core::VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest.size(); i++)
        {
            for (unsigned int j=0; j<3; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i],j,tmp + src->element(offset + i * coordDim + j));
            }

            type::Quat<double> q_src;
            type::Quat<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(offset + i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue(vDest[i], j+3, tmp);
            }
        }

        offset += unsigned(vDest.size() * coordDim);
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(core::VecDerivId(dest));
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        for (unsigned int i=0; i<vDest.size(); i++)
        {
            for (unsigned int j=0; j<derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i],j,tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i], j, tmp + src->element(offset + i * derivDim + j));
            }
        }
        offset += unsigned(vDest.size() * derivDim);
    }

}


template<>
void MechanicalObject<defaulttype::Rigid3Types>::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->setLightingEnabled(false);

	if (showIndices.getValue())
	{
        drawIndices(vparams);
	}

    if (showVectors.getValue())
    {
        drawVectors(vparams);
    }

    if (showObject.getValue())
    {
        const float scale = showObjectScale.getValue();
        const helper::ReadAccessor<Data<VecCoord> > x = *this->read(core::VecCoordId::position());
        const size_t vsize = d_size.getValue();
        for (size_t i = 0; i < vsize; ++i)
        {
            vparams->drawTool()->pushMatrix();
            float glTransform[16];
            ///TODO: check if the drawtool use OpenGL-shaped matrix
            x[i].writeOpenGlMatrix ( glTransform );
            vparams->drawTool()->multMatrix( glTransform );
            vparams->drawTool()->scale ( scale );

			if (getContext()->isSleeping())
			{
				vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ), sofa::type::RGBAColor::gray());
			}
			else switch( drawMode.getValue() )
            {
                case 1:
                    vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ), sofa::type::RGBAColor::green());
                    break;
                case 2:
                    vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ), sofa::type::RGBAColor::red());
                    break;
                case 3:
                    vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ), sofa::type::RGBAColor::blue());
                    break;
                case 4:
                    vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ), sofa::type::RGBAColor::yellow());
                    break;
                case 5:
                    vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ), sofa::type::RGBAColor::magenta());
                    break;
                case 6:
                    vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ), sofa::type::RGBAColor::cyan());
                    break;
                default:
                    vparams->drawTool()->drawFrame ( type::Vec3(), type::Quat<SReal>(), type::Vec3 ( 1_sreal,1_sreal,1_sreal ) );
            }

            vparams->drawTool()->popMatrix();
        }
    }

}

} // namespace sofa::component::statecontainer
