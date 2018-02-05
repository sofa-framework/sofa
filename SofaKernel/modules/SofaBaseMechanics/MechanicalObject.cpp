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
#define SOFA_COMPONENT_CONTAINER_MECHANICALOBJECT_CPP
#include <SofaBaseMechanics/MechanicalObject.inl>
#include <sofa/helper/Quater.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace container
{

using namespace core::behavior;
using namespace defaulttype;

//template <>
//bool MechanicalObject<Vec3dTypes>::addBBox(SReal* minBBox, SReal* maxBBox)
//{
//    cerr << "MechanicalObject<Vec3dTypes>::addBBox, before min=" << *minBBox <<", max=" << *maxBBox << endl;

//    // participating to bbox only if it is drawn
//    if( !showObject.getValue() ) return false;

//    const VecCoord& x = *this->read(sofa::core::ConstVecCoordId::position())->getValue();
//    for( std::size_t i=0; i<x.size(); i++ )
//    {
//        Vec<3,Real> p;
//        DataTypes::get( p[0], p[1], p[2], x[i] );
//        cerr<<"MechanicalObject<Vec3dTypes>::addBBox, p=" << p << endl;

//        assert( DataTypes::spatial_dimensions <= 3 );

//        for( unsigned int j=0 ; j<DataTypes::spatial_dimensions; ++j )
//        {
//            if(p[j]<minBBox[j]) minBBox[j]=p[j];
//            if(p[j]>maxBBox[j]) maxBBox[j]=p[j];
//        }
//    }
//    cerr << "MechanicalObject<Vec3dTypes>::addBBox, after min=" << *minBBox <<", max=" << *maxBBox << endl;
//    return true;
//}


SOFA_DECL_CLASS(MechanicalObject)

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
#ifdef SOFA_FLOAT
        .add< MechanicalObject<Vec3fTypes> >(true) // default template
#else
        .add< MechanicalObject<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Vec3fTypes> >()
#endif
#endif
#ifndef SOFA_FLOAT
        .add< MechanicalObject<Vec2dTypes> >()
        .add< MechanicalObject<Vec1dTypes> >()
        .add< MechanicalObject<Vec6dTypes> >()
        .add< MechanicalObject<Rigid3dTypes> >()
        .add< MechanicalObject<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Vec2fTypes> >()
        .add< MechanicalObject<Vec1fTypes> >()
        .add< MechanicalObject<Vec6fTypes> >()
        .add< MechanicalObject<Rigid3fTypes> >()
        .add< MechanicalObject<Rigid2fTypes> >()
#endif
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
#ifndef SOFA_FLOAT
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec3dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec2dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec1dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec6dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid3dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec3fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec2fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec1fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec6fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid3fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid2fTypes>;
#endif



#ifndef SOFA_FLOAT


template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::applyRotation (const defaulttype::Quat q)
{
    helper::WriteAccessor< Data<VecCoord> > x = *this->write(core::VecCoordId::position());

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i].getCenter() = q.rotate(x[i].getCenter());
        x[i].getOrientation() = q * x[i].getOrientation();
    }
}

template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::addFromBaseVectorDifferentSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset )
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

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
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
void MechanicalObject<defaulttype::Rigid3dTypes>::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset)
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

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
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

        offset += vDest.size() * coordDim;
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
        offset += vDest.size() * derivDim;
    }

}


template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->saveLastState();
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
        helper::ReadAccessor<Data<VecCoord> > x = *this->read(core::VecCoordId::position());
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
				vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0.5,0.5,0.5,1) );
			}
			else switch( drawMode.getValue() )
            {
                case 1:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,1,0,1) );
                    break;
                case 2:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,0,0,1) );
                    break;
                case 3:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    break;
                case 4:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,1,0,1) );
                    break;
                case 5:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,0,1,1) );
                    break;
                case 6:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,1,1,1) );
                    break;
                default:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ) );
            }

            vparams->drawTool()->popMatrix();
        }
    }
    vparams->drawTool()->restoreLastState();
}

#endif

#ifndef SOFA_DOUBLE
template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::applyRotation (const defaulttype::Quat q)
{
    helper::WriteAccessor< Data<VecCoord> > x = *this->write(core::VecCoordId::position());

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i].getCenter() = q.rotate(x[i].getCenter());
        x[i].getOrientation() = q * x[i].getOrientation();
    }
}


template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::addFromBaseVectorDifferentSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset )
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

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
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
                Real tmp=(Real)q_dest[j];
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
void MechanicalObject<defaulttype::Rigid3fTypes>::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset)
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

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
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
                Real tmp=(Real)q_dest[j];
                DataTypeInfo<Coord>::setValue(vDest[i], j+3, tmp);
            }
        }

        offset += vDest.size() * coordDim;
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
        offset += vDest.size() * derivDim;
    }

}

// template <>
//     bool MechanicalObject<Vec1fTypes>::addBBox(SReal* /*minBBox*/, SReal* /*maxBBox*/)
// {
//     return false; // ignore 1D DOFs for 3D bbox
// }

template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->saveLastState();
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
        helper::ReadAccessor<Data<VecCoord> > x = *this->read(core::VecCoordId::position());
        const size_t vsize = d_size.getValue();
        for (size_t i = 0; i < vsize; ++i)
        {
            vparams->drawTool()->pushMatrix();
            float glTransform[16];
            x[i].writeOpenGlMatrix ( glTransform );
            vparams->drawTool()->multMatrix( glTransform );
            vparams->drawTool()->scale ( scale );

            switch( drawMode.getValue() )
            {
                case 1:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,1,0,1) );
                    break;
                case 2:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,0,0,1) );
                    break;
                case 3:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    break;
                case 4:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,1,0,1) );
                    break;
                case 5:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,0,1,1) );
                    break;
                case 6:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,1,1,1) );
                    break;
                default:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ) );
            }

            vparams->drawTool()->popMatrix();
        }
    }
    vparams->drawTool()->restoreLastState();
}

#endif

}

} // namespace component

} // namespace sofa
