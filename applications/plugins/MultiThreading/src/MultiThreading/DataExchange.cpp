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
#define SOFA_MULTITHREADING_PLUGIN_DATAEXCHANGE_CPP
#include <MultiThreading/DataExchange.inl>

#include <MultiThreading/config.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>

//#include <sofa/gpu/cuda/CudaTypes.h>



namespace sofa
{


	namespace defaulttype
	{

		template<> class DataTypeName< type::vector<sofa::type::Vec3d> > { public: static const std::string name() { return "vector<Vec3d>"; } };
		template<> class DataTypeName< type::vector<sofa::type::Vec2d> > { public: static const std::string name() { return "vector<Vec2d>"; } };
		template<> class DataTypeName< type::vector<double> > { public: static const std::string name() { return "vector<double>"; } };
		template<> class DataTypeName< type::vector<sofa::type::Vec3f> > { public: static const std::string name() { return "vector<Vec3f>"; } };
		template<> class DataTypeName< type::vector<sofa::type::Vec2f> > { public: static const std::string name() { return "vector<Vec2f>"; } };
		template<> class DataTypeName< type::vector<float> > { public: static const std::string name() { return "vector<float>"; } };
		template<> class DataTypeName<double> { public: static const std::string name() { return "double"; } };
		template<> class DataTypeName<float> { public: static const std::string name() { return "float"; } };
		template<> class DataTypeName< type::vector<int> > { public: static const std::string name() { return "vector<int>"; } };
		template<> class DataTypeName< type::vector<unsigned int> > { public: static const std::string name() { return "vector<unsigned_int>"; } };
		template<> class DataTypeName<bool> { public: static const std::string name() { return "bool"; } };
		
		//template<> struct DataTypeName< sofa::gpu::cuda::CudaVector<sofa::gpu::cuda::CudaVec2fTypes> > { static const std::string name() { return "cudavector<CudaVec2f>"; } };
		

	} // namespace defaulttype


	namespace core
	{

        SOFA_EVENT_CPP(DataExchangeEvent)

// Register in the Factory
int DataExchangeClass = core::RegisterObject("DataExchange")
.add< DataExchange< sofa::type::vector<sofa::type::Vec3d> > >(true)
.add< DataExchange< sofa::type::vector<sofa::type::Vec2d> > >()
.add< DataExchange< sofa::type::vector<double> > >()
.add< DataExchange< sofa::type::Vec3d > >()
.add< DataExchange< double > >()

.add< DataExchange< sofa::type::vector<sofa::type::Vec3f> > >()
.add< DataExchange< sofa::type::vector<sofa::type::Vec2f> > >()
.add< DataExchange< sofa::type::vector<float> > >()
.add< DataExchange< sofa::type::Vec3f > >()
.add< DataExchange< float > >()

.add< DataExchange< sofa::type::vector<int> > >()
.add< DataExchange< sofa::type::vector<unsigned int> > >()
.add< DataExchange< bool > >()

//.add< DataExchange< sofa::gpu::cuda::CudaVector<sofa::gpu::cuda::CudaVec2fTypes> >()
;




template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec3d> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec2d> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<double> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::Vec3d >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< double >;

template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec3f> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec2f> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<float> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::Vec3f >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< float >;

template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<int> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<unsigned int> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< bool >;

//template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::gpu::cuda::CudaVector<sofa::gpu::cuda::CudaVec2fTypes> >;



	} // namespace core

} // namespace sofa
