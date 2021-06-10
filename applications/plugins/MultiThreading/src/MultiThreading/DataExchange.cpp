#include "DataExchange.inl"

#include <MultiThreading/config.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>

//#include <sofa/gpu/cuda/CudaTypes.h>



namespace sofa
{


	namespace defaulttype
	{

		template<> struct DataTypeName< type::vector<sofa::type::Vec3d> > { static const char* name() { return "vector<Vec3d>"; } };
		template<> struct DataTypeName< type::vector<sofa::type::Vec2d> > { static const char* name() { return "vector<Vec2d>"; } };
		template<> struct DataTypeName< type::vector<double> > { static const char* name() { return "vector<double>"; } };
		template<> struct DataTypeName< type::vector<sofa::type::Vec3f> > { static const char* name() { return "vector<Vec3f>"; } };
		template<> struct DataTypeName< type::vector<sofa::type::Vec2f> > { static const char* name() { return "vector<Vec2f>"; } };
		template<> struct DataTypeName< type::vector<float> > { static const char* name() { return "vector<float>"; } };
		template<> struct DataTypeName<double> { static const char* name() { return "double"; } };
		template<> struct DataTypeName<float> { static const char* name() { return "float"; } };
		template<> struct DataTypeName< type::vector<int> > { static const char* name() { return "vector<int>"; } };
		template<> struct DataTypeName< type::vector<unsigned int> > { static const char* name() { return "vector<unsigned_int>"; } };
		template<> struct DataTypeName<bool> { static const char* name() { return "bool"; } };
		
		//template<> struct DataTypeName< sofa::gpu::cuda::CudaVector<sofa::gpu::cuda::CudaVec2fTypes> > { static const char* name() { return "cudavector<CudaVec2f>"; } };
		

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
