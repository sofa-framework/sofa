#include "DataExchange.inl"

#include <MultiThreading/config.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>

//#include <sofa/gpu/cuda/CudaTypes.h>



namespace sofa
{


	namespace defaulttype
	{

		template<> struct DataTypeName< helper::vector<sofa::defaulttype::Vec3d> > { static const char* name() { return "vector<Vec3d>"; } };
		template<> struct DataTypeName< helper::vector<sofa::defaulttype::Vec2d> > { static const char* name() { return "vector<Vec2d>"; } };
		template<> struct DataTypeName< helper::vector<double> > { static const char* name() { return "vector<double>"; } };
		template<> struct DataTypeName< helper::vector<sofa::defaulttype::Vec3f> > { static const char* name() { return "vector<Vec3f>"; } };
		template<> struct DataTypeName< helper::vector<sofa::defaulttype::Vec2f> > { static const char* name() { return "vector<Vec2f>"; } };
		template<> struct DataTypeName< helper::vector<float> > { static const char* name() { return "vector<float>"; } };
		template<> struct DataTypeName<double> { static const char* name() { return "double"; } };
		template<> struct DataTypeName<float> { static const char* name() { return "float"; } };
		template<> struct DataTypeName< helper::vector<int> > { static const char* name() { return "vector<int>"; } };
		template<> struct DataTypeName< helper::vector<unsigned int> > { static const char* name() { return "vector<unsigned_int>"; } };
		template<> struct DataTypeName<bool> { static const char* name() { return "bool"; } };
		
		//template<> struct DataTypeName< sofa::gpu::cuda::CudaVector<sofa::gpu::cuda::CudaVec2fTypes> > { static const char* name() { return "cudavector<CudaVec2f>"; } };
		

	} // namespace defaulttype


	namespace core
	{

        SOFA_EVENT_CPP(DataExchangeEvent)

		SOFA_DECL_CLASS(DataExchange)

// Register in the Factory
int DataExchangeClass = core::RegisterObject("DataExchange")
.add< DataExchange< sofa::helper::vector<sofa::defaulttype::Vec3d> > >(true)
.add< DataExchange< sofa::helper::vector<sofa::defaulttype::Vec2d> > >()
.add< DataExchange< sofa::helper::vector<double> > >()
.add< DataExchange< sofa::defaulttype::Vec3d > >()
.add< DataExchange< double > >()

.add< DataExchange< sofa::helper::vector<sofa::defaulttype::Vec3f> > >()
.add< DataExchange< sofa::helper::vector<sofa::defaulttype::Vec2f> > >()
.add< DataExchange< sofa::helper::vector<float> > >()
.add< DataExchange< sofa::defaulttype::Vec3f > >()
.add< DataExchange< float > >()

.add< DataExchange< sofa::helper::vector<int> > >()
.add< DataExchange< sofa::helper::vector<unsigned int> > >()
.add< DataExchange< bool > >()
//.add< DataExchange< sofa::gpu::cuda::CudaVector<sofa::gpu::cuda::CudaVec2fTypes> >()
;




template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<sofa::defaulttype::Vec3d> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<sofa::defaulttype::Vec2d> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<double> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::defaulttype::Vec3d >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< double >;

template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<sofa::defaulttype::Vec3f> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<sofa::defaulttype::Vec2f> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<float> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::defaulttype::Vec3f >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< float >;

template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<int> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::helper::vector<unsigned int> >;
template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< bool >;
//template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::gpu::cuda::CudaVector<sofa::gpu::cuda::CudaVec2fTypes> >;



	} // namespace core

} // namespace sofa
