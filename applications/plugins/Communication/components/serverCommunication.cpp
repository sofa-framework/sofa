/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include "serverCommunication.inl"

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{

///// STD::STRING

template <>
void ServerCommunication<std::string>::handleEvent(Event * event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        pthread_mutex_lock(&mutex);
        for( size_t i=0 ; i < this->d_data.size(); ++i )
        {
            ReadAccessor<Data<std::string> > in = this->d_data[i];
            WriteAccessor<Data<std::string> > data = this->d_data_copy[i];
            std::stringstream stream;
            stream.str(in);
            stream >> data;
//            messageStream.str() >> data;

        }
        pthread_mutex_unlock(&mutex);

#if BENCHMARK
        // Uncorrect results if frequency == 1hz, due to tv_usec precision
        gettimeofday(&t1, NULL);
        if(d_refreshRate.getValue() <= 1.0)
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
        else
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);
#endif
    }
}

///// VEC3D

template <>
void ServerCommunication<vector<Vec3d>>::handleEvent(Event * event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        pthread_mutex_lock(&mutex);
        for( size_t i=0 ; i < this->d_data.size(); ++i )
        {
            ReadAccessor<Data<vector<Vec3d>>> in = this->d_data[i];
            WriteAccessor<Data<vector<Vec3d>>> data = this->d_data_copy[i];
            data.resize(in.size());
            for(unsigned int j = 0; j < in.size()/2; j++)
            {
                for(int k=0; k<3; k++)
                    data[j][k] = in[j][k];
            }
        }
        pthread_mutex_unlock(&mutex);

#if BENCHMARK
        // Uncorrect results if frequency == 1hz, due to tv_usec precision
        gettimeofday(&t1, NULL);
        if(d_refreshRate.getValue() <= 1.0)
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
        else
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);
#endif
    }
}

///// VEC3F

template <>
void ServerCommunication<vector<Vec3f>>::handleEvent(Event * event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        pthread_mutex_lock(&mutex);
        for( size_t i=0 ; i < this->d_data.size(); ++i )
        {
            ReadAccessor<Data<vector<Vec3f>>> in = this->d_data[i];
            WriteAccessor<Data<vector<Vec3f>>> data = this->d_data_copy[i];
            data.resize(in.size());
            for(unsigned int j = 0; j < in.size()/2; j++)
            {
                for(int k=0; k<3; k++)
                    data[j][k] = in[j][k];
            }
        }
        pthread_mutex_unlock(&mutex);


#if BENCHMARK
        // Uncorrect results if frequency == 1hz, due to tv_usec precision
        gettimeofday(&t1, NULL);
        if(d_refreshRate.getValue() <= 1.0)
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
        else
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);
#endif
    }
}

//////////////////////////////// Template name definition

template<>
std::string ServerCommunication<double>::templateName(const ServerCommunication<double>* object)
{
    SOFA_UNUSED(object);
    return "double";
}

template<>
std::string ServerCommunication<float>::templateName(const ServerCommunication<float>* object)
{
    SOFA_UNUSED(object);
    return "float";
}

template<>
std::string ServerCommunication<int>::templateName(const ServerCommunication<int>* object)
{
    SOFA_UNUSED(object);
    return "int";
}

template<>
std::string ServerCommunication<std::string>::templateName(const ServerCommunication<std::string>* object)
{
    SOFA_UNUSED(object);
    return "string";
}

template<>
std::string ServerCommunication<vector<Vec3d>>::templateName(const ServerCommunication<vector<Vec3d>>* object){
    SOFA_UNUSED(object);
    return "Vec3d";
}

template<>
std::string ServerCommunication<vector<Vec3f>>::templateName(const ServerCommunication<vector<Vec3f>>* object){
    SOFA_UNUSED(object);
    return "Vec3f";
}


///////////////////////////////////////////////////////////////////////////////////////////////////////

// Force template specialization for the most common sofa floating point related type.
// This goes with the extern template declaration in the .h. Declaring extern template
// avoid the code generation of the template for each compilation unit.
// see: http://www.stroustrup.com/C++11FAQ.html#extern-templates
#ifndef SOFA_FLOAT
template class SOFA_CORE_API ServerCommunication<float>;
template class SOFA_CORE_API ServerCommunication<vector<Vec3f>>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CORE_API ServerCommunication<double>;
template class SOFA_CORE_API ServerCommunication<vector<Vec3d>>;
#endif
template class SOFA_CORE_API ServerCommunication<int>;
template class SOFA_CORE_API ServerCommunication<std::string>;

} /// communication

} /// component

} /// sofa
