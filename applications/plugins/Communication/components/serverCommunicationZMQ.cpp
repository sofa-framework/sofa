#ifndef SOFA_SERVERCOMMUNICATIONZMQ_CPP
#define SOFA_SERVERCOMMUNICATIONZMQ_CPP

#include "serverCommunicationZMQ.inl"

namespace sofa
{

namespace component
{

namespace communication
{

///// VEC3D


template<>
void ServerCommunicationZMQ<vector<Vec3d>>::convertDataToMessage(std::string& messageStr)
{
    for(unsigned int i=0; i<d_data.size(); i++)
    {
        ReadAccessor<Data<vector<Vec3d>>> data = d_data[i];
        messageStr += std::to_string(data.size()) + " ";
        for(unsigned int j=0; j<data.size(); j++)
            for(int k=0; k<3; k++)
                messageStr += std::to_string(data[j][k]) + " ";
    }
}

template<>
void ServerCommunicationZMQ<vector<Vec3d>>::convertStringStreamToData(std::stringstream *stream)
{
    for (unsigned int i= 0; i<d_data.size(); i++)
    {
        WriteAccessor<Data<vector<Vec3d>>> data = d_data[i];
        int dataSize = 0;
        (*stream) >> dataSize;
        data.resize(dataSize);

        for(unsigned int j=0; j<data.size(); j++)
            for(int k=0; k<3; k++)
                (*stream) >> data[j][k];
    }
}

template<>
void ServerCommunicationZMQ<vector<Vec3d>>::checkDataSize(const unsigned int& nbDataFieldReceived)
{
    if(d_data.size()>0)
    {
        ReadAccessor<Data<vector<Vec3d>>> data = d_data[0];
        if(nbDataFieldReceived!=d_nbDataField.getValue()*4*data.size())
            msg_warning(this) << "Something wrong with the size of data received. Please check template.";
    }
}


///// VEC3F


template<>
void ServerCommunicationZMQ<vector<Vec3f>>::convertDataToMessage(std::string& messageStr)
{
    for(unsigned int i=0; i<d_data.size(); i++)
    {
        ReadAccessor<Data<vector<Vec3f>>> data = d_data[i];
        messageStr += std::to_string(data.size()) + " ";
        for(unsigned int j=0; j<data.size(); j++)
            for(int k=0; k<3; k++)
                messageStr += std::to_string(data[j][k]) + " ";
    }
}

template<>
void ServerCommunicationZMQ<vector<Vec3f>>::convertStringStreamToData(std::stringstream *stream)
{
    for (unsigned int i= 0; i<d_data.size(); i++)
    {
        WriteAccessor<Data<vector<Vec3f>>> data = d_data[i];
        int dataSize = 0;
        (*stream) >> dataSize;
        data.resize(dataSize);

        for(unsigned int j=0; j<data.size(); j++)
            for(int k=0; k<3; k++)
                (*stream) >> data[j][k];
    }
}

template<>
void ServerCommunicationZMQ<vector<Vec3f>>::checkDataSize(const unsigned int& nbDataFieldReceived)
{
    if(d_data.size()>0)
    {
        ReadAccessor<Data<vector<Vec3f>>> data = d_data[0];
        if(nbDataFieldReceived!=d_nbDataField.getValue()*4*data.size())
            msg_warning(this) << "Something wrong with the size of data received. Please check template.";
    }
}


///// STRING

template<>
void ServerCommunicationZMQ<std::string>::convertDataToMessage(std::string& messageStr)
{
    for(unsigned int i=0; i<this->d_data_copy.size(); i++)
    {
        ReadAccessor<Data<std::string>> data = this->d_data_copy[i];
        std::string tmp = data;
        tmp += " ";
        messageStr += tmp;
    }
}

//////////////////////////////// Template name definition

template<>
std::string ServerCommunicationZMQ<double>::templateName(const ServerCommunicationZMQ<double>* object){
    SOFA_UNUSED(object);
    return "double";
}

template<>
std::string ServerCommunicationZMQ<float>::templateName(const ServerCommunicationZMQ<float>* object){
    SOFA_UNUSED(object);
    return "float";
}

template<>
std::string ServerCommunicationZMQ<int>::templateName(const ServerCommunicationZMQ<int>* object){
    SOFA_UNUSED(object);
    return "int";
}

template<>
std::string ServerCommunicationZMQ<unsigned int>::templateName(const ServerCommunicationZMQ<unsigned int>* object){
    SOFA_UNUSED(object);
    return "unsigned int";
}

template<>
std::string ServerCommunicationZMQ<vector<Vec3d>>::templateName(const ServerCommunicationZMQ<vector<Vec3d>>* object){
    SOFA_UNUSED(object);
    return "Vec3d";
}

template<>
std::string ServerCommunicationZMQ<vector<Vec3f>>::templateName(const ServerCommunicationZMQ<vector<Vec3f>>* object){
    SOFA_UNUSED(object);
    return "Vec3f";
}

template<>
std::string ServerCommunicationZMQ<std::string>::templateName(const ServerCommunicationZMQ<std::string>* object)
{
    SOFA_UNUSED(object);
    return "string";
}



////////////////////////////////////////////    FACTORY    ////////////////////////////////////////////
using sofa::core::RegisterObject ;

// Registering the component
// see: http://wiki.sofa-framework.org/wiki/ObjectFactory
// 1-SOFA_DECL_CLASS(componentName) : Set the class name of the component
// 2-RegisterObject("description") + .add<> : Register the component
// 3-.add<>(true) : Set default template
SOFA_DECL_CLASS(ServerCommunicationZMQ)

int ServerCommunicationZMQClass = RegisterObject("This component is used to build a communication between two simulations")
#ifdef SOFA_WITH_DOUBLE
.add< ServerCommunicationZMQ<double> >(true)
.add< ServerCommunicationZMQ<vector<Vec3d>> >()
#endif
#ifdef SOFA_WITH_FLOAT
.add< ServerCommunicationZMQ<float> >()
.add< ServerCommunicationZMQ<vector<Vec3f>> >()
#endif
.add< ServerCommunicationZMQ<int> >()
.add< ServerCommunicationZMQ<std::string> >(true)
;

///////////////////////////////////////////////////////////////////////////////////////////////////////

// Force template specialization for the most common sofa floating point related type.
// This goes with the extern template declaration in the .h. Declaring extern template
// avoid the code generation of the template for each compilation unit.
// see: http://www.stroustrup.com/C++11FAQ.html#extern-templates
#ifdef SOFA_WITH_DOUBLE
template class SOFA_COMMUNICATION_API ServerCommunicationZMQ<double>;
template class SOFA_COMMUNICATION_API ServerCommunicationZMQ<vector<Vec3d>>;
#endif
#ifdef SOFA_WITH_FLOAT
template class SOFA_COMMUNICATION_API ServerCommunicationZMQ<float>;
template class SOFA_COMMUNICATION_API ServerCommunicationZMQ<vector<Vec3f>>;
#endif
template class SOFA_COMMUNICATION_API ServerCommunicationZMQ<int>;
template class SOFA_COMMUNICATION_API ServerCommunicationZMQ<std::string >;


}   //namespace controller
}   //namespace component
}   //namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationZMQ_CPP

