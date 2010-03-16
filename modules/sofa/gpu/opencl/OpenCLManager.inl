#ifndef OPENCLMANAGER_INL
#define OPENCLMANAGER_INL
#include "OpenCLManager.h"


namespace sofa
{

namespace helper
{

int OpenCLManager::_numDevices = 0;

cl_context OpenCLManager::_context = NULL;

cl_command_queue* OpenCLManager::_queues = NULL;

cl_device_id* OpenCLManager::_devices = NULL;

cl_int OpenCLManager::_error=CL_SUCCESS;
}

}

#endif // OPENCLMANAGER_INL
