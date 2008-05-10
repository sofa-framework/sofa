#ifndef MYCUDA_H
#define MYCUDA_H

#include <string.h>

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C" {
    extern int mycudaInit(int device=-1);
    extern void mycudaMalloc(void **devPtr, size_t size);
    extern void mycudaMallocPitch(void **devPtr, size_t* pitch, size_t width, size_t height);
    extern void mycudaFree(void *devPtr);
    extern void mycudaMallocHost(void **hostPtr, size_t size);
    extern void mycudaFreeHost(void *hostPtr);
//extern void mycudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern void mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToHost(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyHostToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    extern void mycudaMemcpyDeviceToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    extern void mycudaMemcpyDeviceToHost2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);

    extern void mycudaGLRegisterBufferObject(int id);
    extern void mycudaGLUnregisterBufferObject(int id);

    extern void mycudaGLMapBufferObject(void** ptr, int id);
    extern void mycudaGLUnmapBufferObject(int id);

    extern void mycudaLogError(int err, const char* src);
    extern int myprintf(const char* fmt, ...);
    extern const char* mygetenv(const char* name);

}

enum MycudaVerboseLevel
{
    LOG_NONE = 0,
    LOG_ERR = 1,
    LOG_INFO = 2,
    LOG_TRACE = 3
};

extern MycudaVerboseLevel mycudaVerboseLevel;

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
