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
    extern int mycudaInit(int device);
    extern void mycudaMalloc(void **devPtr, size_t size);
    extern void mycudaFree(void *devPtr);
    extern void mycudaMallocHost(void **hostPtr, size_t size);
    extern void mycudaFreeHost(void *hostPtr);
//extern void mycudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern void mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToHost(void *dst, const void *src, size_t count);

    extern void mycudaLogError(int err, const char* src);
    extern int myprintf(const char* fmt, ...);

    extern void mycudaGLRegisterBufferObject(int id);
    extern void mycudaGLUnregisterBufferObject(int id);

    extern void mycudaGLMapBufferObject(void** ptr, int id);
    extern void mycudaGLUnmapBufferObject(int id);

}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
