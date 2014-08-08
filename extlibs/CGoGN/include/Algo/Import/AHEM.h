#ifndef __AHEM_H__
#define __AHEM_H__




#include <string>



#ifdef _WIN32

typedef __int8				stInt8;
typedef __int16				stInt16;
typedef	__int32				stInt32;
typedef	__int64				stInt64;

typedef unsigned __int8		stUInt8;
typedef unsigned __int16	stUInt16;
typedef	unsigned __int32	stUInt32;
typedef	unsigned __int64	stUInt64;


#else

#include <stdint.h>

typedef int8_t		stInt8;
typedef int16_t		stInt16;
typedef	int32_t		stInt32;
typedef	int64_t		stInt64;

typedef uint8_t		stUInt8;
typedef uint16_t	stUInt16;
typedef	uint32_t	stUInt32;
typedef	uint64_t	stUInt64;

#endif


#ifdef _WIN32
#include <InitGuid.h>
#define DEFINE_GUID_INFUNC(name, dw, w0, w1, b0, b1, b2, b3, b4, b5, b6, b7)	static const GUID name = { dw, w0, w1, { b0, b1, b2, b3, b4, b5, b6, b7 } }
#else
#include <uuid/uuid.h>
#define DEFINE_GUID(name, dw, w0, w1, b0, b1, b2, b3, b4, b5, b6, b7)  UUID_DEFINE(name, dw & 0xff, (dw >> 8) & 0xff, (dw >> 16) & 0xff, (dw >> 24) & 0xff, w0 & 0xff, (w0 >> 8) & 0xff, w1 & 0xff, (w1 >> 8) & 0xff, b0, b1, b2, b3, b4, b5, b6, b7)
#define DEFINE_GUID_INFUNC DEFINE_GUID
typedef uuid_t GUID;
#endif


inline bool IsEqualGUID(const stUInt8* v1, const GUID& v2)
{
	const stUInt32* w1 = (stUInt32*)v1;
	const stUInt32* w2 = (stUInt32*)&v2;

	return w1[0] == w2[0] && w1[1] == w2[1] && w1[2] == w2[2] && w1[3] == w2[3];
}



// {1B8E15D6-FE16-44E2-BDC8-9FBA3420B267}
DEFINE_GUID(AHEMDATATYPE_INT8, 0x1b8e15d6, 0xfe16, 0x44e2, 0xbd, 0xc8, 0x9f, 0xba, 0x34, 0x20, 0xb2, 0x67);
// {3D772789-9D2A-4527-8709-5B3E08BDD071}
DEFINE_GUID(AHEMDATATYPE_UINT8, 0x3d772789, 0x9d2a, 0x4527, 0x87, 0x9, 0x5b, 0x3e, 0x8, 0xbd, 0xd0, 0x71);
// {E6D2DD95-A4EF-4EA3-82FF-12F4BAB40C51}
DEFINE_GUID(AHEMDATATYPE_INT16, 0xe6d2dd95, 0xa4ef, 0x4ea3, 0x82, 0xff, 0x12, 0xf4, 0xba, 0xb4, 0xc, 0x51);
// {5A926579-9586-413B-BEF5-992E0A9CA8E5}
DEFINE_GUID(AHEMDATATYPE_UINT16, 0x5a926579, 0x9586, 0x413b, 0xbe, 0xf5, 0x99, 0x2e, 0xa, 0x9c, 0xa8, 0xe5);
// {717210F9-4AB4-4ED3-9ACC-8AC89AA6BE55}
DEFINE_GUID(AHEMDATATYPE_INT32, 0x717210f9, 0x4ab4, 0x4ed3, 0x9a, 0xcc, 0x8a, 0xc8, 0x9a, 0xa6, 0xbe, 0x55);
// {9EA6E534-B251-44C7-B40C-C33CCCD6C18F}
DEFINE_GUID(AHEMDATATYPE_UINT32, 0x9ea6e534, 0xb251, 0x44c7, 0xb4, 0xc, 0xc3, 0x3c, 0xcc, 0xd6, 0xc1, 0x8f);
// {F10D01EF-D677-453B-B7A7-E3C85E56A18D}
DEFINE_GUID(AHEMDATATYPE_INT64, 0xf10d01ef, 0xd677, 0x453b, 0xb7, 0xa7, 0xe3, 0xc8, 0x5e, 0x56, 0xa1, 0x8d);
// {C0FD73C4-8A06-4CB0-86D4-1B305BCEC3C7}
DEFINE_GUID(AHEMDATATYPE_UINT64, 0xc0fd73c4, 0x8a06, 0x4cb0, 0x86, 0xd4, 0x1b, 0x30, 0x5b, 0xce, 0xc3, 0xc7);
// {2BD6E606-E3E3-4873-AA23-21D552EB11B3}
DEFINE_GUID(AHEMDATATYPE_FLOAT32, 0x2bd6e606, 0xe3e3, 0x4873, 0xaa, 0x23, 0x21, 0xd5, 0x52, 0xeb, 0x11, 0xb3);
// {4CC5D944-4336-49B0-901D-842E30E87D3E}
DEFINE_GUID(AHEMDATATYPE_FLOAT64, 0x4cc5d944, 0x4336, 0x49b0, 0x90, 0x1d, 0x84, 0x2e, 0x30, 0xe8, 0x7d, 0x3e);



// {0FDAF25D-2389-4A6D-BE3B-F8C00D118F23}
DEFINE_GUID(AHEMATTRIBUTE_POSITION, 0xfdaf25d, 0x2389, 0x4a6d, 0xbe, 0x3b, 0xf8, 0xc0, 0xd, 0x11, 0x8f, 0x23);




#define AHEM_MAGIC 0x4148454D

#ifdef _WIN32
typedef enum : stUInt32
#else
typedef enum
#endif
{
	AHEMATTROWNER_UNKNOWN		= 0,
	AHEMATTROWNER_HALFEDGE		= 1,
	AHEMATTROWNER_VERTEX		= 2,
	AHEMATTROWNER_FACE			= 3,
	AHEMATTROWNER_HE_FACECORNER	= 4,
#ifndef _WIN32
	AHEMATTROWNER_FORCE_DWORD	= 0xffffffff
#endif
} AHEMAttributeOwner;


#pragma pack(push, 1)

typedef struct
{
	stUInt32 meshChunkSize;

	stUInt32 heCount;
	stUInt32 vxCount;
	stUInt32 faceCount;

	stUInt32 faceMaxSize;
} AHEMTopologyHeader;


typedef struct
{
	stUInt32			fileStartOffset;
	stUInt32			attributeChunkSize;

	stUInt8				semantic[16];			// Semantic GUID
	
	AHEMAttributeOwner	owner;	
	stUInt32			attrPodSize;			// 0 means non-pod-stored attribute
	stUInt8				dataType[16];			// datatype GUID
	stUInt32			dimension;
	
	stUInt32			nameSize;
} AHEMAttributeDescriptor;



#define AHEM_MAGIC 0x4148454D

typedef struct
{
	stUInt32			magic;
	stUInt32			version;

	AHEMTopologyHeader	meshHdr;
	stUInt32			meshFileStartOffset;

	stUInt32			attributesChunkNumber;
} AHEMHeader;


typedef struct
{
	stUInt32 batchLength;
	stUInt32 batchFaceSize;
} AHEMFaceBatchDescriptor;


#pragma pack(pop)




#endif			// __AHEM_H__
