# - Try to find ffmpeg executable
# Once done this will define
#
# FFMPEG_EXEC_FOUND - system has ffmpeg executable
# FFMPEG_EXEC_FILE - the ffmpeg executable file

find_program(FFMPEG_EXEC
	NAMES ffmpeg
	HINTS ${_FFMPEG_EXEC_DIRS} /usr/bin /usr/local/bin /opt/local/bin /sw/bin
)

if (FFMPEG_EXEC )
	if (NOT FFMPEG_exec_FIND_QUIETLY)
        message(STATUS "-- FFmpeg executable was found  "  ${FFMPEG_EXEC})
	endif()
    set(FFMPEG_EXEC_FOUND TRUE)
	set(FFMPEG_EXEC_FILE "${FFMPEG_EXEC}")
endif()