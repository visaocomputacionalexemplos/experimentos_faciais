cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

message("Compilando experimentos")

project(vc_identificar_piscada VERSION 1.0.0 DESCRIPTION "VisaoComputacionalExperimentos")

#c++11flags
if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 11)
endif ()

#dependencies
find_package(OpenCV REQUIRED)

#includes
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include)


#file to builder
add_executable(vc_identificar_piscada
    main.cpp
    utils.cpp
    detectar_rosto.cpp
    piscadas_facemark_lbf.cpp
    piscadas_facemark_aam.cpp
    piscadas_facemark_kazemi.cpp
)

#add ide libraries
target_link_libraries(vc_identificar_piscada
    opencv_core
    opencv_imgproc
    opencv_videoio
    opencv_highgui
    opencv_objdetect
    opencv_face)

install(TARGETS vc_identificar_piscada)
