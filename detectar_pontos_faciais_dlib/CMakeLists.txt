cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

message("Compilando experimentos")

project(vc_detectar_pontos_faciais_dlib VERSION 1.0.0 DESCRIPTION "VisaoComputacionalExperimentos")

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
find_package(dlib REQUIRED)

#includes
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include)


#file to builder
add_executable(vc_detectar_pontos_faciais_dlib
    main.cpp
    detectar_rosto.cpp
    detectar_pontos_faciais_aam.cpp
)

#add ide libraries
target_link_libraries(vc_detectar_pontos_faciais_dlib
    dlib::dlib
    opencv_core
    opencv_videoio
    opencv_highgui)

install(TARGETS vc_detectar_pontos_faciais_dlib)
