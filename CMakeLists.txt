cmake_minimum_required (VERSION 2.8.11)
project (merging)

add_definitions("-std=c++11")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package( Boost COMPONENTS system filesystem REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions(${PCL_DEFINITIONS})

SET( SOURCES
    Frame3D/FileUtils.cpp
    Frame3D/FileUtils.h
    Frame3D/Frame3D.cpp
    Frame3D/Frame3D.h
    main.cpp
)

add_executable (merge ${SOURCES})

target_link_libraries( merge
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)

