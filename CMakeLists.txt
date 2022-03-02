cmake_minimum_required(VERSION 2.8.3)

project(ipm)

if(${CMAKE_VERSION} VERSION_LESS "3.2.0")
  set( CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}" )
else()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Glog_INCLUDE_DIR}
)

set(THIRD_PARTY_LIBS
  ${OpenCV_LIBS}
  ${GLOG_LIBRARY}
)

FILE(GLOB HRD_FILES "include/ipm_based_on_vp/*.h*")
FILE(GLOB SRC_FILES "src/*.c*")

add_library(${PROJECT_NAME} SHARED
  ${HRD_FILES}
  ${SRC_FILES}
)
target_link_libraries(${PROJECT_NAME}
  ${THIRD_PARTY_LIBS}
)

add_executable(test_ipm 
  test/test_ipm.cpp)
target_link_libraries(test_ipm
  ${PROJECT_NAME}
)