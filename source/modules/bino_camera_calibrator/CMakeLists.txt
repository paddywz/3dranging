cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(bino_camera_calibrator LANGUAGES CXX)
add_executable(${PROJECT_NAME})

find_package(OpenCV CONFIG REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "OpenCV found")
endif()

file(GLOB myInc ./include/*.h)
file(GLOB mySrc ./source/*.cpp)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/include/bino_camera_calibrator.h DESTINATION ${CMAKE_SOURCE_DIR}/include/3dranging/)

target_sources(
  ${PROJECT_NAME}
  PUBLIC ${myInc}
  PRIVATE ${mySrc})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
