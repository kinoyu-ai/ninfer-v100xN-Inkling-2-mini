target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tokenizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/artifact/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/program.cpp)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
