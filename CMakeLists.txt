cmake_minimum_required(VERSION 3.16)
project(mygeom LANGUAGES CXX)

# 基本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# macOS のランタイムパス（@rpathの補助）
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# Geant4（UIが必要。可視化も使うときは REQUIRED ui vis）
find_package(Geant4 REQUIRED ui)
include(${Geant4_USE_FILE})

# 実行ファイル
add_executable(mygeom
  src/main.cc
  src/DetectorConstruction.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SteppingAction.cc
)

# リンク
target_link_libraries(mygeom PRIVATE ${Geant4_LIBRARIES})

# 実行時に Geant4 の .dylib を見つけやすく（あなたの “min” インストール先に合わせる）
if(APPLE AND EXISTS "$ENV{HOME}/opt/geant4-11.3.2-min/lib")
  set(_g4lib "$ENV{HOME}/opt/geant4-11.3.2-min/lib")
  set_target_properties(mygeom PROPERTIES
    BUILD_RPATH "${_g4lib}"
    INSTALL_RPATH "${_g4lib}")
endif()

# ★ 重要：src/run.mac をコピーしない！
# macros/ ディレクトリがあれば build 側へ丸ごとコピー（実行時は ./macros/run.mac を使う）
if(EXISTS "${CMAKE_SOURCE_DIR}/macros")
  add_custom_target(copy_macros ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/macros
            ${CMAKE_BINARY_DIR}/macros)
  add_dependencies(mygeom copy_macros)
endif()

# デバッグ表示
message(STATUS "Geant4_DIR = ${Geant4_DIR}")
message(STATUS "Using Geant4 libs: ${Geant4_LIBRARIES}")
