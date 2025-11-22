.PHONY: build clean-build setting clean-outputs run recon3d full3d all

# 可変パラメータ（必要に応じて上書き可能）
PY      ?= python3
ITERS   ?= 200
GRID3D  ?= configs/grid3d.yml

clean-build:
	@rm -rf build
	@mkdir build

build:
	@cd build && \
	CMAKE_IGNORE_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib" \
	CMAKE_INCLUDE_PATH="/opt/homebrew/opt/expat/include" \
	CMAKE_LIBRARY_PATH="/opt/homebrew/opt/expat/lib" \
	cmake .. -DGeant4_DIR="$$(echo $$HOME)/opt/geant4-11.3.2-min/lib/cmake/Geant4" && \
	$(MAKE) -j2

run:
	@cd build && ./mygeom ../macros/run.mac

recon3d:
	@$(PY) scripts/generate_pairs.py --hit build/hits.csv --out pairs.csv --policy first
	@$(PY) scripts/rays_from_pairs.py build/outputs/pairs.csv rays.csv
	@$(PY) scripts/build_W3D.py --pairs build/outputs/pairs.csv --rays build/outputs/rays.csv --grid3d $(GRID3D) --outW W3D_coo.npz --outy y_theta2_3d.npy
	@$(PY) scripts/recon_mlem3d.py --iters $(ITERS)
	@$(PY) scripts/render_recon3d_png.py
	@$(PY) scripts/render_recon3d_view.py --view side --out build/outputs/recon3d_side.png

full3d: run recon3d

all: clean-build build run recon3d
