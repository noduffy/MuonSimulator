.PHONY: build clean-build setting clean-outputs run simple cgls recon3d full3d all all-test check

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

simple:
	@$(PY) scripts/generate_pairs.py --hit build/hits.csv --out pairs.csv
	@$(PY) scripts/separate_muons.py
	@$(PY) scripts/plot_poca_3d_simple.py

cgls:
	@$(PY) scripts/generate_pairs.py --hit build/hits.csv --out pairs.csv
	@$(PY) scripts/separate_muons.py
	@$(PY) scripts/build_system_matrix.py
	@$(PY) scripts/recon_cgls_3d.py
	@$(PY) scripts/recon_cgls_3d_progressive.py
	
#	@$(PY) scripts/render_recon_x_3d.py

recon3d:
	@$(PY) scripts/generate_pairs.py --hit build/hits.csv --out pairs.csv
	@$(PY) scripts/rays_from_pairs.py build/outputs/pairs.csv rays.csv
	@$(PY) scripts/build_W3D.py --pairs build/outputs/pairs.csv --rays build/outputs/rays.csv --grid3d $(GRID3D) --outW W3D_coo.npz --outy y_theta2_3d.npy
	@$(PY) scripts/recon_mlem3d.py --iters $(ITERS)
	@$(PY) scripts/render_recon3d_png.py
	@$(PY) scripts/render_recon3d_view.py --view side --out build/outputs/recon3d_side.png

recon3d-test:
	@$(PY) scripts/generate_pairs.py --hit build/hits.csv --out pairs.csv
	@$(PY) scripts/rays_from_pairs.py build/outputs/pairs.csv rays.csv
	@$(PY) scripts/build_W3D_poca.py --pairs build/outputs/pairs.csv --rays build/outputs/rays.csv --grid3d $(GRID3D) --outW W3D_poca.npz --outy y_theta2_3d.npy
	@$(PY) scripts/recon_mlem3d.py --iters $(ITERS) --W W3D_poca.npz
	@$(PY) scripts/render_recon3d_view.py --vol build/outputs/recon3d_vol.npy --out recon3d_render_poca.png
	@$(PY) scripts/render_recon3d_view.py --vol build/outputs/recon3d_vol.npy --view side --out recon3d_side_poca.png

full3d: run recon3d

all: clean-build build run

all-test: clean-build build run recon3d-test

check:
	@$(PY) scripts/inspect_x_values.py