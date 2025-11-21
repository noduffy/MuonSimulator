.PHONY: build clean-outputs run recon3d full3d

# 可変パラメータ（必要に応じて上書き可能）
PY      ?= python3
ITERS   ?= 200
GRID3D  ?= configs/grid3d.yml

build:
	@cmake -S . -B build
	@cmake --build build -j2

clean-outputs:
	@mkdir -p build/outputs
	@find build/outputs -mindepth 1 -maxdepth 1 -delete

run: build clean-outputs
	@cd build && ./mygeom ../macros/run.mac

recon3d:
	@$(PY) scripts/generate_pairs.py --hit build/hits.csv --out pairs.csv --policy first
	@$(PY) scripts/rays_from_pairs.py build/outputs/pairs.csv rays.csv
	@$(PY) scripts/build_W3D.py --pairs build/outputs/pairs.csv --rays build/outputs/rays.csv --grid3d $(GRID3D) --outW W3D_coo.npz --outy y_theta2_3d.npy
	@$(PY) scripts/recon_mlem3d.py --iters $(ITERS)
	@$(PY) scripts/render_recon3d_png.py
	@$(PY) scripts/render_recon3d_view.py --view side --out recon3d_side.png

full3d: run recon3d
