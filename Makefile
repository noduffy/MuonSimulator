.PHONY: build clean-build setting clean-outputs run make-csv cgls mix make-maps prob-map fusion all check

# --- 設定変数 ---
PY      ?= python3
ITERS   ?= 200
GRID3D  ?= configs/grid3d.yml

# --- ディレクトリ定義 ---
SRC_PRE   = scripts/preprocessing
SRC_REC   = scripts/reconstruction
SRC_VIS   = scripts/visualization
SRC_FUS   = scripts/fusion

# ==============================================================================
# 1. ビルド & 実行
# ==============================================================================
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

all: clean-build build run

# ==============================================================================
# 2. データ生成
# ==============================================================================
make-csv:
	@echo "--- Generating Pairs ---"
	@$(PY) $(SRC_PRE)/generate_pairs.py --hit build/hits.csv --out pairs.csv
	@echo "--- Separating Muons ---"
	@$(PY) $(SRC_PRE)/separate_muons.py

# ==============================================================================
# 3. 再構成 (通常手法)
# ==============================================================================
cgls:
	@echo "--- Building System Matrix ---"
	@$(PY) $(SRC_REC)/build_system_matrix.py --input scattered_muons.csv --mode poca
	@echo "--- Running Reconstruction ---"
	@$(PY) $(SRC_REC)/recon_cgls_3d_progressive.py --out_dir progressive_cgls

# ==============================================================================
# 4. 手法C (Fusion / Probability Map)
# ==============================================================================

# make-maps: 分子(flux_straight.npy) と 分母(flux_all.npy) を作成する
# ※ ratio=1.0 (全データ使用) にしないと確率が狂うので注意
# Step 1: Flux Mapの作成
make-maps:
	@echo "--- Creating Flux Map (All) ---"
	@$(PY) $(SRC_FUS)/build_flux_map.py --scat scattered_muons.csv --straight straight_muons.csv --ratio 1.0 --out flux_all.npy
	@echo "--- Creating Flux Map (Straight Only) ---"
	@$(PY) $(SRC_FUS)/build_flux_map.py --straight straight_muons.csv --ratio 1.0 --out flux_straight.npy

# Step 2: 確率マップ(図2)の作成
prob-map:
	@echo "--- Calculating Probability Map (Method B) ---"
	@$(PY) $(SRC_FUS)/calc_prob_map.py \
		--all flux_all.npy \
		--straight flux_straight.npy \
		--out_npy prob_map.npy \
		--out_png prob_map_render.png

# Step 3: 画像融合(図3) - Method C
# 依存関係: cgls(Method A) と prob-map(Method B) が完了していること
fusion:
	@echo "--- Fusing Images (Method C) ---"
	@$(PY) $(SRC_FUS)/fuse_images.py \
		--scat progressive_cgls/x_iter_0020.npy \
		--prob prob_map.npy \
		--out_npy fused_result.npy \
		--out_png fused_render.png
	@echo "[Done] Method C completed."

# 手法D: 確率マップを事前情報として用いたCGLS
method-d:
	@echo "--- Running Method D (Constrained Reconstruction) ---"
	@$(PY) scripts/reconstruction/recon_method_d.py \
		--prob_map prob_map.npy \
		--out_dir method_d_result \
		--max_iter 100 \
		--interval 10

# ==============================================================================
# ユーティリティ
# ==============================================================================
clean-outputs:
	@rm -rf build/outputs/*

vis-setup:
	@echo "--- Visualizing Simulation Setup ---"
	@$(PY) scripts/visualization/visualize_setup_3d.py --out setup_render.png

check:
	@$(PY) scripts/inspect_x_values.py