# ninfer-v100xN

[English](README.md) · [中文](README.zh-CN.md) · [Inkling-2-Mini](docs/inkling2-mini.md) · [GUI](docs/gui.md) · [TUI](docs/tui.md) · [Docker](docs/docker.md) · [詳細ベンチマーク](docs/performance.md) · [HTTP API](docs/serving.md)

## フォーク追加機能: ネイティブ・ロスレス Inkling-2-Mini (`inkling2-mini/nvfp4-mxfp8`)

[`tuxKOH/ninfer-v100xN`](https://github.com/tuxKOH/ninfer-v100xN) をベースとする本フォークは、**Codex** と **Antigravity (Agy)** を用いて開発・最適化しました。[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)（論理 48 ループ層、Layer 0–19 NVFP4 + Layer 20–23 MXFP8 + BF16 dense/KV + BF16 DSpark 投機的ドラフター、再量子化なし）に対するネイティブ・ロスレス対応と Volta `sm_70` 最適化を追加しています。

**2 × Tesla V100-SXM2 16 GB**（PCIe P2P、電力上限 **150 W**、CUDA 12.8、CUDA Graphs 有効）での実測結果:

### Decode スループット（vs `llama.cpp`）

| ワークロード / モード | NInfer（ロスレス NVFP4+MXFP8） | `llama.cpp` `UD-Q4_K_XL` | `llama.cpp` `UD-Q8_K_XL` |
|---|---:|---:|---:|
| エッセイ 256 tok、C=1 Normal / DSpark-2 / DSpark-3（e2e） | **142.8** / **129.0** / **129.0 tok/s** | 30.7 tok/s | 29.3 tok/s |
| エッセイ 256 tok、C=1 / C=4 `temp=0.7, top_p=0.95`（CUDA Graph） | **139.5** / **270.6 tok/s** | — | — |
| エッセイ 256 tok、C=4 Normal / DSpark-2（合計） | **271.9** / **209.6 tok/s** | 78.3 tok/s | 55.9 tok/s |
| エッセイ 256 tok、C=8 Normal / DSpark-2（合計） | **358.4** / **238.3 tok/s** | — | — |
| Python コードリファクタ 256 tok、C=1 Normal / DSpark-2 / DSpark-3（engine） | **150.1** / **174.8** / **176.3 tok/s** | — | — |
| ~4.2K tok ニードル検索、C=1 Normal / DSpark-3 | **130.7** / **152.6 tok/s** | — | — |
| ~15.4K–16.3K tok 長文デコード、C=1 Normal / DSpark-3 | **114.8** / **110.7 tok/s** | — | — |

### Cold Prefill スループット・TTFT（512〜65,536 トークン）

`ninfer-serve`（`--tp 2 --devices 0,1 --max-concurrency 1 --max-context 70000 --no-thinking --greedy`）、ウォームアップ（1,024 tok）除外、全件コールド実行（`prefix_cache_hit_tokens = 0`）、512〜16,385 tok は 3 回平均・32,768〜65,536 tok は 2 回平均:

| 入力トークン数 | Normal (`chunk=1024`) | Normal (`chunk=2048`) | DSpark-3 (`chunk=1024`) |
|---:|---:|---:|---:|
| 512 | **1,033.7 tok/s**（TTFT 0.50 s） | — | 1,001.5 tok/s（0.52 s） |
| 1,023 | **1,014.5 tok/s**（TTFT 1.01 s） | — | 982.2 tok/s（1.05 s） |
| 2,048 | **940.8 tok/s**（TTFT 2.18 s） | 928.3 tok/s（2.22 s） | 907.3 tok/s（2.27 s） |
| 4,096 | **882.8 tok/s**（TTFT 4.65 s） | 876.1 tok/s（4.70 s） | 851.0 tok/s（4.83 s） |
| 8,192 | **810.5 tok/s**（TTFT 10.13 s） | 809.3 tok/s（10.18 s） | 784.6 tok/s（10.47 s） |
| 16,385 | 703.5 tok/s（TTFT 23.35 s） | **709.1 tok/s**（23.18 s） | 684.2 tok/s（24.00 s） |
| 32,768 | 566.4 tok/s（TTFT 57.99 s） | **576.6 tok/s**（56.96 s） | 554.2 tok/s（59.22 s） |
| 65,536 | 406.5 tok/s（TTFT 161.47 s） | **417.8 tok/s**（157.10 s） | 401.2 tok/s（163.62 s） |

- 短〜中長文（512〜8K）では既定の `--prefill-chunk 1024` が **810〜1,034 tok/s** と高速で、16K 以上の長文では `--prefill-chunk 2048` が最大 +2.8%（64K で **417.8 tok/s**）上回ります。
- DSpark-3 有効時の Prefill オーバーヘッドは全コンテキスト長で **1.3%〜3.6%** に収まります。
- **325/325** の FP64/TP2/CUDA-Graph 演算子単体テスト（全ホットカーネル 0 B スタック/スピル）、PyTorch リファレンスとの **32/32** 完全トークン一致（Normal Graph/Eager・DSpark-2/3/5/7・Graph キャプチャ Direct BF16 サンプリング経路）、HF `AutoTokenizer` 一致、4K/16K SWA-1024 / Paged-KV 長文デコード、ネイティブ 262K 最大コンテキスト計画、および `C=1..8` 並列サービングを検証済みです。複数トークン処理（Prefill・DSpark 検証・C≥2）の MoE 出力は `atomicAdd` で集約するため、長い生成は実行ごとにビット再現しません（DSpark 単発計測の tok/s にはドラフト受容率の揺らぎを含みます）。
- **V100-32GB×2 互換性**: VRAM 計画や P2P / NVLink / 非 P2P（CUDA UVA ステージング）フォールバック経路は 16 GB VRAM や PCIe 固定の前提を持たないため、同一の `sm_70` ビルドで **2× V100-32GB**（NVLink / PCIe P2P / 非 P2P）および単体 **V100-32GB (`--tp 1`)** に対応します（物理 32 GB V100 実機での単独ベンチマーク計測は未記録です）。

### Inkling-2-Mini クイックスタート

```bash
# 1. Volta (sm_70) 向けビルド
cmake -S . -B build-sm70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
ninja -C build-sm70 ninfer ninfer-serve ninfer_inkling2_ops_test

# 2. Hugging Face チェックポイントを無劣化変換（mtp.safetensors も自動同梱）
python3 tools/convert/inkling2_mini/convert.py \
  --model-dir /path/to/Inkling-2-Mini \
  --output /path/to/inkling2_mini_nvfp4_mxfp8.ninfer

# 3. HTTP サーバー起動（OpenAI / Anthropic 互換、高スループット Normal C=1..8）
./build-sm70/apps/ninfer-serve /path/to/inkling2_mini_nvfp4_mxfp8.ninfer \
  --tp 2 --devices 0,1 --max-concurrency 8 --max-context 8192 --port 8080

# 4. DSpark-3 投機デコード起動（コード生成・長文低遅延向け）
./build-sm70/apps/ninfer-serve /path/to/inkling2_mini_nvfp4_mxfp8.ninfer \
  --tp 2 --devices 0,1 --spec mtp --draft-tokens 3 \
  --max-concurrency 4 --max-context 8192 --port 8080
```

変換オプション、Volta 最適化の詳細、制限事項については [docs/inkling2-mini.md](docs/inkling2-mini.md) を参照してください。

---

## 上流 `ninfer-v100xN`（Qwen3.8-27B / QUASAR NVFP4）概要

**v0.1.0 以前のデコード実装と比較して、同一の 85K トークン・コード生成タスクにおけるコミット済みスループットは 143.46 から 198.10 tok/s へ約 38.1% 向上しました。** 両計測とも V100 16 GB × 2、180K 容量、TP2/MTP4、INT8 KV、CUDA Graphs、greedy サンプリングを使用しています。現行結果は実験的な長文アテンション（long attention）と高速デコード（fast decode）の両方を有効にしており、出力が変化する場合があり、この組み合わせでの GPQA 計測はありません。
[現行結果](docs/performance.md#quasar-v3-mtp4-retained-decode-profile-2026-10-19) · [旧バージョン](docs/performance.md#quasar-v3-final-long-context-decode-evaluation)

先行する 220 W ホストの 4 枚構成 NVLink 最適化では、85K 入力・QUASAR/MTP4・両実験フラグ有効時に **283.118 / 283.367 committed wall tok/s**（平均 **283.242 ± 0.176**、**220 W / 1200 MHz** 制限維持）を記録しました。1Cat のコレクティブ通信と Split-D 機構を取り入れたことで、Prefill は 1743 から平均 **2334.73 tok/s** に向上しています。TP4 の通常モード、プレフィックス、実 85K Graph/eager 状態チェックに合格しています（Split-D と実験プロファイルは出力を変える可能性があり、現組み合わせの GPQA は未実施）。別ホストの **300 W** 環境では、同一の 85K/MTP4 フィクスチャで **322.138 / 322.218 tok/s**（平均 **322.178**）、**Prefill 2731.60 tok/s** を計測しています（異なるホスト・電力枠での計測であり、単独のソフトウェア高速化ではありません）。これらの TP4 変更は **v0.2.0** でリリースされ、Linux SM70 ビルド済みパッケージおよび Docker イメージが提供されます。[計測手法と結果](docs/tp4.md)。

**2 × Tesla V100-SXM2 16 GB**（CUDA 12.8 / SM70）向けの高速な単一リクエスト Qwen3.8-27B 推論エンジンです。ローカル検証機は SXM2-to-PCIe アダプタ経由の **NVLink なし PCIe P2P** 構成です。推奨モデルは **QUASAR NVFP4 v3**（TP2、INT8 group-64 KV、CUDA Graphs）です。テキスト専用起動時は Vision 重みを一切ロードしません。オプションの TP2 画像/動画推論は同一アーティファクトを再利用し、GUI/TUI にデフォルト OFF の Vision スイッチを備えています。

メンテナの主観評価では「FP8 と同等の質感」とされています。元チェックポイント作者の報告では GPQA-D が 90.91%（BF16: 91.41%）、AIME'26 が双方 100% です。これらは外部のソースチェックポイント結果であり、本環境での変換後スコアやロスレスの証明ではありません。[出典・独立比較・制約事項](docs/model-quality.md)。

## v0.2.2 · 1 枚の 16 GB V100 で動作する QUASAR

1 枚の V100-SXM2 16 GB で QUASAR NVFP4 テキスト＋MTP4 が動作します。同じ 85,000 トークンのコード入力で、コールド Prefill は **374.5 → 939.0 tok/s**、コミット済み Decode は **87.4 → 113.9 tok/s** に向上しました（ベースラインは初期の単カード開発版）。9,472 トークンの GPU ウィンドウを保持し、古い KV はホスト RAM から検索取得します。近似検索と実験的数値経路により出力が変わる可能性があり、このプロファイルでの品質評価は未実施です。[単カード起動コマンド](docs/prebuilt.md#single-v100-16-gb)

## v0.2.1: TP2 / TP4 並列化とサービング修正

QUASAR NVFP4 / MTP4 で、TP2・TP4 にパック済みバッチ射影、ダイレクトパケット発行、融合 sum/norm ハンドオフを導入しました。完全な **2K コーディング会話**（1 リクエストあたり 512 トークン出力）での結果:

| 構成 | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| 4 × V100 16 GB、NVLink TP4 | 312.1 | 436.6 | 702.0 | **1083.4** |
| 2 × V100 16 GB、NVLink TP2 | 223.5 | 312.4 | 471.4 | **677.9** |
| 2 × V100 16 GB、PCIe P2P（NVLink なし） | 220.8 | 297.2 | 436.0 | **603.9** |

数値は Prefill を除くエグゼキュータラウンド計測の合計コミット済み Decode tok/s（CUDA Graphs、greedy、最適化ヘッド MTP4、両実験プロファイル ON）です。C8 のドラフト受容率は TP4 で **74.4%**、両 TP2 構成で **76.4%** です。2 台の TP2 ホストは NVLink 以外の条件も異なります。なお、この新しい短文プロンプトは完全なタスクとアシスタントプレフィックスを含んでおり、以前の **493.1 / 554.0**（切り詰め raw prefix 使用）とは別枠の計測です。

**85K C1/C2/C4/C8 マトリクス**では、純粋な Decode に加えて実バッチサイズとウェーブ全体遅延も報告しています。180K 共有 KV 容量では 85K リクエストは最大 2 本しか同時に載らないため、C8 投入でも 8 本同時 Decode にはなりません。C8 は TP4 NVLink / TP2 NVLink / ローカル PCIe TP2 でそれぞれ **317.8 / 200.3 / 198.2 tok/s**（平均実バッチ約 **1.23**）に達します。同一ホスト対照は **1Cat 1.5.2 / QUASAR NVFP4 / MTP4**（FP16 活性化 / E4M3 KV）で、2K/85K × C1/C2/C4/C8 の全 8 条件を計測しています（数値プロファイルと計測面が異なるため同品質比較ではありません）。詳細は [完全マトリクス・受容率・対照と手法](docs/performance.md#tp4-concurrent-packed-batch-decode-2026-11-05) を参照してください。

![Complete-code concurrency measurements](docs/assets/tp4-concurrency/quasar-concurrency.svg)

TP4 ではバッチ対応の短コンテキストアテンションとリクエスト別 GDN record/norm を追加し、独立 FP64 演算子チェック・C2/C4/C8 通常および longfast の Graph/eager 出力/受容チェック・実 85K 状態/リプレイチェックを通過しています（**新規 GPQA は未実施**、long attention と fast decode は既定 OFF）。また、トークン化時のタイムアウト/キャンセル修正（#6）と、`prompt_cache_key`・思考ヒント付きの DSH Responses 対応（#4）を含みます。

## v0.2.0: TP4、モデル常駐ホットアンロード、Docker パッケージ

QUASAR TP4 Text/MTP4 で検証済み long attention と fast decode（既定 OFF）に対応しました。300 W の 4×V100 ホストで **85K 入力時 322.18 committed tok/s**・**Prefill 2731.60 tok/s**（同一ホスト TP2 対照は 199.23 tok/s）、旧 220 W ホストでは **283.24 tok/s** です。[ワークロード・数値検証・非 P2P 対照](docs/tp4.md)。

GUI/TUI はアイドル時にモデルをホスト RAM へホットアンロードし GPU へ再ロードする機能に対応しました（重みは RAM に維持、KV/プレフィックス/メディアキャッシュは消去）。実験的な WSL2 TP2 トランスポートも同梱されます。リリースにはソース、[Linux SM70 ビルド済みパッケージ](docs/prebuilt.md)、[ロード可能 Docker イメージ](docs/docker.md)、チェックサム、バンドル FFmpeg ソースが含まれます。サービングには 3 API 共通の出力上限 `--max-output-tokens` が追加されました。

## v0.1.2: Vision 長文アテンションとオプション CPU Vision

`--volta-long-attention` が完全 INT8 KV 付き TP2 画像/動画（None/MTP）に対応しました。`--vision --vision-device cpu` は視覚エンコーダをホスト RAM に置いて CPU で計算し、BF16 視覚埋め込みのみ GPU へ転送します（エンコードは遅くなりますが GPU キューを塞がず既存 Decode を継続可能）。CPU/GPU の数値差と long attention により出力が変わる可能性があり、新規 GPQA は未主張です。fast decode は引き続きテキスト専用です。

## v0.1.1: 並列実行、履歴ブランチキャッシュ、Responses API

実験的 long attention が TP2/MTP4 で **1〜8 並列リクエスト** に対応しました（共有 KV 容量を超過した分はキュー待機）。[計測と手法](docs/performance.md#experimental-long-context-concurrency-2026-10-20)。

![Concurrent decode throughput and actual batch](docs/assets/quasar-concurrency.svg)

プレフィックス再利用が履歴ブランチと複数会話ツリーを保持するようになり、`ABD → ABC → ABDE` のような分岐でも保存済み ABD フロンティアを復元できます。`--prefix-cache-tokens` の既定値は **384,000**（QUASAR INT8 KV + MTP でホスト RAM 約 **12.8 GiB**、アーカイブ状態は別途 2 GiB 上限、`0` で履歴無効化）。Vision・DFlash・疎 RAM-KV では履歴ブランチは未使用です。[キャッシュ仕様と制限](docs/maintainer/paged-kv-cache.md#104-historical-branches)。

Responses API は `reasoning: {"effort": "high"}` 等を受け付け、ロード済み Qwen テンプレートへマップします（`none` で思考無効化）。[プロトコル対応](docs/serving.md)。

## QUASAR 計測 — 2026-10-19

85,000 入力トークンにおける 4 種テキストタスク・2 種の保存済み GUI 設定（TP2、MTP4、INT8 group-64 KV、CUDA Graphs、chunk 2560、fast decode OFF）でのコミット済みウォール Decode 速度:

| 85,000 入力トークン時 | Vision ロード済み、248,000 容量 | テキスト専用、262,144 容量 + long attention |
|---|---:|---:|
| 中国語ストーリー | 86.44 tok/s | 103.43 tok/s |
| 英→中翻訳 | 143.57 tok/s | 179.43 tok/s |
| 32 件 JSONL | 185.04 tok/s | **230.40 tok/s** |
| 5 スロット論理パズル | 165.22 tok/s | **209.63 tok/s** |

85K 時の Prefill は平均約 1314〜1321 tok/s、初回トークン出力は約 64〜65 秒です。[全コンテキスト長・受容率・計測詳細](docs/performance.md#quasar-v3-saved-configuration-mtp4-speed-matrix-2026-10-19)。

## フルキャッシュ時の初回トークン遅延（TTFT）

**300 トークン全一致キャッシュ（計算 Prefill ゼロ）** の条件下で、MTP ブリッジ準備・遅延状態復元より先に初回トークンを発行するよう最適化しました（15 回中央値、64 トークン出力）:

| キャッシュ経路 | 中央値（改善前 → 後） | 削減率 |
|---|---:|---:|
| Raw トークン、RAM チェックポイント | 25.963 → **1.996 ms** | 92.31% |
| Chat、GPU rewrite チェックポイント | 5.313 → **2.640 ms** | 50.31% |

[計測手法と検証](docs/performance.md#300-token-fully-cached-ttft-2026-10-19)。

## llama.cpp / LM Studio との比較（Q4_K_M 履歴対照）

**QUASAR 同士の現行 llama.cpp ベンチマークは存在しません。** 過去の対照は双方 Q4_K_M、85,000 入力、180,000 容量、Q8/INT8 KV、greedy、MTP 最大 3 ドラフト（PCIe P2P 有効化前）で実施されました:

| 履歴 Q4_K_M エンジン | Decode tok/s | Prefill |
|---|---:|---:|
| NInfer（staging 経路、2 回平均） | 50.68 ± 0.04 | 1251.44 tok/s |
| LM Studio llama.cpp CUDA 2.33.0（1 回） | 35.4977 | 185.374 s |

アーティファクトや通信経路が異なるため、QUASAR の倍率比較には使えません。[詳細テーブルと比較条件](docs/llama-comparison.md)。

## ビルドと起動（QUASAR / 共通）

Docker は初回起動時に QUASAR NVFP4 v3 を自動ダウンロードし、デフォルトで TUI を開きます（NVIDIA Container Toolkit 必須。[Docker 設定](docs/docker.md)）:

```bash
docker compose build
docker compose run --rm --service-ports ninfer
```

ソースからビルドするには Linux、CUDA 12.8、NVIDIA ドライバ、CMake 3.28+、C++20 コンパイラ、Ninja、pkg-config、FFmpeg 開発ライブラリ、libcurl、LP64 BLAS（OpenBLAS 推奨）が必要です:

```bash
bash tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/_deps/install" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_TESTING=OFF
cmake --build build-v100 --target ninfer ninfer-serve -j2
```

QUASAR v3 アーティファクトのダウンロード:

```bash
hf download MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer SHA256SUMS \
  --revision v3 --local-dir /Models/ninfer-V100X2/quasar-v3
(cd /Models/ninfer-V100X2/quasar-v3 && sha256sum --check SHA256SUMS)
```

GUI（PyQt6 / PyQt5、[手順](docs/gui.md)）または TUI（Python 3.11 標準ライブラリのみ、[手順](docs/tui.md)）で起動:

```bash
bash tools/v100/ninfer-gui.sh
# または
bash tools/v100/ninfer-tui.sh
```

API サーバーを直接起動（QUASAR MTP4 / 180K）:

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 180000 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 4 --lm-head-draft --default-max-tokens 65536 \
  --request-log-jsonl profiles/serve.jsonl --log-stats-interval-ms 1000 --no-thinking
```

DFlash7 を使う場合は容量 98304、チャンク 1024、`--spec dflash --draft-tokens 7` を指定し、`--lm-head-draft` を外します。CLI は `build-v100/apps/ninfer` に同じ推論フラグと `--prompt TEXT --max-new N` を渡します。マルチ GPU マシンでは `--tp 2 --devices 4,6` のように任意ペアを選択できます（`CUDA_VISIBLE_DEVICES=4,6` 指定時は論理 ID `0,1` を使用）。

## その他のドキュメント

| ドキュメント | 内容 |
|---|---|
| [Inkling-2-Mini (V100 TP2)](docs/inkling2-mini.md) | ネイティブ・ロスレス NVFP4/MXFP8 変換、DSpark 投機デコード、2× V100 Decode/Prefill 計測 |
| [TP4 / NVLink](docs/tp4.md) | 4 枚構成ホスト、QUASAR / 公式 NVFP4 / ネイティブ FP8 と同機 TP2 対照 |
| [その他 V100 計測](docs/performance-v100.md) | 公式 NVFP4、Q4_K_M、P2P/staging 対照、プレフィックステスト |
| [RAM-KV / PP2 実験](docs/experiments.md) | 近似検索と起動時固定の層パイプライン並列 |
| [代替モデル / Vision](docs/alternatives.md) | Q4_K_M 変換と QUASAR 画像/動画サポート |
| [V100 実装詳細](docs/v100-implementation.md) | カーネル、通信、数値検証の詳細 |
| [デコード実装調査](docs/v100-decode-survey.md) | 調査プロジェクト、採用機構、残課題候補 |
| [上流継承結果](docs/performance-upstream.md) | 他ハードウェア/モデル、並列および長文脈キャンペーン |
| [ドキュメント索引](docs/README.md) | CLI、API、GUI、テスト、メンテナ向けリファレンス |

推奨プロファイルは **テンソル並列（TP2）** です。PP2 は独立した起動時固定実験であり、公開 TP2 数値にはパイプライン並列の利得は含まれません。

## 謝辞とライセンス

V100X2 実装は [Neroued/ninfer](https://github.com/Neroued/ninfer)（RTX 3060 TP2 フォーク）および [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) に基づき、Volta NVFP4 / TP2 動作は [plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo) とクロスチェックされています。TP2/TP4 BF16 パケットプッシュ通信は [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)（出典クレジット: [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100)）の 2 エポック転送を応用し、射影チューニングは QPN カーネルおよび [v100-skinny](https://github.com/dnv2003/v100-skinny) を参照しています。DFlash2 は [Inco AI の実装](https://inco.ai/blog/dflash2/) と [Qwen3.8-27B ドラフトチェックポイント](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) に準拠し、[KVMem](https://github.com/kvmem/kvmem-llama.cpp) はオプション RAM-KV 実験の参考となっています。

[Li3age](https://github.com/Li3age) 氏より [PR #1](https://github.com/tuxKOH/ninfer-V100X2/pull/1) で TP2 Vision と視覚トークン予算設定が、[verssionhack](https://github.com/verssionhack) 氏より [PR #3](https://github.com/tuxKOH/ninfer-V100X2/pull/3) で Prefill ステップ間の Decode バーストスケジューリングが貢献されています。

計測モデルは [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)、[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)（[Neroued](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) パッケージ）、[QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)（[MirkoCovizzi](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer) 変換）に由来します。Inkling-2-Mini 対応は [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) の重みを再量子化なしで直接変換します。本フォークの Inkling-2-Mini 対応および Volta 最適化は **Codex** と **Antigravity (Agy)** により開発・最適化されました。

NInfer および本フォークは [Apache-2.0](LICENSE) ライセンスで提供されます。必要な帰属表示は [NOTICE](NOTICE) を参照してください。サードパーティ依存関係は各々のライセンスに従います。
