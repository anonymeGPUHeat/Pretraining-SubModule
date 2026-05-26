#!/usr/bin/env python3
"""
PTX Dataset Analyzer
====================
Analyzes a dataset of PTX files for:
  - Instruction coverage & frequency
  - Domain/application classification
  - L2 cache contention indicators
    * Memory coalescing patterns
    * Memory uncoalescing (large-stride) patterns
    * Shared memory bank conflict risk
    * Atomic contention
    * Cache modifier usage (.ca/.cg/.cs/.cv/.lu)
    * Prefetch usage
  - Warp divergence indicators
  - Register & shared memory pressure
  - PTX version / SM target distribution

Usage:
    python3 ptx_analyzer.py --dir /path/to/ptx_files [--out report.json] [--csv]

Requires: Python 3.8+, no third-party packages (stdlib only).
Optional: pandas, matplotlib for richer output (auto-detected).
"""

import re
import os
import sys
import json
import hashlib
import logging
import argparse
import time
from collections import  Counter
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PTX ISA 8.x  –  Instruction taxonomy (verified against PTX ISA documentation)
# ─────────────────────────────────────────────────────────────────────────────

# Every key is a canonical opcode prefix (before the first '.')
INSTRUCTION_CATEGORIES: Dict[str, List[str]] = {
    "integer_arith":   ["add", "sub", "mul", "mad", "mul24", "mad24", "sad",
                        "div", "rem", "abs", "neg", "min", "max"],
    "float_arith":     ["fma", "rcp", "sqrt", "rsqrt", "sin", "cos", "lg2",
                        "ex2", "tanh"],
    "comparison":      ["setp", "set", "selp", "slct"],
    "logic_shift":     ["and", "or", "xor", "not", "cnot", "shl", "shr"],
    "data_movement":   ["mov", "shfl", "prmt"],
    "load_store":      ["ld", "st", "ldu", "prefetch", "prefetchu"],
    "atomic":          ["atom", "red"],
    "control_flow":    ["bra", "call", "ret", "exit"],
    "synchronization": ["bar", "membar", "fence", "vote", "activemask", "match"],
    "conversion":      ["cvt", "cvta", "cvtp"],
    "bit_manip":       ["bfe", "bfi", "bfind", "brev", "popc", "clz"],
    "video_dp":        ["dp4a", "dp2a", "vadd", "vsub", "vabsdiff", "vmin",
                        "vmax", "vshl", "vshr", "vmad", "vset"],
    "tensor_core":     ["wmma", "mma", "ldmatrix", "stmatrix"],
    "async_copy":      ["cp"],          # cp.async.*
    "texture_surface": ["tex", "tld4", "txq", "suld", "sust", "sured", "suq"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Real-world domain classifier
# ─────────────────────────────────────────────────────────────────────────────
# Each domain entry contains:
#   "desc"       – human-readable description
#   "signals"    – list of (regex_pattern, score_weight) tuples
#                  score = sum(weight * occurrence_count) for each signal
#   "exclusions" – if ANY exclusion pattern fires, domain is skipped entirely
#
# Scoring is based on known PTX fingerprints from real GPU workloads:
#   AI/DL   : wmma/mma, dp4a, bf16, cp.async, ldmatrix, ex2/lg2 (softmax)
#   Graphics: tex/tld4/suld/sust, tid.y/z, sin/cos approx
#   RT      : rsqrt, divergent branches, xorshift RNG
#   Physics : rsqrt, atom.add.f32, tid.z, min/max.f32
#   HPC     : f64 fma, shfl.bfly (FFT), large smem (stencil/GEMM)
#   Crypto  : xor/and/or/shl/shr bulk, prmt, bfe/bfi/brev/popc, no fp
#   Bio     : max/min/selp s32 (DP scoring), byte ops (nucleotide)
#   Finance : ex2/lg2 (BS/MC), f64 atom.add, no tex/wmma
#   DB/Analytics: atom.cas (hash-table), bfe (radix sort), setp s32
#   MD      : rsqrt+rcp+atom.add.f32, 3D grid, no tex/wmma
#   Graph   : atom.min/cas, vote.any (BFS frontier), irregular ld.cg
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_CLASSIFIER: Dict[str, Dict] = {

    # ── AI / Deep Learning ────────────────────────────────────────────────
    "AI_TRAINING": {
        "desc": "Neural network training (GEMM, backprop, weight update, gradient ops)",
        "signals": [
            # Tensor core ops (strong signals)
            (r"\bwmma\.mma\b",                         15),
            (r"\bmma\.sync\b",                         15),
            (r"\bwmma\.load\b",                         8),
            (r"\bwmma\.store\b",                        8),
            (r"\bldmatrix\b",                           8),
            (r"\bstmatrix\b",                           8),
            (r"\bcp\.async\b",                          5),
            (r"\.bf16\b",                               6),
            (r"\.f16x2\b",                              5),
            (r"atom\.global\.add\.f32",                 4),
            (r"atom\.global\.add\.f64",                 7),
            (r"\bfma\.rn\.f16\b",                       4),
            (r"\bfma\.rn\.bf16\b",                      5),
            (r"\.shared\s+\.\w+\s+\w+\s*\[\s*[4-9]\d{3,}", 4),
            # Non-tensor-core AI signals (common in CUDA DL kernels without tensor cores)
            (r"\bfma\.rn\.ftz\.f32\b",                  3),  # fused multiply-add with flush-to-zero (common in NN)
            (r"\bmax\.ftz\.f32\b",                       4),  # ReLU activation pattern
            (r"\bmax\.f32\b",                             2),  # ReLU activation pattern (non-ftz)
        ],
        "exclusions": [r"\btex\b", r"\bsuld\b"],
    },

    "AI_INFERENCE": {
        "desc": "Neural network inference (int8/fp16 quantized forward pass, TensorRT)",
        "signals": [
            (r"\bwmma\b",                               8),
            (r"\bmma\.sync\b",                         10),
            (r"\bdp4a\b",                              12),
            (r"\bdp2a\b",                               8),
            (r"\bvmad\b",                               5),
            (r"\.s8\b",                                 5),
            (r"\.u8\b",                                 4),
            (r"\bex2\.approx\b",                        4),   # softmax
            (r"\blg2\.approx\b",                        4),   # log-softmax
            (r"\brcp\.approx\b",                        3),
            (r"\bshfl\.sync\b",                         3),
            (r"\bactivemask\b",                         3),
        ],
        "exclusions": [],
    },

    "AI_LLM": {
        "desc": "Large Language Model kernels (FlashAttention, KV-cache, MoE routing)",
        "signals": [
            (r"\bshfl\.sync\.bfly\b",                  10),  # flash-attention warp reduction
            (r"\bshfl\.sync\b",                         5),
            (r"\bmatch\.sync\b",                        8),
            (r"atom\.global\.max\b",                    6),
            (r"atom\.global\.exch\b",                   8),   # MoE top-k
            (r"st\.global\.cs\b",                       5),   # KV-cache streaming store
            (r"\bcp\.async\.cg\b",                      6),
            (r"\bsin\.approx\b",                        3),   # RoPE embeddings
            (r"\bcos\.approx\b",                        3),
            (r"\bfma\.rn\.f16\b",                       3),
            (r"\bfma\.rn\.bf16\b",                      4),
        ],
        "exclusions": [],
    },

    # ── Graphics / Rendering ──────────────────────────────────────────────
    "GRAPHICS_RENDERING": {
        "desc": "Rasterization, pixel/vertex shading, post-processing, DLSS-style upscaling",
        "signals": [
            (r"\btex\b",                               12),
            (r"\btld4\b",                              10),
            (r"\btxq\b",                                8),
            (r"\bsuld\b",                               8),
            (r"\bsust\b",                               8),
            (r"%tid\.y\b",                              5),
            (r"%ctaid\.y\b",                            4),
            (r"\.f16x2\b",                              4),
            (r"\bsin\.approx\b",                        3),
            (r"\bcos\.approx\b",                        3),
            (r"\bex2\.approx\b",                        2),
            (r"st\.global\.wb\b",                       2),
        ],
        "exclusions": [],
    },

    "RAY_TRACING": {
        "desc": "Ray tracing / path tracing (BVH traversal, intersection tests, OptiX)",
        "signals": [
            (r"\brsqrt\.approx\b",                      6),   # ray normalization (also in physics/MD)
            (r"\bsqrt\.approx\b",                       3),
            # Removed overly-generic signals: predicated branches, setp.*.f32,
            # fma.rn.f32, xor.b32, shr.u32, shl.b32 all fire on ANY f32 kernel
            (r"\bneg\.f32\b",                           1),
            # BVH traversal specific patterns (stronger signals)
            (r"min\.f32.*max\.f32|max\.f32.*min\.f32",  6),  # AABB slab test
            (r"\brcp\.approx\.f32\b",                   3),  # 1/t for ray parametric
        ],
        "exclusions": [r"\btex\b"],
    },

    # ── Game Physics / Simulation ─────────────────────────────────────────
    "PHYSICS_SIMULATION": {
        "desc": "Rigid body, cloth, fluid dynamics, collision detection (PhysX/Bullet/Warp)",
        "signals": [
            (r"%tid\.z\b",                              8),   # 3D grid
            (r"%ctaid\.z\b",                            8),
            (r"atom\.global\.add\.f32",                 5),
            (r"atom\.global\.min\.f32\b",               5),
            (r"atom\.global\.max\.f32\b",               5),
            (r"\brsqrt\.approx\b",                      5),   # normalize vectors
            (r"\bsqrt\.approx\b",                       4),
            (r"\bmin\.f32\b",                           2),
            (r"\bmax\.f32\b",                           2),
            (r"\bfma\.rn\.f32\b",                       2),
            (r"\bneg\.f32\b",                           2),
        ],
        "exclusions": [r"\bwmma\b", r"\btex\b"],
    },

    # ── Scientific HPC ────────────────────────────────────────────────────
    "HPC_LINEAR_ALGEBRA": {
        "desc": "BLAS/LAPACK: GEMM, TRSM, Cholesky, LU (cuBLAS/CUTLASS/MAGMA)",
        "signals": [
            (r"\bwmma\b",                               6),
            (r"\bfma\.rn\.f64\b",                      10),
            (r"\bfma\.rn\.f32\b",                       3),
            (r"\bmad\.f64\b",                           8),
            (r"\.f64\b",                                4),
            (r"\badd\.f64\b",                           4),
            (r"\bmul\.f64\b",                           4),
            (r"\.shared\s+\.\w+\s+\w+\s*\[\s*[1-9]\d{2,}", 4),
            (r"\bbar\.sync\b",                          3),
        ],
        "exclusions": [r"\bdp4a\b"],
    },

    "HPC_STENCIL": {
        "desc": "Stencil / finite-difference: CFD, weather modeling, seismic imaging",
        "signals": [
            (r"%tid\.z\b",                              8),
            (r"%ctaid\.z\b",                            8),
            (r"\.f64\b",                                4),
            (r"\bfma\.rn\.f64\b",                       6),
            (r"\.shared\s+\.\w+\s+\w+\s*\[\s*[1-9]\d{3,}", 5),
            (r"\bbar\.sync\b",                          2),
            (r"ld\.global\b",                           1),
        ],
        "exclusions": [r"\bwmma\b", r"\btex\b"],
    },

    "HPC_FFT": {
        "desc": "Fast Fourier Transform: spectral methods, signal processing (cuFFT)",
        "signals": [
            (r"\bshfl\.sync\.bfly\b",                  20),  # butterfly network
            (r"\bshfl\.bfly\b",                        15),
            (r"\bsin\.approx\b",                        5),  # twiddle factors
            (r"\bcos\.approx\b",                        5),
            (r"\bfma\.rn\.f32\b",                       2),
            (r"\bfma\.rn\.f64\b",                       3),
            (r"\bneg\.f32\b",                           2),
            (r"\bneg\.f64\b",                           3),
            (r"\bbar\.sync\b",                          2),
        ],
        "exclusions": [r"\bwmma\b"],
    },

    "HPC_SPARSE": {
        "desc": "Sparse linear algebra: SpMV, SpMM, graph analytics (cuSPARSE)",
        "signals": [
            (r"atom\.global\.add\b",                    5),
            (r"red\.global\b",                          5),
            (r"ld\.global\.cg\b",                       4),
            (r"st\.global\.cg\b",                       4),
            # Removed: mad.lo.s32, add.s32, shr.s32 are ubiquitous in ALL kernels
        ],
        "exclusions": [r"\bwmma\b", r"\.f64\b"],
    },

    "HPC_REDUCTION": {
        "desc": "Parallel reduction / prefix scan / histogram (sum, max, argmax)",
        "signals": [
            (r"\bshfl\.sync\.down\b",                  15),
            (r"\bshfl\.down\b",                        12),
            (r"\bshfl\.sync\b",                         5),
            (r"atom\.global\.add\b",                    5),
            (r"atom\.global\.max\b",                    5),
            (r"atom\.global\.min\b",                    5),
            (r"\bvote\.sync\.all\b",                    6),
            (r"\bvote\.sync\.any\b",                    6),
            (r"\bbar\.red\b",                           8),
            (r"\bpopc\b",                               4),
        ],
        "exclusions": [],
    },

    # ── Image / Video ─────────────────────────────────────────────────────
    "IMAGE_PROCESSING": {
        "desc": "Image filtering, convolution, morphology, color conversion (NPP/OpenCV)",
        "signals": [
            (r"\btex\b",                                8),
            (r"\bsuld\b",                               8),
            (r"\bsust\b",                               6),
            (r"%tid\.y\b",                              5),
            (r"\.u8\b",                                 6),
            (r"\.u16\b",                                4),
            (r"\bprmt\b",                               6),  # byte permute = pixel packing
            (r"\bcvt\.\w+\.u8\b",                       4),
            (r"\bmin\.u32\b",                           2),
            (r"\bmax\.u32\b",                           2),
        ],
        "exclusions": [r"\bwmma\b"],
    },

    "VIDEO_CODEC": {
        "desc": "Video encode/decode, motion estimation, DCT/IDCT (NVENC/NVDEC)",
        "signals": [
            (r"\bsad\b",                               12),
            (r"\bvabsdiff\b",                          10),
            (r"\bvmad\b",                               6),
            (r"\bvadd\b",                               4),
            (r"\bvmin\b",                               4),
            (r"\bvmax\b",                               4),
            (r"\.s16\b",                                4),
            (r"\bprmt\b",                               3),
            (r"\bmul24\b",                              4),
            (r"\bmad24\b",                              5),
        ],
        "exclusions": [],
    },

    # ── Security / Crypto ─────────────────────────────────────────────────
    "CRYPTOGRAPHY": {
        "desc": "AES, SHA-2/3, hash functions, elliptic curve crypto",
        "signals": [
            (r"\bxor\.b32\b",                           4),
            (r"\bxor\.b64\b",                           5),
            (r"\band\.b32\b",                           3),
            (r"\bor\.b32\b",                            3),
            (r"\bshl\.b32\b",                           3),
            (r"\bshr\.b32\b",                           3),
            (r"\bshr\.u32\b",                           3),
            (r"\bprmt\b",                               5),  # AES byte shuffle
            (r"\bbfe\b",                                4),
            (r"\bbfi\b",                                4),
            (r"\bbrev\b",                               5),
            (r"\bpopc\b",                               4),
            (r"\bclz\b",                                3),
        ],
        "exclusions": [r"\bfma\.rn\b", r"\btex\b", r"\bwmma\b"],
    },

    # ── Bioinformatics / Genomics ─────────────────────────────────────────
    "BIOINFORMATICS": {
        "desc": "DNA alignment, Smith-Waterman, genome assembly, variant calling",
        "signals": [
            # Strong signals: must co-occur with byte-level character ops
            # selp.s32 in tight loop with max.s32 is the DP scoring pattern
            (r"\bselp\.s32\b",                          3),  # DP max(0, score) — only weak without other evidence
            (r"\bmax\.s32\b",                           2),  # DP scoring (common in DP but also in index clamping)
            (r"\bmin\.s32\b",                           1),  # weak: common in all kernels
            # Byte-level ops for nucleotide encoding (stronger when combined)
            (r"ld\.global\.\w*\.u8\b",                  6),  # byte load = nucleotide read
            (r"\bcvt\.\w+\.u8\b",                       4),  # byte conversion
            (r"\bprmt\.b32\b",                           3),  # byte permutation for packed nucleotides
        ],
        # Exclude kernels that use FP math (real bio kernels are almost purely integer)
        "exclusions": [r"\bfma\.rn\.f64\b", r"\bwmma\b", r"\bfma\.rn\.f32\b",
                        r"\bfma\.rn\.ftz\.f32\b"],
    },

    # ── Computational Finance ─────────────────────────────────────────────
    "FINANCE_QUANT": {
        "desc": "Monte Carlo option pricing, VaR, risk models, Black-Scholes",
        "signals": [
            (r"\bex2\.approx\b",                        6),  # exp in BS/MC
            (r"\blg2\.approx\b",                        6),
            (r"\bsqrt\.approx\b",                       5),
            (r"\brcp\.approx\b",                        4),
            (r"\.f64\b",                                4),
            (r"\bfma\.rn\.f64\b",                       6),
            (r"atom\.global\.add\.f64",                 8),
        ],
        "exclusions": [r"\bwmma\b", r"\btex\b", r"%tid\.y\b"],
    },

    # ── Data Analytics / Database ─────────────────────────────────────────
    "DATA_ANALYTICS": {
        "desc": "SQL query execution, sort, hash-join, group-by (cuDF/Spark-RAPIDS)",
        "signals": [
            (r"atom\.global\.cas\b",                   10),  # hash-table probe
            (r"atom\.global\.exch\b",                   8),
            (r"atom\.global\.add\.u32\b",               5),
            (r"atom\.global\.add\.s32\b",               5),
            (r"\bbfe\.u32\b",                           4),  # radix sort digit extract
            (r"\bselp\.u32\b",                          4),
            (r"\bselp\.s32\b",                          4),
            # Removed: setp.*.s32, setp.*.u32, mad.lo.u32 are ubiquitous in ALL kernels
        ],
        "exclusions": [r"\bwmma\b", r"\btex\b"],
    },

    # ── Molecular Dynamics / Chemistry ───────────────────────────────────
    "MOLECULAR_DYNAMICS": {
        "desc": "Force-field MD: GROMACS/AMBER/NAMD, DFT, quantum chemistry",
        "signals": [
            (r"\brsqrt\.approx\b",                      6),  # Lennard-Jones r^-1
            (r"\brcp\.approx\b",                        4),
            (r"\bfma\.rn\.f64\b",                       6),
            (r"\bfma\.rn\.f32\b",                       2),
            (r"atom\.global\.add\.f32",                 6),  # force accumulation
            (r"atom\.global\.add\.f64",                 8),
            (r"%tid\.z\b",                              5),  # 3D spatial decomp
            (r"\bsqrt\.approx\b",                       4),
            (r"\bex2\.approx\b",                        3),  # exp(-E/kT)
        ],
        "exclusions": [r"\bwmma\b", r"\btex\b"],
    },

    # ── Graph Algorithms ──────────────────────────────────────────────────
    "GRAPH_ALGORITHMS": {
        "desc": "BFS, SSSP, PageRank, GNN, sparse graph traversal (Gunrock/cuGraph)",
        "signals": [
            (r"atom\.global\.min\b",                    8),  # BFS distance update
            (r"atom\.global\.cas\b",                    8),
            (r"\bvote\.sync\.any\b",                    8),  # active frontier
            (r"\bvote\.sync\.ballot\b",                 6),
            (r"\bactivemask\b",                         5),
            (r"ld\.global\.cg\b",                       5),  # irregular global access
            # Removed: add.s32 and setp.lt.s32 are ubiquitous in ALL kernels
        ],
        "exclusions": [r"\bfma\.rn\.f64\b", r"\bwmma\b", r"\btex\b",
                        r"\bfma\.rn\.ftz\.f32\b"],
    },
}

# Friendly display order for reports
DOMAIN_DISPLAY_ORDER = [
    "AI_TRAINING", "AI_INFERENCE", "AI_LLM",
    "GRAPHICS_RENDERING", "RAY_TRACING",
    "PHYSICS_SIMULATION",
    "HPC_LINEAR_ALGEBRA", "HPC_STENCIL", "HPC_FFT", "HPC_SPARSE", "HPC_REDUCTION",
    "IMAGE_PROCESSING", "VIDEO_CODEC",
    "CRYPTOGRAPHY", "BIOINFORMATICS", "FINANCE_QUANT",
    "DATA_ANALYTICS", "MOLECULAR_DYNAMICS", "GRAPH_ALGORITHMS",
]

# Domain groups for aggregated reporting
DOMAIN_GROUPS = {
    "AI / Deep Learning":        ["AI_TRAINING", "AI_INFERENCE", "AI_LLM"],
    "Graphics / Games":          ["GRAPHICS_RENDERING", "RAY_TRACING", "PHYSICS_SIMULATION"],
    "HPC / Scientific":          ["HPC_LINEAR_ALGEBRA", "HPC_STENCIL", "HPC_FFT",
                                  "HPC_SPARSE", "HPC_REDUCTION", "MOLECULAR_DYNAMICS"],
    "Media Processing":          ["IMAGE_PROCESSING", "VIDEO_CODEC"],
    "Security / Crypto":         ["CRYPTOGRAPHY"],
    "Data / Analytics":          ["DATA_ANALYTICS", "GRAPH_ALGORITHMS", "BIOINFORMATICS"],
    "Finance":                   ["FINANCE_QUANT"],
}

# ─────────────────────────────────────────────────────────────────────────────
# LLM-based domain classifier (optional, enabled with --llm)
# ─────────────────────────────────────────────────────────────────────────────

LLM_SYSTEM_PROMPT = """You are a GPU computing expert. You will receive the PTX assembly source
code of a CUDA kernel. Read the code — especially the kernel function name
(after .entry), the instruction mix, and memory access patterns — and
classify the kernel into exactly ONE domain from the list below.

Domains:
  AI_TRAINING          — neural-net training: convolution, GEMM/matmul, backprop, weight update, fused linear+ReLU, MLP, batch-norm
  AI_INFERENCE         — neural-net inference: int8/fp16 quantized forward pass, TensorRT
  AI_LLM               — large language model: FlashAttention, KV-cache, MoE routing
  GRAPHICS_RENDERING   — rasterization, pixel/vertex shading, post-processing
  RAY_TRACING          — ray/path tracing, BVH traversal, ray-triangle intersection
  PHYSICS_SIMULATION   — rigid body, cloth, fluid dynamics, SPH, collision
  HPC_LINEAR_ALGEBRA   — dense linear algebra: GEMM, TRSM, Cholesky, LU
  HPC_STENCIL          — stencil / finite-difference: CFD, weather, seismic
  HPC_FFT              — Fast Fourier Transform, spectral methods
  HPC_SPARSE           — sparse linear algebra: SpMV, SpMM
  HPC_REDUCTION        — parallel reduction, prefix scan, histogram
  IMAGE_PROCESSING     — image filtering, morphology, color conversion
  VIDEO_CODEC          — video encode/decode, motion estimation, DCT
  CRYPTOGRAPHY         — AES, SHA, hash, elliptic curve
  BIOINFORMATICS       — DNA alignment, Smith-Waterman, genome assembly
  FINANCE_QUANT        — Monte Carlo, Black-Scholes, risk models
  DATA_ANALYTICS       — SQL, sort, hash-join, group-by
  MOLECULAR_DYNAMICS   — MD force-field, Lennard-Jones, DFT
  GRAPH_ALGORITHMS     — BFS, SSSP, PageRank, sparse graph traversal
  UNKNOWN              — cannot determine

IMPORTANT: The kernel function name is the strongest clue — names like
conv2d, matmul, relu, mlp, linear, gemm, softmax, batchnorm indicate
AI_TRAINING. Read the name FIRST, then confirm with the instruction mix.

Reply with ONLY the domain name, nothing else."""

# Maximum number of PTX source characters to include in the LLM prompt.
# Qwen2.5-3B-Instruct supports 32K tokens but V100-16GB limits batch memory.
# ~4 chars/token → keep it under ~12K chars for a safe 4K-token input.
LLM_MAX_SRC_CHARS = 12000


class LLMDomainClassifier:
    """Classifies PTX kernel domains using a HuggingFace causal LM."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 cache_path: Optional[str] = None,
                 device: str = "auto"):
        self.model_name = model_name
        self.cache_path = cache_path
        self._cache: Dict[str, str] = {}
        self._model = None
        self._tokenizer = None
        self._device = device
        self._valid_domains = set(DOMAIN_DISPLAY_ORDER) | {"UNKNOWN"}

        # Load cache from disk
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self._cache = json.load(f)
                logging.info(f"LLM cache loaded: {len(self._cache)} entries from {cache_path}")
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise RuntimeError(
                "LLM classifier requires 'transformers' and 'torch'.\n"
                "Install with: pip install transformers torch accelerate"
            )

        print(f"  Loading LLM: {self.model_name} ...", end="", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()
        print(" ready.")

    @staticmethod
    def _extract_ptx_bodies(src: str, max_chars: int = LLM_MAX_SRC_CHARS) -> str:
        """Extract the most informative parts of a PTX file for the LLM.

        Strategy:
        1. Always include the header (.version, .target, .address_size)
        2. Find .entry / .func boundaries by line scanning (no backtracking regex)
        3. For each kernel body, include up to a per-kernel budget
        4. If the file is small enough, include it entirely
        """
        if len(src) <= max_chars:
            return src

        lines = src.split('\n')
        parts: List[str] = []
        budget = max_chars

        # --- Header: first 40 lines (version, target, address_size, externs)
        header = '\n'.join(lines[:min(40, len(lines))])
        parts.append(header)
        budget -= len(header)
        if budget <= 0:
            return header[:max_chars]

        # --- Find kernel start lines by simple line-level scan
        entry_re = re.compile(r'^\s*\.(?:visible\s+)?(?:entry|func)\s+')
        kernel_starts: List[int] = []
        for i, line in enumerate(lines):
            if entry_re.match(line):
                kernel_starts.append(i)

        if not kernel_starts:
            # No kernels — take first max_chars of the file
            return src[:max_chars] + '\n// ... [truncated]'

        # Build kernel boundaries: [start, next_start or EOF)
        kernel_ranges = []
        for idx, start in enumerate(kernel_starts):
            end = kernel_starts[idx + 1] if idx + 1 < len(kernel_starts) else len(lines)
            kernel_ranges.append((start, end))

        # Distribute remaining budget equally across kernels
        per_kernel = max(400, budget // max(len(kernel_ranges), 1))

        for start, end in kernel_ranges:
            body = '\n'.join(lines[start:end])
            if len(body) <= per_kernel:
                parts.append(body)
            else:
                # Take the beginning (signature + first instructions)
                # and end (ret/exit) of the kernel
                half = per_kernel // 2
                parts.append(body[:half])
                parts.append('// ... [middle truncated]')
                parts.append(body[-min(half, 300):])

        result = '\n'.join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + '\n// ... [truncated]'
        return result

    def _build_fingerprint(self, result: Dict, src_snippet: str = "") -> str:
        """Build the user prompt for the LLM: filename + raw PTX code.

        The LLM reads the actual code (kernel names, instructions, memory
        patterns) to decide the domain — no pre-digested statistics.
        """
        parts: List[str] = []
        parts.append(f"File: {result.get('file', '?')}")

        if src_snippet:
            ptx_code = self._extract_ptx_bodies(src_snippet)
            parts.append(ptx_code)
        else:
            parts.append("(no PTX source available)")

        return "\n".join(parts)

    def _cache_key(self, fingerprint: str) -> str:
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def classify(self, result: Dict, src_snippet: str = "") -> str:
        """Classify a single PTX file's domain via the LLM.

        Args:
            result:      Per-file analysis dict from analyze_file().
            src_snippet: The full (comment-stripped) PTX source code.
                         Automatically truncated to fit the context window.
        """
        import torch

        fingerprint = self._build_fingerprint(result, src_snippet)
        key = self._cache_key(fingerprint)

        # Check cache
        if key in self._cache:
            return self._cache[key]

        # Empty stubs (0-instruction header-only files) → DATA_MODULE
        if result.get("total_static_instructions", 0) == 0:
            self._cache[key] = "DATA_MODULE"
            return "DATA_MODULE"

        self._load_model()

        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user",   "content": fingerprint},
        ]

        # Use the chat template if available, else fall back to manual formatting
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = (f"<|system|>\n{LLM_SYSTEM_PROMPT}<|end|>\n"
                    f"<|user|>\n{fingerprint}<|end|>\n<|assistant|>\n")

        # Tokenize — cap at 4096 tokens to fit V100-16GB comfortably
        inputs = self._tokenizer(text, return_tensors="pt",
                                  truncation=True, max_length=4096)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse: extract the first valid domain label from the response.
        # Sort by length (longest first) so specific names match before
        # shorter substrings (e.g. AI_TRAINING before AI_LLM).
        domain = "UNKNOWN"
        response_upper = response.upper().replace(" ", "_").replace("-", "_")
        for d in sorted(self._valid_domains - {"UNKNOWN"}, key=len, reverse=True):
            if d in response_upper:
                domain = d
                break

        self._cache[key] = domain
        return domain

    def classify_batch(self, results: List[Dict],
                       src_snippets: Optional[List[str]] = None,
                       batch_size: int = 8) -> List[str]:
        """Classify a list of PTX analysis results in batches.

        Args:
            results:      List of per-file analysis dicts.
            src_snippets: Parallel list of comment-stripped PTX sources.
                          If None, fingerprints are built without code.
            batch_size:   Number of files per LLM forward pass.
        """
        import torch

        domains = []
        uncached_indices = []
        uncached_fingerprints = []

        # Separate cached from uncached
        for i, r in enumerate(results):
            snippet = src_snippets[i] if src_snippets else ""
            fp = self._build_fingerprint(r, snippet)
            key = self._cache_key(fp)
            if key in self._cache:
                domains.append(self._cache[key])
            elif r.get("total_static_instructions", 0) == 0:
                self._cache[key] = "UNKNOWN"
                domains.append("UNKNOWN")
            else:
                domains.append(None)  # placeholder
                uncached_indices.append(i)
                uncached_fingerprints.append(fp)

        if not uncached_fingerprints:
            return domains

        self._load_model()

        # Process in batches
        for batch_start in range(0, len(uncached_fingerprints), batch_size):
            batch_fps = uncached_fingerprints[batch_start:batch_start + batch_size]
            batch_idx = uncached_indices[batch_start:batch_start + batch_size]

            texts = []
            for fp in batch_fps:
                messages = [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user",   "content": fp},
                ]
                if hasattr(self._tokenizer, "apply_chat_template"):
                    text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    text = (f"<|system|>\n{LLM_SYSTEM_PROMPT}<|end|>\n"
                            f"<|user|>\n{fp}<|end|>\n<|assistant|>\n")
                texts.append(text)

            # Tokenize batch with left-padding for generation
            self._tokenizer.padding_side = "left"
            inputs = self._tokenizer(texts, return_tensors="pt",
                                      padding=True, truncation=True,
                                      max_length=4096)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode each response
            for j, (idx, fp) in enumerate(zip(batch_idx, batch_fps)):
                new_tokens = outputs[j][inputs["input_ids"].shape[1]:]
                response = self._tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()

                domain = "UNKNOWN"
                response_upper = response.upper().replace(" ", "_").replace("-", "_")
                for d in self._valid_domains:
                    if d in response_upper:
                        domain = d
                        break

                key = self._cache_key(fp)
                self._cache[key] = domain
                domains[idx] = domain

        return domains

    def save_cache(self):
        """Persist the cache to disk."""
        if self.cache_path and self._cache:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
            logging.info(f"LLM cache saved: {len(self._cache)} entries → {self.cache_path}")


# Module-level LLM classifier instance (set in main() when --llm is used)
_llm_classifier: Optional[LLMDomainClassifier] = None


# Cache modifier suffixes on ld/st — affect L2 bypass / eviction policy
CACHE_MODIFIERS = {
    ".ca": "cache-all (L1+L2)",
    ".cg": "cache-global (L2 only, bypass L1)",
    ".cs": "cache-streaming (likely evict-first)",
    ".cv": "cache-volatile (bypass all caches)",
    ".lu": "last-use (evict after use)",
    ".wb": "write-back (default store)",
    ".wt": "write-through (bypass L1 on store)",
}

# Stride thresholds (in bytes) for coalescing heuristics
COALESCED_STRIDE_MAX   = 16   # ≤ 16-byte stride → likely coalesced for 32 threads (128B transaction)
BANK_CONFLICT_STRIDE   = 128  # multiples that hit same shared mem bank (32 banks × 4 B)


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns (compiled once)
# ─────────────────────────────────────────────────────────────────────────────

RE_VERSION       = re.compile(r'\.version\s+([\d.]+)')
RE_TARGET        = re.compile(r'\.target\s+(sm_\w+)')
RE_ENTRY         = re.compile(r'\.(visible\s+)?\.?entry\s+(\w+)')
RE_COMMENT_LINE  = re.compile(r'//.*$', re.MULTILINE)
RE_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)

# Instruction: optional predicate + opcode (handles @%p0 prefix)
RE_INSTRUCTION   = re.compile(
    r'^\s+(?:@!?%\w+\s+)?'          # optional predicate guard
    r'([a-z][a-z0-9]*'              # base opcode
    r'(?:\.[a-z0-9_]+)*)'           # type/modifier suffixes
    r'\s',
    re.MULTILINE
)

# Memory access patterns
RE_LD_GLOBAL     = re.compile(r'\bld(?:u)?\.global(\.\w+)*\b')
RE_ST_GLOBAL     = re.compile(r'\bst\.global(\.\w+)*\b')
RE_LD_SHARED     = re.compile(r'\bld\.shared(\.\w+)*\b')
RE_ST_SHARED     = re.compile(r'\bst\.shared(\.\w+)*\b')
RE_LD_LOCAL      = re.compile(r'\bld\.local(\.\w+)*\b')
RE_ST_LOCAL      = re.compile(r'\bst\.local(\.\w+)*\b')
RE_LD_CONST      = re.compile(r'\bld\.const(\.\w+)*\b')
RE_ATOM_GLOBAL   = re.compile(r'\batom\.global(\.\w+)*\b')
RE_ATOM_SHARED   = re.compile(r'\batom\.shared(\.\w+)*\b')
RE_RED_GLOBAL    = re.compile(r'\bred\.global(\.\w+)*\b')
RE_PREFETCH      = re.compile(r'\bprefetch(?:u)?\b')
RE_CP_ASYNC      = re.compile(r'\bcp\.async\b')

# Cache modifier on ld/st
RE_CACHE_MOD     = re.compile(
    r'\b(?:ld|st|ldu)\.(?:global|shared|local|const|param)'
    r'(\.ca|\.cg|\.cs|\.cv|\.lu|\.wb|\.wt)'
)

# Shared memory declarations: .shared .bXX name[size]
RE_SHARED_DECL   = re.compile(
    r'\.shared\s+\.\w+\s+(\w+)\s*\[\s*(\d+)\s*\]'
)

# Register declarations: .reg .bXX %name<count>
RE_REG_DECL      = re.compile(r'\.reg\s+\.\w+\s+%\w+<(\d+)>')

# Stride pattern: mul.lo.<type> %rX, %rY, <const>
RE_STRIDE_MUL    = re.compile(
    r'\bmul\.(?:lo|hi|wide)\.\w+\s+(%\w+),\s*%\w+,\s*(\d+)'
)

# Shared memory indexed access: [smemname + %rX]  or [%rdX + offset]
RE_SHARED_ACCESS = re.compile(r'\[(?:\w+\s*\+\s*)?(%\w+)(?:\s*\+\s*\d+)?\]')

# Warp divergence: predicated branch
RE_PRED_BRANCH   = re.compile(r'@!?%p\w+\s+bra\b')
RE_SETP          = re.compile(r'\bsetp\b')

# Special registers (for SM utilization awareness)
RE_SREG          = re.compile(r'%(?:tid|ctaid|ntid|nctaid|warpid|laneid|'
                               r'gridid|lanemask_lt|lanemask_le|lanemask_gt|'
                               r'lanemask_ge|lanemask_eq|clock|clock64|'
                               r'pm\d+|smid|nsmid)\b')

# Vectorized loads: .v2 / .v4
RE_VECTOR_LD     = re.compile(r'\bld(?:u)?\.(?:global|shared)\.\w*\.v[24]\b')
RE_VECTOR_ST     = re.compile(r'\bst\.(?:global|shared)\.\w*\.v[24]\b')

# Precision type suffixes recognized in PTX opcodes
PRECISION_TYPES = {
    "f16":  re.compile(r'\.f16\b'),
    "bf16": re.compile(r'\.bf16\b'),
    "f32":  re.compile(r'\.f32\b'),
    "f64":  re.compile(r'\.f64\b'),
    "s8":   re.compile(r'\.s8\b'),
    "s16":  re.compile(r'\.s16\b'),
    "s32":  re.compile(r'\.s32\b'),
    "s64":  re.compile(r'\.s64\b'),
    "u8":   re.compile(r'\.u8\b'),
    "u16":  re.compile(r'\.u16\b'),
    "u32":  re.compile(r'\.u32\b'),
    "u64":  re.compile(r'\.u64\b'),
    "b16":  re.compile(r'\.b16\b'),
    "b32":  re.compile(r'\.b32\b'),
    "b64":  re.compile(r'\.b64\b'),
    "pred": re.compile(r'\.pred\b'),
}

# Opcodes classified by functional role for compute/overhead/memory split
COMPUTE_BASES = {"fma", "mul", "mad", "mul24", "mad24", "add", "sub", "div",
                 "rem", "abs", "neg", "min", "max", "rcp", "sqrt", "rsqrt",
                 "sin", "cos", "lg2", "ex2", "tanh", "dp4a", "dp2a",
                 "wmma", "mma", "setp", "set", "selp", "slct",
                 "vadd", "vsub", "vabsdiff", "vmin", "vmax", "vmad", "vset"}
MEMORY_BASES  = {"ld", "st", "ldu", "prefetch", "prefetchu", "atom", "red",
                 "tex", "tld4", "suld", "sust", "sured", "cp"}
OVERHEAD_BASES = {"mov", "cvt", "cvta", "cvtp", "shfl", "prmt",
                  "shl", "shr", "and", "or", "xor", "not", "cnot",
                  "bfe", "bfi", "bfind", "brev", "popc", "clz",
                  "bra", "call", "ret", "exit",
                  "bar", "membar", "fence", "vote", "activemask", "match"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-file analysis
# ─────────────────────────────────────────────────────────────────────────────

def strip_comments(src: str) -> str:
    src = RE_BLOCK_COMMENT.sub(' ', src)
    src = RE_COMMENT_LINE.sub('', src)
    return src


def classify_instruction(opcode: str) -> str:
    """Return the category for a full opcode like 'ld.global.f32'."""
    base = opcode.split('.')[0]
    for cat, bases in INSTRUCTION_CATEGORIES.items():
        if base in bases:
            return cat
    return "other"


def detect_stride_patterns(src: str) -> Dict:
    """
    Heuristic: look for  mul.lo.* %rX, %rY, CONST  immediately before
    a global/shared memory access.  CONST is the per-thread stride in elements.
    Returns coalesced / uncoalesced / shared_bank_risk counts.
    """
    coalesced = uncoalesced = shared_bank_risk = 0
    stride_values: List[int] = []

    for m in RE_STRIDE_MUL.finditer(src):
        stride_reg   = m.group(1)
        stride_bytes = int(m.group(2))
        stride_values.append(stride_bytes)

        # Look ahead ~5 lines for an ld/st using this register
        pos   = m.end()
        chunk = src[pos: pos + 400]
        uses_global = bool(RE_LD_GLOBAL.search(chunk) or RE_ST_GLOBAL.search(chunk))
        uses_shared = bool(RE_LD_SHARED.search(chunk) or RE_ST_SHARED.search(chunk))

        if uses_global:
            if stride_bytes <= COALESCED_STRIDE_MAX:
                coalesced += 1
            else:
                uncoalesced += 1

        if uses_shared:
            # Bank conflict if stride is a multiple of 128 bytes (32 banks × 4B)
            # or exactly a power-of-two ≥ 128
            if stride_bytes > 0 and (stride_bytes % BANK_CONFLICT_STRIDE == 0
                                      or (stride_bytes >= 128 and
                                          (stride_bytes & (stride_bytes - 1)) == 0)):
                shared_bank_risk += 1

    return {
        "coalesced_patterns":      coalesced,
        "uncoalesced_patterns":    uncoalesced,
        "shared_bank_risk":        shared_bank_risk,
        "stride_values_histogram": dict(Counter(stride_values)),
    }


def analyze_file(path: Path) -> Dict:
    try:
        src_raw = path.read_text(errors='replace')
    except OSError as e:
        return {"error": str(e)}

    src = strip_comments(src_raw)

    result: Dict = {
        "file":     path.name,
        "path":     str(path),
        "size_kb":  round(path.stat().st_size / 1024, 2),
    }

    # ── Header metadata ──────────────────────────────────────────────────────
    vm = RE_VERSION.search(src)
    tm = RE_TARGET.search(src)
    result["ptx_version"] = vm.group(1) if vm else "unknown"
    result["sm_target"]   = tm.group(1) if tm else "unknown"

    entries = RE_ENTRY.findall(src)
    result["kernel_count"] = len(entries)
    result["kernel_names"] = [e[1] for e in entries]

    # ── Register pressure ────────────────────────────────────────────────────
    reg_counts = [int(x) for x in RE_REG_DECL.findall(src)]
    result["total_virtual_regs"]   = sum(reg_counts)
    result["max_vreg_per_kernel"]  = max(reg_counts) if reg_counts else 0
    result["mean_vreg_per_kernel"] = (round(sum(reg_counts) / len(reg_counts), 1)
                                      if reg_counts else 0)

    # ── Shared memory ────────────────────────────────────────────────────────
    shared_decls = RE_SHARED_DECL.findall(src)
    shared_sizes = [int(s) for _, s in shared_decls]
    result["shared_mem_allocations"] = len(shared_decls)
    result["shared_mem_total_bytes"] = sum(shared_sizes)
    result["shared_mem_max_bytes"]   = max(shared_sizes) if shared_sizes else 0

    # ── Instruction counts ───────────────────────────────────────────────────
    all_opcodes   = RE_INSTRUCTION.findall(src)
    instr_counter = Counter(all_opcodes)
    result["total_static_instructions"] = len(all_opcodes)

    # Category totals
    cat_counter: Counter = Counter()
    for op, cnt in instr_counter.items():
        cat_counter[classify_instruction(op)] += cnt
    result["instruction_category_counts"] = dict(cat_counter)

    # Top-20 most frequent opcodes
    result["top20_opcodes"] = dict(instr_counter.most_common(20))

    # Unique opcode coverage
    result["unique_opcodes"]  = len(instr_counter)
    result["opcode_histogram"] = dict(instr_counter)

    # ── Precision breakdown ──────────────────────────────────────────────────
    precision_counter: Counter = Counter()
    for op, cnt in instr_counter.items():
        for prec, pat in PRECISION_TYPES.items():
            if pat.search(op):
                precision_counter[prec] += cnt
                break  # first match wins (e.g. cvt.f32.f16 counts as f32)
    result["precision_counts"] = dict(precision_counter)
    total_typed = sum(precision_counter.values())
    result["precision_pct"] = {
        p: round(c / total_typed * 100, 2) for p, c in precision_counter.items()
    } if total_typed > 0 else {}

    # ── Compute / Overhead / Memory split ────────────────────────────────────
    compute_total = overhead_total = memory_total = 0
    for op, cnt in instr_counter.items():
        base = op.split('.')[0]
        if base in COMPUTE_BASES:
            compute_total += cnt
        elif base in MEMORY_BASES:
            memory_total += cnt
        elif base in OVERHEAD_BASES:
            overhead_total += cnt
        else:
            overhead_total += cnt  # unknown → overhead
    result["functional_split"] = {
        "compute":  compute_total,
        "memory":   memory_total,
        "overhead": overhead_total,
    }
    n_all = compute_total + memory_total + overhead_total
    result["functional_split_pct"] = {
        "compute":  round(compute_total / n_all * 100, 1),
        "memory":   round(memory_total  / n_all * 100, 1),
        "overhead": round(overhead_total / n_all * 100, 1),
    } if n_all > 0 else {}

    # ── Memory access breakdown ──────────────────────────────────────────────
    mem = {
        "ld_global":    len(RE_LD_GLOBAL.findall(src)),
        "st_global":    len(RE_ST_GLOBAL.findall(src)),
        "ld_shared":    len(RE_LD_SHARED.findall(src)),
        "st_shared":    len(RE_ST_SHARED.findall(src)),
        "ld_local":     len(RE_LD_LOCAL.findall(src)),
        "st_local":     len(RE_ST_LOCAL.findall(src)),
        "ld_const":     len(RE_LD_CONST.findall(src)),
        "atom_global":  len(RE_ATOM_GLOBAL.findall(src)),
        "atom_shared":  len(RE_ATOM_SHARED.findall(src)),
        "red_global":   len(RE_RED_GLOBAL.findall(src)),
        "prefetch":     len(RE_PREFETCH.findall(src)),
        "cp_async":     len(RE_CP_ASYNC.findall(src)),
        "vector_ld":    len(RE_VECTOR_LD.findall(src)),
        "vector_st":    len(RE_VECTOR_ST.findall(src)),
    }
    result["memory_access_counts"] = mem

    # Cache modifier usage
    cache_mod_hits = RE_CACHE_MOD.findall(src)
    result["cache_modifier_counts"] = dict(Counter(cache_mod_hits))

    # ── L2 contention analysis ───────────────────────────────────────────────
    stride_info = detect_stride_patterns(src)
    result["stride_analysis"] = stride_info

    global_total = mem["ld_global"] + mem["st_global"]
    # L2 pressure score: atomics count 4×, uncoalesced 3×, global 1×
    l2_score = (
        global_total * 1 +
        mem["atom_global"] * 4 +
        mem["red_global"]  * 3 +
        stride_info["uncoalesced_patterns"] * 3
    )
    result["l2_pressure_score"] = l2_score

    # Coalescing ratio: coalesced / (coalesced + uncoalesced)
    c = stride_info["coalesced_patterns"]
    u = stride_info["uncoalesced_patterns"]
    result["coalescing_ratio"] = round(c / (c + u), 3) if (c + u) > 0 else None

    # Shared-to-global memory ratio (higher → less L2 pressure if smem reuse is good)
    shared_total = mem["ld_shared"] + mem["st_shared"]
    result["shared_to_global_ratio"] = (
        round(shared_total / global_total, 3) if global_total > 0 else None
    )

    # Atomic contention density (atomics per 1000 instructions)
    if result["total_static_instructions"] > 0:
        result["atomic_density_per1k"] = round(
            (mem["atom_global"] + mem["red_global"]) / result["total_static_instructions"] * 1000, 2
        )
    else:
        result["atomic_density_per1k"] = 0

    # ── Warp divergence ──────────────────────────────────────────────────────
    result["warp_divergence"] = {
        "predicated_branches": len(RE_PRED_BRANCH.findall(src)),
        "setp_count":          len(RE_SETP.findall(src)),
    }

    # ── Special register usage ───────────────────────────────────────────────
    sreg_hits = RE_SREG.findall(src)
    result["special_register_uses"] = len(sreg_hits)
    result["special_register_types"] = dict(Counter(sreg_hits))

    # ── Domain classification (scoring-based) ────────────────────────────────
    # Minimum score threshold: domains scoring below this are discarded
    # to reduce weak false-positive classifications
    MIN_DOMAIN_SCORE = 8

    domain_scores: Dict[str, int] = {}
    for domain, cfg in DOMAIN_CLASSIFIER.items():
        if any(re.search(ex, src) for ex in cfg.get("exclusions", [])):
            continue
        score = 0
        for pattern, weight in cfg["signals"]:
            hits = len(re.findall(pattern, src))
            score += hits * weight
        if score >= MIN_DOMAIN_SCORE:
            domain_scores[domain] = score

    # ── Filename-based hinting for AI/ML kernels ─────────────────────────────
    # This dataset consists of compiler-generated PTX from CUDA DL kernels.
    # Kernels named conv2d, matmul, mlp, relu, linear, etc. are clearly
    # neural-network workloads but lack tensor-core ops (compiled for sm_87
    # without wmma/mma.sync). Use filename as a tiebreaker hint.
    filename_lower = path.name.lower()
    kernel_name_str = " ".join(result.get("kernel_names", [])).lower()
    combined_names = filename_lower + " " + kernel_name_str

    ai_name_patterns = [
        "conv2d", "conv_", "matmul", "gemm", "mlp", "linear", "relu",
        "resnet", "transformer", "attention", "batchnorm", "layernorm",
        "softmax", "sigmoid", "dropout", "pooling", "depthwise",
        "pointwise", "backward", "gradient", "weight_update",
    ]
    has_ai_name = any(p in combined_names for p in ai_name_patterns)

    if has_ai_name and "AI_TRAINING" not in domain_scores:
        # If AI name detected but AI_TRAINING didn't score (no tensor core ops),
        # add a baseline AI_TRAINING score from non-tensor-core signals
        ai_baseline = 0
        # Count fma.rn.ftz.f32 (flush-to-zero common in NN f32 kernels)
        fma_ftz_count = len(re.findall(r'\bfma\.rn\.ftz\.f32\b', src))
        relu_count = len(re.findall(r'\bmax\.ftz\.f32\b', src)) + len(re.findall(r'\bmax\.f32\b', src))
        ai_baseline += fma_ftz_count * 3 + relu_count * 4
        if ai_baseline >= MIN_DOMAIN_SCORE:
            domain_scores["AI_TRAINING"] = ai_baseline

    sorted_domains = sorted(domain_scores.items(), key=lambda x: -x[1])
    result["domain_scores"]  = domain_scores
    result["domains"]        = [d for d, _ in sorted_domains]

    # 0-instruction files with actual content are data-declaration stubs
    # (e.g. .global .align declarations, no .entry kernel body)
    if result["total_static_instructions"] == 0 and result["size_kb"] > 0:
        result["primary_domain"] = "DATA_MODULE"
    else:
        result["primary_domain"] = sorted_domains[0][0] if sorted_domains else "UNKNOWN"

    result["top3_domains"]   = sorted_domains[:3]

    # ── LLM-based domain classification (primary when active) ─────────────────
    # When an LLM classifier is available it is the authoritative source.
    # The regex scores are kept for reference / fallback only.
    if _llm_classifier is not None:
        result["regex_domain"] = result["primary_domain"]
        try:
            llm_domain = _llm_classifier.classify(result, src_snippet=src)
        except Exception as e:
            logging.warning(f"LLM classify failed for {path.name}: {e}")
            llm_domain = "UNKNOWN"
        result["llm_domain"] = llm_domain

        # LLM is primary; fall back to regex only when LLM says UNKNOWN
        if llm_domain != "UNKNOWN":
            result["primary_domain"] = llm_domain
        # else: keep whatever regex/DATA_MODULE decided

    primary = result["primary_domain"]
    result["domain_group"] = next(
        (grp for grp, members in DOMAIN_GROUPS.items() if primary in members),
        "Other"
    )

    # ── Compute intensity heuristic ──────────────────────────────────────────
    compute_ops = sum(
        cat_counter.get(c, 0)
        for c in ("integer_arith", "float_arith", "bit_manip",
                  "video_dp", "tensor_core")
    )
    memory_ops = (global_total + shared_total +
                  mem["ld_local"] + mem["st_local"] + mem["ld_const"])
    result["compute_ops"]  = compute_ops
    result["memory_ops"]   = memory_ops
    result["arithmetic_intensity"] = (
        round(compute_ops / memory_ops, 3) if memory_ops > 0 else None
    )

    # ── L2 contention risk level ─────────────────────────────────────────────
    if l2_score == 0:
        risk = "NONE"
    elif l2_score < 50:
        risk = "LOW"
    elif l2_score < 200:
        risk = "MEDIUM"
    elif l2_score < 600:
        risk = "HIGH"
    else:
        risk = "CRITICAL"
    result["l2_contention_risk"] = risk

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(records: List[Dict]) -> Dict:
    valid = [r for r in records if "error" not in r]
    if not valid:
        return {}

    # Instruction coverage across entire dataset
    global_opcode_counter: Counter = Counter()
    global_cat_counter:    Counter = Counter()
    for r in valid:
        global_opcode_counter.update(r.get("opcode_histogram", {}))
        global_cat_counter.update(r.get("instruction_category_counts", {}))

    total_instr = sum(global_opcode_counter.values())

    # Instruction coverage: per-base coverage (rough) + per-qualified-opcode coverage
    all_known_bases = set()
    for bases in INSTRUCTION_CATEGORIES.values():
        all_known_bases.update(bases)
    seen_bases = {op.split('.')[0] for op in global_opcode_counter}
    base_coverage_pct = round(len(seen_bases & all_known_bases) / len(all_known_bases) * 100, 1)

    # Type-qualified coverage: how many distinct fully-qualified opcodes the dataset contains
    # vs. a realistic estimate of common ISA variants (base × typical type suffixes)
    COMMON_SUFFIXES = [
        ".f16", ".bf16", ".f32", ".f64",
        ".s8", ".s16", ".s32", ".s64",
        ".u8", ".u16", ".u32", ".u64",
        ".b16", ".b32", ".b64", ".pred",
    ]
    expected_qualified = set()
    for base in all_known_bases:
        expected_qualified.add(base)  # base alone (bra, ret, exit, bar...)
        for suf in COMMON_SUFFIXES:
            expected_qualified.add(base + suf)
    seen_qualified = set(global_opcode_counter.keys())
    qualified_overlap = seen_qualified & expected_qualified
    qualified_coverage_pct = round(
        len(qualified_overlap) / len(expected_qualified) * 100, 1
    ) if expected_qualified else 0.0

    # Domain distribution (scoring-based)
    primary_domain_counter: Counter = Counter(r.get("primary_domain", "UNKNOWN") for r in valid)
    domain_group_counter:   Counter = Counter(r.get("domain_group", "Other") for r in valid)

    # Per-domain file counts (files where domain is in top-3)
    domain_counter: Counter = Counter()
    for r in valid:
        for d in r.get("domains", []):
            domain_counter[d] += 1

    # Mean confidence score per primary domain
    domain_score_sums: dict = {}
    domain_score_cnts: dict = {}
    for r in valid:
        pd = r.get("primary_domain", "UNKNOWN")
        sc = r.get("domain_scores", {}).get(pd, 0)
        domain_score_sums[pd] = domain_score_sums.get(pd, 0) + sc
        domain_score_cnts[pd] = domain_score_cnts.get(pd, 0) + 1
    domain_mean_scores = {
        d: round(domain_score_sums[d] / domain_score_cnts[d], 1)
        for d in domain_score_sums
    }

    # Memory stats
    def _sum(key, sub=None):
        if sub:
            return sum(r.get(key, {}).get(sub, 0) for r in valid)
        return sum(r.get(key, 0) for r in valid)

    # L2 risk distribution
    risk_dist = Counter(r.get("l2_contention_risk", "UNKNOWN") for r in valid)

    # Coalescing stats
    coal_vals = [r["coalescing_ratio"] for r in valid if r.get("coalescing_ratio") is not None]
    mean_coal = round(sum(coal_vals) / len(coal_vals), 3) if coal_vals else None

    # SM target distribution
    sm_dist = Counter(r.get("sm_target", "unknown") for r in valid)

    # PTX version distribution
    ver_dist = Counter(r.get("ptx_version", "unknown") for r in valid)

    # Shared memory stats
    shared_sizes = [r.get("shared_mem_max_bytes", 0) for r in valid if r.get("shared_mem_max_bytes", 0) > 0]
    kernels_using_smem = sum(1 for r in valid if r.get("shared_mem_allocations", 0) > 0)

    # Atomics
    total_atom_global = _sum("memory_access_counts", "atom_global")
    total_red_global  = _sum("memory_access_counts", "red_global")
    total_atom_shared = _sum("memory_access_counts", "atom_shared")

    # Warp divergence
    total_pred_branches = sum(
        r.get("warp_divergence", {}).get("predicated_branches", 0) for r in valid
    )

    # Tensor core usage (AI_TRAINING or AI_INFERENCE primary)
    files_with_tensor = sum(1 for r in valid
                            if r.get("primary_domain", "") in ("AI_TRAINING", "AI_INFERENCE", "AI_LLM"))

    # Async copy (Ampere+ pipeline)
    files_with_async = sum(
        1 for r in valid if r.get("memory_access_counts", {}).get("cp_async", 0) > 0
    )

    # Vector loads
    total_vld = _sum("memory_access_counts", "vector_ld")
    total_vst = _sum("memory_access_counts", "vector_st")

    # Files with prefetch
    files_with_prefetch = sum(
        1 for r in valid if r.get("memory_access_counts", {}).get("prefetch", 0) > 0
    )

    # Cache modifier distribution across dataset
    cache_mod_total: Counter = Counter()
    for r in valid:
        cache_mod_total.update(r.get("cache_modifier_counts", {}))

    # Stride histogram aggregated
    stride_hist: Counter = Counter()
    for r in valid:
        stride_hist.update(r.get("stride_analysis", {}).get("stride_values_histogram", {}))

    # Precision breakdown aggregated
    global_precision: Counter = Counter()
    for r in valid:
        global_precision.update(r.get("precision_counts", {}))
    total_typed = sum(global_precision.values())

    # Functional split aggregated
    global_compute  = sum(r.get("functional_split", {}).get("compute", 0) for r in valid)
    global_memory   = sum(r.get("functional_split", {}).get("memory", 0) for r in valid)
    global_overhead = sum(r.get("functional_split", {}).get("overhead", 0) for r in valid)
    func_total = global_compute + global_memory + global_overhead

    return {
        "dataset_summary": {
            "total_files":          len(records),
            "valid_files":          len(valid),
            "error_files":          len(records) - len(valid),
            "total_kernels":        sum(r.get("kernel_count", 0) for r in valid),
            "total_static_instructions": total_instr,
        },
        "ptx_metadata": {
            "ptx_version_distribution": dict(ver_dist),
            "sm_target_distribution":   dict(sm_dist),
        },
        "instruction_coverage": {
            "unique_opcodes_seen":       len(global_opcode_counter),
            "known_base_coverage_pct":   base_coverage_pct,
            "qualified_coverage_pct":    qualified_coverage_pct,
            "qualified_opcodes_seen":    len(qualified_overlap),
            "qualified_opcodes_expected": len(expected_qualified),
            "top30_opcodes":             dict(global_opcode_counter.most_common(30)),
            "category_totals":           dict(global_cat_counter),
            "category_pct": {
                cat: round(cnt / total_instr * 100, 2)
                for cat, cnt in global_cat_counter.items()
            } if total_instr > 0 else {},
        },
        "domain_coverage": {
            "primary_domain_dist":     dict(primary_domain_counter),
            "domain_group_dist":       dict(domain_group_counter),
            "domain_file_counts":      dict(domain_counter),
            "domain_mean_scores":      domain_mean_scores,
            "files_with_ai_kernels":   files_with_tensor,
            "files_with_async_copy":   files_with_async,
        },
        "l2_contention_analysis": {
            "risk_level_distribution": dict(risk_dist),
            "total_global_loads":      _sum("memory_access_counts", "ld_global"),
            "total_global_stores":     _sum("memory_access_counts", "st_global"),
            "total_atom_global":       total_atom_global,
            "total_red_global":        total_red_global,
            "total_atom_shared":       total_atom_shared,
            "mean_l2_pressure_score":  round(
                sum(r.get("l2_pressure_score", 0) for r in valid) / len(valid), 1
            ),
            "mean_coalescing_ratio":   mean_coal,
            "total_uncoalesced_patterns": sum(
                r.get("stride_analysis", {}).get("uncoalesced_patterns", 0) for r in valid
            ),
            "total_coalesced_patterns": sum(
                r.get("stride_analysis", {}).get("coalesced_patterns", 0) for r in valid
            ),
            "total_shared_bank_risk":   sum(
                r.get("stride_analysis", {}).get("shared_bank_risk", 0) for r in valid
            ),
            "stride_value_histogram":   {str(k): v for k, v in
                                         sorted(stride_hist.items(), key=lambda x: x[1], reverse=True)[:30]},
            "cache_modifier_distribution": dict(cache_mod_total),
            "vector_ld_count":         total_vld,
            "vector_st_count":         total_vst,
            "files_with_prefetch":     files_with_prefetch,
        },
        "memory_breakdown": {
            "ld_global":   _sum("memory_access_counts", "ld_global"),
            "st_global":   _sum("memory_access_counts", "st_global"),
            "ld_shared":   _sum("memory_access_counts", "ld_shared"),
            "st_shared":   _sum("memory_access_counts", "st_shared"),
            "ld_local":    _sum("memory_access_counts", "ld_local"),
            "st_local":    _sum("memory_access_counts", "st_local"),
            "ld_const":    _sum("memory_access_counts", "ld_const"),
            "atom_global": total_atom_global,
            "atom_shared": total_atom_shared,
            "red_global":  total_red_global,
            "cp_async":    _sum("memory_access_counts", "cp_async"),
        },
        "shared_memory_analysis": {
            "files_using_shared_mem":   kernels_using_smem,
            "max_smem_bytes_any_kernel": max(shared_sizes) if shared_sizes else 0,
            "mean_smem_bytes":          round(sum(shared_sizes) / len(shared_sizes), 1) if shared_sizes else 0,
            "total_smem_bytes_all":     sum(r.get("shared_mem_total_bytes", 0) for r in valid),
        },
        "warp_divergence": {
            "total_predicated_branches": total_pred_branches,
            "mean_pred_branches_per_file": round(total_pred_branches / len(valid), 1),
            "files_with_divergence": sum(
                1 for r in valid
                if r.get("warp_divergence", {}).get("predicated_branches", 0) > 0
            ),
        },
        "register_pressure": {
            "mean_max_vreg_per_file": round(
                sum(r.get("max_vreg_per_kernel", 0) for r in valid) / len(valid), 1
            ),
            "max_vreg_any_file": max(r.get("max_vreg_per_kernel", 0) for r in valid),
        },
        "precision_breakdown": {
            "counts":  dict(global_precision),
            "pct": {
                p: round(c / total_typed * 100, 2) for p, c in global_precision.items()
            } if total_typed > 0 else {},
        },
        "functional_split": {
            "compute":  global_compute,
            "memory":   global_memory,
            "overhead": global_overhead,
            "compute_pct":  round(global_compute  / func_total * 100, 1) if func_total else 0,
            "memory_pct":   round(global_memory   / func_total * 100, 1) if func_total else 0,
            "overhead_pct": round(global_overhead  / func_total * 100, 1) if func_total else 0,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text report printer
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "CRITICAL": "\033[91m",
    "HIGH":     "\033[93m",
    "MEDIUM":   "\033[94m",
    "LOW":      "\033[92m",
    "NONE":     "\033[90m",
}
RESET = "\033[0m"


def bar(value: float, total: float, width: int = 30) -> str:
    if total <= 0:
        return " " * width
    filled = int(round(value / total * width))
    return "█" * filled + "░" * (width - filled)


def print_report(agg: Dict, records: List[Dict], use_color: bool = True) -> None:
    def c(color, text):
        return f"{RISK_COLORS.get(color, '')}{text}{RESET}" if use_color else text

    ds  = agg.get("dataset_summary", {})
    ic  = agg.get("instruction_coverage", {})
    dc  = agg.get("domain_coverage", {})
    l2  = agg.get("l2_contention_analysis", {})
    mb  = agg.get("memory_breakdown", {})
    sm  = agg.get("shared_memory_analysis", {})
    wd  = agg.get("warp_divergence", {})
    rp  = agg.get("register_pressure", {})
    meta = agg.get("ptx_metadata", {})
    prec = agg.get("precision_breakdown", {})
    fspl = agg.get("functional_split", {})

    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  PTX DATASET ANALYSIS REPORT")
    print(f"{'═' * 72}")

    # ── Dataset summary ──────────────────────────────────────────────────────
    print(f"\n{'DATASET SUMMARY':}")
    print(sep)
    print(f"  Files analyzed     : {ds['valid_files']:,}  (errors: {ds['error_files']})")
    print(f"  Total kernels      : {ds['total_kernels']:,}")
    print(f"  Total instructions : {ds['total_static_instructions']:,}")

    sm_dist  = meta.get("sm_target_distribution", {})
    ver_dist = meta.get("ptx_version_distribution", {})
    print(f"  SM targets         : {', '.join(f'{k}×{v}' for k, v in sorted(sm_dist.items()))}")
    print(f"  PTX versions       : {', '.join(f'{k}×{v}' for k, v in sorted(ver_dist.items()))}")

    # ── Instruction coverage ─────────────────────────────────────────────────
    print(f"\n{'INSTRUCTION COVERAGE':}")
    print(sep)
    print(f"  Unique opcodes seen        : {ic['unique_opcodes_seen']}")
    print(f"  Base opcode coverage       : {ic['known_base_coverage_pct']}%  (base opcodes only)")
    print(f"  Type-qualified coverage    : {ic.get('qualified_coverage_pct', '?')}%  "
          f"({ic.get('qualified_opcodes_seen', '?')}/{ic.get('qualified_opcodes_expected', '?')} variants)")

    # ── Functional split ─────────────────────────────────────────────────────
    if fspl:
        print(f"\n  Functional split (compute / memory / overhead):")
        for role in ("compute", "memory", "overhead"):
            cnt = fspl.get(role, 0)
            pct = fspl.get(f"{role}_pct", 0)
            b   = bar(pct, 100, 24)
            print(f"    {role:<12} {b}  {pct:5.1f}%  ({cnt:,})")

    # ── Precision breakdown ──────────────────────────────────────────────────
    if prec.get("pct"):
        print(f"\n  Precision breakdown:")
        prec_pct = prec["pct"]
        prec_cnt = prec.get("counts", {})
        for p, pct in sorted(prec_pct.items(), key=lambda x: -x[1]):
            cnt = prec_cnt.get(p, 0)
            b   = bar(pct, 100, 20)
            print(f"    {p:<6} {b}  {pct:6.2f}%  ({cnt:,})")

    print(f"\n  Category breakdown (% of all instructions):")

    cat_pct  = ic.get("category_pct", {})
    cat_tot  = ic.get("category_totals", {})
    for cat, pct in sorted(cat_pct.items(), key=lambda x: -x[1]):
        cnt = cat_tot.get(cat, 0)
        b   = bar(pct, 100, 24)
        print(f"    {cat:<22} {b}  {pct:6.2f}%  ({cnt:,})")

    print(f"\n  Top-15 opcodes:")
    top30 = ic.get("top30_opcodes", {})
    total_i = ds["total_static_instructions"] or 1
    for op, cnt in list(top30.items())[:15]:
        pct = cnt / total_i * 100
        b   = bar(cnt, list(top30.values())[0], 20)
        print(f"    {op:<30} {b}  {cnt:8,}  ({pct:.2f}%)")

    # ── Domain coverage ──────────────────────────────────────────────────────
    print(f"\n{'DOMAIN COVERAGE':}")
    print(sep)
    total_files   = ds["valid_files"] or 1
    pd            = dc.get("primary_domain_dist", {})
    grp_dist      = dc.get("domain_group_dist", {})
    dd            = dc.get("domain_file_counts", {})
    mean_scores   = dc.get("domain_mean_scores", {})

    # Domain groups summary first
    print(f"  Domain group overview:")
    for grp, cnt in sorted(grp_dist.items(), key=lambda x: -x[1]):
        b = bar(cnt, total_files, 24)
        print(f"    {grp:<28} {b}  {cnt:5} files  ({cnt/total_files*100:.1f}%)")

    # Per-domain breakdown ordered by DOMAIN_DISPLAY_ORDER
    print(f"\n  Primary domain breakdown (score-ranked, 1 per file):")
    print(f"    {'Domain':<28} {'Bar':24}  Files  Pct    AvgScore  Description")
    print(f"    {'-'*28} {'-'*24}  -----  -----  --------  -----------")
    printed = set()
    for d in DOMAIN_DISPLAY_ORDER:
        cnt = pd.get(d, 0)
        if cnt == 0:
            continue
        b    = bar(cnt, total_files, 24)
        pct  = cnt / total_files * 100
        sc   = mean_scores.get(d, 0)
        desc = DOMAIN_CLASSIFIER.get(d, {}).get("desc", "")[:42]
        print(f"    {d:<28} {b}  {cnt:5}  {pct:5.1f}%  {sc:8.1f}  {desc}")
        printed.add(d)
    # any domains not in display order
    for d, cnt in sorted(pd.items(), key=lambda x: -x[1]):
        if d not in printed and cnt > 0:
            b    = bar(cnt, total_files, 24)
            pct  = cnt / total_files * 100
            sc   = mean_scores.get(d, 0)
            desc = DOMAIN_CLASSIFIER.get(d, {}).get("desc", "")[:42]
            print(f"    {d:<28} {b}  {cnt:5}  {pct:5.1f}%  {sc:8.1f}  {desc}")

    # Also-detected domains (co-occurrences beyond primary)
    if dd:
        print(f"\n  Also-detected domains (files where domain scored > 0):")
        for d in DOMAIN_DISPLAY_ORDER:
            cnt = dd.get(d, 0)
            if cnt > 0:
                print(f"    {d:<28} {cnt:5} files")

    print(f"\n  AI kernels (training/inf/LLM primary) : {dc['files_with_ai_kernels']:,} files")
    print(f"  Async copy pipeline (cp.async)        : {dc['files_with_async_copy']:,} files")

    # ── L2 contention ────────────────────────────────────────────────────────
    print(f"\n{'L2 CONTENTION ANALYSIS':}")
    print(sep)

    risk_dist = l2.get("risk_level_distribution", {})
    print(f"  Risk level distribution:")
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"):
        cnt = risk_dist.get(level, 0)
        b   = bar(cnt, total_files, 24)
        label = c(level, f"{level:<10}")
        print(f"    {label} {b}  {cnt:5} files")

    print(f"\n  Mean L2 pressure score   : {l2['mean_l2_pressure_score']}")
    print(f"  Total global loads       : {l2['total_global_loads']:,}")
    print(f"  Total global stores      : {l2['total_global_stores']:,}")
    print(f"  Total atom.global        : {l2['total_atom_global']:,}")
    print(f"  Total red.global         : {l2['total_red_global']:,}")
    print(f"  Total atom.shared        : {l2['total_atom_shared']:,}")

    # ── Coalescing ───────────────────────────────────────────────────────────
    print(f"\n{'MEMORY COALESCING ANALYSIS':}")
    print(sep)
    coal  = l2["total_coalesced_patterns"]
    unco  = l2["total_uncoalesced_patterns"]
    total_patterns = coal + unco
    cr    = l2.get("mean_coalescing_ratio")
    print(f"  Coalesced patterns    : {coal:,}")
    print(f"  Uncoalesced patterns  : {unco:,}")
    if total_patterns > 0:
        print(f"  Dataset coalescing %  : {coal/total_patterns*100:.1f}%")
    print(f"  Mean coalescing ratio : {cr}")
    print(f"  Shared bank-risk hits : {l2['total_shared_bank_risk']:,}")
    print(f"  Vector loads (v2/v4)  : {l2['vector_ld_count']:,}")
    print(f"  Vector stores (v2/v4) : {l2['vector_st_count']:,}")
    print(f"  Files with prefetch   : {l2['files_with_prefetch']:,}")

    if l2.get("stride_value_histogram"):
        print(f"\n  Stride multiplier histogram (top-15, bytes):")
        for stride, cnt in list(sorted(
                l2["stride_value_histogram"].items(),
                key=lambda x: -x[1]))[:15]:
            tag = " ← COALESCED" if int(stride) <= COALESCED_STRIDE_MAX else \
                  " ← BANK CONFLICT RISK" if int(stride) >= BANK_CONFLICT_STRIDE else ""
            print(f"    stride {int(stride):>8} B : {cnt:6,}{tag}")

    cm = l2.get("cache_modifier_distribution", {})
    if cm:
        print(f"\n  Cache modifier usage:")
        for mod, cnt in sorted(cm.items(), key=lambda x: -x[1]):
            desc = CACHE_MODIFIERS.get(mod, "?")
            print(f"    {mod:<6} {desc:<35} : {cnt:,}")

    # ── Shared memory ────────────────────────────────────────────────────────
    print(f"\n{'SHARED MEMORY ANALYSIS':}")
    print(sep)
    print(f"  Files using shared mem   : {sm['files_using_shared_mem']:,}")
    print(f"  Max smem any kernel      : {sm['max_smem_bytes_any_kernel']:,} bytes"
          f"  ({sm['max_smem_bytes_any_kernel']//1024} KB)")
    print(f"  Mean smem per use        : {sm['mean_smem_bytes']:,} bytes")
    print(f"  Shared ld/st total       : {mb['ld_shared']:,} / {mb['st_shared']:,}")

    # ── Memory breakdown ─────────────────────────────────────────────────────
    print(f"\n{'MEMORY SPACE BREAKDOWN':}")
    print(sep)
    spaces = [
        ("Global  ld/st", mb["ld_global"],  mb["st_global"]),
        ("Shared  ld/st", mb["ld_shared"],  mb["st_shared"]),
        ("Local   ld/st", mb["ld_local"],   mb["st_local"]),
        ("Const   ld",    mb["ld_const"],   0),
        ("cp.async",      mb["cp_async"],   0),
    ]
    for label, ld, st in spaces:
        print(f"  {label:<18} loads: {ld:>8,}   stores: {st:>8,}")

    # ── Warp divergence ──────────────────────────────────────────────────────
    print(f"\n{'WARP DIVERGENCE':}")
    print(sep)
    print(f"  Total predicated branches : {wd['total_predicated_branches']:,}")
    print(f"  Mean per file             : {wd['mean_pred_branches_per_file']}")
    print(f"  Files with divergence     : {wd['files_with_divergence']:,}")

    # ── Register pressure ────────────────────────────────────────────────────
    print(f"\n{'REGISTER PRESSURE':}")
    print(sep)
    print(f"  Mean max vregs per file  : {rp['mean_max_vreg_per_file']}")
    print(f"  Max vregs any file       : {rp['max_vreg_any_file']}")

    # ── Per-file high-risk summary ───────────────────────────────────────────
    high_risk = [r for r in records
                 if r.get("l2_contention_risk") in ("CRITICAL", "HIGH")]
    if high_risk:
        print(f"\n{'HIGH / CRITICAL L2 RISK FILES':}")
        print(sep)
        for r in sorted(high_risk, key=lambda x: -x.get("l2_pressure_score", 0))[:20]:
            risk_label = c(r["l2_contention_risk"], r["l2_contention_risk"])
            print(f"  [{risk_label}] {r['file']}")
            print(f"           score={r['l2_pressure_score']}  "
                  f"uncoalesced={r['stride_analysis']['uncoalesced_patterns']}  "
                  f"atoms={r['memory_access_counts'].get('atom_global', 0)}  "
                  f"coal_ratio={r.get('coalescing_ratio')}")

    print(f"\n{'═' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Optional CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(records: List[Dict], out_path: str) -> None:
    import csv
    flat_keys = [
        "file", "size_kb", "ptx_version", "sm_target", "kernel_count",
        "total_static_instructions", "unique_opcodes",
        "max_vreg_per_kernel", "shared_mem_max_bytes",
        "l2_pressure_score", "l2_contention_risk",
        "coalescing_ratio", "shared_to_global_ratio",
        "arithmetic_intensity", "atomic_density_per1k",
        "primary_domain",
    ]

    func_keys = ["compute", "memory", "overhead"]

    def mem(r, k):
        return r.get("memory_access_counts", {}).get(k, 0)

    extra_keys = ["ld_global", "st_global", "ld_shared", "st_shared",
                  "atom_global", "red_global", "cp_async", "vector_ld"]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys + func_keys + extra_keys +
                           ["coalesced_patterns", "uncoalesced_patterns",
                            "shared_bank_risk", "predicated_branches"])
        w.writeheader()
        for r in records:
            if "error" in r:
                continue
            row = {k: r.get(k, "") for k in flat_keys}
            fs = r.get("functional_split", {})
            for k in func_keys:
                row[k] = fs.get(k, 0)
            for k in extra_keys:
                row[k] = mem(r, k)
            sa = r.get("stride_analysis", {})
            row["coalesced_patterns"]   = sa.get("coalesced_patterns", 0)
            row["uncoalesced_patterns"] = sa.get("uncoalesced_patterns", 0)
            row["shared_bank_risk"]     = sa.get("shared_bank_risk", 0)
            row["predicated_branches"]  = r.get("warp_divergence", {}).get("predicated_branches", 0)
            w.writerow(row)
    print(f"  CSV written → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional matplotlib charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_charts(agg: Dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping charts)")
        return

    ic = agg["instruction_coverage"]
    l2 = agg["l2_contention_analysis"]
    dc = agg["domain_coverage"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PTX Dataset Analysis", fontsize=14, fontweight="bold")

    # 1. Category pie
    ax = axes[0][0]
    cat_tot = ic.get("category_totals", {})
    labels  = list(cat_tot.keys())
    values  = list(cat_tot.values())
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140,
           textprops={"fontsize": 7})
    ax.set_title("Instruction Categories")

    # 2. Top-15 opcodes bar
    ax = axes[0][1]
    top15 = dict(list(ic.get("top30_opcodes", {}).items())[:15])
    ax.barh(list(top15.keys())[::-1], list(top15.values())[::-1], color="steelblue")
    ax.set_title("Top-15 Opcodes")
    ax.set_xlabel("Count")
    ax.tick_params(axis="y", labelsize=8)

    # 3. Domain group distribution (cleaner than per-domain)
    ax = axes[0][2]
    grp = dc.get("domain_group_dist", dc.get("primary_domain_dist", {}))
    grp_colors = ["#e74c3c","#f39c12","#3498db","#2ecc71","#9b59b6","#1abc9c","#e67e22"]
    ax.bar(list(grp.keys()), list(grp.values()),
           color=grp_colors[:len(grp)])
    ax.set_title("Domain Group Distribution")
    ax.set_ylabel("Files")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    # 4. L2 risk pie
    ax = axes[1][0]
    risk_d = l2.get("risk_level_distribution", {})
    risk_colors = {"CRITICAL": "#e74c3c", "HIGH": "#f39c12",
                   "MEDIUM": "#3498db", "LOW": "#2ecc71", "NONE": "#95a5a6"}
    ax.pie(
        [risk_d.get(k, 0) for k in risk_colors],
        labels=list(risk_colors.keys()),
        colors=list(risk_colors.values()),
        autopct="%1.0f%%", startangle=140,
        textprops={"fontsize": 9}
    )
    ax.set_title("L2 Contention Risk Levels")

    # 5. Memory space bar
    ax = axes[1][1]
    mb = agg.get("memory_breakdown", {})
    spaces   = ["Global\nload", "Global\nstore", "Shared\nload", "Shared\nstore",
                "Local\nload", "Const\nload", "Atomic\nglobal", "cp.async"]
    values   = [mb.get("ld_global", 0), mb.get("st_global", 0),
                mb.get("ld_shared", 0), mb.get("st_shared", 0),
                mb.get("ld_local",  0), mb.get("ld_const",  0),
                mb.get("atom_global", 0), mb.get("cp_async", 0)]
    ax.bar(spaces, values, color=["#e74c3c", "#c0392b", "#3498db", "#2980b9",
                                   "#9b59b6", "#8e44ad", "#e67e22", "#16a085"])
    ax.set_title("Memory Space Access Counts")
    ax.set_ylabel("Instructions")
    ax.tick_params(axis="x", labelsize=7)

    # 6. Stride histogram
    ax = axes[1][2]
    sh = l2.get("stride_value_histogram", {})
    if sh:
        top_strides = sorted(sh.items(), key=lambda x: -x[1])[:12]
        keys   = [str(k) for k, _ in top_strides]
        vals   = [v for _, v in top_strides]
        colors = ["#2ecc71" if int(k) <= COALESCED_STRIDE_MAX else "#e74c3c"
                  for k in keys]
        ax.bar(keys, vals, color=colors)
        ax.set_title("Stride Multiplier Histogram\n(green=coalesced, red=risk)")
        ax.set_xlabel("Stride (bytes)")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelsize=7)
    else:
        ax.text(0.5, 0.5, "No stride data", ha="center", va="center")
        ax.set_title("Stride Histogram")

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "ptx_analysis_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Charts written → {chart_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a dataset of PTX files for instruction/domain coverage "
                    "and L2 contention (coalescing, shared memory, atomics)."
    )
    parser.add_argument("--dir",     required=True,         help="Directory containing .ptx files (recursive)")
    parser.add_argument("--out",     default="ptx_report.json", help="JSON output path (default: ptx_report.json)")
    parser.add_argument("--csv",     action="store_true",   help="Export per-file CSV alongside JSON")
    parser.add_argument("--charts",  action="store_true",   help="Generate matplotlib charts (requires matplotlib)")
    parser.add_argument("--no-color",action="store_true",   help="Disable ANSI color in terminal output")
    parser.add_argument("--glob",    default="**/*.ptx",    help="Glob pattern (default: **/*.ptx)")
    parser.add_argument("--llm",     action="store_true",   help="Use a HuggingFace LLM for domain classification")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--llm-device", default="auto",
                        help="Device for LLM: 'auto', 'cuda', 'cpu' (default: auto)")
    parser.add_argument("--llm-batch", type=int, default=16,
                        help="Batch size for LLM inference (default: 16)")
    parser.add_argument("--llm-cache", default=None,
                        help="Path to LLM classification cache JSON (auto-created)")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    ptx_files = sorted(root.glob(args.glob))
    ptx_files = ptx_files[:60]
    if not ptx_files:
        print(f"No PTX files found under {root} with pattern '{args.glob}'", file=sys.stderr)
        sys.exit(1)

    # ── Optional LLM classifier setup ─────────────────────────────────────
    global _llm_classifier
    if args.llm:
        cache_path = args.llm_cache
        if cache_path is None:
            cache_path = os.path.join(
                os.path.dirname(os.path.abspath(args.out)), "llm_domain_cache.json"
            )
        _llm_classifier = LLMDomainClassifier(
            model_name=args.llm_model,
            cache_path=cache_path,
            device=args.llm_device,
        )
        print(f"LLM domain classifier enabled: {args.llm_model}")
        print(f"  Cache: {cache_path}")
        print(f"  Batch size: {args.llm_batch}")

    total = len(ptx_files)
    print(f"Found {total} PTX file(s) under {root}")
    llm_tag = " + LLM" if args.llm else ""
    print(f"\n[1/5] Analyzing files (regex{llm_tag}) ...")

    records: List[Dict] = []
    t0 = time.time()
    for i, f in enumerate(ptx_files):
        records.append(analyze_file(f))
        done = i + 1
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        r = records[-1]
        dom = r.get("primary_domain", "?")
        n_instr = r.get("total_static_instructions", 0)
        pct = done / total * 100
        print(
            f"\r  [{done:>{len(str(total))}}/{total}] {pct:5.1f}%  "
            f"{f.name[:45]:<45s}  {dom:<22s}  {n_instr:>6} instrs  "
            f"({rate:.0f} files/s, ETA {eta:.0f}s)",
            end="", flush=True,
        )
    elapsed_total = time.time() - t0
    print(f"\n  Done: {total} files in {elapsed_total:.1f}s ({total/elapsed_total:.0f} files/s)\n")

    # Save LLM cache after analysis
    if _llm_classifier is not None:
        _llm_classifier.save_cache()
        print("[2/5] LLM cache saved.")
    else:
        print("[2/5] No LLM cache (regex-only mode).")

    print("[3/5] Aggregating dataset statistics ...")
    agg = aggregate(records)
    print("  Done.\n")

    # Print human-readable report
    print("[4/5] Generating report ...")
    print_report(agg, records, use_color=not args.no_color)

    # Save JSON
    print("[5/5] Writing output files ...")
    out_data = {"aggregate": agg, "per_file": records}
    with open(args.out, "w") as jf:
        json.dump(out_data, jf, indent=2)
    print(f"  JSON report  → {args.out}")

    # Optional CSV
    if args.csv:
        csv_path = args.out.replace(".json", ".csv")
        export_csv(records, csv_path)

    # Optional charts
    if args.charts:
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        plot_charts(agg, out_dir)

    print(f"\nAll done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()



'''
python dataset.py --dir ../raw/ --out report.json --llm --llm-model "Qwen/Qwen2.5-3B-Instruct" --llm-device cuda --llm-batch 32 --llm-cache ./llm_cache.json --csv --charts

'''