// liftoff/liftoff.cuh — Single-header include-all
// LIFTOFF: Warp-Level Primitives Library
// Zero-dependency, header-only CUDA C++ library
// Author: Maaran | ML Systems Engineering Research
#pragma once

// Core layer
#include "core/config.cuh"
#include "core/types.cuh"
#include "core/profile.cuh"

// Primitive layer
#include "primitives/intrinsics.cuh"
#include "primitives/reduce.cuh"
#include "primitives/scan.cuh"
#include "primitives/ballot.cuh"
#include "primitives/broadcast.cuh"
#include "primitives/sort.cuh"
#include "primitives/topk.cuh"

// Cooperative groups layer
#include "cooperative/warptile.cuh"

// Application kernels
#include "kernels/softmax.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/topk_sampling.cuh"
#include "kernels/dot_product.cuh"
#include "kernels/attention_reduce.cuh"
#include "kernels/gelu_reduce.cuh"
#include "kernels/histogram.cuh"

