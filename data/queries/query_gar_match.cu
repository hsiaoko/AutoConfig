#include "core/common/consts.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/task/gpu_task/kernel/algorithms/hash.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using GARGraphArrays = sics::matrixgraph::core::data_structures::GARGraphArrays;
using GARPatternArrays = sics::matrixgraph::core::data_structures::GARPatternArrays;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
using DeviceOwnedBufferInt32 = sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<int32_t>;

GARMatchKernelWrapper* GARMatchKernelWrapper::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new GARMatchKernelWrapper();
  }
  return ptr_;
}

static __forceinline__ __device__ bool LabelDegreeFilter(
    const GARPatternArrays& p, const GARGraphArrays& g, int u_idx,
    uint32_t v_idx) {
  if (u_idx < 0 || u_idx >= p.n_nodes) return false;
  if (v_idx >= static_cast<uint32_t>(g.n_vertices)) return false;

  const int32_t u_label = p.node_label_idx[u_idx];
  const int32_t v_label = g.v_label_idx[v_idx];
  if (u_label != v_label) return false;

  int p_out = 0, p_in = 0, g_out = 0, g_in = 0;
  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_src[e] == u_idx) ++p_out;
    if (p.edge_dst[e] == u_idx) ++p_in;
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_src[e] == v_idx) ++g_out;
    if (g.e_dst[e] == v_idx) ++g_in;
  }
  return g_out >= p_out && g_in >= p_in;
}

static __global__ void GARBuildVertexValidMaskKernel(ParametersGARFilter params,
                                                     int u_idx,
                                                     int32_t* valid_mask) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  
  for (uint32_t v = tid; v < static_cast<uint32_t>(params.n_vertices_g);
       v += step) {
    valid_mask[v] = GARVertexFilter(p_view, g_view, u_idx, v) ? 1 : 0;
  }
}

static __global__ void GARFilterEdgeCandidatesKernel(
    ParametersGARFilter params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (int ge = static_cast<int>(tid); ge < params.n_edges_g;
       ge += static_cast<int>(step)) {
    if (params.g_e_label_idx &&
        params.g_e_label_idx[ge] != params.p_edge_label) {
      continue;
    }
    const uint32_t gs = params.g_e_src[ge];
    const uint32_t gd = params.g_e_dst[ge];
    
    if (params.src_valid_mask && params.src_valid_mask[gs] == 0) continue;
    if (params.dst_valid_mask && params.dst_valid_mask[gd] == 0) continue;
    
    int slot = atomicAdd(params.cand_count, 1);
    if (slot < params.cand_capacity) {
      params.cand_src[slot] = gs;
      params.cand_dst[slot] = gd;
    }
  }
}

static __global__ void GARExpandEmbeddingsKernel(
    const int32_t* curr_embeddings, int curr_count, int n_nodes, int p_u,
    int p_v, const uint32_t* cand_src, const uint32_t* cand_dst, int n_cand,
    int32_t* next_embeddings, int* next_count, int next_capacity) {
  const unsigned long long tid =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long step =
      static_cast<unsigned long long>(blockDim.x) * gridDim.x;
  const unsigned long long total_pairs =
      static_cast<unsigned long long>(curr_count) *
      static_cast<unsigned long long>(n_cand);
  
  for (unsigned long long k = tid; k < total_pairs; k += step) {
    const int emb_idx =
        static_cast<int>(k / static_cast<unsigned long long>(n_cand));
    const int cand_idx =
        static_cast<int>(k % static_cast<unsigned long long>(n_cand));
    
    const int32_t* row =
        curr_embeddings + static_cast<size_t>(emb_idx) * n_nodes;
    const uint32_t gs = cand_src[cand_idx];
    const uint32_t gd = cand_dst[cand_idx];
    
    int slot = atomicAdd(next_count, 1);
    if (slot >= next_capacity) continue;
    
    int32_t* out_row = next_embeddings + static_cast<size_t>(slot) * n_nodes;
    for (int i = 0; i < n_nodes; ++i) out_row[i] = row[i];
    out_row[p_u] = static_cast<int32_t>(gs);
    out_row[p_v] = static_cast<int32_t>(gd);
  }
}

int GARMatchKernelWrapper::GARMatch(const GARGraphArrays& g,
                                    const GARPatternArrays& p,
                                    GARMatchArrays* out) {
  // CUDA kernel stage 1: candidate filtering
  DeviceOwnedBufferInt32 d_g_v_id;
  DeviceOwnedBufferInt32 d_g_e_src;
  DeviceOwnedBufferInt32 d_g_e_dst;
  
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  
  for (int pe = 0; pe < p.n_edges; ++pe) {
    DeviceOwnedBufferInt32 d_src_valid(sizeof(int32_t) *
                                       static_cast<size_t>(g.n_vertices));
    DeviceOwnedBufferInt32 d_cand_count(sizeof(int));

    GARBuildVertexValidMaskKernel<<<dimGrid, dimBlock>>>(
        params, pu, d_src_valid.GetPtr());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GARFilterEdgeCandidatesKernel<<<dimGrid, dimBlock>>>(params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Join phase (CUDA): expand frontier
  DeviceOwnedBufferInt32 d_frontier_a(emb_bytes);
  DeviceOwnedBufferInt32 d_frontier_b(emb_bytes);
  
  for (int pe = 1; pe < p.n_edges && h_curr_count > 0; ++pe) {
    GARExpandEmbeddingsKernel<<<dimGrid, dimBlock>>>(
        curr_ptr, h_curr_count, p.n_nodes, pu, pv,
        d_c_src.GetPtr(), d_c_dst.GetPtr(), n_cand,
        next_ptr, d_next_count.GetPtr(), max_embeddings);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  return 0;
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
