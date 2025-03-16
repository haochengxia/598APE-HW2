#include "constants.h"
#include "custom_distributions.h"
#include "node_detail.h"
#include "philox_engine.h"
#include "reg_stack.h"
#include <algorithm>
#include <cstdint>
#include <fitness.h>
#include <node.h>
#include <numeric>
#include <program.h>
#include <random>
#include <stack>
#include <immintrin.h>

namespace genetic {

#ifdef USE_ICPX
struct Instruction {
    enum class Op {
        LOAD_CONST, 
        LOAD_VAR, 
        BINARY_OP, 
        UNARY_OP 
    };
    
    Op op;
    union {
        float const_val; 
        int var_idx; 
        node::type op_type; 
    };
    int arity; 
};

std::vector<Instruction> compile_program(const program_t prog) {
    std::vector<Instruction> instructions;
    const node* const nodes = prog->nodes;
    const int prog_len = prog->len;
    
    for (int i = prog_len - 1; i >= 0; --i) {
        const node& curr = nodes[i];
        
        if (curr.t == node::type::constant) {
            instructions.push_back({
                Instruction::Op::LOAD_CONST,
                {.const_val = curr.u.val},
                0
            });
        }
        else if (curr.t == node::type::variable) {
            instructions.push_back({
                Instruction::Op::LOAD_VAR,
                {.var_idx = curr.u.fid},
                0
            });
        }
        else {
            int ar = detail::arity(curr.t);
            instructions.push_back({
                ar > 1 ? Instruction::Op::BINARY_OP : Instruction::Op::UNARY_OP,
                {.op_type = curr.t},
                ar
            });
        }
    }
    return instructions;
}

template <int MaxSize = MAX_STACK_SIZE, int BatchSize = 32>
void execute_kernel(const program_t d_progs, const float *data, float *y_pred,
                    const uint64_t n_rows, const uint64_t n_progs) {
    
    constexpr int SIMD_WIDTH = 8;  // AVX2 processes 8 floats at once
    constexpr int BATCH_SIMD = BatchSize / SIMD_WIDTH;  // Number of SIMD operations per batch
    
    #pragma omp parallel for if(n_progs > 1)
    for (uint64_t pid = 0; pid < n_progs; ++pid) {
        const auto instructions = compile_program(d_progs + pid);
        const uint64_t output_offset = pid * n_rows;
        
        #pragma omp parallel for if(n_rows > 1000)
        for (uint64_t row_id = 0; row_id < n_rows; row_id += BatchSize) {
            const int actual_batch = std::min(BatchSize, static_cast<int>(n_rows - row_id));
            const int full_simd_iters = actual_batch / SIMD_WIDTH;
            
            // Aligned stack for both batch and SIMD processing
            alignas(32) float stack_data[MaxSize][BatchSize];
            int stack_top = 0;
            
            for (const auto& inst : instructions) {
                switch (inst.op) {
                    case Instruction::Op::LOAD_CONST: {
                        __m256 const_val = _mm256_set1_ps(inst.const_val);
                        // Process SIMD_WIDTH elements at a time
                        for (int b = 0; b < full_simd_iters; ++b) {
                            _mm256_store_ps(&stack_data[stack_top][b * SIMD_WIDTH], const_val);
                        }
                        // Handle remaining elements
                        if (actual_batch % SIMD_WIDTH != 0) {
                            alignas(32) float temp[SIMD_WIDTH];
                            _mm256_store_ps(temp, const_val);
                            for (int i = full_simd_iters * SIMD_WIDTH; i < actual_batch; ++i) {
                                stack_data[stack_top][i] = temp[i % SIMD_WIDTH];
                            }
                        }
                        stack_top++;
                        break;
                    }
                    
                    case Instruction::Op::LOAD_VAR: {
                        const float* src = data + inst.var_idx * n_rows + row_id;
                        // Process SIMD_WIDTH elements at a time
                        for (int b = 0; b < full_simd_iters; ++b) {
                            __m256 var_val = _mm256_loadu_ps(src + b * SIMD_WIDTH);
                            _mm256_store_ps(&stack_data[stack_top][b * SIMD_WIDTH], var_val);
                        }
                        // Handle remaining elements
                        for (int i = full_simd_iters * SIMD_WIDTH; i < actual_batch; ++i) {
                            stack_data[stack_top][i] = src[i];
                        }
                        stack_top++;
                        break;
                    }
                    
                    case Instruction::Op::BINARY_OP: {
                        stack_top--;
                        // Process SIMD_WIDTH elements at a time
                        for (int b = 0; b < full_simd_iters; ++b) {
                            __m256 op2 = _mm256_load_ps(&stack_data[stack_top][b * SIMD_WIDTH]);
                            __m256 op1 = _mm256_load_ps(&stack_data[stack_top-1][b * SIMD_WIDTH]);
                            
                            __m256 result = detail::evaluate_node_simd(
                                {.t = inst.op_type},
                                data, n_rows, row_id + b * SIMD_WIDTH,
                                (__m256[]){op1, op2}
                            );
                            
                            _mm256_store_ps(&stack_data[stack_top-1][b * SIMD_WIDTH], result);
                        }
                        // Handle remaining elements
                        for (int i = full_simd_iters * SIMD_WIDTH; i < actual_batch; ++i) {
                            const float inputs[2] = {
                                stack_data[stack_top-1][i],
                                stack_data[stack_top][i]
                            };
                            stack_data[stack_top-1][i] = detail::evaluate_node(
                                {.t = inst.op_type},
                                data,
                                n_rows,
                                row_id + i,
                                inputs
                            );
                        }
                        break;
                    }
                    
                    case Instruction::Op::UNARY_OP: {
                        stack_top--;
                        // Process SIMD_WIDTH elements at a time
                        for (int b = 0; b < full_simd_iters; ++b) {
                            __m256 op = _mm256_load_ps(&stack_data[stack_top][b * SIMD_WIDTH]);
                            
                            __m256 result = detail::evaluate_node_simd(
                                {.t = inst.op_type},
                                data, n_rows, row_id + b * SIMD_WIDTH,
                                (__m256[]){op, _mm256_setzero_ps()}
                            );
                            
                            _mm256_store_ps(&stack_data[stack_top][b * SIMD_WIDTH], result);
                        }
                        // Handle remaining elements
                        for (int i = full_simd_iters * SIMD_WIDTH; i < actual_batch; ++i) {
                            const float inputs[2] = {
                                stack_data[stack_top][i],
                                0.0f
                            };
                            stack_data[stack_top][i] = detail::evaluate_node(
                                {.t = inst.op_type},
                                data,
                                n_rows,
                                row_id + i,
                                inputs
                            );
                        }
                        stack_top++;
                        break;
                    }
                }
            }
            
            // Store results
            for (int b = 0; b < full_simd_iters; ++b) {
                __m256 final_result = _mm256_load_ps(&stack_data[stack_top-1][b * SIMD_WIDTH]);
                _mm256_storeu_ps(y_pred + output_offset + row_id + b * SIMD_WIDTH, final_result);
            }
            // Handle remaining elements
            for (int i = full_simd_iters * SIMD_WIDTH; i < actual_batch; ++i) {
                y_pred[output_offset + row_id + i] = stack_data[stack_top-1][i];
            }
        }
    }
}
#else
#ifdef ENABLE_OMP
template <int MaxSize = MAX_STACK_SIZE>
void execute_kernel(const program_t d_progs, const float *data, float *y_pred,
                    const uint64_t n_rows, const uint64_t n_progs) {
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    const program_t curr_p = d_progs + pid;
    const node* const nodes = curr_p->nodes;
    const int prog_len = curr_p->len;
    const uint64_t output_offset = pid * n_rows;

    #pragma omp parallel for if(n_rows > 100)
    for (uint64_t row_id = 0; row_id < n_rows; ++row_id) {
      alignas(64) float stack_data[MaxSize];
      int stack_top = 0;

      float res = 0.0f;
      float in[2] = {0.0f, 0.0f};

      const node* curr_node = nodes + prog_len - 1;
      for (int end = prog_len - 1; end >= 0; --end, --curr_node) {
        if (detail::is_nonterminal(curr_node->t)) {
          const int ar = detail::arity(curr_node->t);
          in[0] = stack_data[--stack_top];
          if (ar > 1) {
            in[1] = stack_data[--stack_top];
          }
        }
        res = detail::evaluate_node(*curr_node, data, n_rows, row_id, in);
        stack_data[stack_top++] = res;
      }

      y_pred[output_offset + row_id] = stack_data[--stack_top];
    }
  }
}
#else
#ifdef ENABLE_OMP_BATCHED
template <int MaxSize = MAX_STACK_SIZE, int BatchSize = 32>
void execute_kernel(const program_t d_progs, const float *data, float *y_pred,
                    const uint64_t n_rows, const uint64_t n_progs) {
    for (uint64_t pid = 0; pid < n_progs; ++pid) {
        const program_t curr_p = d_progs + pid;
        const node* const nodes = curr_p->nodes;
        const int prog_len = curr_p->len;
        const uint64_t output_offset = pid * n_rows;

        #pragma omp parallel for if(n_rows > 100)
        for (uint64_t row_id = 0; row_id < n_rows; row_id += BatchSize) {
            const int actual_batch = std::min(BatchSize, static_cast<int>(n_rows - row_id));
            
            alignas(64) float stack_data[MaxSize][BatchSize];
            alignas(64) float batch_in[2][BatchSize];
            alignas(64) float eval_inputs[2];
            int stack_top = 0;
            
            const node* curr_node = nodes + prog_len - 1;
            for (int end = prog_len - 1; end >= 0; --end, --curr_node) {
                if (detail::is_nonterminal(curr_node->t)) {
                    const int ar = detail::arity(curr_node->t);
                    
                    #pragma omp simd
                    for (int b = 0; b < actual_batch; ++b) {
                        batch_in[0][b] = stack_data[stack_top-1][b];
                    }
                    
                    if (ar > 1) {
                        #pragma omp simd
                        for (int b = 0; b < actual_batch; ++b) {
                            batch_in[1][b] = stack_data[stack_top-2][b];
                        }
                        stack_top -= 2;
                    } else {
                        stack_top -= 1;
                    }
                }

                #pragma omp simd
                for (int b = 0; b < actual_batch; ++b) {
                    eval_inputs[0] = batch_in[0][b];
                    eval_inputs[1] = curr_node->arity() > 1 ? batch_in[1][b] : 0.0f;
                    
                    stack_data[stack_top][b] = detail::evaluate_node(
                        *curr_node, 
                        data, 
                        n_rows, 
                        row_id + b, 
                        eval_inputs
                    );
                }
                stack_top++;
            }

            #pragma omp simd
            for (int b = 0; b < actual_batch; ++b) {
                y_pred[output_offset + row_id + b] = stack_data[stack_top-1][b];
            }
        }
    }
}
#else
#ifdef ENABLE_OMP_PRECOMPILE_BATCHED
struct Instruction {
    enum class Op {
        LOAD_CONST,
        LOAD_VAR,
        COMPUTE
    };
    
    Op op;
    union {
        float const_val;
        int var_idx;
        node::type op_type;
    };
    int arity;
};

std::vector<Instruction> compile_program(const program_t prog) {
    std::vector<Instruction> instructions;
    const node* const nodes = prog->nodes;
    const int prog_len = prog->len;
    
    for (int i = prog_len - 1; i >= 0; --i) {
        const node& curr = nodes[i];
        
        if (curr.t == node::type::constant) {
            instructions.push_back({
                .op = Instruction::Op::LOAD_CONST,
                .const_val = curr.u.val,
                .arity = 0
            });
        }
        else if (curr.t == node::type::variable) {
            instructions.push_back({
                .op = Instruction::Op::LOAD_VAR,
                .var_idx = curr.u.fid,
                .arity = 0
            });
        }
        else {
            instructions.push_back({
                .op = Instruction::Op::COMPUTE,
                .op_type = curr.t,
                .arity = detail::arity(curr.t)
            });
        }
    }
    return instructions;
}

template <int MaxSize = MAX_STACK_SIZE, int BatchSize = 32>
void execute_kernel(const program_t d_progs, const float *data, float *y_pred,
                    const uint64_t n_rows, const uint64_t n_progs) {
    
    if (n_progs > 1) {
        #pragma omp parallel for schedule(dynamic)
        for (uint64_t pid = 0; pid < n_progs; ++pid) {
            const auto instructions = compile_program(d_progs + pid);
            const uint64_t output_offset = pid * n_rows;
            
            alignas(64) float stack_data[MaxSize][BatchSize];
            alignas(64) float inputs[2][BatchSize];
            node temp_node;
            
            for (uint64_t row_id = 0; row_id < n_rows; row_id += BatchSize) {
                const int actual_batch = std::min(BatchSize, static_cast<int>(n_rows - row_id));
                int stack_top = 0;
                
                for (const auto& inst : instructions) {
                    switch (inst.op) {
                        case Instruction::Op::LOAD_CONST: {
                            #pragma omp simd
                            for (int b = 0; b < actual_batch; ++b) {
                                stack_data[stack_top][b] = inst.const_val;
                            }
                            stack_top++;
                            break;
                        }
                        
                        case Instruction::Op::LOAD_VAR: {
                            const float* src = data + inst.var_idx * n_rows + row_id;
                            #pragma omp simd
                            for (int b = 0; b < actual_batch; ++b) {
                                stack_data[stack_top][b] = src[b];
                            }
                            stack_top++;
                            break;
                        }
                        
                        case Instruction::Op::COMPUTE: {
                            temp_node.t = inst.op_type;
                            
                            #pragma omp simd
                            for (int b = 0; b < actual_batch; ++b) {
                                inputs[0][b] = stack_data[stack_top-1][b];
                                if (inst.arity > 1) {
                                    inputs[1][b] = stack_data[stack_top-2][b];
                                }
                            }
                            
                            #pragma omp simd
                            for (int b = 0; b < actual_batch; ++b) {
                                const float curr_inputs[2] = {
                                    inputs[0][b],
                                    inst.arity > 1 ? inputs[1][b] : 0.0f
                                };
                                
                                stack_data[stack_top - inst.arity][b] = detail::evaluate_node(
                                    temp_node,
                                    data,
                                    n_rows,
                                    row_id + b,
                                    curr_inputs
                                );
                            }
                            stack_top = stack_top - inst.arity + 1;
                            break;
                        }
                    }
                }
                
                #pragma omp simd
                for (int b = 0; b < actual_batch; ++b) {
                    y_pred[output_offset + row_id + b] = stack_data[stack_top-1][b];
                }
            }
        }
    } else {
        const auto instructions = compile_program(d_progs);
        
        #pragma omp parallel for schedule(dynamic) if(n_rows > 10000)
        for (uint64_t row_id = 0; row_id < n_rows; row_id += BatchSize) {
            const int actual_batch = std::min(BatchSize, static_cast<int>(n_rows - row_id));
            
            float stack_data[MaxSize][BatchSize];
            int stack_top = 0;
            
            node temp_node;
            float inputs[2];
            
            for (const auto& inst : instructions) {
                switch (inst.op) {
                    case Instruction::Op::LOAD_CONST:
                        for (int b = 0; b < actual_batch; ++b) {
                            stack_data[stack_top][b] = inst.const_val;
                        }
                        stack_top++;
                        break;
                        
                    case Instruction::Op::LOAD_VAR:
                        for (int b = 0; b < actual_batch; ++b) {
                            stack_data[stack_top][b] = data[inst.var_idx * n_rows + row_id + b];
                        }
                        stack_top++;
                        break;
                        
                    case Instruction::Op::COMPUTE:
                        temp_node.t = inst.op_type;
                        
                        for (int b = 0; b < actual_batch; ++b) {
                            inputs[0] = stack_data[stack_top-1][b];
                            if (inst.arity > 1) {
                                inputs[1] = stack_data[stack_top-2][b];
                            }
                            
                            stack_data[stack_top - inst.arity][b] = detail::evaluate_node(
                                temp_node,
                                data,
                                n_rows,
                                row_id + b,
                                inputs
                            );
                        }
                        stack_top = stack_top - inst.arity + 1;
                        break;
                }
            }
            
            for (int b = 0; b < actual_batch; ++b) {
                y_pred[row_id + b] = stack_data[stack_top-1][b];
            }
        }
    }
}
#else
/**
 * Execution kernel for a single program. We assume that the input data
 * is stored in column major format.
 */
template <int MaxSize = MAX_STACK_SIZE>
void execute_kernel(const program_t d_progs, const float *data, float *y_pred,
                    const uint64_t n_rows, const uint64_t n_progs) {
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t row_id = 0; row_id < n_rows; ++row_id) {

      stack<float, MaxSize> eval_stack;
      program_t curr_p = d_progs + pid; // Current program

      int end = curr_p->len - 1;
      node *curr_node = curr_p->nodes + end;

      float res = 0.0f;
      float in[2] = {0.0f, 0.0f};

      while (end >= 0) {
        if (detail::is_nonterminal(curr_node->t)) {
          int ar = detail::arity(curr_node->t);
          in[0] = eval_stack.pop(); // Min arity of function is 1
          if (ar > 1)
            in[1] = eval_stack.pop();
        }
        res = detail::evaluate_node(*curr_node, data, n_rows, row_id, in);
        eval_stack.push(res);
        curr_node--;
        end--;
      }

      // Outputs stored in col-major format
      y_pred[pid * n_rows + row_id] = eval_stack.pop();
    }
  }
}
#endif
#endif
#endif
#endif
program::program()
    : len(0), depth(0), raw_fitness_(0.0f), metric(metric_t::mse),
      mut_type(mutation_t::none), nodes(nullptr) {}

program::~program() { delete[] nodes; }

program::program(const program &src)
    : len(src.len), depth(src.depth), raw_fitness_(src.raw_fitness_),
      metric(src.metric), mut_type(src.mut_type) {
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);
}

program &program::operator=(const program &src) {
  len = src.len;
  depth = src.depth;
  raw_fitness_ = src.raw_fitness_;
  metric = src.metric;
  mut_type = src.mut_type;

  // Copy nodes
  delete[] nodes;
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);

  return *this;
}

void compute_metric(int n_rows, int n_progs, const float *y,
                    const float *y_pred, const float *w, float *score,
                    const param &params) {
  // Call appropriate metric function based on metric defined in params
  if (params.metric == metric_t::pearson) {
    weightedPearson(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::spearman) {
    weightedSpearman(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::mae) {
    meanAbsoluteError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::mse) {
    meanSquareError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::rmse) {
    rootMeanSquareError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::logloss) {
    logLoss(n_rows, n_progs, y, y_pred, w, score);
  } else {
    // This should not be reachable
  }
}

void execute(const program_t &d_progs, const int n_rows, const int n_progs,
             const float *data, float *y_pred) {
  execute_kernel(d_progs, data, y_pred, static_cast<uint64_t>(n_rows),
                 static_cast<uint64_t>(n_progs));
}

void find_fitness(program_t d_prog, float *score, const param &params,
                  const int n_rows, const float *data, const float *y,
                  const float *sample_weights) {

  // Compute predicted values
  std::vector<float> y_pred(n_rows);
  execute(d_prog, n_rows, 1, data, y_pred.data());

  // Compute error
  compute_metric(n_rows, 1, y, y_pred.data(), sample_weights, score, params);
}

void find_batched_fitness(int n_progs, program_t d_progs, float *score,
                          const param &params, const int n_rows,
                          const float *data, const float *y,
                          const float *sample_weights) {
#ifdef ENABLE_COMMON
  static thread_local struct BufferPool {
    float* buffer = nullptr;
    size_t capacity = 0;
    
    ~BufferPool() {
      if (buffer) {
        free(buffer);
      }
    }
    
    float* get_buffer(size_t required_size) {
      if (capacity < required_size) {
        if (buffer) {
          free(buffer);
        }
        buffer = (float*)aligned_alloc(64, required_size * sizeof(float));
        capacity = required_size;
      }
      return buffer;
    }
  } buffer_pool;

  const size_t required_size = (uint64_t)n_rows * (uint64_t)n_progs;
  float* y_pred = buffer_pool.get_buffer(required_size);

  execute(d_progs, n_rows, n_progs, data, y_pred);

  compute_metric(n_rows, n_progs, y, y_pred, sample_weights, score, params);
#else
  std::vector<float> y_pred((uint64_t)n_rows * (uint64_t)n_progs);
  execute(d_progs, n_rows, n_progs, data, y_pred.data());

  // Compute error
  compute_metric(n_rows, n_progs, y, y_pred.data(), sample_weights, score,
                 params);
#endif
}

#ifdef ENABLE_COMMON
void set_batched_fitness(int n_progs, std::vector<program> &h_progs,
                         const param &params, const int n_rows,
                         const float *data, const float *y,
                         const float *sample_weights) {
    static thread_local struct ScorePool {
        float* buffer = nullptr;
        size_t capacity = 0;
        
        ~ScorePool() {
            if (buffer) {
                free(buffer);
            }
        }
        
        float* get_buffer(size_t required_size) {
            if (capacity < required_size) {
                if (buffer) {
                    free(buffer);
                }
                buffer = (float*)aligned_alloc(64, required_size * sizeof(float));
                capacity = required_size;
            }
            return buffer;
        }
    } score_pool;

    float* score = score_pool.get_buffer(n_progs);

    find_batched_fitness(n_progs, h_progs.data(), score, params, n_rows,
                         data, y, sample_weights);

    #pragma omp simd
    for (int i = 0; i < n_progs; ++i) {
        h_progs[i].raw_fitness_ = score[i];
    }
}

void set_fitness(program &h_prog, const param &params, const int n_rows,
                 const float *data, const float *y,
                 const float *sample_weights) {
    static thread_local struct SingleScorePool {
        float buffer[1];
    } score_pool;

    find_fitness(&h_prog, score_pool.buffer, params, n_rows, data, y, sample_weights);

    h_prog.raw_fitness_ = score_pool.buffer[0];
}
#else
void set_batched_fitness(int n_progs, std::vector<program> &h_progs,
                         const param &params, const int n_rows,
                         const float *data, const float *y,
                         const float *sample_weights) {
    std::vector<float> score(n_progs);
    find_batched_fitness(n_progs, h_progs.data(), score.data(), params, n_rows,
                         data, y, sample_weights);
    for (int i = 0; i < n_progs; ++i) {
        h_progs[i].raw_fitness_ = score[i];
    }
}

void set_fitness(program &h_prog, const param &params, const int n_rows,
                 const float *data, const float *y,
                 const float *sample_weights) {
    std::vector<float> score(1);
    find_fitness(&h_prog, score.data(), params, n_rows, data, y, sample_weights);
    h_prog.raw_fitness_ = score[0];
}
#endif

float get_fitness(const program &prog, const param &params) {
  int crit = params.criterion();
  float penalty = params.parsimony_coefficient * prog.len * (2 * crit - 1);
  return (prog.raw_fitness_ - penalty);
}

/**
 * @brief Get a random subtree of the current program nodes (on CPU)
 *
 * @param pnodes  AST represented as a list of nodes
 * @param len     The total number of nodes in the AST
 * @param rng     Random number generator for subtree selection
 * @return A tuple [first,last) which contains the required subtree
 */
std::pair<int, int> get_subtree(node *pnodes, int len, PhiloxEngine &rng) {
  int start, end;
  start = end = 0;

  // Specify RNG
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  float bound = dist_uniform(rng);

  // Specify subtree start probs acc to Koza's selection approach
  std::vector<float> node_probs(len, 0.1);
  float sum = 0.1 * len;

  for (int i = 0; i < len; ++i) {
    if (pnodes[i].is_nonterminal()) {
      node_probs[i] = 0.9;
      sum += 0.8;
    }
  }

  // Normalize vector
  for (int i = 0; i < len; ++i) {
    node_probs[i] /= sum;
  }

  // Compute cumulative sum
  std::partial_sum(node_probs.begin(), node_probs.end(), node_probs.begin());

  start = std::lower_bound(node_probs.begin(), node_probs.end(), bound) -
          node_probs.begin();
  end = start;

  // Iterate until all function arguments are satisfied in current subtree
  int num_args = 1;
  while (num_args > end - start) {
    node curr;
    curr = pnodes[end];
    if (curr.is_nonterminal())
      num_args += curr.arity();
    ++end;
  }

  return std::make_pair(start, end);
}

#ifdef ENABLE_COMMON
int get_depth(const program &p_out) {
  static constexpr int MAX_DEPTH = 64;
  int arity_array[MAX_DEPTH];
  int stack_size = 0;
  int depth = 0;

  const node* const nodes = p_out.nodes;
  const int len = p_out.len;

  for (int i = 0; i < len; ++i) {
    depth = std::max(depth, stack_size);

    if (detail::is_nonterminal(nodes[i].t)) {
      arity_array[stack_size++] = detail::arity(nodes[i].t);
    } else {
      if (stack_size == 0) break;
      
      do {
        if (--arity_array[stack_size - 1] > 0) break;
        --stack_size;
      } while (stack_size > 0);
    }
  }
  return depth;
}
#else
int get_depth(const program &p_out) {
  int depth = 0;
  std::stack<int> arity_stack;
  for (auto i = 0; i < p_out.len; ++i) {
    node curr(p_out.nodes[i]);

    // Update depth
    int sz = arity_stack.size();
    depth = std::max(depth, sz);

    // Update stack
    if (curr.is_nonterminal()) {
      arity_stack.push(curr.arity());
    } else {
      // Only triggered for a depth 0 node
      if (arity_stack.empty())
        break;

      int e = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(e - 1);

      while (arity_stack.top() == 0) {
        arity_stack.pop();
        if (arity_stack.empty())
          break;

        e = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(e - 1);
      }
    }
  }

  return depth;
}
#endif

void build_program(program &p_out, const param &params, PhiloxEngine &rng) {
  // Define data structures needed for tree
  std::stack<int> arity_stack;
  std::vector<node> nodelist;
  nodelist.reserve(1 << (MAX_STACK_SIZE));

  // Specify Distributions with parameters
  uniform_int_distribution_custom<int> dist_function(
      0, params.function_set.size() - 1);
  uniform_int_distribution_custom<int> dist_initDepth(params.init_depth[0],
                                                      params.init_depth[1]);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constVal(params.const_range[0],
                                                        params.const_range[1]);
  bernoulli_distribution_custom dist_nodeChoice(params.terminalRatio);
  bernoulli_distribution_custom dist_coinToss(0.5);

  // Initialize nodes
  int max_depth = dist_initDepth(rng);
  node::type func = params.function_set[dist_function(rng)];
  node curr_node(func);
  nodelist.push_back(curr_node);
  arity_stack.push(curr_node.arity());

  init_method_t method = params.init_method;
  if (method == init_method_t::half_and_half) {
    // Choose either grow or full for this tree
    bool choice = dist_coinToss(rng);
    method = choice ? init_method_t::grow : init_method_t::full;
  }

  // Fill tree
  while (!arity_stack.empty()) {
    int depth = arity_stack.size();
    p_out.depth = std::max(depth, p_out.depth);
    bool node_choice = dist_nodeChoice(rng);

    if ((node_choice == false || method == init_method_t::full) &&
        depth < max_depth) {
      // Add a function to node list
      curr_node = node(params.function_set[dist_function(rng)]);
      nodelist.push_back(curr_node);
      arity_stack.push(curr_node.arity());
    } else {
      // Add terminal
      int terminal_choice = dist_terminalChoice(rng);
      if (terminal_choice == params.num_features) {
        // Add constant
        float val = dist_constVal(rng);
        curr_node = node(val);
      } else {
        // Add variable
        int fid = terminal_choice;
        curr_node = node(fid);
      }

      // Modify nodelist
      nodelist.push_back(curr_node);

      // Modify stack
      int e = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(e - 1);
      while (arity_stack.top() == 0) {
        arity_stack.pop();
        if (arity_stack.empty()) {
          break;
        }

        e = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(e - 1);
      }
    }
  }

  // Set new program parameters - need to do a copy as
  // nodelist will be deleted using RAII semantics
  p_out.nodes = new node[nodelist.size()];
  std::copy(nodelist.begin(), nodelist.end(), p_out.nodes);

  p_out.len = nodelist.size();
  p_out.metric = params.metric;
  p_out.raw_fitness_ = 0.0f;
}

void point_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // deep-copy program
  p_out = prog;

  // Specify RNGs
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constantVal(
      params.const_range[0], params.const_range[1]);

  // Fill with uniform numbers
  std::vector<float> node_probs(p_out.len);
  std::generate(node_probs.begin(), node_probs.end(),
                [&dist_uniform, &rng] { return dist_uniform(rng); });

  // Mutate nodes
  int len = p_out.len;
  for (int i = 0; i < len; ++i) {
    node curr(prog.nodes[i]);

    if (node_probs[i] < params.p_point_replace) {
      if (curr.is_terminal()) {
        int choice = dist_terminalChoice(rng);

        if (choice == params.num_features) {
          // Add a randomly generated constant
          curr = node(dist_constantVal(rng));
        } else {
          // Add a variable with fid=choice
          curr = node(choice);
        }
      } else if (curr.is_nonterminal()) {
        // Replace current function with another function of the same arity
        int ar = curr.arity();
        // CUML_LOG_DEBUG("Arity is %d, curr function is
        // %d",ar,static_cast<std::underlying_type<node::type>::type>(curr.t));
        std::vector<node::type> fset = params.arity_set.at(ar);
        uniform_int_distribution_custom<int> dist_fset(0, fset.size() - 1);
        int choice = dist_fset(rng);
        curr = node(fset[choice]);
      }

      // Update p_out with updated value
      p_out.nodes[i] = curr;
    }
  }
}

void crossover(const program &prog, const program &donor, program &p_out,
               const param &params, PhiloxEngine &rng) {
  // Get a random subtree of prog to replace
  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  // Set metric of output program
  p_out.metric = prog.metric;

  // MAX_STACK_SIZE can only handle tree of depth MAX_STACK_SIZE -
  // max(func_arity=2) + 1 Thus we continuously hoist the donor subtree. Actual
  // indices in donor
  int donor_start = 0;
  int donor_end = donor.len;
  int output_depth = 0;
  int iter = 0;
  do {
    ++iter;
    // Get donor subtree
    std::pair<int, int> donor_slice =
        get_subtree(donor.nodes + donor_start, donor_end - donor_start, rng);

    // Get indices w.r.t current subspace [donor_start,donor_end)
    int donor_substart = donor_slice.first;
    int donor_subend = donor_slice.second;

    // Update relative indices to global indices
    donor_substart += donor_start;
    donor_subend += donor_start;

    // Update to new subspace
    donor_start = donor_substart;
    donor_end = donor_subend;

    // Evolve on current subspace
    p_out.len =
        (prog_start) + (donor_end - donor_start) + (prog.len - prog_end);
    delete[] p_out.nodes;
    p_out.nodes = new node[p_out.len];

    // Copy slices using std::copy
    std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
    std::copy(donor.nodes + donor_start, donor.nodes + donor_end,
              p_out.nodes + prog_start);
    std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
              p_out.nodes + (prog_start) + (donor_end - donor_start));

    output_depth = get_depth(p_out);
  } while (output_depth >= MAX_STACK_SIZE);

  // Set the depth of the final program
  p_out.depth = output_depth;
}

void subtree_mutation(const program &prog, program &p_out, const param &params,
                      PhiloxEngine &rng) {
  // Generate a random program and perform crossover
  program new_program;
  build_program(new_program, params, rng);
  crossover(prog, new_program, p_out, params, rng);
}

void hoist_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // Replace program subtree with a random sub-subtree

  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  std::pair<int, int> sub_slice =
      get_subtree(prog.nodes + prog_start, prog_end - prog_start, rng);
  int sub_start = sub_slice.first;
  int sub_end = sub_slice.second;

  // Update subtree indices to global indices
  sub_start += prog_start;
  sub_end += prog_start;

  p_out.len = (prog_start) + (sub_end - sub_start) + (prog.len - prog_end);
  p_out.nodes = new node[p_out.len];
  p_out.metric = prog.metric;

  // Copy node slices using std::copy
  std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
  std::copy(prog.nodes + sub_start, prog.nodes + sub_end,
            p_out.nodes + prog_start);
  std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
            p_out.nodes + (prog_start) + (sub_end - sub_start));

  // Update depth
  p_out.depth = get_depth(p_out);
}

} // namespace genetic
