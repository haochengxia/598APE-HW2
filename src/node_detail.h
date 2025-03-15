#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <node.h>

namespace genetic {
namespace detail {

static constexpr float MIN_VAL = 0.001f;

inline bool is_terminal(node::type t) {
  return t == node::type::variable || t == node::type::constant;
}

inline bool is_nonterminal(node::type t) { return !is_terminal(t); }

inline int arity(node::type t) {
  if (node::type::unary_begin <= t && t <= node::type::unary_end) {
    return 1;
  }
  if (node::type::binary_begin <= t && t <= node::type::binary_end) {
    return 2;
  }
  return 0;
}

#ifdef ENABLE_SIMD
inline __m256 evaluate_node_simd(const node &n, const float *data, 
                                const uint64_t stride, const uint64_t idx, 
                                const __m256 *in) {
    if (n.t == node::type::constant) {
        return _mm256_set1_ps(n.u.val);
    } else if (n.t == node::type::variable) {
        return _mm256_loadu_ps(data + (stride * n.u.fid) + idx);
    } else {
        __m256 abs_inval = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), in[0]);
        __m256 abs_inval1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), in[1]);
        
        switch (n.t) {
        // binary operators
        case node::type::add:
            return _mm256_add_ps(in[0], in[1]);
        case node::type::mul:
            return _mm256_mul_ps(in[0], in[1]);
        case node::type::sub:
            return _mm256_sub_ps(in[0], in[1]);
        case node::type::div: {
            __m256 min_mask = _mm256_cmp_ps(abs_inval1, _mm256_set1_ps(MIN_VAL), _CMP_LT_OQ);
            __m256 div_result = _mm256_div_ps(in[0], in[1]);
            return _mm256_blendv_ps(div_result, _mm256_set1_ps(1.0f), min_mask);
        }
        case node::type::max:
            return _mm256_max_ps(in[0], in[1]);
        case node::type::min:
            return _mm256_min_ps(in[0], in[1]);
        case node::type::pow:
            return _mm256_pow_ps(in[0], in[1]);
        case node::type::atan2:
            return _mm256_atan2_ps(in[0], in[1]);
        case node::type::fdim: {
            __m256 diff = _mm256_sub_ps(in[0], in[1]);
            return _mm256_max_ps(diff, _mm256_setzero_ps());
        }
            
        // unary operators
        case node::type::abs:
            return abs_inval;
        case node::type::neg:
            return _mm256_xor_ps(in[0], _mm256_set1_ps(-0.0f));
        case node::type::sq:
            return _mm256_mul_ps(in[0], in[0]);
        case node::type::cube:
            return _mm256_mul_ps(_mm256_mul_ps(in[0], in[0]), in[0]);
        case node::type::sqrt:
            return _mm256_sqrt_ps(abs_inval);
        case node::type::cbrt:
            return _mm256_cbrt_ps(in[0]);
        case node::type::inv: {
            __m256 min_mask = _mm256_cmp_ps(abs_inval, _mm256_set1_ps(MIN_VAL), _CMP_LT_OQ);
            __m256 inv_result = _mm256_div_ps(_mm256_set1_ps(1.0f), in[0]);
            return _mm256_blendv_ps(inv_result, _mm256_set1_ps(0.0f), min_mask);
        }
        case node::type::rcbrt:
            return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_cbrt_ps(in[0]));
        case node::type::rsqrt:
            return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(abs_inval));
            
        // trigonometric functions
        case node::type::sin:
            return _mm256_sin_ps(in[0]);
        case node::type::cos:
            return _mm256_cos_ps(in[0]);
        case node::type::tan:
            return _mm256_tan_ps(in[0]);
        case node::type::asin:
            return _mm256_asin_ps(in[0]);
        case node::type::acos:
            return _mm256_acos_ps(in[0]);
        case node::type::atan:
            return _mm256_atan_ps(in[0]);
            
        // hyperbolic functions
        case node::type::sinh:
            return _mm256_sinh_ps(in[0]);
        case node::type::cosh:
            return _mm256_cosh_ps(in[0]);
        case node::type::tanh:
            return _mm256_tanh_ps(in[0]);
        case node::type::asinh:
            return _mm256_asinh_ps(in[0]);
        case node::type::acosh:
            return _mm256_acosh_ps(in[0]);
        case node::type::atanh:
            return _mm256_atanh_ps(in[0]);
            
        // exponential and logarithmic functions
        case node::type::exp:
            return _mm256_exp_ps(in[0]);
        case node::type::log: {
            __m256 safe_input = _mm256_max_ps(abs_inval, _mm256_set1_ps(MIN_VAL));
            return _mm256_log_ps(safe_input);
        }
            
        default:
            return _mm256_setzero_ps();
        }
    }
}
#endif
inline float evaluate_node(const node &n, const float *data, const uint64_t stride,
                    const uint64_t idx, const float *in) {
  if (n.t == node::type::constant) {
    return n.u.val;
  } else if (n.t == node::type::variable) {
    return data[(stride * n.u.fid) + idx];
  } else {
    auto abs_inval = fabsf(in[0]), abs_inval1 = fabsf(in[1]);
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
    // binary operators
    case node::type::add:
      return in[0] + in[1];
    case node::type::atan2:
      return atan2f(in[0], in[1]);
    case node::type::div:
      return abs_inval1 < MIN_VAL ? 1.0f : (in[0]/in[1]);//fdividef(in[0], in[1]);
    case node::type::fdim:
      return fdimf(in[0], in[1]);
    case node::type::max:
      return fmaxf(in[0], in[1]);
    case node::type::min:
      return fminf(in[0], in[1]);
    case node::type::mul:
      return in[0] * in[1];
    case node::type::pow:
      return powf(in[0], in[1]);
    case node::type::sub:
      return in[0] - in[1];
    // unary operators
    case node::type::abs:
      return abs_inval;
    case node::type::acos:
      return acosf(in[0]);
    case node::type::acosh:
      return acoshf(in[0]);
    case node::type::asin:
      return asinf(in[0]);
    case node::type::asinh:
      return asinhf(in[0]);
    case node::type::atan:
      return atanf(in[0]);
    case node::type::atanh:
      return atanhf(in[0]);
    case node::type::cbrt:
      return cbrtf(in[0]);
    case node::type::cos:
      return cosf(in[0]);
    case node::type::cosh:
      return coshf(in[0]);
    case node::type::cube:
      return in[0] * in[0] * in[0];
    case node::type::exp:
      return expf(in[0]);
    case node::type::inv:
      return abs_inval < MIN_VAL ? 0.f : 1.f / in[0];
    case node::type::log:
      return abs_inval < MIN_VAL ? 0.f : logf(abs_inval);
    case node::type::neg:
      return -in[0];
    case node::type::rcbrt:
      return static_cast<float>(1.0) / cbrtf(in[0]);
    case node::type::rsqrt:
      return static_cast<float>(1.0) / sqrtf(abs_inval);
    case node::type::sin:
      return sinf(in[0]);
    case node::type::sinh:
      return sinhf(in[0]);
    case node::type::sq:
      return in[0] * in[0];
    case node::type::sqrt:
      return sqrtf(abs_inval);
    case node::type::tan:
      return tanf(in[0]);
    case node::type::tanh:
      return tanhf(in[0]);
    // shouldn't reach here!
    default:
      return 0.f;
    };
  }
}

} // namespace detail
} // namespace genetic
