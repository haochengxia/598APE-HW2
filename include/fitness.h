#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace genetic {

template <typename math_t = float>
void weightedPearson(const uint64_t n_samples, const uint64_t n_progs,
                     const math_t *Y, const math_t *X, const math_t *W,
                     math_t *out) {
  // Find Karl Pearson's correlation coefficient

  // cudaStream_t stream = h.get_stream();

  std::vector<math_t> corr(n_samples * n_progs); // correlation matrix
  std::vector<math_t> y_tmp(n_samples);
  std::vector<math_t> x_tmp(n_samples * n_progs);

  math_t y_mu;                       // output mean
  std::vector<math_t> x_mu(n_progs); // predicted output mean

  std::vector<math_t> y_diff(n_samples);           // normalized output
  std::vector<math_t> x_diff(n_samples * n_progs); // normalized predicted
                                                   // output
  math_t y_std;                                    // output stddev
  std::vector<math_t> x_std(n_progs);              // predicted output stddev

  math_t N = static_cast<math_t>(n_samples); // (math_t)n_samples;

  // Sum of weights
  math_t WS = (math_t)0;
  for (uint64_t i = 0; i < n_samples; ++i) {
    WS += W[i];
  }

  {
    // Find y_tmp
    for (uint64_t i = 0; i < n_samples; ++i) {
      y_tmp[i] = Y[i] * W[i] * N / WS;
    }

    // Find y_mu
    for (uint64_t i = 0; i < n_samples; ++i) {
      y_mu += y_tmp[i] / N;
    }
  }

  // Find x_mu - bcast W along columns
  {
    // find x_tmp
    for (uint64_t pid = 0; pid < n_progs; ++pid) {
      for (uint64_t i = 0; i < n_samples; ++i) {
        x_tmp[pid * n_samples + i] = X[pid * n_samples + i] * W[i] * N / WS;
      }
    }

    // find x_mu
    for (uint64_t pid = 0; pid < n_progs; ++pid) {
      for (uint64_t i = 0; i < n_samples; ++i) {
        x_mu[pid] += x_tmp[pid * n_samples + i] / N;
      }
    }
  }

  // Find y_diff
  for (uint64_t i = 0; i < n_samples; ++i) {
    y_diff[i] = Y[i] - y_mu;
  }

  // raft::stats::meanCenter(y_diff.data(), Y, y_mu.data(), (uint64_t)1,
  // n_samples,
  //                         false, true, stream);

  // Find x_diff
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t i = 0; i < n_samples; ++i) {
      x_diff[pid * n_samples + i] = X[pid * n_samples + i] - x_mu[pid];
    }
  }

  // Find y_std
  {
    y_std = static_cast<math_t>(0);
    for (uint64_t i = 0; i < n_samples; ++i) {
      y_std += y_diff[i] * y_diff[i] * W[i];
    }
    y_std = std::sqrt(y_std);
  }

  // Find x_std
  {
    for (uint64_t pid = 0; pid < n_progs; ++pid) {
      for (uint64_t i = 0; i < n_samples; ++i) {
        x_std[pid] +=
            x_diff[pid * n_samples + i] * x_diff[pid * n_samples + i] * W[i];
      }
      x_std[pid] = std::sqrt(x_std[pid]);
    }
  }

  // Cross covariance
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t i = 0; i < n_samples; ++i) {
      corr[pid * n_samples + i] =
          N * W[i] * x_diff[pid * n_samples + i] * y_diff[i] / y_std;
    }
  }

  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t i = 0; i < n_samples; ++i) {
      corr[pid * n_samples + i] = corr[pid * n_samples + i] / x_std[pid];
    }
  }

  // Find out = corr_mu, the correlation coefficient
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    out[pid] = static_cast<math_t>(0);
    for (uint64_t i = 0; i < n_samples; ++i) {
      out[pid] += corr[pid * n_samples + i] / N;
    }
  }
}

struct rank_functor {
  template <typename math_t> math_t operator()(math_t data) {
    if (data == static_cast<math_t>(0))
      return 0;
    else
      return 1;
  }
};

template <typename math_t = float>
void weightedSpearman(const uint64_t n_samples, const uint64_t n_progs,
                      const math_t *Y, const math_t *Y_pred, const math_t *W,
                      math_t *out) {
  // Get ranks for Y in Yrank
  std::vector<math_t> Ycopy(Y, Y + n_samples);
  std::vector<size_t> rank_idx(n_samples, 0);
  std::vector<math_t> rank_diff(n_samples, 0);
  std::vector<math_t> Yrank(n_samples, 0);

  std::iota(rank_idx.begin(), rank_idx.end(), 0);
  std::sort(rank_idx.begin(), rank_idx.end(), [&Ycopy](size_t i1, size_t i2) {
    return Ycopy[i1] < Ycopy[i2];
  }); // sort_by_key

  // Sort Y
  std::vector<math_t> Ysorted(n_samples);
  for (uint64_t i = 0; i < n_samples; ++i) {
    Ysorted[i] = Ycopy[rank_idx[i]];
  }

  std::adjacent_difference(Ysorted.begin(), Ysorted.end(), rank_diff.begin());
  std::transform(rank_diff.begin(), rank_diff.end(), rank_diff.begin(),
                 rank_functor());
  rank_diff[0] = 1;
  std::inclusive_scan(rank_diff.begin(), rank_diff.end(),
                      rank_diff.begin()); // the actual rank

  // no permutation_iterator in cpp :(
  for (uint64_t i = 0; i < n_samples; ++i) {
    Yrank[rank_idx[i]] = rank_diff[i];
  }

  // Get ranks for Y_pred in Ypredrank
  std::vector<math_t> Ypredcopy(Y_pred, Y_pred + n_samples * n_progs);
  std::vector<math_t> Ypredrank(n_samples * n_progs, 0);

  for (std::size_t pid = 0; pid < n_progs; ++pid) {
    std::iota(rank_idx.begin(), rank_idx.end(), 0);
    std::sort(rank_idx.begin(), rank_idx.end(),
              [&Ypredcopy, pid, n_samples](size_t i1, size_t i2) {
                return Ypredcopy[pid * n_samples + i1] <
                       Ypredcopy[pid * n_samples + i2];
              });

    for (uint64_t i = 0; i < n_samples; ++i) {
      Ysorted[i] = Ypredcopy[pid * n_samples + rank_idx[i]];
    }

    std::adjacent_difference(Ysorted.begin(), Ysorted.end(), rank_diff.begin());
    std::transform(rank_diff.begin(), rank_diff.end(), rank_diff.begin(),
                   rank_functor());
    rank_diff[0] = 1;
    std::inclusive_scan(rank_diff.begin(), rank_diff.end(), rank_diff.begin());

    for (uint64_t i = 0; i < n_samples; ++i) {
      Ypredrank[pid * n_samples + rank_idx[i]] = rank_diff[i];
    }
  }

  // Compute pearson's coefficient on the ranks
  weightedPearson(n_samples, n_progs, Yrank.data(), Ypredrank.data(), W, out);
}

template <typename math_t = float>
void meanAbsoluteError(const uint64_t n_samples, const uint64_t n_progs,
                       const math_t *Y, const math_t *Y_pred, const math_t *W,
                       math_t *out) {
  std::vector<math_t> error(n_samples * n_progs);
  math_t N = static_cast<math_t>(n_samples);

  // Weight Sum
  math_t WS = static_cast<math_t>(0);
  for (uint64_t i = 0; i < n_samples; ++i) {
    WS += W[i];
  }

  // Compute absolute differences
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t i = 0; i < n_samples; ++i) {
      error[pid * n_samples + i] =
          N * W[i] * abs(Y_pred[pid * n_samples + i] - Y[i]) / WS;
    }
  }

  // Average along rows
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    out[pid] = static_cast<math_t>(0);
    for (uint64_t i = 0; i < n_samples; ++i) {
      out[pid] += error[pid * n_samples + i] / N;
    }
  }
}

#ifdef ENABLE_COMMON
template <typename math_t = float>
void meanSquareError(const uint64_t n_samples, const uint64_t n_progs,
                     const math_t *Y, const math_t *Y_pred, const math_t *W,
                     math_t *out) {
    const math_t N = static_cast<math_t>(n_samples);
    
    math_t WS = static_cast<math_t>(0);
    #pragma omp simd reduction(+:WS)
    for (uint64_t i = 0; i < n_samples; ++i) {
        WS += W[i];
    }
    
    const math_t scale = N / WS;
    
    #pragma omp parallel for if(n_progs > 1)
    for (uint64_t pid = 0; pid < n_progs; ++pid) {
        math_t sum = 0;
        const math_t* Y_pred_prog = Y_pred + pid * n_samples;
        
        #pragma omp simd reduction(+:sum)
        for (uint64_t i = 0; i < n_samples; ++i) {
            const math_t diff = Y_pred_prog[i] - Y[i];
            sum += W[i] * diff * diff;
        }
        
        out[pid] = sum * scale / N;
    }
}
#else
template <typename math_t = float>
void meanSquareError(const uint64_t n_samples, const uint64_t n_progs,
                     const math_t *Y, const math_t *Y_pred, const math_t *W,
                     math_t *out) {

  std::vector<math_t> error(n_samples * n_progs);
  math_t N = static_cast<math_t>(n_samples);

  // Weight Sum
  math_t WS = static_cast<math_t>(0);
  for (uint64_t i = 0; i < n_samples; ++i) {
    WS += W[i];
  }

  // Compute absolute differences
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t i = 0; i < n_samples; ++i) {
      error[pid * n_samples + i] = N * W[i] *
                                   (Y_pred[pid * n_samples + i] - Y[i]) *
                                   (Y_pred[pid * n_samples + i] - Y[i]) / WS;
    }
  }

  // Average along rows
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    out[pid] = static_cast<math_t>(0);
    for (uint64_t i = 0; i < n_samples; ++i) {
      out[pid] += error[pid * n_samples + i] / N;
    }
  }
}
#endif
template <typename math_t = float>
void rootMeanSquareError(const uint64_t n_samples, const uint64_t n_progs,
                         const math_t *Y, const math_t *Y_pred, const math_t *W,
                         math_t *out) {

  // Find MSE
  meanSquareError(n_samples, n_progs, Y, Y_pred, W, out);

  // Take sqrt on all entries
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    out[pid] = sqrt(out[pid]);
  }
}

template <typename math_t = float>
void logLoss(const uint64_t n_samples, const uint64_t n_progs, const math_t *Y,
             const math_t *Y_pred, const math_t *W, math_t *out) {
  // Logistic error per sample
  std::vector<math_t> error(n_samples * n_progs);
  math_t N = (math_t)n_samples;

  // Weight Sum
  math_t WS = static_cast<math_t>(0);
  for (uint64_t i = 0; i < n_samples; ++i) {
    WS += W[i];
  }

  // Compute logistic loss as described in
  // http://fa.bianp.net/blog/2019/evaluate_logistic/
  // in an attempt to avoid encountering nan values. Modified for weighted
  // logistic regression.
  // PS - In 2021, I spent 2 sleepless nights trying to just compute this, then
  // adapt it for the weighted version (turned out pre-multiplying N just
  // worked). Improving numerical stability in CUDA is ... :)

  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t i = 0; i < n_samples; ++i) {
      math_t logsig;
      math_t yp = Y_pred[pid * n_samples + i];
      math_t y = Y[i];
      math_t w = W[i];
      if (yp < -33.3)
        logsig = yp;
      else if (yp <= -18)
        logsig = yp - expf(yp);
      else if (yp <= 37)
        logsig = -log1pf(expf(-yp));
      else
        logsig = -expf(-yp);
      error[pid * n_samples + i] = ((1 - y) * yp - logsig) * (N * w / WS);
    }
  }

  // Take average along rows
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    out[pid] = static_cast<math_t>(0);
    for (uint64_t i = 0; i < n_samples; ++i) {
      out[pid] += error[pid * n_samples + i] / N;
    }
  }
}

} // namespace genetic
