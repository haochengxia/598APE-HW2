/**
 * Performance benchmark for the Genetic Programming library
 * Measures execution time for symbolic regression and classification
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
// Include ctimer for end-to-end timing
#include "common.h"
#include "ctimer.h"
#include "fitness.h"
#include "genetic.h"
#include "node.h"
#include "program.h"

// Utility functions
namespace utils {

// Load dataset from CSV file
std::pair<std::vector<std::vector<float>>, std::vector<float>>
load_dataset(const std::string &filename) {
  std::vector<std::vector<float>> X;
  std::vector<float> y;

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  // Skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string value;

    while (std::getline(ss, value, ',')) {
      row.push_back(std::stof(value));
    }

    // Last column is target
    y.push_back(row.back());
    row.pop_back();
    X.push_back(row);
  }

  return {X, y};
}

// Convert 2D vector to column-major vector
std::vector<float>
flatten_column_major(const std::vector<std::vector<float>> &data) {
  if (data.empty())
    return {};

  size_t rows = data.size();
  size_t cols = data[0].size();
  std::vector<float> flattened(rows * cols);

  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
      flattened[j * rows + i] = data[i][j];
    }
  }

  return flattened;
}

// Split dataset into training and testing sets
template <typename T>
std::pair<T, T> train_test_split(const T &data, float test_size = 0.2) {
  T train, test;
  size_t test_count = static_cast<size_t>(data.size() * test_size);
  size_t train_count = data.size() - test_count;

  // Create a vector of indices
  std::vector<size_t> indices(data.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Split into train and test
  train.reserve(train_count);
  test.reserve(test_count);

  for (size_t i = 0; i < train_count; ++i) {
    train.push_back(data[indices[i]]);
  }

  for (size_t i = train_count; i < data.size(); ++i) {
    test.push_back(data[indices[i]]);
  }

  return {train, test};
}

// Calculate mean squared error
float mean_squared_error(const std::vector<float> &y_true,
                         const std::vector<float> &y_pred) {
  if (y_true.size() != y_pred.size()) {
    throw std::runtime_error("Arrays must have the same length");
  }

  float sum = 0.0f;
  for (size_t i = 0; i < y_true.size(); ++i) {
    float diff = y_true[i] - y_pred[i];
    sum += diff * diff;
  }

  return sum / y_true.size();
}

#ifdef ENABLE_COMMON
float accuracy(const std::vector<float> &y_true,
               const std::vector<float> &y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::runtime_error("Arrays must have the same length");
    }

    const size_t n = y_true.size();
    const float threshold = 0.5f;
    size_t correct = 0;

    #pragma omp simd reduction(+:correct)
    for (size_t i = 0; i < n; ++i) {
        const bool pred_class = y_pred[i] >= threshold;
        const bool true_class = y_true[i] >= threshold;
        correct += (pred_class == true_class);
    }

    return static_cast<float>(correct) / n;
}
#else
// Calculate accuracy for classification
float accuracy(const std::vector<float> &y_true,
               const std::vector<float> &y_pred) {
  if (y_true.size() != y_pred.size()) {
    throw std::runtime_error("Arrays must have the same length");
  }

  size_t correct = 0;
  for (size_t i = 0; i < y_true.size(); ++i) {
    if ((y_pred[i] >= 0.5f && y_true[i] >= 0.5f) ||
        (y_pred[i] < 0.5f && y_true[i] < 0.5f)) {
      correct++;
    }
  }

  return static_cast<float>(correct) / y_true.size();
}
#endif
} // namespace utils

using namespace genetic;

void insertionSortPrograms(genetic::program *programs, int size) {
  // set any nan values to +inf
  for (int i = 0; i < size; ++i) {
    if (std::isnan(programs[i].raw_fitness_)) {
      programs[i].raw_fitness_ = std::numeric_limits<float>::max();
    }
  }

  // do insertion sort
  for (int i = 1; i < size; i++) {
    genetic::program key(programs[i]);
    int j = i - 1;

    while (j >= 0 && (programs[j].raw_fitness_ > key.raw_fitness_)) {
      programs[j + 1] = programs[j];
      j--;
    }
    programs[j + 1] = key;
  }
}

void find_top_2_programs(program* programs, int size, program*& best1, program*& best2) {
    // deal with nan values
    best1 = nullptr;
    best2 = nullptr;
    float min1 = std::numeric_limits<float>::max();
    float min2 = std::numeric_limits<float>::max();

    // first scan to find min1
    for (int i = 0; i < size; ++i) {
        float fitness = programs[i].raw_fitness_;
        if (std::isnan(fitness)) continue;
        
        if (fitness < min1) {
            min1 = fitness;
            best1 = &programs[i];
        }
    }

    // if no valid min1 found, return
    if (!best1) return;

    // second scan to find min2
    for (int i = 0; i < size; ++i) {
        float fitness = programs[i].raw_fitness_;
        if (std::isnan(fitness)) continue;
        
        if (&programs[i] != best1 && fitness < min2) {
            min2 = fitness;
            best2 = &programs[i];
        }
    }
}

void run_symbolic_regression(const std::string &dataset_file) {
  std::cout << "\n===== Symbolic Regression Benchmark =====\n" << std::endl;

  // Initialize ctimer for end-to-end timing
  ctimer_t end_to_end_timer;
  ctimer_start(&end_to_end_timer);

  // Load dataset
  std::cout << "Loading dataset..." << std::endl;

  auto dataset = utils::load_dataset(dataset_file);
  auto X = dataset.first;
  auto y = dataset.second;

  std::cout << "Dataset dimensions: " << X.size() << " samples x "
            << X[0].size() << " features" << std::endl;

  // Split dataset
  auto X_split = utils::train_test_split(X);
  auto y_split = utils::train_test_split(y);

  std::vector<std::vector<float>> X_train = X_split.first;
  std::vector<std::vector<float>> X_test = X_split.second;
  std::vector<float> y_train = y_split.first;
  std::vector<float> y_test = y_split.second;

  // Flatten data for genetic library (column-major)
  std::vector<float> X_train_flat = utils::flatten_column_major(X_train);
  std::vector<float> X_test_flat = utils::flatten_column_major(X_test);

  // Create weights (all 1.0)
  std::vector<float> sample_weights(y_train.size(), 1.0f);

  // Set parameters
  genetic::param params;
  params.population_size = 512;
  params.generations = 16;
  params.tournament_size = 4;
  params.init_depth[0] = 2;
  params.init_depth[1] = 6;
  params.init_method = genetic::init_method_t::half_and_half;
  params.num_features = X_train[0].size(); // Number of features
  params.terminalRatio = 0.05;
  // Function set
  {
    using namespace genetic;
    params.function_set = {node::type::add,  node::type::sub, node::type::mul,
                           node::type::abs,  node::type::sin, node::type::cos,
                           node::type::fdim, node::type::log};
    // Arity set
    params.arity_set = {
        {1,
         {node::type::abs, node::type::sin, node::type::cos, node::type::log}},
        {2,
         {node::type::add, node::type::sub, node::type::mul,
          node::type::fdim}}};
  }

  params.metric = genetic::metric_t::mse; // Use MSE as the fitness metric
  params.parsimony_coefficient = 0.00f;   // Penalize complexity
  params.p_crossover = 0.80f;             // High crossover probability
  params.p_subtree_mutation = 0.05f;
  params.p_hoist_mutation = 0.01f;
  params.p_point_mutation = 0.01f;
  params.max_samples = 1.0f;        // Use all samples
  params.random_state = 2025000000; // For reproducibility

  // Running the symbolic regression
  std::cout << "Training symbolic regressor with " << params.population_size
            << " population size and " << params.generations << " generations"
            << std::endl;

  // Create history vector to store programs
  genetic::program_t final_programs;
  final_programs = new genetic::program[params.population_size]();
  std::vector<std::vector<genetic::program>> history;

  // Train the model
  genetic::symFit(X_train_flat.data(), y_train.data(), sample_weights.data(),
                  X_train.size(),    // Number of rows
                  X_train[0].size(), // Number of columns
                  params, final_programs, history);

  // Debug printing
  // for (int i = 0; i < params.population_size; ++i) {
  //   std::cout << "Gen 0: " << i << "=" << genetic::stringify(history[0][i])
  //             << std::endl;
  // }

  // Predict on top 2 candidates
#ifdef ENABLE_REMOVE_INSERTION_SORT
  program *best1, *best2;
  find_top_2_programs(final_programs, params.population_size, best1, best2);
  std::vector<float> y_pred1(X_test.size());
  genetic::symRegPredict(X_test_flat.data(), X_test.size(), best1, y_pred1.data());

  std::vector<float> y_pred2(X_test.size());
  genetic::symRegPredict(X_test_flat.data(), X_test.size(), best2, y_pred2.data());
#else
  insertionSortPrograms(final_programs, params.population_size);
  std::vector<float> y_pred1(X_test.size());
  genetic::symRegPredict(X_test_flat.data(), X_test.size(), &final_programs[0],
                         y_pred1.data());

  std::vector<float> y_pred2(X_test.size());
  genetic::symRegPredict(X_test_flat.data(), X_test.size(), &final_programs[1],
                         y_pred2.data());
#endif

  // Calculate MSE on test set
  float mse = utils::mean_squared_error(y_test, y_pred1);
  float mse2 = utils::mean_squared_error(y_test, y_pred2);

  // Extract the best programs and print some stats
  if (history.back().size() > 0) {
#ifdef ENABLE_REMOVE_INSERTION_SORT
    genetic::program_t best_program1 = best1;
    genetic::program_t best_program2 = best2;
#else
    genetic::program_t best_program1 = &final_programs[0];
    genetic::program_t best_program2 = &final_programs[1];
#endif
    std::cout << "Best program 1 details:" << std::endl;
    std::cout << "- Length: " << best_program1->len << " nodes" << std::endl;
    std::cout << "- Depth: " << best_program1->depth << std::endl;
    std::cout << "- Raw fitness: " << best_program1->raw_fitness_ << std::endl;
    std::cout << "- Test MSE: " << mse << std::endl;

    // Convert to string representation
    std::string program_str = genetic::stringify(*best_program1);
    std::cout << "- Program: " << program_str << std::endl;

    std::cout << "Best program 2 details:" << std::endl;
    std::cout << "- Length: " << best_program2->len << " nodes" << std::endl;
    std::cout << "- Depth: " << best_program2->depth << std::endl;
    std::cout << "- Raw fitness: " << best_program2->raw_fitness_ << std::endl;
    std::cout << "- Test MSE: " << mse2 << std::endl;

    // Convert to string representation
    std::string program_str2 = genetic::stringify(*best_program2);
    std::cout << "- Program: " << program_str2 << std::endl;
  }

  // Stop end-to-end timer and print results
  ctimer_stop(&end_to_end_timer);
  ctimer_measure(&end_to_end_timer);
  ctimer_print(end_to_end_timer, "Symbolic Regression (End-to-End)");

  delete[] final_programs;
}

void run_symbolic_classification(const std::string &dataset_file) {
  std::cout << "\n===== Symbolic Classification Benchmark =====\n" << std::endl;

  // Initialize ctimer for end-to-end timing
  ctimer_t end_to_end_timer;
  ctimer_start(&end_to_end_timer);

  // Load dataset
  std::cout << "Loading dataset..." << std::endl;

  auto dataset = utils::load_dataset(dataset_file);
  auto X = dataset.first;
  auto y = dataset.second;

  std::cout << "Dataset dimensions: " << X.size() << " samples x "
            << X[0].size() << " features" << std::endl;

  // Split dataset
  auto X_split = utils::train_test_split(X);
  auto y_split = utils::train_test_split(y);

  std::vector<std::vector<float>> X_train = X_split.first;
  std::vector<std::vector<float>> X_test = X_split.second;
  std::vector<float> y_train = y_split.first;
  std::vector<float> y_test = y_split.second;

  // Flatten data for genetic library (column-major)
  std::vector<float> X_train_flat = utils::flatten_column_major(X_train);
  std::vector<float> X_test_flat = utils::flatten_column_major(X_test);

  // Create weights (all 1.0)
  std::vector<float> sample_weights(y_train.size(), 1.0f);

  // Set parameters
  genetic::param params;
  params.population_size = 16384;
  params.generations = 16;
  params.tournament_size = 16;
  params.init_depth[0] = 2;
  params.init_depth[1] = 6;
  params.init_method = genetic::init_method_t::half_and_half;
  params.num_features = X_train[0].size(); // Number of features
  params.terminalRatio = 0.05;

  // Function set for classification
  params.function_set = {node::type::add,  node::type::sub, node::type::mul,
                         node::type::sin,  node::type::cos, node::type::sq,
                         node::type::sqrt, node::type::abs, node::type::fdim};

  // Don't worry if you see stuff like sqrt(-5) -> we consider only the absolute
  // value in that case
  params.arity_set = {
      {1,
       {node::type::abs, node::type::sin, node::type::cos, node::type::sq,
        node::type::sqrt}},
      {2,
       {node::type::add, node::type::sub, node::type::mul, node::type::fdim}}};

  params.metric = genetic::metric_t::logloss; // Use log loss forclassification
  params.transformer =
      genetic::transformer_t::sigmoid; // Use sigmoid for binary classification
  params.parsimony_coefficient = 0.01f;
  params.p_crossover = 0.80f; // High crossover probability
  params.p_subtree_mutation = 0.05f;
  params.p_hoist_mutation = 0.01f;
  params.p_point_mutation = 0.01f;
  params.max_samples = 1.0f;  // Use all samples
  params.random_state = 2025; // For reproducibility

  // Running the symbolic classification
  std::cout << "Training symbolic classifier with " << params.population_size
            << " population size and " << params.generations << " generations "
            << std::endl;

  // Create history vector to store programs
  genetic::program_t final_programs;
  final_programs = new genetic::program[params.population_size]();

  std::vector<std::vector<genetic::program>> history;

  // Train the model
  genetic::symFit(X_train_flat.data(), y_train.data(), sample_weights.data(),
                  X_train.size(),    // Number of rows
                  X_train[0].size(), // Number of columns
                  params, final_programs, history);

  // // print and check programs from hsitory
  // for (int i = 0; i < params.population_size; ++i) {
  //   std::cout << "Gen " << 16 << " : "
  //             << genetic::stringify(history[history.size() - 1][i])
  //             << std::endl;
  // }

  // Predict classes for best 2 programs acc to training
  // 
#ifdef ENABLE_REMOVE_INSERTION_SORT
  program *best1, *best2;
  find_top_2_programs(final_programs, params.population_size, best1, best2);
  std::vector<float> y_pred1(X_test.size());
  genetic::symClfPredict(X_test_flat.data(), X_test.size(), params, best1, y_pred1.data());

  std::vector<float> y_pred2(X_test.size());
  genetic::symClfPredict(X_test_flat.data(), X_test.size(), params, best2, y_pred2.data());
#else
  insertionSortPrograms(final_programs, params.population_size);
  std::vector<float> y_pred1(X_test.size());
  genetic::symClfPredict(X_test_flat.data(), X_test.size(), params,
                         &final_programs[0], y_pred1.data());

  std::vector<float> y_pred2(X_test.size());
  genetic::symClfPredict(X_test_flat.data(), X_test.size(), params,
                         &final_programs[1], y_pred2.data());
#endif
  float acc = utils::accuracy(y_test, y_pred1);
  float acc2 = utils::accuracy(y_test, y_pred2);

  // Extract the best programs and print some stats
#ifdef ENABLE_REMOVE_INSERTION_SORT
  genetic::program_t best_program1 = best1;
  genetic::program_t best_program2 = best2;
#else
  genetic::program_t best_program1 = &final_programs[0];
  genetic::program_t best_program2 = &final_programs[1];
#endif
  if (history.back().size() > 0) {
    std::cout << "Best program 1 details:" << std::endl;
    std::cout << "- Length: " << best_program1->len << " nodes" << std::endl;
    std::cout << "- Depth: " << best_program1->depth << std::endl;
    std::cout << "- Raw fitness: " << best_program1->raw_fitness_ << std::endl;
    std::cout << "- Test accuracy: " << acc << std::endl;

    // Convert to string representation
    std::string program_str = genetic::stringify(*best_program1);
    std::cout << "- Program: " << program_str << std::endl;

    genetic::program_t best_program2 = &final_programs[1];
    std::cout << "Best program 2 details:" << std::endl;
    std::cout << "- Length: " << best_program2->len << " nodes" << std::endl;
    std::cout << "- Depth: " << best_program2->depth << std::endl;
    std::cout << "- Raw fitness: " << best_program2->raw_fitness_ << std::endl;
    std::cout << "- Test accuracy: " << acc2 << std::endl;

    // Convert to string representation
    std::string program_str2 = genetic::stringify(*best_program2);
    std::cout << "- Program: " << program_str2 << std::endl;
  }

  // Stop end-to-end timer and print results
  ctimer_stop(&end_to_end_timer);
  ctimer_measure(&end_to_end_timer);
  ctimer_print(end_to_end_timer, "Symbolic Classification (End-to-End)");

  delete[] final_programs;
}

int main(int argc, char *argv[]) {
  try {
    // Default datasets
    std::string regression_dataset = "benchmark/diabetes.csv";
    std::string classification_dataset = "benchmark/cancer.csv";
    std::string housing_dataset = "benchmark/housing.csv";

    std::string arg_dset(argv[1]);

    if (arg_dset == "diabetes") {
      run_symbolic_regression(regression_dataset);
    } else if (arg_dset == "cancer") {
      run_symbolic_classification(classification_dataset);
    } else if (arg_dset == "housing") {
      for (int i = 0; i < 1; ++i) {
        run_symbolic_regression(housing_dataset);
      }
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
