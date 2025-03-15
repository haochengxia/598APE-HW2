#pragma once

namespace genetic {

#ifdef ENABLE_STACK
template <typename DataT, int MaxSize> 
struct stack {
  explicit stack() : top_(0) {
    #ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    #endif
    
    // avoid initialization of array to improve performance
    // regs_ array will be written to when actually used
  }

  // inline to reduce function call overhead
  __attribute__((always_inline)) 
  void push(DataT val) {
    // directly write to the top
    regs_[top_++] = val;
  }

  __attribute__((always_inline)) 
  DataT pop() {
    // directly return the top element
    return regs_[--top_];
  }

  __attribute__((always_inline)) 
  bool empty() const { 
    return top_ == 0; 
  }

  __attribute__((always_inline)) 
  int size() const { 
    return top_; 
  }

  private:
    // use a single pointer to track the top
    int top_;
    // align memory to optimize access
    alignas(64) DataT regs_[MaxSize];
};
#else
/**
 * @brief A fixed capacity stack on device currently used for AST evaluation
 *
 * The idea is to use only the registers to store the elements of the stack,
 * thereby achieving the best performance.
 *
 * @tparam DataT   data type of the stack elements
 * @tparam MaxSize max capacity of the stack
 */
template <typename DataT, int MaxSize> struct stack {
  explicit stack() : elements_(0) {
    for (int i = 0; i < MaxSize; ++i) {
      regs_[i] = DataT(0);
    }
  }

  /** Checks if the stack is empty */
  bool empty() const { return elements_ == 0; }

  /** Current number of elements in the stack */
  int size() const { return elements_; }

  /** Checks if the number of elements in the stack equal its capacity */
  bool full() const { return elements_ == MaxSize; }

  /**
   * @brief Pushes the input element to the top of the stack
   *
   * @param[in] val input element to be pushed
   *
   * @note If called when the stack is already full, then it is a no-op! To keep
   *       the device-side logic simpler, it has been designed this way. Trying
   *       to push more than `MaxSize` elements leads to all sorts of incorrect
   *       behavior.
   */
  void push(DataT val) {
    for (int i = MaxSize - 1; i >= 0; --i) {
      if (elements_ == i) {
        ++elements_;
        regs_[i] = val;
      }
    }
  }

  /**
   * @brief Lazily pops the top element from the stack
   *
   * @return pops the element and returns it, if already reached bottom, then it
   *         returns zero.
   *
   * @note If called when the stack is already empty, then it just returns a
   *       value of zero! To keep the device-side logic simpler, it has been
   *       designed this way. Trying to pop beyond the bottom of the stack leads
   *       to all sorts of incorrect behavior.
   */
  DataT pop() {
    for (int i = 0; i < MaxSize; ++i) {
      if (elements_ == (i + 1)) {
        elements_--;
        return regs_[i];
      }
    }

    return DataT(0);
  }

private:
  int elements_;
  DataT regs_[MaxSize];
}; // struct stack
#endif
} // namespace genetic
