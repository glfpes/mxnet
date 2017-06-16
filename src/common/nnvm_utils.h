/*!
 * Copyright (c) 2017 by Contributors
 * \file nnvm_utils.h
 * \brief NNVM utilility functions.
 */
#ifndef MXNET_COMMON_NNVM_UTILS_H_
#define MXNET_COMMON_NNVM_UTILS_H_

#include <string>
#include <thread>

#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>

#include "../engine/profiler.h"

namespace mxnet {
namespace common {

#if MXNET_USE_PROFILER
  #define MXNET_PROFILER_NNVM_PASS(name, statement) \
    do { \
      auto __pass_stat = engine::Profiler::Get()->AddOprStat(Context::kCPU, 0); \
      __pass_stat->thread_id = std::hash<std::thread::id>()(std::this_thread::get_id()); \
      strcpy(__pass_stat->opr_name, "g:"); \
      strncpy(__pass_stat->opr_name + 2, name, \
              sizeof(__pass_stat->opr_name) - 3); \
      SetOprStart(__pass_stat); \
      statement; \
      SetOprEnd(__pass_stat); \
    } while (false)
#else
  #define MXNET_PROFILER_NNVM_PASS(name, statement) \
  do { \
    statement; \
  } while (false)
#endif

inline nnvm::Graph ApplyPass(nnvm::Graph src, const std::string& pass) {
  nnvm::Graph ret;
  MXNET_PROFILER_NNVM_PASS(pass.c_str(), ret = nnvm::ApplyPass(src, pass));
  return ret;
}

} // namespace common
} // namespace mxnet

#endif // MXNET_COMMON_NNVM_UTILS_H_
