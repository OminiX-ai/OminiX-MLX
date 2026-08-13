/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/metal.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" mlx_metal_device_info_t mlx_metal_device_info(void) {
  // MLX 0.32 replaced metal::device_info() with a device-scoped
  // mlx::core::device_info(Device), and the returned map's keys differ by
  // backend and version. Look each one up defensively so a renamed or absent
  // key yields 0 rather than throwing std::bad_variant_access across the C ABI.
  auto info = mlx::core::device_info(mlx::core::Device::gpu);

  auto str_or = [&](const char* key, const char* fallback) -> std::string {
    auto it = info.find(key);
    if (it != info.end()) {
      if (auto* v = std::get_if<std::string>(&it->second)) {
        return *v;
      }
    }
    return fallback;
  };
  auto size_or = [&](std::initializer_list<const char*> keys) -> size_t {
    for (auto key : keys) {
      auto it = info.find(key);
      if (it != info.end()) {
        if (auto* v = std::get_if<size_t>(&it->second)) {
          return *v;
        }
      }
    }
    return 0;
  };

  mlx_metal_device_info_t c_info;
  std::strncpy(c_info.architecture, str_or("architecture", "unknown").c_str(), 256);
  c_info.architecture[255] = '\0';
  c_info.max_buffer_length = size_or({"max_buffer_length"});
  c_info.max_recommended_working_set_size =
      size_or({"max_recommended_working_set_size"});
  c_info.memory_size = size_or({"memory_size", "total_memory"});
  return c_info;
}

extern "C" int mlx_metal_is_available(bool* res) {
  try {
    *res = mlx::core::metal::is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_start_capture(const char* path) {
  try {
    mlx::core::metal::start_capture(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_stop_capture(void) {
  try {
    mlx::core::metal::stop_capture();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
