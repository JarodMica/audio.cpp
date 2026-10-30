#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::fish_drama3 {

// A low-rank adapter evaluated beside the base weights instead of merged into
// them, so one loaded model serves adapted and unadapted requests. The B factors
// hold either the scaled adapter or zeros; switching rewrites only B, so weight
// pointers, cached graphs and CUDA graph captures stay valid.
class Drama3Lora {
public:
    Drama3Lora(const std::filesystem::path & path, float strength, core::ExecutionContext & execution);
    ~Drama3Lora();

    // Binds the adapter's update for `name` (a weight name without ".weight")
    // to a loaded weight. Names without an update are left unadapted.
    void attach(const std::string & name, const ggml_tensor * weight);
    // Throws when an adapter update has no matching model weight.
    void require_all_attached() const;

    // Returns output + B(A input) for an adapted weight, otherwise output.
    // `first_row`/`rows` select output rows when the weight is a row view.
    ggml_tensor * apply(ggml_context * ctx, ggml_tensor * input, ggml_tensor * output,
                        const ggml_tensor * weight, int64_t first_row = 0, int64_t rows = 0) const;

    void set_enabled(bool enabled);
    bool enabled() const noexcept { return enabled_; }

private:
    struct Factors {
        ggml_tensor * a = nullptr;
        ggml_tensor * b = nullptr;
        int64_t rank = 0;
        int64_t in = 0;
        int64_t out = 0;
        std::vector<float> scaled_b;
        bool attached = false;
    };

    std::unique_ptr<core::BackendWeightStore> store_;
    std::unordered_map<std::string, Factors> by_name_;
    std::unordered_map<const ggml_tensor *, const Factors *> by_weight_;
    bool enabled_ = false;
};

}  // namespace engine::models::fish_drama3
