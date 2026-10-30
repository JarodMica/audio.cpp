#include "engine/community_models/fish_drama3/lora.h"

#include "engine/framework/assets/weight_adapter.h"

#include <stdexcept>
#include <variant>

namespace engine::models::fish_drama3 {

Drama3Lora::Drama3Lora(const std::filesystem::path & path, float strength, core::ExecutionContext & execution) {
    assets::WeightAdapterReadOptions read_options;
    read_options.strength = strength;
    auto adapter = assets::read_weight_adapter(path, read_options);
    if (!adapter.replacements.empty()) throw std::runtime_error("Drama 3 LoRA does not support full weight replacements");
    store_ = std::make_unique<core::BackendWeightStore>(execution.backend(), execution.backend_type(), "drama3.lora",
        (adapter.updates.size() * 2 + 1) * ggml_tensor_overhead());
    for (auto & [name, update] : adapter.updates) {
        const auto * low = std::get_if<assets::LowRankWeightUpdate>(&update.factors);
        if (low == nullptr) throw std::runtime_error("Drama 3 LoRA supports only A/B low-rank factors: " + name);
        Factors factors;
        factors.rank = low->r;
        factors.in = low->in;
        factors.out = low->out;
        factors.scaled_b = low->b;
        for (auto & value : factors.scaled_b) value *= update.scale;
        factors.a = store_->make_f32(core::TensorShape::from_dims({low->r, low->in}), low->a).tensor;
        // B starts at zero: requests are unadapted until they ask for the adapter.
        factors.b = store_->make_f32(core::TensorShape::from_dims({low->out, low->r}),
            std::vector<float>(factors.scaled_b.size(), 0.0F)).tensor;
        by_name_.emplace(name, std::move(factors));
    }
    store_->upload();
}

Drama3Lora::~Drama3Lora() = default;

void Drama3Lora::attach(const std::string & name, const ggml_tensor * weight) {
    const auto it = by_name_.find(name);
    if (it == by_name_.end()) return;
    auto & factors = it->second;
    if (weight == nullptr || weight->ne[0] != factors.in || weight->ne[1] != factors.out) {
        throw std::runtime_error("Drama 3 LoRA shape does not match model weight: " + name);
    }
    if (factors.attached) throw std::runtime_error("Drama 3 LoRA target is bound twice: " + name);
    factors.attached = true;
    by_weight_.emplace(weight, &factors);
}

void Drama3Lora::require_all_attached() const {
    for (const auto & [name, factors] : by_name_) {
        if (!factors.attached) throw std::runtime_error("Drama 3 LoRA target has no model weight: " + name);
    }
}

ggml_tensor * Drama3Lora::apply(ggml_context * ctx, ggml_tensor * input, ggml_tensor * output,
                                const ggml_tensor * weight, int64_t first_row, int64_t rows) const {
    const auto it = by_weight_.find(weight);
    if (it == by_weight_.end()) return output;
    const auto & factors = *it->second;
    if (rows == 0) rows = factors.out;
    if (first_row < 0 || rows <= 0 || first_row + rows > factors.out) throw std::runtime_error("Drama 3 LoRA row view is out of range");
    auto * x = input->type == GGML_TYPE_F32 ? input : ggml_cast(ctx, input, GGML_TYPE_F32);
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    x = ggml_reshape_2d(ctx, x, x->ne[0], ggml_nelements(x) / x->ne[0]);
    auto * b = rows == factors.out ? factors.b
        : ggml_view_2d(ctx, factors.b, factors.rank, rows, factors.b->nb[1], first_row * factors.b->nb[1]);
    auto * delta = ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, factors.a, x));
    if (ggml_nelements(delta) != ggml_nelements(output)) throw std::runtime_error("Drama 3 LoRA output shape mismatch");
    delta = ggml_reshape_4d(ctx, delta, output->ne[0], output->ne[1], output->ne[2], output->ne[3]);
    if (output->type != GGML_TYPE_F32) delta = ggml_cast(ctx, delta, output->type);
    return ggml_add(ctx, output, delta);
}

void Drama3Lora::set_enabled(bool enabled) {
    if (enabled == enabled_) return;
    for (const auto & [name, factors] : by_name_) {
        if (enabled) ggml_backend_tensor_set(factors.b, factors.scaled_b.data(), 0, ggml_nbytes(factors.b));
        else ggml_backend_tensor_memset(factors.b, 0, 0, ggml_nbytes(factors.b));
    }
    enabled_ = enabled;
}

}  // namespace engine::models::fish_drama3
