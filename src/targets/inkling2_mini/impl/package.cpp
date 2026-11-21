#include <ninfer/targets/inkling2_mini/package.h>

#include "artifact/reader.h"
#include "targets/inkling2_mini/impl/artifact/bindings.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::inkling2_mini::detail {

class LoadPlan::Impl {
public:
    explicit Impl(ArtifactLoadPlan target_plan) : plan(std::move(target_plan)) {}
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("inkling2 load plan is empty"); }
    return impl_->plan.materialization;
}

class LoadedModel::Impl {
public:
    explicit Impl(LoadedModelData model_data) : data(std::move(model_data)) {}
    LoadedModelData data;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::inkling2_mini::detail

namespace ninfer::targets::inkling2_mini {
namespace {

constexpr ModelSamplingDefaults kInkling2Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 1.0F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 1.0F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kInkling2Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package 'inkling2_mini'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "nvfp4-mxfp8") {
        return WeightsProfile::Nvfp4Mxfp8;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target 'inkling2_mini'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile /*weights_profile*/) {
    if (binder.device_count() != options.tp * options.pp) {
        throw std::invalid_argument("artifact binder device count does not match tp * pp");
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(detail::bind_artifact(binder, options)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("inkling2 load plan is empty"); }
    auto data = detail::materialize_model(plan.impl_->plan.bindings, std::move(materialized),
                                          plan.impl_->plan.tp, plan.impl_->plan.mtp_enabled,
                                          plan.impl_->plan.draft_tokens);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::make_unique<LoadedModel::Impl>(std::move(data))));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("inkling2 loaded model is empty"); }
    return inkling2_mini::make_frontend(model.impl_->data.frontend, options.max_context);
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile /*weights_profile*/) {
    return inkling2_mini::make_sequence_planner(device, options);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan,
                        ExecutionContext& execution) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("inkling2 loaded model is empty"); }
    return inkling2_mini::create_program(model.impl_->data, std::move(plan), execution);
}

} // namespace ninfer::targets::inkling2_mini
