# Standalone GPU test; intentionally not included by runtests.jl.

module TestGPU

import LinearAlgebra
import NLPModels
import Percival
using Test

# min 1/2 * dot(x, x), subject to sum(x) = n (or >= n), 0 <= x <= 2.
# Both variants have the unique solution x = ones(n).
struct QuadraticModel{T, V} <: NLPModels.AbstractNLPModel{T, V}
  meta::NLPModels.NLPModelMeta{T, V}
  counters::NLPModels.Counters
end

function QuadraticModel(
  ::Type{V};
  n = 100,
  inequality = false,
  bounds_analysis = false,
) where {T, V <: AbstractVector{T}}
  meta = NLPModels.NLPModelMeta{T, V}(
    n,
    x0 = fill!(V(undef, n), T(0.5)),
    lvar = fill!(V(undef, n), zero(T)),
    uvar = fill!(V(undef, n), T(2)),
    ncon = 1,
    lcon = fill!(V(undef, 1), T(n)),
    ucon = fill!(V(undef, 1), inequality ? T(Inf) : T(n)),
    nnzh = n,
    variable_bounds_analysis = bounds_analysis,
    constraint_bounds_analysis = bounds_analysis,
  )
  return QuadraticModel(meta, NLPModels.Counters())
end

function NLPModels.obj(nlp::QuadraticModel, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_obj)
  return LinearAlgebra.dot(x, x) / 2
end

function NLPModels.grad!(nlp::QuadraticModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(nlp, :neval_grad)
  g .= x
  return g
end

function NLPModels.cons!(nlp::QuadraticModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(nlp, :neval_cons)
  return fill!(c, sum(x))
end

function NLPModels.jprod!(
  nlp::QuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jprod)
  return fill!(Jv, sum(v))
end

function NLPModels.jtprod!(
  nlp::QuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  Jty::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jtprod)
  return fill!(Jty, sum(y))
end

function NLPModels.hprod!(
  nlp::QuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  NLPModels.increment!(nlp, :neval_hprod)
  Hv .= obj_weight .* v
  return Hv
end

# SlackModel evaluates the nonlinear block separately.
NLPModels.cons_nln!(nlp::QuadraticModel, x::AbstractVector, c::AbstractVector) =
  NLPModels.cons!(nlp, x, c)
NLPModels.jprod_nln!(
  nlp::QuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
) = NLPModels.jprod!(nlp, x, v, Jv)
NLPModels.jtprod_nln!(
  nlp::QuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  Jty::AbstractVector,
) = NLPModels.jtprod!(nlp, x, y, Jty)

function test_gpu(::Type{V}; cases = (false, true), bounds_analysis = false) where {V}
  @testset "GPU quadratic: inequality=$inequality" for inequality in cases
    nlp = QuadraticModel(V; inequality = inequality, bounds_analysis = bounds_analysis)
    if !inequality
      @test_throws ErrorException(
        "percival(::Val{:ineq}, nlp) should only be called for problems with inequalities. Use percival(nlp)",
      ) Percival.percival(Val(:ineq), nlp)
    end
    function callback(model, solver, stats)
      if stats.iter == 0
        @test solver.x isa V
        @test solver.sub_pb.store_Jv isa V
        @test solver.sub_pb.store_Jtv isa V
      end
      return
    end
    stats = Percival.percival(nlp; callback = callback, atol = 1e-8, rtol = 1e-8)
    @test stats.status == :first_order
    @test stats.solution isa V
    @test maximum(abs.(stats.solution .- 1)) < 1e-6
    @test stats.primal_feas < 1e-6
    @test stats.dual_feas < 1e-6
    @test isapprox(stats.objective, nlp.meta.nvar / 2; atol = 1e-6)
    @test minimum(stats.solution) >= 0
    @test maximum(stats.solution) <= 2
  end
  return
end

import CUDA

function runtests(cases = (false, true))
  if !CUDA.functional()
    error("This standalone test requires a functional CUDA GPU.")
  end
  CUDA.versioninfo()
  CUDA.allowscalar(false)
  TestGPU.test_gpu(typeof(CUDA.zeros(Float64, 0)); cases)
end

end # module TestGPU
