module TestAugLagModel

using Test
import ADNLPModels
import NLPModels
import Percival

@testset "AugLagModel work buffer initialization" begin
  @testset "$T, $ncon constraints" for T in (Float64, BigFloat), ncon in (0, 1, 2)
    x = T[1, 2]
    y = fill(T(3), ncon)
    nlp = ADNLPModels.ADNLPModel(
      x -> sum(abs2, x) / 2,
      x,
      x -> fill(sum(x), ncon),
      ones(T, ncon),
      ones(T, ncon),
    )
    # Solver construction supplies uninitialized vectors, including reference
    # elements for BigFloat. Construction must not read their contents.
    model = Percival.AugLagModel(
      nlp,
      Vector{T}(undef, ncon),
      zero(T),
      Vector{T}(undef, 2),
      zero(T),
      Vector{T}(undef, ncon),
    )
    buffer = model.μc_y
    @test buffer isa Vector{T}
    @test length(buffer) == ncon
    @test buffer !== model.y
    @test buffer !== model.cx

    # Check both the initial solve and reuse with a new point and multipliers.
    for (x, y, μ) in ((x, y, T(2)), (T[2, 3], fill(T(4), ncon), T(3)))
      fx = NLPModels.obj(nlp, x)
      Percival.reinit!(model, nlp, fx, μ, x, y)
      @test model.μc_y === buffer
      @test model.cx == fill(sum(x) - 1, ncon)
      @test model.μc_y == μ .* model.cx .- y
      @test NLPModels.grad(model, x) ≈ x .+ sum(model.μc_y)
    end
  end
end

end # module TestAugLagModel
