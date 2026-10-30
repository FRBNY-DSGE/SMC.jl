using Test, SMC, LinearAlgebra, Distributions, Random

@testset "Cached mutation proposals match the original" begin
    c = 0.7
    μ9 = collect(range(-0.4, 0.4; length=9))
    Σ9 = Matrix{Float64}(I, 9, 9) + 0.1 .* ones(9, 9)
    for (μ, Σ, block, θ_old) in (([0.1, -0.2], [1.0 0.25; 0.25 1.5], [1, 2], [0.4, -0.6]),
                                  (μ9, Σ9, collect(1:9), collect(range(-0.1, 0.7; length=9))),
                                  (μ9, Σ9, [2, 5, 9], [0.1, -0.3, 0.6]))
        original = MvNormal(μ[block], Σ[block, block])
        for α in (0.0, 0.4, 0.9, 1.0)
            proposal = SMC.mutation_block_proposal(μ, Σ, block, c, α)
            Random.seed!(20261020)
            expected_draw = SMC.mvnormal_mixture_draw(θ_old, original; c=c, α=α)
            Random.seed!(20261020)
            actual_draw = SMC.draw_mutation_proposal(θ_old, proposal)
            @test actual_draw == expected_draw

            expected_q = SMC.compute_proposal_densities(actual_draw, θ_old, original;
                                                          c=c, α=α)
            actual_q = SMC.mutation_proposal_densities(actual_draw, θ_old,
                                                        proposal, α)
            @test actual_q == expected_q
        end
    end
end

# N2: the proposal density is the density of the mixture that draw_mutation_proposal draws
# from, including the diagonal component's c·√Σᵢᵢ scale (it used √Σᵢᵢ, so the kernel was not
# invariant for α < 1)
@testset "Mutation proposal density matches the drawn mixture" begin
    c, μ = 0.3, [0.1, -0.2]
    Σ = [1.0 0.25; 0.25 1.5]
    θ_old, θ_new = [0.4, -0.6], [0.2, -0.1]
    for α in (0.0, 0.5, 0.975, 1.0)
        proposal = SMC.mutation_block_proposal(μ, Σ, [1, 2], c, α)
        q(to, from) = log(α * pdf(MvNormal(from, c^2 * Σ), to) +
                          (1 - α) / 2 * pdf(MvNormal(from, c^2 * Diagonal(diag(Σ))), to) +
                          (1 - α) / 2 * pdf(MvNormal(μ, c^2 * Σ), to))
        q0, q1 = SMC.mutation_proposal_densities(θ_new, θ_old, proposal, α)
        @test q0 ≈ q(θ_old, θ_new)
        @test q1 ≈ q(θ_new, θ_old)
        @test all(SMC.compute_proposal_densities(θ_new, θ_old, MvNormal(μ, Σ); c = c, α = α) .≈ (q0, q1))
    end
end

@testset "mvnormal_mixture_draw draws from the RNG it is given" begin
    d = MvNormal([0.1, -0.2, 0.3], [1.0 0.25 0.0; 0.25 1.5 0.1; 0.0 0.1 0.8])
    θ_old = [0.4, -0.6, 0.2]
    # An explicit RNG: the draw does not depend on, or advance, the global RNG
    draws = map((1, 2)) do global_seed
        Random.seed!(global_seed)
        x = SMC.mvnormal_mixture_draw(θ_old, d; c = 0.5, α = 0.9, rng = Xoshiro(11))
        x, rand()
    end
    @test draws[1][1] == draws[2][1]
    Random.seed!(1)
    @test draws[1][2] == rand()
    # The default is the global RNG, as before
    Random.seed!(7)
    x_default = SMC.mvnormal_mixture_draw(θ_old, d; c = 0.5, α = 0.9)
    Random.seed!(7)
    @test x_default == SMC.mvnormal_mixture_draw(θ_old, d; c = 0.5, α = 0.9, rng = Random.default_rng())
end
