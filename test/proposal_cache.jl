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
