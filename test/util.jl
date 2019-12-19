import SparseArrays.SparseMatrixCSC

write_test_output = false
path = dirname(@__FILE__)

###################################################################
# Test: scalar_reshape()
###################################################################
s1 = SMC.scalar_reshape(1, 0)
s2 = SMC.scalar_reshape([1, 0], [1, 0])
###################################################################
@testset "Scalar reshape" begin
    @test s1 == Array{Float64,1}[[1.0], [0.0]]
    @test s2 == Array{Float64,1}[[1.0, 0.0], [1.0, 0.0]]
end


###################################################################
# Test: vector_reshape()
###################################################################
v1 = SMC.vector_reshape(1.0, 2.0)
v2 = SMC.vector_reshape([1, 0], 1, 0)
v3 = SMC.vector_reshape([1.0, 0.0], 1.0, 0.0)

if write_test_output
    JLD2.jldopen("reference/vector_reshape.jld2", true, true, true, IOStream) do file
        file["v1"] = v1
        file["v2"] = v2
        file["v3"] = v3
    end
end

test_v1, test_v2, test_v3 = JLD2.jldopen("reference/vector_reshape.jld2", "r") do file
    file["v1"], file["v2"], file["v3"]
end
###################################################################
@testset "Vector reshape" begin
    @test v1 == test_v1
    @test v2 == test_v2
    @test v3 == test_v3
end


###################################################################
# Test: scalar_reduce()
###################################################################
s1_r = SMC.scalar_reduce([s1]...)
s2_r = SMC.scalar_reduce([s2 for i in 1:5]...)

if write_test_output
    JLD2.jldopen("reference/scalar_reduce.jld2", true, true, true, IOStream) do file
        file["s1_r"] = s1_r
        file["s2_r"] = s2_r
    end
end

test_s1_r, test_s2_r = JLD2.jldopen("reference/scalar_reduce.jld2", "r") do file
    file["s1_r"], file["s2_r"]
end
###################################################################
@testset "Scalar reduce" begin
    @test s1_r == test_s1_r
    @test s2_r == test_s2_r
end


###################################################################
# Test: vector_reduce()
###################################################################
v1_r = SMC.vector_reduce([v1]...)
v2_r = SMC.vector_reduce([v2 for i in 1:5]...)
v3_r = SMC.vector_reduce([v3 for i in 1:5]...)

if write_test_output
    JLD2.jldopen("reference/vector_reduce.jld2", true, true, true, IOStream) do file
        file["v1_r"] = v1_r
        file["v2_r"] = v2_r
        file["v3_r"] = v3_r
    end
end

test_v1_r, test_v2_r, test_v3_r = JLD2.jldopen("reference/vector_reduce.jld2", "r") do file
    file["v1_r"], file["v2_r"], file["v3_r"]
end

###################################################################
@testset "Vector reduce" begin
    @test v1_r == test_v1_r
    @test v2_r == test_v2_r
    @test v3_r == test_v3_r
end

###################################################################
# Test: speye()
###################################################################
@testset "speye" begin
    @test SMC.speye(20) == SparseMatrixCSC{Float64}(I, 20, 20)
    @test SMC.speye(0)  == SparseMatrixCSC{Float64}(I, 0, 0)
    @test SMC.speye(Int64, 20) == SparseMatrixCSC{Int64}(I, 20, 20)
    @test SMC.speye(Complex{Float64}, 20) == SparseMatrixCSC{Complex{Float64}}(I, 20, 20)
end


###################################################################
# Test: <, min, max
###################################################################
@testset "<, min, max" begin
    @test 1 < 2
    @test 1 < Complex(2, 1)
    @test Complex(0.95, 0.5) < 1
    @test Complex(0.5, 20) < Complex(0.99, 0.5)

    @test min(1, 2) == 1
    @test min(Complex(0.99, 0.5), 1) == 0.99
    @test min(1, Complex(20, 0.5)) == 1
    @test min(Complex(0.99, 0.5), Complex(20, 0.5)) == 0.99

    @test max(1, 2) == 2
    @test max(Complex(0.99, 0.5), 1) == 1
    @test max(1, Complex(20, 0.5)) == 20
    @test max(Complex(0.99, 0.5), Complex(20, 0.5)) == 20
end
