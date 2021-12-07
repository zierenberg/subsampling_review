###########
# simple project activate
using Pkg
path_project=join(split(@__DIR__, "/")[1:end-1],"/")
println(path_project)
Pkg.activate(path_project)
###########
include("../src/asm.jl")
include("../src/branching_network.jl")
include("../src/utils.jl")

using Statistics
using LinearAlgebra
using RollingFunctions
using Printf
using StatsBase

"""
    produce_data_avalanches_sandpile()

runs the default simulation of an abelian sandpile model with cyclic (periodic)
boundary conditions. Parameter can be specified if desired.

Model is Abelian sandpile plus periodic boundary conditions and the chance
that toppling does not propagate grains (to make it subcritical)

In fact, check the definition in

[1]S. N. Majumdar and D. Dhar, Physica A: Statistical Mechanics and Its Applications 185, 129 (1992).

for abelian sandpile models (ABS) on periodic boundary conditions where the
dissipation is implemented via a sink to the system (symmetric and solves some
disambiguities)

For now stick to the solution by Levina and Priesemann, but let's keep thinking about an ambiguous solution

L=4;
Ls=(L,L);
seed=1000;
hc=4;

"""
function produce_data_avalanches_sandpile(;
        L::Int=Int(2^7), # linear extension of lattice
        Ls=(L,L),   # 2D lattice
        hc::Int=4,
        p_dis::Float64=1e-4, # dissipation probability per toppling (probably system size dependent)
        seed=1000,  # seed for random number generator
        num_avalanches::Int = Int(1e6),
        periodic=true
    )
    rng = MersenneTwister(seed)

    lattice = grid(Ls, periodic=periodic)
    hs = rand(rng, 0:hc-1, Ls)

    subsample = Dict{String, Any}("l" => Int[2^2, 2^4, 2^6])
    subsample["random"] = [Vector{Int}(undef,l*l) for l in subsample["l"]]
    subsample["window"] = [Vector{Int}(undef,l*l) for l in subsample["l"]]
    # fill the subsampling schemes
    for (i,l) in enumerate(subsample["l"])
        subsample["random"][i] .= sample(rng, vertices(lattice), length(subsample["random"][i]), replace=false)
        range =  floor(Int,L/2)-floor(Int,l/2)+1 : floor(Int,L/2)-floor(Int,l/2)+l
        subsample["window"][i] .= vcat(LinearIndices(size(hs))[range,range]...)
    end

    num_valid_avalanches = 0
    data = Dict()
    data["mean_h"] = Float64[]

    data["duration"] = Dict{String,Any}()
    data["duration"]["full"] = zeros(Int64, num_avalanches)

    data["size"] = Dict{String,Any}()
    data["size"]["full"] = zeros(Int64, num_avalanches)
    data["size"]["random"] = [zeros(Int64, num_avalanches) for i in 1:length(subsample["random"])]
    data["size"]["window"] = [zeros(Int64, num_avalanches) for i in 1:length(subsample["window"])]

    response_steps = 1000
    data["response"] = Dict{String, Any}()
    data["response"]["full"] = zeros(Int64, num_avalanches, response_steps)
    data["response"]["random"] = [zeros(Int64, num_avalanches, response_steps) for i in 1:length(subsample["random"])]
    data["response"]["window"] = [zeros(Int64, num_avalanches, response_steps) for i in 1:length(subsample["window"])]

    progress = Progress(num_avalanches, 1, "sampling: ", offset=0)
    while num_valid_avalanches < num_avalanches
        push!(data["mean_h"], mean(hs))
        # find smart way to check for stationary critical dynamics from state
        # hs (Dhar?) (do this after every 10 non-zero avalanches or so)
        topple_sites_over_time = perturb_and_relax(rng, hs, lattice, hc, p_dis)
        ProgressMeter.update!(progress, num_valid_avalanches)
        if length(topple_sites_over_time) > 0
            num_valid_avalanches += 1

            # full sampling
            data["duration"]["full"][num_valid_avalanches] = length(topple_sites_over_time)
            data["size"]["full"][num_valid_avalanches] = sum(length.(topple_sites_over_time))

            for (t,sites) in enumerate(topple_sites_over_time)
                if t <= response_steps
                    data["response"]["full"][num_valid_avalanches,t] = length(sites)
                end
                # random spatial subsampling
                for (i, test_sample) in enumerate(subsample["random"])
                    a_t = sum(in.(sites, Ref(test_sample)))
                    data["size"]["random"][i][num_valid_avalanches] += a_t
                    if t <= response_steps
                        data["response"]["random"][i][num_valid_avalanches,t] = a_t
                    end
                end

                # field-of-view spatial subsampling
                for (i, test_sample) in enumerate(subsample["window"])
                    a_t = sum(in.(sites, Ref(test_sample)))
                    data["size"]["window"][i][num_valid_avalanches] += a_t
                    if t <= response_steps
                        data["response"]["window"][i][num_valid_avalanches,t] = a_t
                    end
                end
            end
        end
    end

    return subsample, data
end
function produce_data_perturb_relax_bn(;
        ref=4,
        L::Int=Int(2^7), # linear extension of lattice
        Ls=(L,L),          # 1D lattice
        p_rec::Float64=0.644, #
        seed=1000,  # seed for random number generator
        num_avalanches::Int = Int(1e5),
        periodic=true
    )
    rng = MersenneTwister(seed)

    lattice = grid(Ls, periodic=periodic)
    state = zeros(Bool, Ls)
    state_ref = zeros(Int, Ls)
    state_new = zeros(Bool, Ls)

    subsample = Dict{String, Any}("l" => Int[2^2, 2^4, 2^6])
    subsample["random"] = [Vector{Int}(undef,l*l) for l in subsample["l"]]
    subsample["window"] = [Vector{Int}(undef,l*l) for l in subsample["l"]]
    # fill the subsampling schemes
    for (i,l) in enumerate(subsample["l"])
        subsample["random"][i] .= sample(rng, vertices(lattice), length(subsample["random"][i]), replace=false)
        range =  floor(Int,L/2)-floor(Int,l/2)+1 : floor(Int,L/2)-floor(Int,l/2)+l
        subsample["window"][i] .= vcat(LinearIndices(size(state))[range, range]...)
    end

    num_valid_avalanches = 0
    data = Dict()
    data["duration"] = Dict{String,Any}()
    data["duration"]["full"] = zeros(Int64, num_avalanches)

    data["size"] = Dict{String,Any}()
    data["size"]["full"] = zeros(Int64, num_avalanches)
    data["size"]["random"] = [zeros(Int64, num_avalanches) for i in 1:length(subsample["random"])]
    data["size"]["window"] = [zeros(Int64, num_avalanches) for i in 1:length(subsample["window"])]

    progress = Progress(num_avalanches, 1, "sampling: ", offset=0)
    for num_valid_avalanches in 1:num_avalanches
        next!(progress)

        # single perturbation
        state_ref .= 0
        state[rand(rng, vertices(lattice))] = true
        state_ref[state] .= ref
        activity = sum(state)
        while !(activity == 0)
            # full sampling
            data["duration"]["full"][num_valid_avalanches] += 1
            data["size"]["full"][num_valid_avalanches] += activity
            # random spatial subsampling
            for (i, test_sample) in enumerate(subsample["random"])
                a_t = sum(state[test_sample])
                data["size"]["random"][i][num_valid_avalanches] += a_t
            end

            # field-of-view spatial subsampling
            for (i, test_sample) in enumerate(subsample["window"])
                a_t = sum(state[test_sample])
                data["size"]["window"][i][num_valid_avalanches] += a_t
            end
            # next step
            step!(rng, state, state_new, state_ref, ref, lattice, p_rec)
            activity = sum(state)
        end
    end

    return data
end


"""
    produce_data_box_scaling_branching_network(m, h)

simulation of branching network on 2D lattice with nearest neighbor interaction.
Returns estimates of the connected correlation function for box scaling analysis.
"""
function produce_data_box_scaling_branching_network(m, h::Float64;
        L::Int=Int(2^10), # linear extension of lattice
        Ls=(L,L),        # 2D lattice
        seed=1000,  # seed for random number generator
        steps_total::Int = Int(1e5),
        steps_equil::Int = Int(4e4),
        dstep_correlation::Int = 100,
    )
    string(l) = @sprintf("l=%d",l)

    dt=1
    p_ext = 1-exp(-h*dt)

    # initialize random number generator
    rng = MersenneTwister(seed)

    # construct 2D lattice with additional neighbors along the diagonal
    lattice = lattice_nearest_neighbor(Ls[1],Ls[2]);
    p_rec = m/mean(degree(lattice));
    state = zeros(Bool, Ls);
    state_new = zeros(Bool, Ls);
    #state_ref = zeros(Int, Ls);
    #val_ref = 2;

    # setup the spatial subsampling schemes
    data = Dict{String, Any}()
    data["l"] = Int[L/2, L/2^2, L/2^3, L/2^4, L/2^5, L/2^6]
    for l in data["l"]
        data["random/"*string(l)] = sample(rng, vertices(lattice), l*l, replace=false)
        range =  floor(Int,L/2)-floor(Int,l/2)+1 : floor(Int,L/2)-floor(Int,l/2)+l
        data["window/"*string(l)] =  LinearIndices(size(state))[range,range]
    end

    # prepare output dictionary
    # number of active sites over time
    data["activity/full"] = zeros(Int64, steps_total)
    for l in data["l"]
        data["activity/random/"*string(l)] = zeros(Int64, steps_total)
        data["activity/window/"*string(l)] = zeros(Int64, steps_total)
    end
    # time average of spatial correlation function
    data["correlation/full_lag"] = collect(0:floor(Int,L/2))
    data["correlation/full"] = zeros(Float64, length(data["correlation/full_lag"]))
    for l in data["l"]
        l_range = collect(0:floor(Int,l/2))
        data["correlation/window_lag/"*string(l)] = l_range
        data["correlation/window/"*string(l)] = zeros(Float64, length(l_range))
    end

    # actual simulation loop
    progress = Progress(steps_total, 1, "sampling: ", offset=0)
    number_estimates_correlation = 0
    for s in 1:steps_total
        next!(progress)
        # external input independently added to state_new
        for i in 1:length(state)
            if rand(rng) < p_ext
                state_new[i] = true
            end
        end
        # recurrent update step
        step!(rng, state, state_new, lattice, p_rec)
        #step!(rng, state, state_new, lattice, p_rec, state_ref, val_ref)

        # activity
        data["activity/full"][s] = sum(state)
        for l in data["l"]
            # random spatial subsampling
            state_sub = state[data["random/"*string(l)]]
            data["activity/random/"*string(l)][s] = sum(state_sub)
            # field-of-view spatial subsampling
            state_sub = state[data["window/"*string(l)]]
            data["activity/window/"*string(l)][s] = sum(state_sub)
        end

        # start measurements after initial equilibration
        # (needs to be checked afterwards with the time trance of active sites)
        if s > steps_equil
            # correlation estimates
            if s%dstep_correlation ==0
                number_estimates_correlation += 1
                state_cor = state
                data["correlation/full"] .+= correlation_x(state_cor,data["correlation/full_lag"])
                for l in data["l"]
                    state_cor = state[data["window/"*string(l)]]
                    data["correlation/window/"*string(l)] .+= correlation_x(state_cor, data["correlation/window_lag/"*string(l)])
                end
            end
        end
    end
    # normalization of correlation function estimate
    data["correlation/full"] ./= number_estimates_correlation
    for l in data["l"]
        data["correlation/window/"*string(l)] ./= number_estimates_correlation
    end

    return data
end
