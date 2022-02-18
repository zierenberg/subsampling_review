using Random
using LightGraphs
using ProgressMeter
using Distributions

function add_and_check!(hs, hc::Number, site::Int)::Bool
    hs[site] += 1
    if hs[site] >= hc
        return true
    else
        return false
    end
end

"""
add perturbation at random lattice site and let sytem relax until no further topplings
"""
function perturb_and_relax(rng::AbstractRNG, hs::Array{Int64}, lattice::SimpleGraph{Int64}, hc::Int64, p_dis::Float64;
        start_site = missing
    )
    topple_sites_over_time = Vector{Int}[]

    # select a random lattice site to perturb by adding extra grain
    if ismissing(start_site)
        site = rand(rng, vertices(lattice))
    else
        site = start_site
    end
    if add_and_check!(hs,hc,site)
        push!(topple_sites_over_time, [site])
    end

    # iterate over all topple sites that were created in last time step
    time = 1
    while length(topple_sites_over_time) >= time
        for site in topple_sites_over_time[time]
            # here happens the topple (add one grain to every neighboring sites)
            hs[site] -= hc
            # topple occasionally dissipates (required for lattice with periodic boundary conditions)
            if rand(rng) < 1-p_dis
                for neighbor_site in outneighbors(lattice, site)
                    if add_and_check!(hs,hc,neighbor_site)
                        #if true then this mean hs will topple (possible multiple counts here...)
                        if length(topple_sites_over_time) < time+1
                            push!(topple_sites_over_time, [neighbor_site])
                        elseif ! (neighbor_site in topple_sites_over_time[time+1])
                            push!(topple_sites_over_time[time+1], neighbor_site)
                        end
                    end
                end
            end
        end
        time += 1
    end

    return topple_sites_over_time
end

function add_external_input!(rng::AbstractRNG, topple_sites::Vector{Int}, hs, hc, P_input)
    num_input = rand(rng, P_input)
    for i in 1:num_input
        site = rand(rng, 1:length(hs))
        if add_and_check!(hs,hc,site)
            if ! (site in topple_sites)
                push!(topple_sites, site)
            end
        end
    end
end

function step!(rng::AbstractRNG, topple_sites::Vector{Int}, hs, lattice::SimpleGraph{Int64}, hc::Int64, p_dis::Float64)
    topple_sites_next_step = Int[]
    for site in topple_sites
        # here happens the topple (add one grain to every neighboring sites)
        hs[site] -= hc
        # topple occasionally dissipates (required for lattice with periodic
        # boundary conditions)
        if rand(rng) < 1-p_dis
            for neighbor_site in outneighbors(lattice, site)
                if add_and_check!(hs,hc,neighbor_site)
                    #avoid multiple occurances
                    if ! (neighbor_site in topple_sites_next_step)
                        push!(topple_sites_next_step, neighbor_site)
                    end
                end
            end
        end
    end
    return topple_sites_next_step
end
